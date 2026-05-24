"""DDP 的逐参数异步梯度同步实现 (overlap individual)。

与 naive DDP（all-reduce 集中在 backward 之后）不同，
本实现为每个参数注册 ``post_accumulate_grad_hook``，在梯度累加完成后
立即对该参数执行异步 all-reduce，从而将通信与后续层的 backward 计算
overlap，减少训练步骤末尾的通信尾部等待。

注意：逐参数同步会产生大量小粒度通信（每个参数一次 all-reduce），
通信调用开销较大。后续 ``DDPBucketed`` 通过分桶解决了这个问题。
"""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn


class DDPIndividualParameters(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self._handles: list[dist.Work] = []
        self._hook_handles: list[Any] = []
        self._synced_param_ids: set[int] = set()

        # Ensure all ranks start from rank-0 weights.
        for p in self.module.parameters():
            dist.broadcast(p.data, src=0)

        # Register gradient hooks for async per-parameter all-reduce.
        for p in self.module.parameters():
            if not p.requires_grad:
                continue
            if hasattr(p, "register_post_accumulate_grad_hook"):
                h = p.register_post_accumulate_grad_hook(self._make_post_acc_hook(p))
            else:
                h = p.register_hook(self._make_grad_hook(p))
            self._hook_handles.append(h)

    def _make_post_acc_hook(self, param: torch.nn.Parameter):
        def hook(_: torch.Tensor) -> None:
            self._enqueue_all_reduce(param)

        return hook

    def _make_grad_hook(self, param: torch.nn.Parameter):
        def hook(grad: torch.Tensor) -> torch.Tensor:
            self._enqueue_all_reduce(param)
            return grad

        return hook

    def _enqueue_all_reduce(self, param: torch.nn.Parameter) -> None:
        if param.grad is None:
            return
        pid = id(param)
        if pid in self._synced_param_ids:
            return
        self._synced_param_ids.add(pid)
        handle = dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=True)
        self._handles.append(handle)

    def finish_gradient_synchronization(self) -> None:
        for h in self._handles:
            h.wait()
        self._handles.clear()

        world_size = dist.get_world_size()
        for p in self.module.parameters():
            if p.grad is not None:
                p.grad.div_(world_size)

        self._synced_param_ids.clear()

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
