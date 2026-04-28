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
