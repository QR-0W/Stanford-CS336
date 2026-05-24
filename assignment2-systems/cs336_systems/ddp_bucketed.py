"""分布式数据并行（DDP）的 bucketed overlap 实现。

将需要梯度的参数按大小分桶，每个桶内的梯度 ready 后立即启动异步
all-reduce，实现梯度通信与反向传播计算的 overlap，从而减少训练步骤
末尾的通信尾部等待时间。

核心设计：
    - 参数按反向传播顺序（从末尾到开头）逆序分桶，因为 backward 一般
      从模型末尾往前生成梯度，逆序可以更早"收集齐"一个桶。
    - 每个参数注册 ``post_accumulate_grad_hook``，在该参数的梯度累加完成
      后触发桶就绪检查，一旦桶内所有参数梯度 ready 就启动 all-reduce。
    - 桶大小由 ``bucket_size_mb`` 控制：太小则通信次数多（开销大），
      太大则 overlap 机会少（尾部等待长）。
"""

from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.distributed as dist
import torch.nn as nn
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors


class DDPBucketed(nn.Module):
    def __init__(self, module: nn.Module, bucket_size_mb: float):
        super().__init__()
        self.module = module
        self.bucket_size_mb = float(bucket_size_mb)
        self._bucket_size_bytes = int(self.bucket_size_mb * 1024 * 1024)
        self._params = [p for p in self.module.parameters() if p.requires_grad]
        self._hook_handles: list[Any] = []

        for p in self.module.parameters():
            dist.broadcast(p.data, src=0)

        # Use reverse parameter order because backward generally produces gradients
        # from the end of the model toward the beginning.
        self._buckets: list[list[torch.nn.Parameter]] = self._build_buckets(
            list(reversed(self._params)), self._bucket_size_bytes
        )
        self._param_to_bucket_idx = {
            id(p): bucket_idx
            for bucket_idx, bucket in enumerate(self._buckets)
            for p in bucket
        }
        self._reset_bucket_state()

        for p in self._params:
            if hasattr(p, "register_post_accumulate_grad_hook"):
                handle = p.register_post_accumulate_grad_hook(self._make_post_acc_hook(p))
            else:
                handle = p.register_hook(self._make_grad_hook(p))
            self._hook_handles.append(handle)

    @staticmethod
    def _build_buckets(
        params: Sequence[torch.nn.Parameter],
        bucket_size_bytes: int,
    ) -> list[list[torch.nn.Parameter]]:
        if bucket_size_bytes <= 0:
            return [[p for p in params]]

        buckets: list[list[torch.nn.Parameter]] = []
        cur: list[torch.nn.Parameter] = []
        cur_bytes = 0

        for p in params:
            p_bytes = p.numel() * p.element_size()
            if cur and cur_bytes + p_bytes > bucket_size_bytes:
                buckets.append(cur)
                cur = []
                cur_bytes = 0
            cur.append(p)
            cur_bytes += p_bytes

        if cur:
            buckets.append(cur)

        return buckets

    def on_train_batch_start(self) -> None:
        self._reset_bucket_state()

    def _reset_bucket_state(self) -> None:
        self._bucket_ready_counts = [0 for _ in self._buckets]
        self._bucket_seen_params: list[set[int]] = [set() for _ in self._buckets]
        self._bucket_handles: list[dist.Work | None] = [None for _ in self._buckets]
        self._bucket_flats: list[torch.Tensor | None] = [None for _ in self._buckets]
        self._bucket_grads: list[list[torch.Tensor] | None] = [None for _ in self._buckets]

    def _make_post_acc_hook(self, param: torch.nn.Parameter):
        def hook(_: torch.Tensor) -> None:
            self._mark_param_ready(param)

        return hook

    def _make_grad_hook(self, param: torch.nn.Parameter):
        def hook(grad: torch.Tensor) -> torch.Tensor:
            # Older PyTorch versions call this before .grad is populated, so
            # this fallback cannot overlap safely; finish() handles any missing
            # bucket synchronously.
            return grad

        return hook

    def _mark_param_ready(self, param: torch.nn.Parameter) -> None:
        bucket_idx = self._param_to_bucket_idx[id(param)]
        pid = id(param)
        if pid in self._bucket_seen_params[bucket_idx]:
            return
        self._bucket_seen_params[bucket_idx].add(pid)
        self._bucket_ready_counts[bucket_idx] += 1

        bucket = self._buckets[bucket_idx]
        if self._bucket_ready_counts[bucket_idx] == len(bucket):
            self._launch_bucket_all_reduce(bucket_idx)

    def _launch_bucket_all_reduce(self, bucket_idx: int) -> None:
        if self._bucket_handles[bucket_idx] is not None:
            return
        bucket = self._buckets[bucket_idx]
        grads = [p.grad for p in bucket if p.grad is not None]
        if not grads:
            return

        flat = _flatten_dense_tensors(grads)
        handle = dist.all_reduce(flat, op=dist.ReduceOp.SUM, async_op=True)
        self._bucket_handles[bucket_idx] = handle
        self._bucket_flats[bucket_idx] = flat
        self._bucket_grads[bucket_idx] = grads

    def finish_gradient_synchronization(self) -> None:
        world_size = dist.get_world_size()

        for bucket_idx, bucket in enumerate(self._buckets):
            if self._bucket_handles[bucket_idx] is None:
                # Fallback for unusual cases where hooks were unavailable or a
                # bucket did not launch during backward.
                self._launch_bucket_all_reduce(bucket_idx)

            handle = self._bucket_handles[bucket_idx]
            flat = self._bucket_flats[bucket_idx]
            grads = self._bucket_grads[bucket_idx]
            if handle is None or flat is None or grads is None:
                continue

            handle.wait()
            flat.div_(world_size)
            synced = _unflatten_dense_tensors(flat, grads)

            for grad, synced_grad in zip(grads, synced):
                grad.copy_(synced_grad)

        self._reset_bucket_state()

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
