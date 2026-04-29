from __future__ import annotations

from typing import Sequence

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

        for p in self.module.parameters():
            dist.broadcast(p.data, src=0)

        self._buckets: list[list[torch.nn.Parameter]] = self._build_buckets(self._params, self._bucket_size_bytes)

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
        # No async state to reset in this minimal bucketed implementation.
        return None

    def finish_gradient_synchronization(self) -> None:
        world_size = dist.get_world_size()

        for bucket in self._buckets:
            grads = [p.grad for p in bucket if p.grad is not None]
            if not grads:
                continue

            flat = _flatten_dense_tensors(grads)
            dist.all_reduce(flat, op=dist.ReduceOp.SUM, async_op=False)
            flat.div_(world_size)
            synced = _unflatten_dense_tensors(flat, grads)

            idx = 0
            for p in bucket:
                if p.grad is None:
                    continue
                p.grad.copy_(synced[idx])
                idx += 1

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
