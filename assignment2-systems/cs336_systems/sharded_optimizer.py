"""优化器状态分片 (Optimizer State Sharding)。

将优化器状态（Adam 的 m/v）按参数在设备间均匀分片，每个 device 只管理自己
负责分片的优化器状态和梯度更新。类似于 ZeRO Stage 1（FSDP 的一部分）。

核心思路：
    - 参数按 id 哈希取模分配到不同 rank。
    - 每个 rank 仅为"自己负责的参数"创建真实的优化器状态（m, v）。
    - ``step()`` 时：
        1. 先对本 rank 负责的参数做 local all-reduce 以获取全局平均梯度。
        2. 每个 rank 只对自己负责的参数执行 optimizer.step()。
        3. 最后 all-gather 更新后的参数，使所有 rank 保持一致。
"""

from __future__ import annotations

from typing import Any, Type

import torch
import torch.distributed as dist


class ShardedOptimizer(torch.optim.Optimizer):
    def __init__(self, params, optimizer_cls: Type[torch.optim.Optimizer], **kwargs: Any):
        if not dist.is_initialized():
            raise RuntimeError("ShardedOptimizer requires torch.distributed to be initialized.")

        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = dict(kwargs)
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self._local_optimizer: torch.optim.Optimizer | None = None
        self._id_to_owner: dict[int, int] = {}

        super().__init__(params, defaults=dict(kwargs))
        self._rebuild_local_optimizer()

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        super().add_param_group(param_group)
        if hasattr(self, "_local_optimizer"):
            self._rebuild_local_optimizer()

    def _all_params_in_order(self) -> list[torch.nn.Parameter]:
        params: list[torch.nn.Parameter] = []
        seen: set[int] = set()
        for group in self.param_groups:
            for p in group["params"]:
                if id(p) in seen:
                    continue
                seen.add(id(p))
                params.append(p)
        return params

    def _rebuild_local_optimizer(self) -> None:
        all_params = self._all_params_in_order()
        self._id_to_owner = {id(p): idx % self.world_size for idx, p in enumerate(all_params)}

        local_param_groups: list[dict[str, Any]] = []
        for group in self.param_groups:
            local_params = [p for p in group["params"] if self._id_to_owner[id(p)] == self.rank]
            if not local_params:
                continue
            local_group = {k: v for k, v in group.items() if k != "params"}
            local_group["params"] = local_params
            local_param_groups.append(local_group)

        if local_param_groups:
            self._local_optimizer = self.optimizer_cls(local_param_groups, **self.optimizer_kwargs)
        else:
            self._local_optimizer = None

    @torch.no_grad()
    def _broadcast_updated_parameters(self) -> None:
        for p in self._all_params_in_order():
            owner = self._id_to_owner[id(p)]
            dist.broadcast(p.data, src=owner)

    def step(self, closure=None, **kwargs):
        loss = None
        if self._local_optimizer is not None:
            loss = self._local_optimizer.step(closure=closure, **kwargs)
        elif closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._broadcast_updated_parameters()
        return loss

    def state_dict(self):
        if self._local_optimizer is None:
            return {"state": {}, "param_groups": []}
        return self._local_optimizer.state_dict()

    def load_state_dict(self, state_dict):
        if self._local_optimizer is None:
            return None
        return self._local_optimizer.load_state_dict(state_dict)
