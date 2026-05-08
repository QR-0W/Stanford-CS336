from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable


ALLOWED_TRAIN_FLOPS = [
    int(1e13),
    int(3e13),
    int(6e13),
    int(1e14),
    int(3e14),
    int(6e14),
    int(1e15),
    int(3e15),
    int(6e15),
    int(1e16),
    int(3e16),
    int(6e16),
    int(1e17),
    int(3e17),
    int(6e17),
    int(1e18),
]


@dataclass(frozen=True)
class ModelShape:
    num_layers: int
    d_model: int
    num_heads: int

    @property
    def non_embedding_params(self) -> int:
        return estimate_non_embedding_params(self.num_layers, self.d_model)

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class TrainingConfig:
    d_model: int
    num_layers: int
    num_heads: int
    batch_size: int
    learning_rate: float
    train_flops: int

    @property
    def non_embedding_params(self) -> int:
        return estimate_non_embedding_params(self.num_layers, self.d_model)

    def key(self) -> tuple[int, int, int, int, str, int]:
        return (
            self.d_model,
            self.num_layers,
            self.num_heads,
            self.batch_size,
            f"{self.learning_rate:.12g}",
            self.train_flops,
        )

    def to_api_params(self, api_key: str) -> dict[str, int | float | str]:
        params = asdict(self)
        params["api_key"] = api_key
        return params

    def to_record(self, stage: str | None = None) -> dict[str, int | float | str]:
        record = asdict(self)
        record["non_embedding_params"] = self.non_embedding_params
        if stage is not None:
            record["stage"] = stage
        return record


def estimate_non_embedding_params(num_layers: int, d_model: int) -> int:
    return 12 * num_layers * d_model * d_model


def choose_num_heads(d_model: int) -> int:
    target_heads = min(16, max(2, round(d_model / 64)))
    valid_heads = [h for h in range(2, 17) if d_model % h == 0]
    if not valid_heads:
        raise ValueError(f"No valid num_heads in [2, 16] divides d_model={d_model}")
    return min(valid_heads, key=lambda h: (abs(h - target_heads), abs(d_model / h - 64)))


def default_model_shapes() -> list[ModelShape]:
    # A compact log-spaced grid over the API's allowed architecture range.
    layer_width_pairs = [
        (2, 64),
        (4, 128),
        (6, 192),
        (8, 256),
        (10, 384),
        (12, 512),
        (16, 640),
        (20, 768),
        (24, 896),
        (24, 1024),
    ]
    return [
        ModelShape(num_layers=layers, d_model=d_model, num_heads=choose_num_heads(d_model))
        for layers, d_model in layer_width_pairs
    ]


def all_valid_model_shapes() -> list[ModelShape]:
    shapes: list[ModelShape] = []
    for num_layers in range(2, 25):
        for d_model in range(64, 1025):
            valid_heads = [
                h for h in range(2, 17) if d_model % h == 0 and 32 <= d_model / h <= 128
            ]
            if not valid_heads:
                continue
            num_heads = min(valid_heads, key=lambda h: abs(d_model / h - 64))
            shapes.append(
                ModelShape(
                    num_layers=num_layers,
                    d_model=d_model,
                    num_heads=num_heads,
                )
            )
    return shapes


def nearest_shape(target_params: float, shapes: Iterable[ModelShape] | None = None) -> ModelShape:
    shape_list = list(all_valid_model_shapes() if shapes is None else shapes)
    return min(
        shape_list,
        key=lambda shape: abs(
            _safe_log_ratio(shape.non_embedding_params, max(float(target_params), 1.0))
        ),
    )


def _safe_log_ratio(a: float, b: float) -> float:
    import math

    return math.log(max(a, 1.0)) - math.log(max(b, 1.0))


def make_config(
    shape: ModelShape,
    train_flops: int,
    batch_size: int,
    learning_rate: float,
) -> TrainingConfig:
    if train_flops not in ALLOWED_TRAIN_FLOPS:
        raise ValueError(f"train_flops={train_flops} is not one of the API's allowed values")
    if batch_size not in {128, 256}:
        raise ValueError("batch_size must be 128 or 256")
    if not (1e-4 <= learning_rate <= 1e-3):
        raise ValueError("learning_rate must be in [1e-4, 1e-3]")
    return TrainingConfig(
        d_model=shape.d_model,
        num_layers=shape.num_layers,
        num_heads=shape.num_heads,
        batch_size=batch_size,
        learning_rate=learning_rate,
        train_flops=train_flops,
    )


def dedupe_configs(configs: Iterable[tuple[str, TrainingConfig]]) -> list[tuple[str, TrainingConfig]]:
    seen: set[tuple[int, int, int, int, str, int]] = set()
    deduped: list[tuple[str, TrainingConfig]] = []
    for stage, config in configs:
        key = config.key()
        if key in seen:
            continue
        seen.add(key)
        deduped.append((stage, config))
    return deduped
