from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from cs336_scaling.scaling_config import (
    TrainingConfig,
    dedupe_configs,
    default_model_shapes,
    make_config,
)


@dataclass(frozen=True)
class QueryPlan:
    configs: list[tuple[str, TrainingConfig]]

    @property
    def total_flops(self) -> int:
        return sum(config.train_flops for _, config in self.configs)

    def by_stage(self) -> dict[str, list[TrainingConfig]]:
        stages: dict[str, list[TrainingConfig]] = defaultdict(list)
        for stage, config in self.configs:
            stages[stage].append(config)
        return dict(stages)

    def to_records(self) -> list[dict[str, int | float | str]]:
        return [config.to_record(stage=stage) for stage, config in self.configs]


def build_initial_query_plan() -> QueryPlan:
    shapes = default_model_shapes()
    configs: list[tuple[str, TrainingConfig]] = []

    # Cheap hyperparameter pilot: tune LR and batch size before high-cost curves.
    pilot_shapes = [shapes[i] for i in [1, 2, 3, 4, 5]]
    for shape in pilot_shapes:
        for batch_size in [128, 256]:
            for learning_rate in [1e-3, 6e-4, 3e-4, 1e-4]:
                configs.append(
                    (
                        "pilot_hparams",
                        make_config(
                            shape=shape,
                            train_flops=int(1e15),
                            batch_size=batch_size,
                            learning_rate=learning_rate,
                        ),
                    )
                )

    # Baseline IsoFLOPs sweep. The LR/batch-size choice is intentionally a
    # conservative placeholder; after the pilot, rerun this plan with the best
    # values if they differ.
    default_batch_size = 128
    default_learning_rate = 6e-4
    stage1_shape_indices_by_flops = {
        int(1e15): [0, 1, 2, 3, 4, 5, 6],
        int(3e15): [1, 2, 3, 4, 5, 6, 7],
        int(1e16): [2, 3, 4, 5, 6, 7, 8],
        int(3e16): [3, 4, 5, 6, 7, 8, 9],
    }
    for train_flops, shape_indices in stage1_shape_indices_by_flops.items():
        for index in shape_indices:
            configs.append(
                (
                    "stage1_isoflops",
                    make_config(
                        shape=shapes[index],
                        train_flops=train_flops,
                        batch_size=default_batch_size,
                        learning_rate=default_learning_rate,
                    ),
                )
            )

    return QueryPlan(configs=dedupe_configs(configs))


def build_local_self_study_query_plan() -> QueryPlan:
    """Return a fuller local plan that still stays below the 2e18 FLOP cap."""
    shapes = default_model_shapes()
    configs = list(build_initial_query_plan().configs)

    default_learning_rate = 6e-4
    stage2 = {
        int(6e16): [2, 3, 4, 5],
        int(1e17): [2, 3, 4, 5, 6],
        int(3e17): [3, 4, 5],
    }
    for train_flops, shape_indices in stage2.items():
        for index in shape_indices:
            shape = shapes[index]
            batch_size = 256 if shape.non_embedding_params >= 6e7 else 128
            configs.append(
                (
                    "stage2_high_compute",
                    make_config(
                        shape=shape,
                        train_flops=train_flops,
                        batch_size=batch_size,
                        learning_rate=default_learning_rate,
                    ),
                )
            )

    return QueryPlan(configs=dedupe_configs(configs))


def summarize_plan(plan: QueryPlan) -> dict[str, object]:
    stage_summaries = []
    for stage, configs in plan.by_stage().items():
        stage_summaries.append(
            {
                "stage": stage,
                "num_configs": len(configs),
                "total_flops": sum(config.train_flops for config in configs),
            }
        )
    return {
        "num_configs": len(plan.configs),
        "total_flops": plan.total_flops,
        "stages": stage_summaries,
    }
