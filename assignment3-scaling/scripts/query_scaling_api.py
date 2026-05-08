from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cs336_scaling.api import (
    API_KEY_ENV,
    ScalingApiClient,
    config_key,
    load_runs,
    merge_runs,
    save_runs,
)
from cs336_scaling.scaling_config import TrainingConfig


def config_from_record(record: dict[str, int | float | str]) -> TrainingConfig:
    return TrainingConfig(
        d_model=int(record["d_model"]),
        num_layers=int(record["num_layers"]),
        num_heads=int(record["num_heads"]),
        batch_size=int(record["batch_size"]),
        learning_rate=float(record["learning_rate"]),
        train_flops=int(float(record["train_flops"])),
    )


def load_plan(path: Path, stage: str | None) -> list[tuple[str, TrainingConfig]]:
    with path.open() as f:
        payload = json.load(f)
    records = payload["configs"] if isinstance(payload, dict) and "configs" in payload else payload
    configs: list[tuple[str, TrainingConfig]] = []
    for record in records:
        record_stage = str(record.get("stage", "unknown"))
        if stage is not None and record_stage != stage:
            continue
        configs.append((record_stage, config_from_record(record)))
    return configs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Safely query the CS336 scaling-laws training API from a JSON plan."
    )
    parser.add_argument("--plan", type=Path, default=Path("results/scaling_laws/initial_query_plan.json"))
    parser.add_argument("--cache", type=Path, default=Path("results/scaling_laws/api_runs.json"))
    parser.add_argument("--stage", type=str, default=None, help="Only query one stage from the plan.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of new configs to query.")
    parser.add_argument(
        "--max-new-flops",
        type=float,
        default=0.0,
        help="Hard cap for newly-spent FLOPs in this invocation. Required with --execute.",
    )
    parser.add_argument("--execute", action="store_true", help="Actually call /loss and spend budget.")
    parser.add_argument("--status", action="store_true", help="Only print API status and previous-run count.")
    args = parser.parse_args()

    cached_runs = load_runs(args.cache)
    previous_runs = []
    client = None

    if args.execute or args.status or os.environ.get(API_KEY_ENV):
        client = ScalingApiClient.from_env()
        previous_runs = client.previous_runs()

    known_runs = merge_runs(previous_runs, cached_runs)
    known_keys = {config_key(run) for run in known_runs}

    if client is not None:
        print(f"API total_flops_used: {client.total_flops_used()}")
    else:
        print(f"API total_flops_used: unavailable; set {API_KEY_ENV} to include API state")
    print(f"previous_runs: {len(previous_runs)}")
    print(f"cached/known runs: {len(known_runs)}")
    if args.status:
        save_runs(args.cache, known_runs)
        print(f"Updated cache at {args.cache}")
        return

    planned_configs = load_plan(args.plan, args.stage)
    new_configs = [
        (stage, config) for stage, config in planned_configs if config.key() not in known_keys
    ]
    if args.limit is not None:
        new_configs = new_configs[: args.limit]

    planned_new_flops = sum(config.train_flops for _, config in new_configs)
    print(f"plan configs after stage/limit filters: {len(planned_configs)}")
    print(f"new configs to query: {len(new_configs)}")
    print(f"planned new FLOPs: {planned_new_flops:.3e}")

    if not args.execute:
        print("Dry run only. Add --execute and --max-new-flops to query /loss.")
        return

    if client is None:
        client = ScalingApiClient.from_env()

    if args.max_new_flops <= 0:
        raise SystemExit("--execute requires --max-new-flops > 0")
    if planned_new_flops > args.max_new_flops:
        raise SystemExit(
            f"Refusing to spend {planned_new_flops:.3e} FLOPs; "
            f"this exceeds --max-new-flops={args.max_new_flops:.3e}."
        )

    results = []
    spent = 0
    for stage, config in new_configs:
        result = client.loss(config)
        result["stage"] = stage
        results.append(result)
        spent += config.train_flops
        known_runs = merge_runs(known_runs, results)
        save_runs(args.cache, known_runs)
        print(
            f"queried stage={stage} C={config.train_flops:.0e} "
            f"N={config.non_embedding_params:.3e} loss={result['loss']:.6f} "
            f"spent_this_run={spent:.3e}"
        )

    print(f"Finished. Newly spent FLOPs: {spent:.3e}")
    print(f"Updated cache at {args.cache}")


if __name__ == "__main__":
    main()
