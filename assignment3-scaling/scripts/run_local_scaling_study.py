from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cs336_scaling.api import save_runs
from cs336_scaling.local_api import LocalTrainingApi
from cs336_scaling.scaling_plan import build_local_self_study_query_plan, summarize_plan


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a local self-study version of the CS336 scaling_laws experiment."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/scaling_laws"))
    parser.add_argument("--budget", type=float, default=2e18)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan = build_local_self_study_query_plan()
    summary = summarize_plan(plan)
    if plan.total_flops > args.budget:
        raise SystemExit(
            f"Plan costs {plan.total_flops:.3e} FLOPs, exceeding budget {args.budget:.3e}."
        )

    api = LocalTrainingApi()
    runs = []
    for stage, config in plan.configs:
        record = api.loss(config)
        record["stage"] = stage
        runs.append(record)

    runs_path = args.output_dir / "local_runs.json"
    plan_path = args.output_dir / "local_query_plan.json"
    save_runs(runs_path, runs)
    plan_path.write_text(
        json.dumps({"summary": summary, "configs": plan.to_records()}, indent=2, sort_keys=True)
    )

    print(json.dumps(summary, indent=2))
    print(f"Wrote local runs to {runs_path}")
    print(f"Wrote local plan to {plan_path}")


if __name__ == "__main__":
    main()
