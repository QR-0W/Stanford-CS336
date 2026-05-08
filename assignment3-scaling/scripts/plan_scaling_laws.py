from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cs336_scaling.scaling_plan import build_initial_query_plan, summarize_plan


def format_billions(value: float) -> str:
    if value >= 1e9:
        return f"{value / 1e9:.2f}B"
    if value >= 1e6:
        return f"{value / 1e6:.2f}M"
    return f"{value:.0f}"


def write_markdown(records: list[dict[str, int | float | str]], output_path: Path) -> None:
    lines = [
        "# Scaling Laws Query Plan",
        "",
        "This plan is a staged starting point. Do not execute the entire plan before inspecting pilot results.",
        "",
        "| stage | C | params | layers | d_model | heads | batch | lr |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for record in records:
        lines.append(
            "| "
            f"{record['stage']} | "
            f"{record['train_flops']:.0e} | "
            f"{format_billions(float(record['non_embedding_params']))} | "
            f"{record['num_layers']} | "
            f"{record['d_model']} | "
            f"{record['num_heads']} | "
            f"{record['batch_size']} | "
            f"{record['learning_rate']:.1e} |"
        )
    output_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a cautious query plan for scaling_laws.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/scaling_laws"),
        help="Directory to write the plan files.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan = build_initial_query_plan()
    records = plan.to_records()
    summary = summarize_plan(plan)
    payload = {"summary": summary, "configs": records}

    json_path = args.output_dir / "initial_query_plan.json"
    md_path = args.output_dir / "initial_query_plan.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    write_markdown(records, md_path)

    print(json.dumps(summary, indent=2))
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
