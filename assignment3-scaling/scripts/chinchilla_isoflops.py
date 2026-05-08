from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cs336_scaling.isoflops import (
    IsoFlopsOptimum,
    PowerLawFit,
    fit_isoflops_scaling_laws,
    load_isoflops_runs,
    make_summary,
    select_isoflops_optima,
)


def format_scientific(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}e}"


def plot_scaling_law(
    optima: list[IsoFlopsOptimum],
    fit: PowerLawFit,
    y_attr: str,
    ylabel: str,
    title: str,
    output_base: Path,
    max_compute: float,
) -> None:
    compute = np.array([point.compute_budget for point in optima], dtype=np.float64)
    observed = np.array([getattr(point, y_attr) for point in optima], dtype=np.float64)
    fit_x = np.logspace(np.log10(compute.min()), np.log10(max_compute), 256)
    fit_y = fit.predict(fit_x)

    plt.figure(figsize=(7.0, 5.0))
    plt.loglog(compute, observed, "o", label="IsoFLOPs optima used for fit")
    plt.loglog(
        fit_x,
        fit_y,
        "-",
        label=rf"Fit: $y = {fit.coefficient:.2e} C^{{{fit.exponent:.3f}}}$",
    )
    plt.xlabel("Training compute budget C (FLOPs)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, which="both", linestyle=":", linewidth=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_base.with_suffix(".png"), dpi=200)
    plt.savefig(output_base.with_suffix(".pdf"))
    plt.close()


def write_markdown_summary(summary: dict[str, object], output_path: Path) -> None:
    lines = [
        "# Chinchilla IsoFLOPs Results",
        "",
        "## IsoFLOPs Optima",
        "",
        "| C (FLOPs) | N_opt parameters | D_opt tokens | final loss |",
        "| ---: | ---: | ---: | ---: |",
    ]

    for point in summary["optima"]:
        lines.append(
            "| "
            f"{format_scientific(point['compute_budget'])} | "
            f"{format_scientific(point['parameters'])} | "
            f"{format_scientific(point['dataset_tokens'])} | "
            f"{point['final_loss']:.6f} |"
        )

    parameter_fit = summary["parameter_fit"]
    dataset_fit = summary["dataset_fit"]
    lines.extend(
        [
            "",
            "## Fitted Scaling Laws",
            "",
            f"- Model size: `N_opt(C) = {parameter_fit['coefficient']:.6e} * C^{parameter_fit['exponent']:.6f}`",
            f"- Dataset size: `D_opt(C) = {dataset_fit['coefficient']:.6e} * C^{dataset_fit['exponent']:.6f}`",
            "",
            "## Predictions",
            "",
            "| C (FLOPs) | predicted N_opt parameters | predicted D_opt tokens |",
            "| ---: | ---: | ---: |",
        ]
    )

    for prediction in summary["predictions"]:
        lines.append(
            "| "
            f"{format_scientific(prediction['compute_budget'])} | "
            f"{format_scientific(prediction['parameters'])} | "
            f"{format_scientific(prediction['dataset_tokens'])} |"
        )

    lines.append("")
    output_path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce Chinchilla IsoFLOPs scaling-law fits from synthetic runs."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/isoflops_curves.json"),
        help="Path to the IsoFLOPs JSON data.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/chinchilla_isoflops"),
        help="Directory where plots and summaries will be written.",
    )
    parser.add_argument(
        "--target-budgets",
        type=float,
        nargs="+",
        default=[1e23, 1e24],
        help="Compute budgets to extrapolate to.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    runs = load_isoflops_runs(args.data)
    optima = select_isoflops_optima(runs)
    parameter_fit, dataset_fit = fit_isoflops_scaling_laws(optima)
    summary = make_summary(optima, parameter_fit, dataset_fit, args.target_budgets)

    max_compute = max(args.target_budgets)
    plot_scaling_law(
        optima,
        parameter_fit,
        y_attr="parameters",
        ylabel="Compute-optimal model size N",
        title="IsoFLOPs Model-Size Scaling Law",
        output_base=args.output_dir / "model_size_scaling",
        max_compute=max_compute,
    )
    plot_scaling_law(
        optima,
        dataset_fit,
        y_attr="dataset_tokens",
        ylabel="Compute-optimal dataset size D (tokens)",
        title="IsoFLOPs Dataset-Size Scaling Law",
        output_base=args.output_dir / "dataset_size_scaling",
        max_compute=max_compute,
    )

    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    write_markdown_summary(summary, args.output_dir / "summary.md")

    print("IsoFLOPs optima:")
    for point in summary["optima"]:
        print(
            f"  C={format_scientific(point['compute_budget'])}, "
            f"N_opt={format_scientific(point['parameters'])}, "
            f"D_opt={format_scientific(point['dataset_tokens'])}, "
            f"loss={point['final_loss']:.6f}"
        )
    print()
    print(
        "Model-size fit: "
        f"N_opt(C) = {parameter_fit.coefficient:.6e} * C^{parameter_fit.exponent:.6f}"
    )
    print(
        "Dataset-size fit: "
        f"D_opt(C) = {dataset_fit.coefficient:.6e} * C^{dataset_fit.exponent:.6f}"
    )
    print()
    print("Predictions:")
    for prediction in summary["predictions"]:
        print(
            f"  C={format_scientific(prediction['compute_budget'])}: "
            f"N_opt={format_scientific(prediction['parameters'])}, "
            f"D_opt={format_scientific(prediction['dataset_tokens'])}"
        )
    print(f"\nWrote outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
