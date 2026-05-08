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

from cs336_scaling.api import load_runs
from cs336_scaling.api_fit import (
    fit_loss_scaling,
    fit_model_size_scaling,
    select_best_by_compute,
)
from cs336_scaling.scaling_config import nearest_shape


def format_size(value: float) -> str:
    if value >= 1e9:
        return f"{value / 1e9:.2f}B"
    if value >= 1e6:
        return f"{value / 1e6:.2f}M"
    return f"{value:.0f}"


def plot_model_fit(optima, fit, target_compute: float, output_path: Path) -> None:
    compute = np.array([point.train_flops for point in optima], dtype=np.float64)
    params = np.array([point.non_embedding_params for point in optima], dtype=np.float64)
    xs = np.logspace(np.log10(compute.min()), np.log10(target_compute), 256)
    plt.figure(figsize=(7, 5))
    plt.loglog(compute, params, "o", label="observed IsoFLOPs optima")
    plt.loglog(xs, fit.predict(xs), "-", label=f"N={fit.coefficient:.2e} C^{fit.exponent:.3f}")
    plt.xlabel("Training compute C (FLOPs)")
    plt.ylabel("Non-embedding parameters N")
    plt.title("IsoFLOPs Model-Size Fit")
    plt.grid(True, which="both", linestyle=":")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_loss_fit(optima, fit, target_compute: float, output_path: Path) -> None:
    compute = np.array([point.train_flops for point in optima], dtype=np.float64)
    losses = np.array([point.loss for point in optima], dtype=np.float64)
    xs = np.logspace(np.log10(compute.min()), np.log10(target_compute), 256)
    plt.figure(figsize=(7, 5))
    plt.semilogx(compute, losses, "o", label="observed best losses")
    plt.semilogx(xs, fit.predict(xs), "-", label="loss fit")
    plt.xlabel("Training compute C (FLOPs)")
    plt.ylabel("Best final training loss")
    plt.title("IsoFLOPs Loss Fit")
    plt.grid(True, which="both", linestyle=":")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit scaling laws from cached API runs.")
    parser.add_argument("--runs", type=Path, default=Path("results/scaling_laws/api_runs.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/scaling_laws"))
    parser.add_argument("--target-compute", type=float, default=1e19)
    args = parser.parse_args()

    runs = load_runs(args.runs)
    optima = select_best_by_compute(runs)
    if len(optima) < 3:
        raise SystemExit(
            f"Need at least 3 compute budgets with losses to fit scaling laws; found {len(optima)}."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_fit = fit_model_size_scaling(optima)
    loss_fit = fit_loss_scaling(optima)
    predicted_params = float(model_fit.predict(args.target_compute))
    predicted_loss = float(loss_fit.predict(args.target_compute))
    recommended_shape = nearest_shape(predicted_params)

    plot_model_fit(optima, model_fit, args.target_compute, args.output_dir / "api_model_size_fit.png")
    plot_loss_fit(optima, loss_fit, args.target_compute, args.output_dir / "api_loss_fit.png")

    summary = {
        "optima": [point.__dict__ for point in optima],
        "model_fit": model_fit.to_dict(),
        "loss_fit": loss_fit.to_dict(),
        "target_compute": args.target_compute,
        "predicted_non_embedding_params": predicted_params,
        "predicted_loss": predicted_loss,
        "recommended_shape": recommended_shape.to_dict()
        | {"non_embedding_params": recommended_shape.non_embedding_params},
    }
    (args.output_dir / "api_scaling_fit_summary.json").write_text(json.dumps(summary, indent=2))

    lines = [
        "# Scaling-Laws Fit",
        "",
        "## IsoFLOPs Optima",
        "",
        "| C | N_opt | loss | layers | d_model | heads | batch | lr |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for point in optima:
        lines.append(
            "| "
            f"{point.train_flops:.0e} | "
            f"{format_size(point.non_embedding_params)} | "
            f"{point.loss:.6f} | "
            f"{point.num_layers} | "
            f"{point.d_model} | "
            f"{point.num_heads} | "
            f"{point.batch_size} | "
            f"{point.learning_rate:.1e} |"
        )
    lines.extend(
        [
            "",
            "## Prediction",
            "",
            f"- Target compute: `{args.target_compute:.0e}` FLOPs",
            f"- Predicted optimal size: `{format_size(predicted_params)}` non-embedding parameters",
            f"- Predicted loss: `{predicted_loss:.6f}`",
            f"- Nearest API shape: `{recommended_shape.num_layers}` layers, `d_model={recommended_shape.d_model}`, `num_heads={recommended_shape.num_heads}`",
            "",
        ]
    )
    (args.output_dir / "api_scaling_fit_summary.md").write_text("\n".join(lines))

    print(f"Predicted N_opt({args.target_compute:.0e}) = {format_size(predicted_params)}")
    print(f"Predicted loss = {predicted_loss:.6f}")
    print(
        "Nearest API shape: "
        f"layers={recommended_shape.num_layers}, d_model={recommended_shape.d_model}, "
        f"heads={recommended_shape.num_heads}, N={format_size(recommended_shape.non_embedding_params)}"
    )
    print(f"Wrote outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
