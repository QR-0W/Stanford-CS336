#!/usr/bin/env python3
"""
Plot Learning Rate Sweep Results

Reads experiment results from lr_sweep and generates learning curve plots.

Usage:
    python experiments/plot_lr_curves.py --results_dir experiments/lr_sweep_results
"""

import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def load_results(results_dir: Path) -> dict:
    """Load sweep summary and all experiment logs"""
    summary_file = results_dir / "sweep_summary.json"

    if not summary_file.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_file}")

    with open(summary_file) as f:
        summary = json.load(f)

    return summary


def plot_learning_curves(summary: dict, output_dir: Path, metric: str = "train_loss"):
    """Plot learning curves for all LRs on same graph"""

    fig, ax = plt.subplots(figsize=(12, 8))

    # Color palette
    colors = plt.cm.viridis(np.linspace(0, 1, len(summary["results"])))

    for i, result in enumerate(sorted(summary["results"], key=lambda x: x["lr"])):
        if "metrics" not in result or not result["metrics"]:
            continue

        lr = result["lr"]
        metrics = result["metrics"]

        steps = [m["step"] for m in metrics]
        values = [m.get(metric, np.nan) for m in metrics]

        # Filter out None values
        valid_idx = [j for j, v in enumerate(values) if v is not None and not np.isnan(v)]
        steps = [steps[j] for j in valid_idx]
        values = [values[j] for j in valid_idx]

        if steps:
            label = f"lr={lr:.0e}"
            if result.get("final_val_loss"):
                label += f" (val={result['final_val_loss']:.3f})"
            ax.plot(steps, values, label=label, color=colors[i], linewidth=2)

    ax.set_xlabel("Training Steps", fontsize=12)
    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=12)
    ax.set_title(f"Learning Rate Sweep: {metric.replace('_', ' ').title()}", fontsize=14)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Log scale for loss
    if "loss" in metric:
        ax.set_yscale("log")

    plt.tight_layout()

    output_file = output_dir / f"lr_sweep_{metric}.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_file}")

    plt.close()


def plot_val_loss_curves(summary: dict, output_dir: Path):
    """Plot validation loss curves"""

    fig, ax = plt.subplots(figsize=(12, 8))

    colors = plt.cm.viridis(np.linspace(0, 1, len(summary["results"])))

    for i, result in enumerate(sorted(summary["results"], key=lambda x: x["lr"])):
        if "metrics" not in result or not result["metrics"]:
            continue

        lr = result["lr"]
        metrics = result["metrics"]

        # Extract only points with val_loss
        val_data = [(m["step"], m["val_loss"]) for m in metrics if m.get("val_loss") is not None]

        if val_data:
            steps, values = zip(*val_data)
            label = f"lr={lr:.0e} (final={values[-1]:.3f})"
            ax.plot(steps, values, label=label, color=colors[i], linewidth=2, marker="o", markersize=4)

    ax.set_xlabel("Training Steps", fontsize=12)
    ax.set_ylabel("Validation Loss", fontsize=12)
    ax.set_title("Learning Rate Sweep: Validation Loss", fontsize=14)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add target line at 1.45
    ax.axhline(y=1.45, color="red", linestyle="--", linewidth=2, label="Target (1.45)")

    plt.tight_layout()

    output_file = output_dir / "lr_sweep_val_loss.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_file}")

    plt.close()


def plot_lr_vs_final_loss(summary: dict, output_dir: Path):
    """Plot final loss vs learning rate"""

    fig, ax = plt.subplots(figsize=(10, 6))

    lrs = []
    train_losses = []
    val_losses = []
    diverged = []

    for result in summary["results"]:
        lr = result["lr"]
        train_loss = result.get("final_train_loss")
        val_loss = result.get("final_val_loss")

        if train_loss is not None:
            lrs.append(lr)
            train_losses.append(train_loss)
            val_losses.append(val_loss if val_loss else np.nan)
            diverged.append(train_loss > 10 or not result["success"])

    # Plot
    ax.semilogx(lrs, train_losses, "o-", label="Train Loss", markersize=10, linewidth=2)
    ax.semilogx(
        [lr for lr, vl in zip(lrs, val_losses) if not np.isnan(vl)],
        [vl for vl in val_losses if not np.isnan(vl)],
        "s-",
        label="Val Loss",
        markersize=10,
        linewidth=2,
    )

    # Mark diverged
    for lr, tl, div in zip(lrs, train_losses, diverged):
        if div:
            ax.annotate("diverged", (lr, tl), textcoords="offset points", xytext=(0, 10), ha="center", color="red")

    ax.set_xlabel("Learning Rate", fontsize=12)
    ax.set_ylabel("Final Loss", fontsize=12)
    ax.set_title("Final Loss vs Learning Rate", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Target line
    ax.axhline(y=1.45, color="red", linestyle="--", alpha=0.7, label="Target (1.45)")

    plt.tight_layout()

    output_file = output_dir / "lr_vs_final_loss.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_file}")

    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="experiments/lr_sweep_results")
    parser.add_argument("--output_dir", type=str, default=None, help="Output dir for plots (default: same as results)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from: {results_dir}")
    summary = load_results(results_dir)

    print(f"\nFound {len(summary['results'])} experiments")
    print(f"Learning rates: {summary['learning_rates']}")

    # Generate plots
    print("\nGenerating plots...")
    plot_learning_curves(summary, output_dir, metric="train_loss")
    plot_val_loss_curves(summary, output_dir)
    plot_lr_vs_final_loss(summary, output_dir)

    print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
