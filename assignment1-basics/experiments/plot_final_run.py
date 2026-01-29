#!/usr/bin/env python3
"""
Plot Final Run Training Curves

Reads experiment_log.json from the final run and plots training and validation curves.

Usage:
    python experiments/plot_final_run.py --log_file experiments/final_run/runs/final_run_optimal_lr_1e-3/experiment_log.json
"""

import argparse
import json
import matplotlib.pyplot as plt
from pathlib import Path


def plot_curves(log_file: Path, output_file: Path):
    if not log_file.exists():
        print(f"Log file not found: {log_file}")
        return

    with open(log_file) as f:
        data = json.load(f)

    metrics = data.get("metrics", [])
    if not metrics:
        print("No metrics found in log file")
        return

    # Extract data
    steps = [m["step"] for m in metrics]
    train_loss = [m["train_loss"] for m in metrics]
    val_loss_data = [(m["step"], m["val_loss"]) for m in metrics if "val_loss" in m]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot Train Loss
    ax.plot(steps, train_loss, label="Train Loss", alpha=0.6, linewidth=1)

    # Plot Val Loss
    if val_loss_data:
        val_steps, val_losses = zip(*val_loss_data)
        ax.plot(
            val_steps,
            val_losses,
            label=f"Val Loss (final={val_losses[-1]:.4f})",
            linewidth=2,
            marker="o",
            color="orange",
        )

        # Add target line
        ax.axhline(y=1.45, color="red", linestyle="--", label="Target (1.45)")

    ax.set_xlabel("Steps")
    ax.set_ylabel("Cross Entropy Loss")
    ax.set_title(f"Final Run Training Progress (LR={data['config']['lr']})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"Saved plot to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_file", type=str, required=True)
    parser.add_argument("--output", type=str, default="final_run_plot.png")
    args = parser.parse_args()

    plot_curves(Path(args.log_file), Path(args.output))
