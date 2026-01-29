#!/usr/bin/env python3
"""
Generate OWT comparison plots for CS336 Assignment 1
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse

plt.style.use("seaborn-v0_8-whitegrid")


def load_experiment(log_path):
    """Load experiment log and extract metrics."""
    with open(log_path) as f:
        data = json.load(f)

    metrics = data["metrics"]
    steps = [m["step"] for m in metrics]
    train_loss = [m["train_loss"] for m in metrics]
    val_loss = [m.get("val_loss") for m in metrics if "val_loss" in m]
    val_steps = [m["step"] for m in metrics if "val_loss" in m]

    return {
        "steps": steps,
        "train_loss": train_loss,
        "val_steps": val_steps,
        "val_loss": val_loss,
        "config": data.get("config", {}),
        "final_val_loss": val_loss[-1] if val_loss else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--owt_dir", default="experiments/owt")
    parser.add_argument(
        "--baseline_log", default="experiments/final_run_v2/runs/final_run_scheduler_lr1e-3/experiment_log.json"
    )
    parser.add_argument("--output_file", default="experiments/owt/owt_comparison.png")
    args = parser.parse_args()

    owt_dir = Path(args.owt_dir)

    # Load experiments
    ts_baseline = load_experiment(args.baseline_log)
    owt_lr1e3 = load_experiment(owt_dir / "owt_lr1e-3/runs/owt_lr1e-3/experiment_log.json")
    owt_lr5e4 = load_experiment(owt_dir / "owt_lr5e-4/runs/owt_lr5e-4/experiment_log.json")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # TinyStories
    ax.plot(
        ts_baseline["val_steps"],
        ts_baseline["val_loss"],
        label=f"TinyStories (Vocab=10k, LR=1e-3)",
        color="green",
        linestyle="--",
    )

    # OWT
    if owt_lr1e3["val_loss"]:
        ax.plot(owt_lr1e3["val_steps"], owt_lr1e3["val_loss"], label=f"OWT (Vocab=32k, LR=1e-3)", color="blue")

    if owt_lr5e4["val_loss"]:
        ax.plot(owt_lr5e4["val_steps"], owt_lr5e4["val_loss"], label=f"OWT (Vocab=32k, LR=5e-4)", color="orange")

    ax.set_xlabel("Training Steps", fontsize=12)
    ax.set_ylabel("Validation Loss", fontsize=12)
    ax.set_title("TinyStories vs OpenWebText Training", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add annotation about vocab size impact
    ts_random = np.log(10000)
    owt_random = np.log(32000)
    ax.axhline(ts_random, color="green", linestyle=":", alpha=0.3, label="TS Random (-log(1/10k))")
    ax.axhline(owt_random, color="blue", linestyle=":", alpha=0.3, label="OWT Random (-log(1/32k))")

    plt.tight_layout()
    plt.savefig(args.output_file, dpi=150)
    print(f"Saved plot to {args.output_file}")


if __name__ == "__main__":
    main()
