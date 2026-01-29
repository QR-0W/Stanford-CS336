#!/usr/bin/env python3
"""
Generate ablation comparison plots for CS336 Assignment 1
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


def plot_comparison(experiments, title, output_path, ylabel="Validation Loss"):
    """Create comparison plot with multiple experiments."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(experiments)))

    for (name, data), color in zip(experiments.items(), colors):
        if data["val_loss"]:
            final = data["final_val_loss"]
            label = f"{name} (final: {final:.4f})"
            ax.plot(data["val_steps"], data["val_loss"], label=label, color=color, linewidth=2)

    ax.set_xlabel("Training Steps", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablations_dir", default="experiments/ablations")
    parser.add_argument(
        "--baseline_log", default="experiments/final_run_v2/runs/final_run_scheduler_lr1e-3/experiment_log.json"
    )
    parser.add_argument("--output_dir", default="experiments/ablations")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    ablations_dir = Path(args.ablations_dir)

    # Load baseline
    baseline = load_experiment(args.baseline_log)

    # ===== Plot 1: RMSNorm Ablation =====
    print("\n=== RMSNorm Ablation ===")
    rmsnorm_exps = {
        "Baseline (Pre-Norm + RMSNorm)": baseline,
        "No RMSNorm (LR=1e-3)": load_experiment(
            ablations_dir / "no_rmsnorm_lr1e-3/runs/no_rmsnorm_lr1e-3/experiment_log.json"
        ),
        "No RMSNorm (LR=1e-4)": load_experiment(
            ablations_dir / "no_rmsnorm_lr1e-4/runs/no_rmsnorm_lr1e-4/experiment_log.json"
        ),
    }
    for name, data in rmsnorm_exps.items():
        print(f"  {name}: final val_loss = {data['final_val_loss']:.4f}")
    plot_comparison(
        rmsnorm_exps, "RMSNorm Ablation: Effect of Removing Normalization", output_dir / "ablation_rmsnorm.png"
    )

    # ===== Plot 2: Pre-Norm vs Post-Norm =====
    print("\n=== Norm Type Ablation ===")
    norm_exps = {
        "Pre-Norm (default)": baseline,
        "Post-Norm": load_experiment(ablations_dir / "post_norm/runs/post_norm/experiment_log.json"),
    }
    for name, data in norm_exps.items():
        print(f"  {name}: final val_loss = {data['final_val_loss']:.4f}")
    plot_comparison(norm_exps, "Pre-Norm vs Post-Norm Transformer", output_dir / "ablation_norm_type.png")

    # ===== Plot 3: RoPE vs NoPE =====
    print("\n=== Position Embedding Ablation ===")
    rope_exps = {
        "RoPE (default)": baseline,
        "NoPE (No Position Embedding)": load_experiment(ablations_dir / "nope/runs/nope/experiment_log.json"),
    }
    for name, data in rope_exps.items():
        print(f"  {name}: final val_loss = {data['final_val_loss']:.4f}")
    plot_comparison(rope_exps, "RoPE vs No Position Embedding (NoPE)", output_dir / "ablation_rope.png")

    # ===== Plot 4: SwiGLU vs SiLU =====
    print("\n=== FFN Type Ablation ===")
    ffn_exps = {
        "SwiGLU (default)": baseline,
        "SiLU (param-matched)": load_experiment(ablations_dir / "silu_ffn/runs/silu_ffn/experiment_log.json"),
    }
    for name, data in ffn_exps.items():
        print(f"  {name}: final val_loss = {data['final_val_loss']:.4f}")
    plot_comparison(ffn_exps, "SwiGLU vs SiLU FFN (Parameter Matched)", output_dir / "ablation_ffn.png")

    print("\nAll plots saved to:", output_dir)


if __name__ == "__main__":
    main()
