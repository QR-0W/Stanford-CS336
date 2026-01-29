#!/usr/bin/env python3
"""
Learning Rate Sweep Experiment for TinyStories

Runs multiple training experiments with different learning rates in parallel
across multiple GPUs. Results are saved to JSON for later analysis.

Usage:
    python experiments/lr_sweep.py --gpus 0,1,2
"""

import argparse
import json
import subprocess
import time
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed


# ===== 实验配置 =====
BASE_CONFIG = {
    # 数据
    "train_data": "output/tiny_stories_train.npy",
    "val_data": "output/tiny_stories_valid.npy",
    # 模型 (17M non-embedding params)
    "vocab_size": 10000,
    "context_length": 256,
    "d_model": 512,
    "d_ff": 1344,
    "num_layers": 4,
    "num_heads": 16,
    "rope_theta": 10000.0,
    # 训练 (327,680,000 total tokens)
    "batch_size": 32,
    "num_steps": 5000,  # 32 * 5000 * 256 = 40,960,000 (low-resource)
    # 完整版: num_steps = 40000 → 32 * 40000 * 256 = 327,680,000
    # 优化器 (默认值，LR 会被覆盖)
    "weight_decay": 0.1,
    "beta1": 0.9,
    "beta2": 0.95,
    "eps": 1e-8,
    "grad_clip": 1.0,
    # 日志
    "log_interval": 50,
    "eval_interval": 250,
    "eval_steps": 20,
    "save_interval": 10000,  # 不频繁保存
}

# 学习率候选值 (对数尺度扫描)
LEARNING_RATES = [
    1e-5,  # 很小，可能收敛慢
    3e-5,
    1e-4,
    3e-4,  # 常见默认值
    1e-3,
    3e-3,  # 可能发散
    1e-2,  # 大概率发散
]


def run_single_experiment(lr: float, gpu_id: int, output_dir: Path) -> dict:
    """在指定 GPU 上运行单个学习率实验"""
    run_name = f"lr_{lr:.0e}".replace("-", "m").replace("+", "p")

    cmd = [
        "bash",
        "-c",
        f"source ~/miniconda3/etc/profile.d/conda.sh && conda activate Qwen3 && "
        f"cd /mdata/wjx/CS336/assignment1-basics && "
        f"PYTHONPATH=/mdata/wjx/CS336/assignment1-basics python cs336_basics/train.py "
        f"--train_data={BASE_CONFIG['train_data']} "
        f"--val_data={BASE_CONFIG['val_data']} "
        f"--vocab_size={BASE_CONFIG['vocab_size']} "
        f"--context_length={BASE_CONFIG['context_length']} "
        f"--d_model={BASE_CONFIG['d_model']} "
        f"--d_ff={BASE_CONFIG['d_ff']} "
        f"--num_layers={BASE_CONFIG['num_layers']} "
        f"--num_heads={BASE_CONFIG['num_heads']} "
        f"--rope_theta={BASE_CONFIG['rope_theta']} "
        f"--batch_size={BASE_CONFIG['batch_size']} "
        f"--num_steps={BASE_CONFIG['num_steps']} "
        f"--lr={lr} "
        f"--weight_decay={BASE_CONFIG['weight_decay']} "
        f"--beta1={BASE_CONFIG['beta1']} "
        f"--beta2={BASE_CONFIG['beta2']} "
        f"--eps={BASE_CONFIG['eps']} "
        f"--grad_clip={BASE_CONFIG['grad_clip']} "
        f"--log_interval={BASE_CONFIG['log_interval']} "
        f"--eval_interval={BASE_CONFIG['eval_interval']} "
        f"--eval_steps={BASE_CONFIG['eval_steps']} "
        f"--save_interval={BASE_CONFIG['save_interval']} "
        f"--checkpoint_dir={output_dir / 'checkpoints'} "
        f"--log_dir={output_dir / 'runs'} "
        f"--run_name={run_name} "
        f"--device=cuda:0",  # Use cuda:0 since CUDA_VISIBLE_DEVICES isolates the GPU
    ]

    print(f"[GPU {gpu_id}] Starting experiment lr={lr:.0e}")
    start_time = time.time()

    # 运行训练
    env = {"CUDA_VISIBLE_DEVICES": str(gpu_id)}
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env={**subprocess.os.environ, **env},
        cwd="/mdata/wjx/CS336/assignment1-basics",
    )

    elapsed = time.time() - start_time

    # 读取实验日志
    log_file = output_dir / "runs" / run_name / "experiment_log.json"
    experiment_result = {
        "lr": lr,
        "gpu_id": gpu_id,
        "run_name": run_name,
        "elapsed_seconds": elapsed,
        "returncode": result.returncode,
        "success": result.returncode == 0,
    }

    if log_file.exists():
        with open(log_file) as f:
            log_data = json.load(f)
        experiment_result["metrics"] = log_data.get("metrics", [])
        experiment_result["final_train_loss"] = log_data["metrics"][-1]["train_loss"] if log_data["metrics"] else None

        # 找到最后一个有 val_loss 的记录
        val_losses = [m.get("val_loss") for m in log_data["metrics"] if m.get("val_loss")]
        experiment_result["final_val_loss"] = val_losses[-1] if val_losses else None
    else:
        experiment_result["error"] = result.stderr[-2000:] if result.stderr else "No log file found"

    status = "✓" if experiment_result["success"] else "✗"
    print(f"[GPU {gpu_id}] {status} lr={lr:.0e} completed in {elapsed / 60:.1f}min")

    return experiment_result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=str, default="0,1,2", help="Comma-separated GPU IDs")
    parser.add_argument("--output_dir", type=str, default="experiments/lr_sweep_results")
    parser.add_argument("--lrs", type=str, default=None, help="Custom learning rates (comma-separated)")
    args = parser.parse_args()

    gpu_ids = [int(g) for g in args.gpus.split(",")]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    learning_rates = LEARNING_RATES
    if args.lrs:
        learning_rates = [float(lr) for lr in args.lrs.split(",")]

    print("=" * 60)
    print("Learning Rate Sweep Experiment")
    print("=" * 60)
    print(f"GPUs: {gpu_ids}")
    print(f"Learning rates: {learning_rates}")
    print(f"Output dir: {output_dir}")
    print(
        f"Total tokens per run: {BASE_CONFIG['batch_size'] * BASE_CONFIG['num_steps'] * BASE_CONFIG['context_length']:,}"
    )
    print("=" * 60)

    all_results = []
    start_time = datetime.now()

    # 并行运行实验 (每个 GPU 运行一个)
    with ProcessPoolExecutor(max_workers=len(gpu_ids)) as executor:
        futures = {}
        gpu_queue = list(gpu_ids)
        lr_queue = list(learning_rates)

        # 初始提交
        while gpu_queue and lr_queue:
            gpu_id = gpu_queue.pop(0)
            lr = lr_queue.pop(0)
            future = executor.submit(run_single_experiment, lr, gpu_id, output_dir)
            futures[future] = (lr, gpu_id)

        # 等待完成并提交新任务
        while futures:
            done_future = next(as_completed(futures))
            lr, gpu_id = futures.pop(done_future)

            try:
                result = done_future.result()
                all_results.append(result)
            except Exception as e:
                all_results.append({"lr": lr, "gpu_id": gpu_id, "success": False, "error": str(e)})

            # 如果还有待运行的实验，提交到刚释放的 GPU
            if lr_queue:
                next_lr = lr_queue.pop(0)
                future = executor.submit(run_single_experiment, next_lr, gpu_id, output_dir)
                futures[future] = (next_lr, gpu_id)

    # 保存汇总结果
    summary = {
        "timestamp": start_time.isoformat(),
        "total_time_seconds": (datetime.now() - start_time).total_seconds(),
        "config": BASE_CONFIG,
        "learning_rates": learning_rates,
        "results": all_results,
    }

    summary_file = output_dir / "sweep_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    # 打印结果摘要
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    print(f"{'LR':<12} {'Train Loss':<12} {'Val Loss':<12} {'Status':<10}")
    print("-" * 50)

    for r in sorted(all_results, key=lambda x: x["lr"]):
        train_loss = r.get("final_train_loss", "N/A")
        val_loss = r.get("final_val_loss", "N/A")
        status = "OK" if r["success"] else "FAILED"

        if isinstance(train_loss, float):
            train_loss = f"{train_loss:.4f}"
        if isinstance(val_loss, float):
            val_loss = f"{val_loss:.4f}"

        print(f"{r['lr']:<12.0e} {train_loss:<12} {val_loss:<12} {status:<10}")

    print(f"\nResults saved to: {summary_file}")
    print(f"Total time: {summary['total_time_seconds'] / 60:.1f} minutes")


if __name__ == "__main__":
    main()
