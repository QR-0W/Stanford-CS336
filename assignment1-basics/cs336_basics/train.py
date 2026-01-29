"""
CS336 Assignment 1 - Training Script

Complete training loop for TransformerLM with:
- Command-line hyperparameter configuration
- Memory-efficient data loading (np.memmap)
- Periodic checkpointing
- TensorBoard logging
"""

import argparse
import json
import time
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from cs336_basics.transformer import TransformerLM, cross_entropy
from cs336_basics.optimizer import AdamW, get_lr_cosine_schedule
from cs336_basics.get_batch import get_batch
from cs336_basics.checkpointing import save_checkpoint, load_checkpoint


def get_args():
    parser = argparse.ArgumentParser(description="Train a Transformer Language Model")

    # ===== 数据路径 =====
    parser.add_argument("--train_data", type=str, required=True, help="Path to training data (.npy)")
    parser.add_argument("--val_data", type=str, default=None, help="Path to validation data (.npy)")

    # ===== 模型超参数 =====
    parser.add_argument("--vocab_size", type=int, default=50257, help="Vocabulary size")
    parser.add_argument("--context_length", type=int, default=256, help="Context length (max sequence length)")
    parser.add_argument("--d_model", type=int, default=512, help="Model dimension")
    parser.add_argument("--num_layers", type=int, default=6, help="Number of transformer layers")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--d_ff", type=int, default=2048, help="Feed-forward hidden dimension")
    parser.add_argument("--rope_theta", type=float, default=10000.0, help="RoPE theta parameter")

    # ===== 消融实验参数 =====
    parser.add_argument("--no_rmsnorm", action="store_true", help="Disable RMSNorm (ablation)")
    parser.add_argument("--norm_type", type=str, default="pre", choices=["pre", "post"], help="pre or post norm")
    parser.add_argument("--no_rope", action="store_true", help="Disable RoPE (NoPE ablation)")
    parser.add_argument("--ffn_type", type=str, default="swiglu", choices=["swiglu", "silu"], help="FFN type")

    # ===== 训练超参数 =====
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_steps", type=int, default=10000, help="Number of training steps")
    parser.add_argument("--lr", type=float, default=1e-4, help="Max learning rate")
    parser.add_argument("--min_lr", type=float, default=None, help="Min learning rate (default: lr * 0.1)")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="Warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--beta1", type=float, default=0.9, help="AdamW beta1")
    parser.add_argument("--beta2", type=float, default=0.999, help="AdamW beta2")
    parser.add_argument("--eps", type=float, default=1e-8, help="AdamW epsilon")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping (0 to disable)")

    # ===== 日志与检查点 =====
    parser.add_argument("--log_interval", type=int, default=10, help="Log every N steps")
    parser.add_argument("--eval_interval", type=int, default=100, help="Evaluate every N steps")
    parser.add_argument("--eval_steps", type=int, default=20, help="Number of eval batches")
    parser.add_argument("--save_interval", type=int, default=1000, help="Save checkpoint every N steps")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--log_dir", type=str, default="runs", help="TensorBoard log directory")
    parser.add_argument("--run_name", type=str, default=None, help="Run name for logging")

    # ===== 恢复训练 =====
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")

    # ===== 设备 =====
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    return parser.parse_args()


@torch.no_grad()
def evaluate(model, val_data, batch_size, context_length, device, vocab_size, num_batches=20):
    """Evaluate model on validation data."""
    model.eval()
    losses = []
    for _ in range(num_batches):
        inputs, targets = get_batch(val_data, batch_size, context_length, device)
        logits = model(inputs)
        loss = cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def main():
    args = get_args()

    # ===== 设置运行名称 =====
    if args.run_name is None:
        args.run_name = f"lm_d{args.d_model}_l{args.num_layers}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print(f"=" * 60)
    print(f"Training Configuration")
    print(f"=" * 60)
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    print(f"=" * 60)

    # ===== 加载数据 (memory-mapped) =====
    print(f"Loading training data from {args.train_data}...")
    train_data = np.load(args.train_data, mmap_mode="r")
    print(f"  Training tokens: {len(train_data):,}")

    val_data = None
    if args.val_data:
        print(f"Loading validation data from {args.val_data}...")
        val_data = np.load(args.val_data, mmap_mode="r")
        print(f"  Validation tokens: {len(val_data):,}")

    # ===== 初始化模型 =====
    print(f"Initializing model on {args.device}...")
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        use_rmsnorm=not args.no_rmsnorm,
        norm_type=args.norm_type,
        use_rope=not args.no_rope,
        ffn_type=args.ffn_type,
    ).to(args.device)

    # 打印参数量
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {num_params:,}")

    args.min_lr = args.min_lr if args.min_lr is not None else args.lr * 0.1

    # ===== 初始化优化器 =====
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    # ===== 恢复检查点 =====
    start_step = 0
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        start_step = load_checkpoint(args.resume, model, optimizer)
        print(f"  Resumed at step {start_step}")

    # ===== 创建目录 =====
    checkpoint_dir = Path(args.checkpoint_dir) / args.run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    log_dir = Path(args.log_dir) / args.run_name
    writer = SummaryWriter(log_dir=str(log_dir))
    print(f"TensorBoard logs: {log_dir}")
    print(f"Checkpoints: {checkpoint_dir}")

    # ===== 实验日志文件 =====
    experiment_log = {
        "config": vars(args),
        "num_params": num_params,
        "start_time": datetime.now().isoformat(),
        "metrics": [],
    }
    log_file = log_dir / "experiment_log.json"

    # ===== 训练循环 =====
    print(f"\nStarting training from step {start_step}...")
    model.train()
    start_time = time.time()

    for step in range(start_step, args.num_steps):
        # 采样 batch
        inputs, targets = get_batch(train_data, args.batch_size, args.context_length, args.device)

        # 前向传播
        logits = model(inputs)  # (batch, seq, vocab)

        # 计算 loss
        loss = cross_entropy(logits.view(-1, args.vocab_size), targets.view(-1))

        # 更新学习率 (Cosine Schedule)
        lr = get_lr_cosine_schedule(step, args.lr, args.min_lr, args.warmup_steps, args.num_steps)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # 反向传播
        optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        optimizer.step()

        # ===== TensorBoard 日志 =====
        if step % args.log_interval == 0:
            elapsed_time = time.time() - start_time
            writer.add_scalar("train/loss", loss.item(), step)
            writer.add_scalar("train/perplexity", torch.exp(loss).item(), step)
            writer.add_scalar("time/wallclock_seconds", elapsed_time, step)
            writer.add_scalar(
                "time/steps_per_second", (step - start_step + 1) / elapsed_time if elapsed_time > 0 else 0, step
            )
            writer.add_scalar("train/lr", lr, step)

            # 记录到实验日志
            experiment_log["metrics"].append(
                {
                    "step": step,
                    "train_loss": loss.item(),
                    "train_ppl": torch.exp(loss).item(),
                    "lr": lr,
                    "wallclock_seconds": elapsed_time,
                }
            )

            print(
                f"Step {step:>6d} | loss: {loss.item():.4f} | ppl: {torch.exp(loss).item():.2f} | time: {elapsed_time:.1f}s"
            )

        # ===== 验证 =====
        if val_data is not None and step % args.eval_interval == 0:
            val_loss = evaluate(
                model, val_data, args.batch_size, args.context_length, args.device, args.vocab_size, args.eval_steps
            )
            elapsed_time = time.time() - start_time
            writer.add_scalar("val/loss", val_loss, step)
            writer.add_scalar("val/perplexity", np.exp(val_loss), step)

            # 更新实验日志中最后一条记录
            if experiment_log["metrics"] and experiment_log["metrics"][-1]["step"] == step:
                experiment_log["metrics"][-1]["val_loss"] = val_loss
                experiment_log["metrics"][-1]["val_ppl"] = float(np.exp(val_loss))

            print(f"Step {step:>6d} | val_loss: {val_loss:.4f} | val_ppl: {np.exp(val_loss):.2f}")

        # ===== 保存检查点 =====
        if step > 0 and step % args.save_interval == 0:
            ckpt_path = checkpoint_dir / f"checkpoint_{step}.pt"
            save_checkpoint(model, optimizer, step, ckpt_path)
            print(f"  → Saved checkpoint: {ckpt_path}")

    # ===== 保存最终模型和日志 =====
    final_ckpt = checkpoint_dir / "checkpoint_final.pt"
    save_checkpoint(model, optimizer, args.num_steps, final_ckpt)

    # 保存实验日志
    total_time = time.time() - start_time
    experiment_log["end_time"] = datetime.now().isoformat()
    experiment_log["total_time_seconds"] = total_time
    experiment_log["total_steps"] = args.num_steps - start_step

    with open(log_file, "w") as f:
        json.dump(experiment_log, f, indent=2)

    print(f"\nTraining complete!")
    print(f"  Total time: {total_time:.1f}s ({total_time / 60:.1f}min)")
    print(f"  Final checkpoint: {final_ckpt}")
    print(f"  Experiment log: {log_file}")

    writer.close()


if __name__ == "__main__":
    main()
