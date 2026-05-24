from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from statistics import mean, stdev
from timeit import default_timer

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

try:
    from cs336_systems.sharded_optimizer import ShardedOptimizer
except ImportError:
    from sharded_optimizer import ShardedOptimizer


MODEL_SIZE_PRESETS = {
    "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    "xl": {"d_model": 1600, "d_ff": 6400, "num_layers": 48, "num_heads": 25},
}


def _load_transformer_cls():
    try:
        module = importlib.import_module("cs336_basics.model")
        return getattr(module, "BasicsTransformerLM")
    except ImportError:
        try:
            module = importlib.import_module("cs336_basics.transformer")
            return getattr(module, "TransformerLM")
        except ImportError:
            repo_root = Path(__file__).resolve().parents[2]
            assignment1_root = repo_root / "assignment1-basics"
            if assignment1_root.exists() and str(assignment1_root) not in sys.path:
                sys.path.insert(0, str(assignment1_root))
            try:
                module = importlib.import_module("cs336_basics.model")
                return getattr(module, "BasicsTransformerLM")
            except ImportError:
                module = importlib.import_module("cs336_basics.transformer")
                return getattr(module, "TransformerLM")


def _parse_dtype(name: str) -> torch.dtype:
    lookup = {"float32": torch.float32, "fp32": torch.float32, "bfloat16": torch.bfloat16, "bf16": torch.bfloat16}
    key = name.lower()
    if key not in lookup:
        raise ValueError(f"Unsupported dtype: {name}")
    return lookup[key]


def _setup(rank: int, world_size: int, master_addr: str, master_port: str) -> torch.device:
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    return torch.device(f"cuda:{rank}")


def _cleanup() -> None:
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def _build_model(args: argparse.Namespace, device: torch.device, dtype: torch.dtype) -> torch.nn.Module:
    cls = _load_transformer_cls()
    preset = MODEL_SIZE_PRESETS[args.size]
    kwargs = {
        "vocab_size": args.vocab_size,
        "context_length": args.context_length,
        "d_model": preset["d_model"],
        "num_layers": preset["num_layers"],
        "num_heads": preset["num_heads"],
        "d_ff": preset["d_ff"],
        "rope_theta": args.rope_theta,
    }
    try:
        return cls(**kwargs, device=device, dtype=dtype)
    except TypeError:
        model = cls(**kwargs)
        return model.to(device=device, dtype=dtype)


def _broadcast_rank0(model: torch.nn.Module) -> None:
    for p in model.parameters():
        dist.broadcast(p.data, src=0)


def _memory_snapshot(device: torch.device) -> dict[str, float]:
    torch.cuda.synchronize(device)
    return {
        "allocated_gb": torch.cuda.memory_allocated(device) / 1e9,
        "reserved_gb": torch.cuda.memory_reserved(device) / 1e9,
        "max_allocated_gb": torch.cuda.max_memory_allocated(device) / 1e9,
    }


def _local_optimizer_state_numel(optimizer: torch.optim.Optimizer) -> int:
    total = 0
    states = optimizer.state.values()
    if isinstance(optimizer, ShardedOptimizer) and optimizer._local_optimizer is not None:
        states = optimizer._local_optimizer.state.values()
    for state in states:
        for value in state.values():
            if torch.is_tensor(value):
                total += value.numel()
    return total


def _optimizer_for_impl(model: torch.nn.Module, args: argparse.Namespace, impl: str) -> torch.optim.Optimizer:
    kwargs = {
        "lr": args.learning_rate,
        "betas": (args.beta1, args.beta2),
        "eps": args.eps,
        "weight_decay": args.weight_decay,
    }
    if impl == "sharded":
        return ShardedOptimizer(model.parameters(), torch.optim.AdamW, **kwargs)
    if impl == "baseline":
        return torch.optim.AdamW(model.parameters(), **kwargs)
    raise ValueError(f"Unsupported impl: {impl}")


def _worker(rank: int, world_size: int, args: argparse.Namespace, impl: str, out_json: str) -> None:
    device = _setup(rank, world_size, args.master_addr, args.master_port)
    dtype = _parse_dtype(args.dtype)
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)
    torch.cuda.reset_peak_memory_stats(device)

    model = _build_model(args, device, dtype)
    model.train()
    _broadcast_rank0(model)
    optimizer = _optimizer_for_impl(model, args, impl)

    if args.global_batch_size % world_size != 0:
        raise ValueError("global_batch_size must be divisible by world_size")
    local_bs = args.global_batch_size // world_size
    x = torch.randint(0, args.vocab_size, (local_bs, args.context_length), device=device, dtype=torch.long)
    y = torch.randint(0, args.vocab_size, (local_bs, args.context_length), device=device, dtype=torch.long)

    torch.cuda.synchronize(device)
    after_init = _memory_snapshot(device)

    step_ms: list[float] = []
    before_step = {}
    after_step = {}
    for step_idx in range(args.measure_steps):
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device)
        t0 = default_timer()
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        loss.backward()
        torch.cuda.synchronize(device)
        if step_idx == 0:
            before_step = _memory_snapshot(device)
        optimizer.step()
        torch.cuda.synchronize(device)
        if step_idx == 0:
            after_step = _memory_snapshot(device)
        t1 = default_timer()
        step_ms.append((t1 - t0) * 1000.0)

    param_numel = sum(p.numel() for p in model.parameters())
    grad_numel = sum(p.grad.numel() for p in model.parameters() if p.grad is not None)
    local_opt_state_numel = _local_optimizer_state_numel(optimizer)
    rank_result = {
        "rank": rank,
        "impl": impl,
        "step_mean_ms": mean(step_ms),
        "step_std_ms": stdev(step_ms) if len(step_ms) > 1 else 0.0,
        "after_init": after_init,
        "before_optimizer_step": before_step,
        "after_optimizer_step": after_step,
        "param_numel": param_numel,
        "grad_numel": grad_numel,
        "local_optimizer_state_numel": local_opt_state_numel,
    }
    gathered: list[dict] = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, rank_result)

    if rank == 0:
        result = {
            "status": "ok",
            "impl": impl,
            "world_size": world_size,
            "model_size": args.size,
            "dtype": args.dtype,
            "global_batch_size": args.global_batch_size,
            "local_batch_size": local_bs,
            "context_length": args.context_length,
            "measure_steps": args.measure_steps,
            "step_mean_ms": mean(r["step_mean_ms"] for r in gathered),
            "step_std_ms": stdev([r["step_mean_ms"] for r in gathered]) if len(gathered) > 1 else 0.0,
            "max_after_init_allocated_gb": max(r["after_init"]["allocated_gb"] for r in gathered),
            "max_before_step_allocated_gb": max(r["before_optimizer_step"]["allocated_gb"] for r in gathered),
            "max_after_step_allocated_gb": max(r["after_optimizer_step"]["allocated_gb"] for r in gathered),
            "rank_results": gathered,
        }
        Path(out_json).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    _cleanup()


def _run_impl(args: argparse.Namespace, impl: str) -> dict:
    out_json = args.output_dir / f"{impl}.json"
    try:
        mp.spawn(_worker, args=(args.world_size, args, impl, str(out_json)), nprocs=args.world_size, join=True)
        return json.loads(out_json.read_text(encoding="utf-8"))
    except torch.cuda.OutOfMemoryError as exc:
        return {"status": "oom", "impl": impl, "error": str(exc)}
    except Exception as exc:
        msg = str(exc)
        status = "oom" if "out of memory" in msg.lower() else "runtime_error"
        return {"status": status, "impl": impl, "error": msg}


def _to_markdown(rows: list[dict]) -> str:
    lines = [
        "| impl | status | step_ms | after_init_gb | before_step_gb | after_step_gb |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row.get("status") != "ok":
            lines.append(f"| {row.get('impl', '-')} | {row.get('status', 'unknown')} | - | - | - | - |")
            continue
        lines.append(
            "| {impl} | ok | {step:.3f} | {init:.3f} | {before:.3f} | {after:.3f} |".format(
                impl=row["impl"],
                step=row["step_mean_ms"],
                init=row["max_after_init_allocated_gb"],
                before=row["max_before_step_allocated_gb"],
                after=row["max_after_step_allocated_gb"],
            )
        )
    return "\n".join(lines) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Measure optimizer state sharding memory and step-time effects.")
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--size", choices=list(MODEL_SIZE_PRESETS.keys()), default="xl")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--global-batch-size", type=int, default=2)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--vocab-size", type=int, default=50257)
    parser.add_argument("--rope-theta", type=float, default=10000.0)
    parser.add_argument("--measure-steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--master-addr", default="127.0.0.1")
    parser.add_argument("--master-port", default="29551")
    parser.add_argument("--output-dir", type=Path, default=Path("results/benchmarks/optimizer_state_sharding_accounting"))
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.world_size > torch.cuda.device_count():
        raise RuntimeError(f"Need {args.world_size} GPUs, found {torch.cuda.device_count()}")

    rows = [_run_impl(args, "baseline"), _run_impl(args, "sharded")]
    config = {**vars(args), "output_dir": str(args.output_dir)}
    (args.output_dir / "results.json").write_text(
        json.dumps({"config": config, "results": rows}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "table.md").write_text(_to_markdown(rows), encoding="utf-8")
    print(f"JSON results: {args.output_dir / 'results.json'}")
    print(f"Markdown table: {args.output_dir / 'table.md'}")


if __name__ == "__main__":
    main()
