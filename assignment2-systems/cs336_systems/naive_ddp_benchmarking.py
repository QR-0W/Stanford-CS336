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


MODEL_SIZE_PRESETS = {
    "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    "xl": {"d_model": 1600, "d_ff": 6400, "num_layers": 48, "num_heads": 25},
    "2.7b": {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
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


def _parse_dtype(dtype_name: str) -> torch.dtype:
    lookup = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    key = dtype_name.lower()
    if key not in lookup:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return lookup[key]


def _setup_process_group(rank: int, world_size: int, master_addr: str, master_port: str) -> torch.device:
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    return torch.device(f"cuda:{rank}")


def _cleanup_process_group() -> None:
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def _build_model(args: argparse.Namespace, device: torch.device, dtype: torch.dtype) -> torch.nn.Module:
    transformer_cls = _load_transformer_cls()
    preset = MODEL_SIZE_PRESETS[args.size]

    model_kwargs = {
        "vocab_size": args.vocab_size,
        "context_length": args.context_length,
        "d_model": preset["d_model"],
        "num_layers": preset["num_layers"],
        "num_heads": preset["num_heads"],
        "d_ff": preset["d_ff"],
        "rope_theta": args.rope_theta,
    }

    try:
        model = transformer_cls(**model_kwargs, device=device, dtype=dtype)
    except TypeError:
        model = transformer_cls(**model_kwargs)
        model = model.to(device=device, dtype=dtype)

    return model


def _broadcast_model_from_rank0(model: torch.nn.Module) -> None:
    for p in model.parameters():
        dist.broadcast(p.data, src=0)


def _naive_allreduce_grads(model: torch.nn.Module) -> None:
    world_size = dist.get_world_size()
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, async_op=False)
            p.grad.div_(world_size)


def _benchmark_worker(rank: int, world_size: int, args: argparse.Namespace, output_json: str) -> None:
    device = _setup_process_group(rank, world_size, args.master_addr, args.master_port)
    dtype = _parse_dtype(args.dtype)

    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)

    model = _build_model(args=args, device=device, dtype=dtype)
    model.train()
    _broadcast_model_from_rank0(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    if args.global_batch_size % world_size != 0:
        if rank == 0:
            raise ValueError("global_batch_size must be divisible by world_size")
        _cleanup_process_group()
        return

    local_bs = args.global_batch_size // world_size
    local_inputs = torch.randint(
        low=0,
        high=args.vocab_size,
        size=(local_bs, args.context_length),
        device=device,
        dtype=torch.long,
    )
    local_targets = torch.randint(
        low=0,
        high=args.vocab_size,
        size=(local_bs, args.context_length),
        device=device,
        dtype=torch.long,
    )

    for _ in range(args.warmup_steps):
        optimizer.zero_grad(set_to_none=True)
        logits = model(local_inputs)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), local_targets.reshape(-1))
        loss.backward()
        _naive_allreduce_grads(model)
        optimizer.step()
        torch.cuda.synchronize(device)

    step_timings_ms: list[float] = []
    comm_timings_ms: list[float] = []

    for _ in range(args.measure_steps):
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device)
        t0 = default_timer()

        logits = model(local_inputs)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), local_targets.reshape(-1))
        loss.backward()

        torch.cuda.synchronize(device)
        c0 = default_timer()
        _naive_allreduce_grads(model)
        torch.cuda.synchronize(device)
        c1 = default_timer()

        optimizer.step()

        torch.cuda.synchronize(device)
        t1 = default_timer()

        step_timings_ms.append((t1 - t0) * 1000.0)
        comm_timings_ms.append((c1 - c0) * 1000.0)

    rank_summary = {
        "rank": rank,
        "step_mean_ms": mean(step_timings_ms),
        "step_std_ms": stdev(step_timings_ms) if len(step_timings_ms) > 1 else 0.0,
        "comm_mean_ms": mean(comm_timings_ms),
        "comm_std_ms": stdev(comm_timings_ms) if len(comm_timings_ms) > 1 else 0.0,
        "comm_ratio_mean": mean(c / s for c, s in zip(comm_timings_ms, step_timings_ms)),
    }

    gathered: list[dict] = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, rank_summary)

    if rank == 0:
        step_means = [x["step_mean_ms"] for x in gathered]
        comm_means = [x["comm_mean_ms"] for x in gathered]
        comm_ratios = [x["comm_ratio_mean"] for x in gathered]

        result = {
            "status": "ok",
            "backend": "nccl",
            "world_size": world_size,
            "model_size": args.size,
            "dtype": args.dtype,
            "global_batch_size": args.global_batch_size,
            "local_batch_size": local_bs,
            "context_length": args.context_length,
            "warmup_steps": args.warmup_steps,
            "measure_steps": args.measure_steps,
            "rank_summaries": gathered,
            "step_mean_ms": mean(step_means),
            "step_std_ms": stdev(step_means) if len(step_means) > 1 else 0.0,
            "comm_mean_ms": mean(comm_means),
            "comm_std_ms": stdev(comm_means) if len(comm_means) > 1 else 0.0,
            "comm_ratio_mean": mean(comm_ratios),
            "comm_ratio_percent": mean(comm_ratios) * 100.0,
        }
        Path(output_json).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    _cleanup_process_group()


def _to_markdown(result: dict) -> str:
    if result.get("status") != "ok":
        return "| status | error |\n|---|---|\n| {status} | {error} |\n".format(
            status=result.get("status", "unknown"),
            error=result.get("error", ""),
        )

    return (
        "| model_size | world_size | dtype | global_bs | local_bs | context_len | step_mean_ms | comm_mean_ms | comm_ratio_% |\n"
        "|---|---:|---|---:|---:|---:|---:|---:|---:|\n"
        "| {model} | {ws} | {dtype} | {gbs} | {lbs} | {ctx} | {step:.3f} | {comm:.3f} | {ratio:.2f} |\n"
    ).format(
        model=result["model_size"],
        ws=result["world_size"],
        dtype=result["dtype"],
        gbs=result["global_batch_size"],
        lbs=result["local_batch_size"],
        ctx=result["context_length"],
        step=result["step_mean_ms"],
        comm=result["comm_mean_ms"],
        ratio=result["comm_ratio_percent"],
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark naive DDP (per-parameter all-reduce after backward).")
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--size", choices=list(MODEL_SIZE_PRESETS.keys()), default="xl")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--global-batch-size", type=int, default=8)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--vocab-size", type=int, default=50257)
    parser.add_argument("--rope-theta", type=float, default=10000.0)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--measure-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--master-addr", default="127.0.0.1")
    parser.add_argument("--master-port", default="29511")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/benchmarks/naive_ddp_benchmarking"),
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("naive_ddp_benchmarking requires CUDA")
    if args.world_size > torch.cuda.device_count():
        raise RuntimeError(
            f"Requested world_size={args.world_size}, but only {torch.cuda.device_count()} GPU(s) available"
        )

    result_json = args.output_dir / "results.json"

    try:
        mp.spawn(
            _benchmark_worker,
            args=(args.world_size, args, str(result_json)),
            nprocs=args.world_size,
            join=True,
        )
        result = json.loads(result_json.read_text(encoding="utf-8"))
    except torch.cuda.OutOfMemoryError as exc:
        result = {"status": "oom", "error": str(exc)}
    except RuntimeError as exc:
        msg = str(exc)
        if "out of memory" in msg.lower():
            result = {"status": "oom", "error": msg}
        else:
            result = {"status": "runtime_error", "error": msg}
    except Exception as exc:
        result = {"status": "runtime_error", "error": str(exc)}

    (args.output_dir / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (args.output_dir / "table.md").write_text(_to_markdown(result), encoding="utf-8")

    print(f"JSON results: {args.output_dir / 'results.json'}")
    print(f"Markdown table: {args.output_dir / 'table.md'}")


if __name__ == "__main__":
    main()
