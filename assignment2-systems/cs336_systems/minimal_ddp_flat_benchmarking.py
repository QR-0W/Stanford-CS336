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
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors


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
    lookup = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
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
    p = MODEL_SIZE_PRESETS[args.size]
    kwargs = {
        "vocab_size": args.vocab_size,
        "context_length": args.context_length,
        "d_model": p["d_model"],
        "num_layers": p["num_layers"],
        "num_heads": p["num_heads"],
        "d_ff": p["d_ff"],
        "rope_theta": args.rope_theta,
    }
    try:
        return cls(**kwargs, device=device, dtype=dtype)
    except TypeError:
        m = cls(**kwargs)
        return m.to(device=device, dtype=dtype)


def _broadcast_rank0(model: torch.nn.Module) -> None:
    for p in model.parameters():
        dist.broadcast(p.data, src=0)


def _sync_grads_individual(model: torch.nn.Module) -> None:
    ws = dist.get_world_size()
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, async_op=False)
            p.grad.div_(ws)


def _sync_grads_flat(model: torch.nn.Module) -> None:
    ws = dist.get_world_size()
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    if not grads:
        return
    flat = _flatten_dense_tensors(grads)
    dist.all_reduce(flat, op=dist.ReduceOp.SUM, async_op=False)
    flat.div_(ws)
    synced = _unflatten_dense_tensors(flat, grads)
    for g, new_g in zip(grads, synced):
        g.copy_(new_g)


def _worker(rank: int, world_size: int, args: argparse.Namespace, impl: str, out_json: str) -> None:
    device = _setup(rank, world_size, args.master_addr, args.master_port)
    dtype = _parse_dtype(args.dtype)

    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)

    model = _build_model(args, device, dtype)
    model.train()
    _broadcast_rank0(model)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    if args.global_batch_size % world_size != 0:
        raise ValueError("global_batch_size must be divisible by world_size")
    local_bs = args.global_batch_size // world_size

    x = torch.randint(0, args.vocab_size, (local_bs, args.context_length), device=device, dtype=torch.long)
    y = torch.randint(0, args.vocab_size, (local_bs, args.context_length), device=device, dtype=torch.long)

    sync_fn = _sync_grads_flat if impl == "flat" else _sync_grads_individual

    for _ in range(args.warmup_steps):
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        loss.backward()
        sync_fn(model)
        opt.step()
        torch.cuda.synchronize(device)

    step_ms: list[float] = []
    comm_ms: list[float] = []
    for _ in range(args.measure_steps):
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device)
        t0 = default_timer()
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        loss.backward()

        torch.cuda.synchronize(device)
        c0 = default_timer()
        sync_fn(model)
        torch.cuda.synchronize(device)
        c1 = default_timer()

        opt.step()
        torch.cuda.synchronize(device)
        t1 = default_timer()

        step_ms.append((t1 - t0) * 1000.0)
        comm_ms.append((c1 - c0) * 1000.0)

    rank_result = {
        "rank": rank,
        "step_mean_ms": mean(step_ms),
        "comm_mean_ms": mean(comm_ms),
        "comm_ratio": mean(c / s for c, s in zip(comm_ms, step_ms)),
    }
    gathered: list[dict] = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, rank_result)

    if rank == 0:
        step_means = [r["step_mean_ms"] for r in gathered]
        comm_means = [r["comm_mean_ms"] for r in gathered]
        ratios = [r["comm_ratio"] for r in gathered]
        result = {
            "status": "ok",
            "impl": impl,
            "model_size": args.size,
            "world_size": world_size,
            "dtype": args.dtype,
            "global_batch_size": args.global_batch_size,
            "context_length": args.context_length,
            "warmup_steps": args.warmup_steps,
            "measure_steps": args.measure_steps,
            "step_mean_ms": mean(step_means),
            "step_std_ms": stdev(step_means) if len(step_means) > 1 else 0.0,
            "comm_mean_ms": mean(comm_means),
            "comm_std_ms": stdev(comm_means) if len(comm_means) > 1 else 0.0,
            "comm_ratio_percent": mean(ratios) * 100.0,
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
        "| impl | status | step_mean_ms | comm_mean_ms | comm_ratio_% |",
        "|---|---|---:|---:|---:|",
    ]
    for r in rows:
        if r.get("status") != "ok":
            lines.append(f"| {r.get('impl','-')} | {r.get('status','unknown')} | - | - | - |")
        else:
            lines.append(
                "| {impl} | ok | {step:.3f} | {comm:.3f} | {ratio:.2f} |".format(
                    impl=r["impl"], step=r["step_mean_ms"], comm=r["comm_mean_ms"], ratio=r["comm_ratio_percent"]
                )
            )
    return "\n".join(lines) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Benchmark minimal DDP individual vs flattened all-reduce.")
    p.add_argument("--world-size", type=int, default=2)
    p.add_argument("--size", choices=list(MODEL_SIZE_PRESETS.keys()), default="xl")
    p.add_argument("--dtype", default="float32")
    p.add_argument("--global-batch-size", type=int, default=2)
    p.add_argument("--context-length", type=int, default=256)
    p.add_argument("--vocab-size", type=int, default=50257)
    p.add_argument("--rope-theta", type=float, default=10000.0)
    p.add_argument("--warmup-steps", type=int, default=2)
    p.add_argument("--measure-steps", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--master-addr", default="127.0.0.1")
    p.add_argument("--master-port", default="29521")
    p.add_argument("--output-dir", type=Path, default=Path("results/benchmarks/minimal_ddp_flat_benchmarking"))
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.world_size > torch.cuda.device_count():
        raise RuntimeError(f"Need {args.world_size} GPUs, found {torch.cuda.device_count()}")

    rows = [_run_impl(args, "individual"), _run_impl(args, "flat")]
    cfg = {**vars(args), "output_dir": str(args.output_dir)}
    (args.output_dir / "results.json").write_text(json.dumps({"config": cfg, "results": rows}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (args.output_dir / "table.md").write_text(_to_markdown(rows), encoding="utf-8")

    print(f"JSON results: {args.output_dir / 'results.json'}")
    print(f"Markdown table: {args.output_dir / 'table.md'}")


if __name__ == "__main__":
    main()
