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

try:
    from cs336_systems.ddp_bucketed import DDPBucketed
    from cs336_systems.ddp_individual_parameters import DDPIndividualParameters
except ImportError:
    from ddp_bucketed import DDPBucketed
    from ddp_individual_parameters import DDPIndividualParameters


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


def _sync_grads_individual(model: torch.nn.Module) -> None:
    world_size = dist.get_world_size()
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, async_op=False)
            p.grad.div_(world_size)


def _sync_grads_flat(model: torch.nn.Module) -> None:
    world_size = dist.get_world_size()
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    if not grads:
        return
    flat = _flatten_dense_tensors(grads)
    dist.all_reduce(flat, op=dist.ReduceOp.SUM, async_op=False)
    flat.div_(world_size)
    synced = _unflatten_dense_tensors(flat, grads)
    for g, synced_g in zip(grads, synced):
        g.copy_(synced_g)


def _worker(
    rank: int,
    world_size: int,
    args: argparse.Namespace,
    impl: str,
    bucket_size_mb: float | None,
    out_json: str,
) -> None:
    device = _setup(rank, world_size, args.master_addr, args.master_port)
    dtype = _parse_dtype(args.dtype)
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)

    base_model = _build_model(args, device, dtype)
    base_model.train()

    train_batch_start_fn = lambda: None
    if impl == "bucketed_overlap":
        assert bucket_size_mb is not None
        ddp_model = DDPBucketed(base_model, bucket_size_mb=bucket_size_mb)
        run_model = ddp_model
        train_batch_start_fn = ddp_model.on_train_batch_start
        sync_fn = ddp_model.finish_gradient_synchronization
    elif impl == "overlap_individual":
        ddp_model = DDPIndividualParameters(base_model)
        run_model = ddp_model
        sync_fn = ddp_model.finish_gradient_synchronization
    else:
        run_model = base_model
        _broadcast_rank0(base_model)
        sync_fn = (lambda: _sync_grads_flat(base_model)) if impl == "flat" else (lambda: _sync_grads_individual(base_model))

    optimizer = torch.optim.AdamW(
        run_model.parameters(),
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

    for _ in range(args.warmup_steps):
        train_batch_start_fn()
        optimizer.zero_grad(set_to_none=True)
        logits = run_model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        loss.backward()
        sync_fn()
        optimizer.step()
        torch.cuda.synchronize(device)

    step_ms: list[float] = []
    sync_tail_ms: list[float] = []
    for _ in range(args.measure_steps):
        train_batch_start_fn()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device)
        t0 = default_timer()

        with torch.cuda.nvtx.range("forward"):
            logits = run_model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        with torch.cuda.nvtx.range("backward"):
            loss.backward()

        torch.cuda.synchronize(device)
        c0 = default_timer()
        with torch.cuda.nvtx.range("sync_gradients"):
            sync_fn()
        torch.cuda.synchronize(device)
        c1 = default_timer()

        with torch.cuda.nvtx.range("optimizer_step"):
            optimizer.step()
        torch.cuda.synchronize(device)
        t1 = default_timer()

        step_ms.append((t1 - t0) * 1000.0)
        sync_tail_ms.append((c1 - c0) * 1000.0)

    rank_result = {
        "rank": rank,
        "step_mean_ms": mean(step_ms),
        "sync_tail_mean_ms": mean(sync_tail_ms),
        "sync_tail_ratio": mean(c / s for c, s in zip(sync_tail_ms, step_ms)),
    }
    gathered: list[dict] = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, rank_result)

    if rank == 0:
        step_means = [r["step_mean_ms"] for r in gathered]
        tail_means = [r["sync_tail_mean_ms"] for r in gathered]
        tail_ratios = [r["sync_tail_ratio"] for r in gathered]
        result = {
            "status": "ok",
            "impl": impl if bucket_size_mb is None else f"bucketed_overlap_{bucket_size_mb:g}mb",
            "bucket_size_mb": bucket_size_mb,
            "model_size": args.size,
            "world_size": world_size,
            "dtype": args.dtype,
            "global_batch_size": args.global_batch_size,
            "local_batch_size": local_bs,
            "context_length": args.context_length,
            "warmup_steps": args.warmup_steps,
            "measure_steps": args.measure_steps,
            "step_mean_ms": mean(step_means),
            "step_std_ms": stdev(step_means) if len(step_means) > 1 else 0.0,
            "sync_tail_mean_ms": mean(tail_means),
            "sync_tail_std_ms": stdev(tail_means) if len(tail_means) > 1 else 0.0,
            "sync_tail_ratio_percent": mean(tail_ratios) * 100.0,
        }
        Path(out_json).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    _cleanup()


def _run_case(args: argparse.Namespace, impl: str, bucket_size_mb: float | None = None) -> dict:
    suffix = impl if bucket_size_mb is None else f"{impl}_{bucket_size_mb:g}mb"
    out_json = args.output_dir / f"{suffix}.json"
    try:
        mp.spawn(
            _worker,
            args=(args.world_size, args, impl, bucket_size_mb, str(out_json)),
            nprocs=args.world_size,
            join=True,
        )
        return json.loads(out_json.read_text(encoding="utf-8"))
    except torch.cuda.OutOfMemoryError as exc:
        return {"status": "oom", "impl": suffix, "bucket_size_mb": bucket_size_mb, "error": str(exc)}
    except Exception as exc:
        msg = str(exc)
        status = "oom" if "out of memory" in msg.lower() else "runtime_error"
        return {"status": status, "impl": suffix, "bucket_size_mb": bucket_size_mb, "error": msg}


def _to_markdown(rows: list[dict]) -> str:
    lines = [
        "| impl | bucket_mb | status | step_mean_ms | sync_tail_mean_ms | sync_tail_ratio_% |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for row in rows:
        bucket = row.get("bucket_size_mb")
        bucket_s = "-" if bucket is None else f"{bucket:g}"
        if row.get("status") != "ok":
            lines.append(f"| {row.get('impl', '-')} | {bucket_s} | {row.get('status', 'unknown')} | - | - | - |")
        else:
            lines.append(
                "| {impl} | {bucket} | ok | {step:.3f} | {tail:.3f} | {ratio:.2f} |".format(
                    impl=row["impl"],
                    bucket=bucket_s,
                    step=row["step_mean_ms"],
                    tail=row["sync_tail_mean_ms"],
                    ratio=row["sync_tail_ratio_percent"],
                )
            )
    return "\n".join(lines) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark bucketed overlap DDP across bucket sizes.")
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--size", choices=list(MODEL_SIZE_PRESETS.keys()), default="xl")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--global-batch-size", type=int, default=2)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--vocab-size", type=int, default=50257)
    parser.add_argument("--rope-theta", type=float, default=10000.0)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--measure-steps", type=int, default=5)
    parser.add_argument("--bucket-sizes-mb", nargs="+", type=float, default=[1, 10, 100, 1000])
    parser.add_argument("--include-baselines", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--master-addr", default="127.0.0.1")
    parser.add_argument("--master-port", default="29541")
    parser.add_argument("--output-dir", type=Path, default=Path("results/benchmarks/ddp_bucketed_benchmarking"))
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.world_size > torch.cuda.device_count():
        raise RuntimeError(f"Need {args.world_size} GPUs, found {torch.cuda.device_count()}")

    rows: list[dict] = []
    if args.include_baselines:
        rows.extend([
            _run_case(args, "individual"),
            _run_case(args, "flat"),
            _run_case(args, "overlap_individual"),
        ])
    rows.extend(_run_case(args, "bucketed_overlap", bucket_size) for bucket_size in args.bucket_sizes_mb)

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
