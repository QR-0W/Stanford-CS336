from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, stdev
from timeit import default_timer
from typing import Callable

import torch

from benchmarking_script import _benchmark_step, _build_model
from pytorch_attention import _attention as eager_attention
from pytorch_attention import _causal_mask, _sync


def _compile_attention_fn(enable_compile: bool) -> Callable:
    def fn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        return eager_attention(q, k, v, mask)

    if not enable_compile:
        return fn
    return torch.compile(fn, fullgraph=False)


def _bench_attention_single(
    attention_fn: Callable,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None,
    warmup_steps: int,
    steps: int,
    device: torch.device,
) -> dict:
    with torch.no_grad():
        for _ in range(warmup_steps):
            _ = attention_fn(q.detach(), k.detach(), v.detach(), mask)
            _sync(device)

    fwd_timings: list[float] = []
    with torch.no_grad():
        for _ in range(steps):
            start = default_timer()
            _ = attention_fn(q.detach(), k.detach(), v.detach(), mask)
            _sync(device)
            fwd_timings.append(default_timer() - start)

    for _ in range(warmup_steps):
        q.grad = None
        k.grad = None
        v.grad = None
        out = attention_fn(q, k, v, mask)
        out.backward(torch.randn_like(out))
        _sync(device)

    bwd_timings: list[float] = []
    mem_before_bwd: list[int] = []
    for _ in range(steps):
        q.grad = None
        k.grad = None
        v.grad = None
        out = attention_fn(q, k, v, mask)
        _sync(device)
        mem_before_bwd.append(int(torch.cuda.memory_allocated(device)))
        grad_out = torch.randn_like(out)
        start = default_timer()
        out.backward(grad_out)
        _sync(device)
        bwd_timings.append(default_timer() - start)

    mem_mib = [x / (1024.0 * 1024.0) for x in mem_before_bwd]
    return {
        "forward_timings_seconds": fwd_timings,
        "forward_mean_seconds": mean(fwd_timings),
        "forward_std_seconds": stdev(fwd_timings) if len(fwd_timings) > 1 else math.nan,
        "backward_timings_seconds": bwd_timings,
        "backward_mean_seconds": mean(bwd_timings),
        "backward_std_seconds": stdev(bwd_timings) if len(bwd_timings) > 1 else math.nan,
        "memory_before_backward_mean_mib": mean(mem_mib),
        "memory_before_backward_max_mib": max(mem_mib),
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0),
    }


def run_attention_comparison(args: argparse.Namespace) -> list[dict]:
    device = torch.device(args.device)
    rows: list[dict] = []

    for compiled in [False, True]:
        attention_fn = _compile_attention_fn(enable_compile=compiled)
        for d_model in args.attention_d_models:
            for seq_len in args.attention_seq_lens:
                record = {
                    "implementation": "compiled" if compiled else "eager",
                    "batch_size": args.attention_batch_size,
                    "d_model": d_model,
                    "seq_len": seq_len,
                    "status": "unknown",
                }
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats(device)

                    q = torch.randn(
                        args.attention_batch_size,
                        seq_len,
                        d_model,
                        device=device,
                        dtype=torch.float32,
                        requires_grad=True,
                    )
                    k = torch.randn_like(q, requires_grad=True)
                    v = torch.randn_like(q, requires_grad=True)
                    mask = _causal_mask(args.attention_batch_size, seq_len, device)

                    stats = _bench_attention_single(
                        attention_fn=attention_fn,
                        q=q,
                        k=k,
                        v=v,
                        mask=mask,
                        warmup_steps=args.warmup_steps,
                        steps=args.measure_steps,
                        device=device,
                    )
                    record.update(stats)
                    record["status"] = "ok"

                    del q, k, v, mask
                    torch.cuda.empty_cache()
                except torch.cuda.OutOfMemoryError as exc:
                    record["status"] = "oom"
                    record["error"] = str(exc)
                    torch.cuda.empty_cache()
                except RuntimeError as exc:
                    record["status"] = "runtime_error"
                    record["error"] = str(exc)
                    torch.cuda.empty_cache()

                rows.append(record)
    return rows


def _build_transformer_namespace(args: argparse.Namespace, size: str, mode: str) -> argparse.Namespace:
    return argparse.Namespace(
        size=size,
        d_model=0,
        d_ff=0,
        num_layers=0,
        num_heads=0,
        vocab_size=args.transformer_vocab_size,
        context_length=args.transformer_context_length,
        rope_theta=args.transformer_rope_theta,
        mode=mode,
        optimizer="torch_adamw",
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        beta1=args.beta1,
        beta2=args.beta2,
        eps=args.eps,
        enable_nvtx=False,
    )


def _bench_transformer_single(
    model: torch.nn.Module,
    mode: str,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    warmup_steps: int,
    measure_steps: int,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> dict:
    for _ in range(warmup_steps):
        _ = _benchmark_step(
            model=model,
            inputs=inputs,
            targets=targets,
            device=device,
            mode=mode,
            enable_nvtx=False,
            optimizer=optimizer,
        )

    timings: list[float] = []
    for _ in range(measure_steps):
        if mode == "backward":
            elapsed = _benchmark_step(
                model=model,
                inputs=inputs,
                targets=targets,
                device=device,
                mode=mode,
                enable_nvtx=False,
                optimizer=optimizer,
            )
            if elapsed is None:
                raise RuntimeError("backward mode did not return timing")
        else:
            start = default_timer()
            _ = _benchmark_step(
                model=model,
                inputs=inputs,
                targets=targets,
                device=device,
                mode=mode,
                enable_nvtx=False,
                optimizer=optimizer,
            )
            elapsed = default_timer() - start
        timings.append(elapsed)

    return {
        "timings_seconds": timings,
        "mean_seconds": mean(timings),
        "std_seconds": stdev(timings) if len(timings) > 1 else math.nan,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0),
    }


def run_transformer_comparison(args: argparse.Namespace) -> list[dict]:
    device = torch.device(args.device)
    rows: list[dict] = []

    for compiled in [False, True]:
        for size in args.transformer_sizes:
            for mode in args.transformer_modes:
                record = {
                    "implementation": "compiled" if compiled else "eager",
                    "size": size,
                    "mode": mode,
                    "batch_size": args.transformer_batch_size,
                    "context_length": args.transformer_context_length,
                    "status": "unknown",
                }
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats(device)

                    model_args = _build_transformer_namespace(args, size=size, mode=mode)
                    model = _build_model(model_args, device=device, dtype=torch.float32)
                    if compiled:
                        model = torch.compile(model, fullgraph=False)
                    if mode == "forward":
                        model.eval()
                    else:
                        model.train()

                    optimizer = None
                    if mode == "train_step":
                        optimizer = torch.optim.AdamW(
                            model.parameters(),
                            lr=args.learning_rate,
                            betas=(args.beta1, args.beta2),
                            eps=args.eps,
                            weight_decay=args.weight_decay,
                        )

                    inputs = torch.randint(
                        low=0,
                        high=args.transformer_vocab_size,
                        size=(args.transformer_batch_size, args.transformer_context_length),
                        device=device,
                        dtype=torch.long,
                    )
                    targets = torch.randint(
                        low=0,
                        high=args.transformer_vocab_size,
                        size=(args.transformer_batch_size, args.transformer_context_length),
                        device=device,
                        dtype=torch.long,
                    )

                    stats = _bench_transformer_single(
                        model=model,
                        mode=mode,
                        inputs=inputs,
                        targets=targets,
                        warmup_steps=args.warmup_steps,
                        measure_steps=args.measure_steps,
                        device=device,
                        optimizer=optimizer,
                    )
                    record.update(stats)
                    record["status"] = "ok"

                    del model, inputs, targets, optimizer
                    torch.cuda.empty_cache()
                except torch.cuda.OutOfMemoryError as exc:
                    record["status"] = "oom"
                    record["error"] = str(exc)
                    torch.cuda.empty_cache()
                except RuntimeError as exc:
                    record["status"] = "runtime_error"
                    record["error"] = str(exc)
                    torch.cuda.empty_cache()

                rows.append(record)
    return rows


def _attention_table(rows: list[dict]) -> str:
    lines = [
        "| impl | d_model | seq_len | status | fwd_ms | bwd_ms | mem_before_bwd_mib | peak_alloc_mib |",
        "|---|---:|---:|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        if r["status"] != "ok":
            lines.append(f"| {r['implementation']} | {r['d_model']} | {r['seq_len']} | {r['status']} | - | - | - | - |")
            continue
        lines.append(
            f"| {r['implementation']} | {r['d_model']} | {r['seq_len']} | ok | {r['forward_mean_seconds'] * 1000:.3f} | {r['backward_mean_seconds'] * 1000:.3f} | {r['memory_before_backward_mean_mib']:.2f} | {r['peak_allocated_mib']:.2f} |"
        )
    return "\n".join(lines) + "\n"


def _transformer_table(rows: list[dict]) -> str:
    lines = [
        "| impl | size | mode | status | mean_ms | peak_alloc_mib |",
        "|---|---|---|---|---:|---:|",
    ]
    for r in rows:
        if r["status"] != "ok":
            lines.append(f"| {r['implementation']} | {r['size']} | {r['mode']} | {r['status']} | - | - |")
            continue
        lines.append(
            f"| {r['implementation']} | {r['size']} | {r['mode']} | ok | {r['mean_seconds'] * 1000:.3f} | {r['peak_allocated_mib']:.2f} |"
        )
    return "\n".join(lines) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark torch.compile for attention and full Transformer.")
    parser.add_argument("--device", choices=["cuda"], default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--measure-steps", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=Path("results/benchmarks/torch_compile"))

    parser.add_argument("--attention-batch-size", type=int, default=8)
    parser.add_argument("--attention-d-models", nargs="+", type=int, default=[16, 32, 64, 128])
    parser.add_argument("--attention-seq-lens", nargs="+", type=int, default=[256, 1024, 4096, 8192, 16384])

    parser.add_argument("--transformer-sizes", nargs="+", default=["small", "medium", "large"])
    parser.add_argument("--transformer-modes", nargs="+", default=["forward", "forward_backward", "train_step"])
    parser.add_argument("--transformer-batch-size", type=int, default=4)
    parser.add_argument("--transformer-context-length", type=int, default=256)
    parser.add_argument("--transformer-vocab-size", type=int, default=10_000)
    parser.add_argument("--transformer-rope-theta", type=float, default=10_000.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-8)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for torch_compile benchmark.")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    attention_rows = run_attention_comparison(args)
    transformer_rows = run_transformer_comparison(args)

    payload = {
        "config": {**vars(args), "output_dir": str(args.output_dir)},
        "attention_comparison": attention_rows,
        "transformer_comparison": transformer_rows,
    }

    (args.output_dir / "results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "attention_table.md").write_text(_attention_table(attention_rows), encoding="utf-8")
    (args.output_dir / "transformer_table.md").write_text(_transformer_table(transformer_rows), encoding="utf-8")

    print(f"Wrote {args.output_dir / 'results.json'}")
    print(f"Wrote {args.output_dir / 'attention_table.md'}")
    print(f"Wrote {args.output_dir / 'transformer_table.md'}")


if __name__ == "__main__":
    main()
