from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, stdev
from timeit import default_timer

import torch

try:
    from cs336_basics.transformer import scaled_dot_product_attention as student_scaled_dot_product_attention
except Exception:
    student_scaled_dot_product_attention = None


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _causal_mask(batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
    tri = torch.tril(torch.ones((seq_len, seq_len), device=device, dtype=torch.bool))
    return tri.unsqueeze(0).expand(batch_size, -1, -1)


def _attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if student_scaled_dot_product_attention is not None:
        return student_scaled_dot_product_attention(q, k, v, mask)

    d_k = q.shape[-1]
    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v)


def _benchmark_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None,
    steps: int,
    warmup_steps: int,
    device: torch.device,
) -> dict:
    with torch.no_grad():
        for _ in range(warmup_steps):
            _ = _attention(q, k, v, mask)
            _sync(device)

        timings: list[float] = []
        for _ in range(steps):
            start = default_timer()
            _ = _attention(q, k, v, mask)
            _sync(device)
            timings.append(default_timer() - start)

    return {
        "forward_timings_seconds": timings,
        "forward_mean_seconds": mean(timings),
        "forward_std_seconds": stdev(timings) if len(timings) > 1 else math.nan,
    }


def _benchmark_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None,
    steps: int,
    warmup_steps: int,
    device: torch.device,
) -> dict:
    for _ in range(warmup_steps):
        q.grad = None
        k.grad = None
        v.grad = None
        out = _attention(q, k, v, mask)
        grad_out = torch.randn_like(out)
        out.backward(grad_out)
        _sync(device)

    timings: list[float] = []
    memory_before_backward_bytes: list[int] = []

    for _ in range(steps):
        q.grad = None
        k.grad = None
        v.grad = None

        out = _attention(q, k, v, mask)
        _sync(device)
        memory_before_backward_bytes.append(int(torch.cuda.memory_allocated(device)))

        grad_out = torch.randn_like(out)
        start = default_timer()
        out.backward(grad_out)
        _sync(device)
        timings.append(default_timer() - start)

    mib = 1024.0 * 1024.0
    mem_mib = [x / mib for x in memory_before_backward_bytes]
    return {
        "backward_timings_seconds": timings,
        "backward_mean_seconds": mean(timings),
        "backward_std_seconds": stdev(timings) if len(timings) > 1 else math.nan,
        "memory_before_backward_bytes": memory_before_backward_bytes,
        "memory_before_backward_mean_mib": mean(mem_mib),
        "memory_before_backward_max_mib": max(mem_mib),
    }


def _run_single_config(args: argparse.Namespace, d_model: int, seq_len: int) -> dict:
    device = torch.device(args.device)
    dtype = torch.float32

    if device.type != "cuda":
        raise RuntimeError("This benchmark script supports CUDA only.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    q = torch.randn(args.batch_size, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(args.batch_size, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(args.batch_size, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)
    mask = _causal_mask(args.batch_size, seq_len, device) if args.causal else None

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    forward_stats = _benchmark_forward(
        q=q.detach(),
        k=k.detach(),
        v=v.detach(),
        mask=mask,
        steps=args.steps,
        warmup_steps=args.warmup_steps,
        device=device,
    )
    backward_stats = _benchmark_backward(
        q=q,
        k=k,
        v=v,
        mask=mask,
        steps=args.steps,
        warmup_steps=args.warmup_steps,
        device=device,
    )

    peak_allocated_bytes = int(torch.cuda.max_memory_allocated(device))
    peak_reserved_bytes = int(torch.cuda.max_memory_reserved(device))

    result = {
        "batch_size": args.batch_size,
        "d_model": d_model,
        "seq_len": seq_len,
        "causal": args.causal,
        "status": "ok",
        **forward_stats,
        **backward_stats,
        "peak_allocated_mib": peak_allocated_bytes / (1024.0 * 1024.0),
        "peak_reserved_mib": peak_reserved_bytes / (1024.0 * 1024.0),
    }

    del q, k, v, mask
    torch.cuda.empty_cache()
    return result


def _to_markdown(results: list[dict]) -> str:
    lines = [
        "| d_model | seq_len | status | fwd_mean_ms | bwd_mean_ms | mem_before_bwd_mean_mib | mem_before_bwd_max_mib | peak_allocated_mib |",
        "|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        if r["status"] != "ok":
            lines.append(f"| {r['d_model']} | {r['seq_len']} | {r['status']} | - | - | - | - | - |")
            continue
        lines.append(
            "| {d_model} | {seq_len} | ok | {fwd:.3f} | {bwd:.3f} | {mb_mean:.2f} | {mb_max:.2f} | {peak:.2f} |".format(
                d_model=r["d_model"],
                seq_len=r["seq_len"],
                fwd=r["forward_mean_seconds"] * 1000.0,
                bwd=r["backward_mean_seconds"] * 1000.0,
                mb_mean=r["memory_before_backward_mean_mib"],
                mb_max=r["memory_before_backward_max_mib"],
                peak=r["peak_allocated_mib"],
            )
        )
    return "\n".join(lines) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark PyTorch attention across sizes.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--d-models", nargs="+", type=int, default=[16, 32, 64, 128])
    parser.add_argument("--seq-lens", nargs="+", type=int, default=[256, 1024, 4096, 8192, 16384])
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--device", choices=["cuda"], default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--causal", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/benchmarks/pytorch_attention"))
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    for d_model in args.d_models:
        for seq_len in args.seq_lens:
            record = {
                "batch_size": args.batch_size,
                "d_model": d_model,
                "seq_len": seq_len,
                "causal": args.causal,
                "status": "unknown",
            }
            try:
                record = _run_single_config(args=args, d_model=d_model, seq_len=seq_len)
            except torch.cuda.OutOfMemoryError as exc:
                record["status"] = "oom"
                record["error"] = str(exc)
                torch.cuda.empty_cache()
            except RuntimeError as exc:
                record["status"] = "runtime_error"
                record["error"] = str(exc)
                torch.cuda.empty_cache()

            results.append(record)
            with (args.output_dir / "summary.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    payload = {"config": {**vars(args), "output_dir": str(args.output_dir)}, "results": results}
    out_json = args.output_dir / "results.json"
    out_md = args.output_dir / "table.md"

    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    out_md.write_text(_to_markdown(results), encoding="utf-8")

    ok_count = sum(1 for r in results if r.get("status") == "ok")
    oom_count = sum(1 for r in results if r.get("status") == "oom")
    print(f"Completed {len(results)} configs: ok={ok_count}, oom={oom_count}")
    print(f"JSON results: {out_json}")
    print(f"Markdown table: {out_md}")


if __name__ == "__main__":
    main()
