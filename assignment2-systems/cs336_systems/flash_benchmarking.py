from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import triton

try:
    from cs336_systems.flash_attention import FlashAttention2Triton
    from cs336_systems.pytorch_attention import _attention as pytorch_attention
except Exception:
    from flash_attention import FlashAttention2Triton
    from pytorch_attention import _attention as pytorch_attention


def _parse_dtype(name: str) -> torch.dtype:
    key = name.lower()
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if key not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[key]


def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    tri = torch.tril(torch.ones((seq_len, seq_len), device=device, dtype=torch.bool))
    return tri.unsqueeze(0)


def _make_inputs(
    batch_size: int,
    seq_len: int,
    d_model: int,
    dtype: torch.dtype,
    device: torch.device,
    requires_grad: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype, requires_grad=requires_grad)
    k = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype, requires_grad=requires_grad)
    v = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype, requires_grad=requires_grad)
    return q, k, v


def _flash_forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return FlashAttention2Triton.apply(q, k, v, True)


def _pytorch_forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return pytorch_attention(q, k, v, mask)


def _run_do_bench(fn, warmup_ms: int, rep_ms: int) -> float:
    return float(triton.testing.do_bench(fn, warmup=warmup_ms, rep=rep_ms))


def _bench_impl(
    impl: str,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
    warmup_ms: int,
    rep_ms: int,
) -> dict:
    if impl == "flash":
        fwd_impl = _flash_forward

        def run_forward(xq, xk, xv):
            return fwd_impl(xq, xk, xv)

    elif impl == "pytorch":

        def run_forward(xq, xk, xv):
            return _pytorch_forward(xq, xk, xv, mask)

    else:
        raise ValueError(f"Unsupported impl: {impl}")

    with torch.no_grad():
        q_fwd, k_fwd, v_fwd = q.detach(), k.detach(), v.detach()

        def forward_fn():
            _ = run_forward(q_fwd, k_fwd, v_fwd)

        forward_ms = _run_do_bench(forward_fn, warmup_ms=warmup_ms, rep_ms=rep_ms)

    q_bwd = q.detach().clone().requires_grad_(True)
    k_bwd = k.detach().clone().requires_grad_(True)
    v_bwd = v.detach().clone().requires_grad_(True)
    out_bwd = run_forward(q_bwd, k_bwd, v_bwd)
    grad_out = torch.randn_like(out_bwd)

    def backward_fn():
        q_bwd.grad = None
        k_bwd.grad = None
        v_bwd.grad = None
        out_bwd.backward(grad_out, retain_graph=True)

    backward_ms = _run_do_bench(backward_fn, warmup_ms=warmup_ms, rep_ms=rep_ms)

    q_e2e = q.detach().clone().requires_grad_(True)
    k_e2e = k.detach().clone().requires_grad_(True)
    v_e2e = v.detach().clone().requires_grad_(True)

    def e2e_fn():
        q_e2e.grad = None
        k_e2e.grad = None
        v_e2e.grad = None
        out = run_forward(q_e2e, k_e2e, v_e2e)
        out.backward(torch.ones_like(out))

    e2e_ms = _run_do_bench(e2e_fn, warmup_ms=warmup_ms, rep_ms=rep_ms)

    return {
        "forward_ms": forward_ms,
        "backward_ms": backward_ms,
        "end_to_end_ms": e2e_ms,
    }


def _to_markdown(rows: list[dict]) -> str:
    lines = [
        "| impl | dtype | d_model | seq_len | status | fwd_ms | bwd_ms | fwd_bwd_ms |",
        "|---|---|---:|---:|---|---:|---:|---:|",
    ]
    for row in rows:
        if row["status"] != "ok":
            lines.append(
                f"| {row['impl']} | {row['dtype']} | {row['d_model']} | {row['seq_len']} | {row['status']} | - | - | - |"
            )
            continue
        lines.append(
            "| {impl} | {dtype} | {d_model} | {seq_len} | ok | {fwd:.3f} | {bwd:.3f} | {e2e:.3f} |".format(
                impl=row["impl"],
                dtype=row["dtype"],
                d_model=row["d_model"],
                seq_len=row["seq_len"],
                fwd=row["forward_ms"],
                bwd=row["backward_ms"],
                e2e=row["end_to_end_ms"],
            )
        )
    return "\n".join(lines) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark FlashAttention-2 vs regular PyTorch attention.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-lens", nargs="+", type=int, default=[2**i for i in range(7, 17)])
    parser.add_argument("--d-models", nargs="+", type=int, default=[16, 32, 64, 128])
    parser.add_argument("--dtypes", nargs="+", type=str, default=["bfloat16", "float32"])
    parser.add_argument("--device", choices=["cuda"], default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-ms", type=int, default=100)
    parser.add_argument("--rep-ms", type=int, default=200)
    parser.add_argument("--output-dir", type=Path, default=Path("results/benchmarks/flash_benchmarking"))
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("flash_benchmarking requires CUDA.")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    rows: list[dict] = []
    for dtype_name in args.dtypes:
        dtype = _parse_dtype(dtype_name)
        for d_model in args.d_models:
            if d_model < 16 or (d_model & (d_model - 1)) != 0:
                raise ValueError(f"d_model must be power-of-two and >= 16: got {d_model}")
            for seq_len in args.seq_lens:
                if seq_len < 128 or (seq_len & (seq_len - 1)) != 0:
                    raise ValueError(f"seq_len must be power-of-two and >= 128: got {seq_len}")

                mask = _causal_mask(seq_len, device)
                base = {
                    "batch_size": args.batch_size,
                    "dtype": dtype_name,
                    "d_model": d_model,
                    "seq_len": seq_len,
                    "causal": True,
                }

                for impl in ["flash", "pytorch"]:
                    record = {**base, "impl": impl, "status": "unknown"}
                    try:
                        torch.cuda.empty_cache()
                        q, k, v = _make_inputs(
                            batch_size=args.batch_size,
                            seq_len=seq_len,
                            d_model=d_model,
                            dtype=dtype,
                            device=device,
                            requires_grad=False,
                        )
                        stats = _bench_impl(
                            impl=impl,
                            q=q,
                            k=k,
                            v=v,
                            mask=mask,
                            warmup_ms=args.warmup_ms,
                            rep_ms=args.rep_ms,
                        )
                        record.update(stats)
                        record["status"] = "ok"
                    except torch.cuda.OutOfMemoryError as exc:
                        record["status"] = "oom"
                        record["error"] = str(exc)
                        torch.cuda.empty_cache()
                    except RuntimeError as exc:
                        err = str(exc)
                        err_lower = err.lower()
                        if (
                            "out of memory" in err_lower
                            or ("cuda error" in err_lower and "memory" in err_lower)
                            or "out of resource" in err_lower
                            or "shared memory" in err_lower
                        ):
                            record["status"] = "oom"
                        else:
                            record["status"] = "runtime_error"
                        record["error"] = err
                        torch.cuda.empty_cache()
                    except Exception as exc:  # Catch Triton OutOfResources and similar non-RuntimeError failures.
                        err = str(exc)
                        err_lower = err.lower()
                        if (
                            "out of memory" in err_lower
                            or ("cuda error" in err_lower and "memory" in err_lower)
                            or "out of resource" in err_lower
                            or "shared memory" in err_lower
                        ):
                            record["status"] = "oom"
                        else:
                            record["status"] = "runtime_error"
                        record["error"] = err
                        torch.cuda.empty_cache()

                    rows.append(record)
                    with (args.output_dir / "summary.jsonl").open("a", encoding="utf-8") as f:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    payload = {
        "config": {
            **vars(args),
            "output_dir": str(args.output_dir),
        },
        "results": rows,
    }
    (args.output_dir / "results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (args.output_dir / "table.md").write_text(_to_markdown(rows), encoding="utf-8")

    ok = sum(1 for x in rows if x.get("status") == "ok")
    oom = sum(1 for x in rows if x.get("status") == "oom")
    print(f"Completed {len(rows)} runs: ok={ok}, oom={oom}")
    print(f"JSON results: {args.output_dir / 'results.json'}")
    print(f"Markdown table: {args.output_dir / 'table.md'}")


if __name__ == "__main__":
    main()
