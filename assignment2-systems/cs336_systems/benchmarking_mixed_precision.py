from __future__ import annotations

import argparse
import contextlib
import json
import math
from pathlib import Path
from statistics import mean, stdev
from timeit import default_timer

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from cs336_systems.benchmarking_script import MODEL_SIZE_PRESETS, _build_model, _sync_if_needed
except Exception:
    from benchmarking_script import MODEL_SIZE_PRESETS, _build_model, _sync_if_needed


class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.ln(x)
        x = self.fc2(x)
        return x


def _autocast_context(device: torch.device, enabled: bool, dtype: torch.dtype):
    if enabled:
        return torch.autocast(device_type=device.type, dtype=dtype)
    return contextlib.nullcontext()


def inspect_toy_model_fp16(device: torch.device) -> dict:
    if device.type != "cuda":
        raise RuntimeError("Toy mixed-precision inspection requires CUDA.")

    torch.manual_seed(42)
    model = ToyModel(in_features=8, out_features=4).to(device=device, dtype=torch.float32)
    model.train()

    activations: dict[str, str] = {}

    def fc1_hook(_module, _inputs, output):
        activations["fc1_output_dtype"] = str(output.dtype)

    def ln_hook(_module, _inputs, output):
        activations["ln_output_dtype"] = str(output.dtype)

    h1 = model.fc1.register_forward_hook(fc1_hook)
    h2 = model.ln.register_forward_hook(ln_hook)

    x = torch.randn(16, 8, device=device, dtype=torch.float32)
    y = torch.randint(0, 4, (16,), device=device, dtype=torch.long)

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        logits = model(x)
        loss = F.cross_entropy(logits, y)

    loss.backward()
    _sync_if_needed(device)

    grad_dtype = None
    for p in model.parameters():
        if p.grad is not None:
            grad_dtype = str(p.grad.dtype)
            break

    h1.remove()
    h2.remove()

    return {
        "param_dtype": str(next(model.parameters()).dtype),
        "fc1_output_dtype": activations.get("fc1_output_dtype", "unknown"),
        "ln_output_dtype": activations.get("ln_output_dtype", "unknown"),
        "logits_dtype": str(logits.dtype),
        "loss_dtype": str(loss.dtype),
        "grad_dtype": grad_dtype,
    }


def _single_step(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
    mode: str,
    use_mixed_precision: bool,
    mixed_precision_dtype: torch.dtype,
) -> float | None:
    context = _autocast_context(device, use_mixed_precision, mixed_precision_dtype)

    if mode == "forward":
        with torch.no_grad():
            with context:
                _ = model(inputs)
        _sync_if_needed(device)
        return None

    if mode == "backward":
        with context:
            logits = model(inputs)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        _sync_if_needed(device)
        start = default_timer()
        loss.backward()
        _sync_if_needed(device)
        elapsed = default_timer() - start
        model.zero_grad(set_to_none=True)
        return elapsed

    raise ValueError(f"Unsupported mode: {mode}")


def benchmark_model_sizes(args: argparse.Namespace) -> list[dict]:
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    results: list[dict] = []
    for size in args.sizes:
        for mode in ["forward", "backward"]:
            for use_mixed in [False, True]:
                precision_label = "mixed_bf16" if use_mixed else "full_fp32"

                run_record = {
                    "size": size,
                    "mode": mode,
                    "precision": precision_label,
                    "batch_size": args.batch_size,
                    "context_length": args.context_length,
                    "vocab_size": args.vocab_size,
                    "warmup_steps": args.warmup_steps,
                    "measure_steps": args.measure_steps,
                    "status": "ok",
                }

                try:
                    model_args = argparse.Namespace(
                        size=size,
                        d_model=MODEL_SIZE_PRESETS[size]["d_model"],
                        d_ff=MODEL_SIZE_PRESETS[size]["d_ff"],
                        num_layers=MODEL_SIZE_PRESETS[size]["num_layers"],
                        num_heads=MODEL_SIZE_PRESETS[size]["num_heads"],
                        vocab_size=args.vocab_size,
                        context_length=args.context_length,
                        rope_theta=args.rope_theta,
                    )
                    model = _build_model(model_args, device=device, dtype=torch.float32)
                    if mode == "forward":
                        model.eval()
                    else:
                        model.train()

                    inputs = torch.randint(
                        low=0,
                        high=args.vocab_size,
                        size=(args.batch_size, args.context_length),
                        device=device,
                        dtype=torch.long,
                    )
                    targets = torch.randint(
                        low=0,
                        high=args.vocab_size,
                        size=(args.batch_size, args.context_length),
                        device=device,
                        dtype=torch.long,
                    )

                    for _ in range(args.warmup_steps):
                        _ = _single_step(
                            model=model,
                            inputs=inputs,
                            targets=targets,
                            device=device,
                            mode=mode,
                            use_mixed_precision=use_mixed,
                            mixed_precision_dtype=torch.bfloat16,
                        )

                    timings: list[float] = []
                    for _ in range(args.measure_steps):
                        if mode == "backward":
                            elapsed = _single_step(
                                model=model,
                                inputs=inputs,
                                targets=targets,
                                device=device,
                                mode=mode,
                                use_mixed_precision=use_mixed,
                                mixed_precision_dtype=torch.bfloat16,
                            )
                            if elapsed is None:
                                raise RuntimeError("Backward mode must return elapsed timing.")
                        else:
                            start = default_timer()
                            _ = _single_step(
                                model=model,
                                inputs=inputs,
                                targets=targets,
                                device=device,
                                mode=mode,
                                use_mixed_precision=use_mixed,
                                mixed_precision_dtype=torch.bfloat16,
                            )
                            elapsed = default_timer() - start
                        timings.append(elapsed)

                    run_record["timings_seconds"] = timings
                    run_record["mean_seconds"] = mean(timings)
                    run_record["std_seconds"] = stdev(timings) if len(timings) > 1 else math.nan

                    del model, inputs, targets
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

                except torch.cuda.OutOfMemoryError as exc:
                    run_record["status"] = "oom"
                    run_record["error"] = str(exc)
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                except RuntimeError as exc:
                    run_record["status"] = "runtime_error"
                    run_record["error"] = str(exc)
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

                results.append(run_record)

    return results


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark mixed precision behavior for toy model and Transformer.")
    parser.add_argument("--sizes", nargs="+", default=["small", "medium", "large", "xl", "2.7b"])
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--rope-theta", type=float, default=10_000.0)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measure-steps", type=int, default=10)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/benchmarks/benchmarking_mixed_precision/results.json"),
    )
    parser.add_argument("--skip-toy-inspection", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    device = torch.device(args.device)

    payload: dict[str, object] = {
        "config": {
            "sizes": args.sizes,
            "vocab_size": args.vocab_size,
            "batch_size": args.batch_size,
            "context_length": args.context_length,
            "warmup_steps": args.warmup_steps,
            "measure_steps": args.measure_steps,
            "device": args.device,
        }
    }

    if not args.skip_toy_inspection:
        payload["toy_model_fp16_autocast"] = inspect_toy_model_fp16(device)

    payload["size_benchmarks"] = benchmark_model_sizes(args)

    text = json.dumps(payload, indent=2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
