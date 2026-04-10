from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import math
import sys
from pathlib import Path
from statistics import mean, stdev
from timeit import default_timer
from typing import Optional

import torch
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
            # Fallback for plain conda/venv runs where cs336-basics is not installed.
            repo_root = Path(__file__).resolve().parents[2]
            assignment1_root = repo_root / "assignment1-basics"
            if assignment1_root.exists():
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


def _sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _load_adamw_cls():
    try:
        module = importlib.import_module("cs336_basics.optimizer")
        return getattr(module, "AdamW")
    except (ImportError, AttributeError):
        repo_root = Path(__file__).resolve().parents[2]
        assignment1_root = repo_root / "assignment1-basics"
        if assignment1_root.exists() and str(assignment1_root) not in sys.path:
            sys.path.insert(0, str(assignment1_root))
        module = importlib.import_module("cs336_basics.optimizer")
        return getattr(module, "AdamW")


def _nvtx_range(name: str, enabled: bool):
    if enabled and torch.cuda.is_available():
        return torch.cuda.nvtx.range(name)
    return contextlib.nullcontext()


def _build_model(args: argparse.Namespace, device: torch.device, dtype: torch.dtype) -> torch.nn.Module:
    transformer_cls = _load_transformer_cls()

    if args.size is not None:
        preset = MODEL_SIZE_PRESETS[args.size]
        d_model = preset["d_model"]
        d_ff = preset["d_ff"]
        num_layers = preset["num_layers"]
        num_heads = preset["num_heads"]
    else:
        d_model = args.d_model
        d_ff = args.d_ff
        num_layers = args.num_layers
        num_heads = args.num_heads

    model_kwargs = {
        "vocab_size": args.vocab_size,
        "context_length": args.context_length,
        "d_model": d_model,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "d_ff": d_ff,
        "rope_theta": args.rope_theta,
    }

    # Some student implementations may not expose dtype/device args.
    try:
        model = transformer_cls(**model_kwargs, device=device, dtype=dtype)
    except TypeError:
        model = transformer_cls(**model_kwargs)
        model = model.to(device=device, dtype=dtype)

    return model


def _benchmark_step(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
    mode: str,
    enable_nvtx: bool,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Optional[float]:
    if mode == "forward":
        with _nvtx_range("forward", enable_nvtx):
            with torch.no_grad():
                _ = model(inputs)
    elif mode == "forward_backward":
        with _nvtx_range("forward", enable_nvtx):
            logits = model(inputs)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        with _nvtx_range("backward", enable_nvtx):
            loss.backward()
        model.zero_grad(set_to_none=True)
    elif mode == "backward":
        with _nvtx_range("forward_for_backward", enable_nvtx):
            logits = model(inputs)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        # Exclude forward compute from the backward-only timing.
        _sync_if_needed(device)
        with _nvtx_range("backward", enable_nvtx):
            start = default_timer()
            loss.backward()
        _sync_if_needed(device)
        elapsed = default_timer() - start
        model.zero_grad(set_to_none=True)
        return elapsed
    elif mode == "train_step":
        if optimizer is None:
            raise ValueError("train_step mode requires an optimizer.")
        with _nvtx_range("forward", enable_nvtx):
            logits = model(inputs)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        with _nvtx_range("backward", enable_nvtx):
            loss.backward()
        with _nvtx_range("optimizer_step", enable_nvtx):
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    _sync_if_needed(device)
    return None


def run_benchmark(args: argparse.Namespace) -> dict:
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")

    device = torch.device(args.device)
    dtype = _parse_dtype(args.dtype)

    if device.type == "cpu" and dtype in {torch.float16, torch.bfloat16}:
        raise RuntimeError("float16/bfloat16 benchmarking on CPU is not supported by this script.")

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    model = _build_model(args, device, dtype)
    if args.mode == "forward":
        model.eval()
    else:
        model.train()

    optimizer: Optional[torch.optim.Optimizer] = None
    if args.mode == "train_step":
        if args.optimizer == "torch_adamw":
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=args.learning_rate,
                betas=(args.beta1, args.beta2),
                eps=args.eps,
                weight_decay=args.weight_decay,
            )
        elif args.optimizer == "cs336_adamw":
            adamw_cls = _load_adamw_cls()
            optimizer = adamw_cls(
                model.parameters(),
                lr=args.learning_rate,
                betas=(args.beta1, args.beta2),
                eps=args.eps,
                weight_decay=args.weight_decay,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {args.optimizer}")

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

    with _nvtx_range("warmup", args.enable_nvtx):
        for _ in range(args.warmup_steps):
            _ = _benchmark_step(
                model=model,
                inputs=inputs,
                targets=targets,
                device=device,
                mode=args.mode,
                enable_nvtx=args.enable_nvtx,
                optimizer=optimizer,
            )

    timings: list[float] = []
    with _nvtx_range("measure", args.enable_nvtx):
        for _ in range(args.measure_steps):
            if args.mode == "backward":
                elapsed = _benchmark_step(
                    model=model,
                    inputs=inputs,
                    targets=targets,
                    device=device,
                    mode=args.mode,
                    enable_nvtx=args.enable_nvtx,
                    optimizer=optimizer,
                )
                if elapsed is None:
                    raise RuntimeError("Internal error: backward mode expected a timing result.")
            else:
                start = default_timer()
                _ = _benchmark_step(
                    model=model,
                    inputs=inputs,
                    targets=targets,
                    device=device,
                    mode=args.mode,
                    enable_nvtx=args.enable_nvtx,
                    optimizer=optimizer,
                )
                elapsed = default_timer() - start
            timings.append(elapsed)

    params_count = sum(p.numel() for p in model.parameters())
    result = {
        "mode": args.mode,
        "size": args.size,
        "batch_size": args.batch_size,
        "context_length": args.context_length,
        "vocab_size": args.vocab_size,
        "warmup_steps": args.warmup_steps,
        "measure_steps": args.measure_steps,
        "device": str(device),
        "dtype": str(dtype),
        "enable_nvtx": args.enable_nvtx,
        "optimizer": args.optimizer if args.mode == "train_step" else None,
        "num_parameters": int(params_count),
        "timings_seconds": timings,
        "mean_seconds": mean(timings),
        "std_seconds": stdev(timings) if len(timings) > 1 else math.nan,
    }
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark Transformer forward/backward runtime.")
    parser.add_argument("--size", choices=sorted(MODEL_SIZE_PRESETS.keys()), default="small")
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--rope-theta", type=float, default=10_000.0)
    parser.add_argument(
        "--mode",
        choices=["forward", "forward_backward", "backward", "train_step"],
        default="forward_backward",
    )
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measure-steps", type=int, default=10)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--enable-nvtx", action="store_true")
    parser.add_argument("--optimizer", choices=["cs336_adamw", "torch_adamw"], default="cs336_adamw")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-8)

    # Optional custom overrides when not using presets.
    parser.add_argument("--d-model", type=int, default=768)
    parser.add_argument("--d-ff", type=int, default=3072)
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--num-heads", type=int, default=12)

    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    result = run_benchmark(args)
    payload = json.dumps(result, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
