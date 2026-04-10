from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path
from timeit import default_timer

import torch
import torch.nn.functional as F

try:
    from cs336_systems.benchmarking_script import MODEL_SIZE_PRESETS, _build_model, _load_adamw_cls, _sync_if_needed
except Exception:
    from benchmarking_script import MODEL_SIZE_PRESETS, _build_model, _load_adamw_cls, _sync_if_needed


def _autocast_context(device: torch.device, enabled: bool):
    if enabled:
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    return contextlib.nullcontext()


def _mb(x: int | float) -> float:
    return float(x) / (1024.0 * 1024.0)


def _start_memory_history(max_entries: int) -> None:
    torch.cuda.memory._record_memory_history(
        enabled="all",
        context="all",
        stacks="python",
        max_entries=max_entries,
        clear_history=True,
    )


def _make_model_args(args: argparse.Namespace, context_length: int) -> argparse.Namespace:
    preset = MODEL_SIZE_PRESETS[args.size]
    return argparse.Namespace(
        size=args.size,
        d_model=preset["d_model"],
        d_ff=preset["d_ff"],
        num_layers=preset["num_layers"],
        num_heads=preset["num_heads"],
        vocab_size=args.vocab_size,
        context_length=context_length,
        rope_theta=args.rope_theta,
    )


def _run_forward(model: torch.nn.Module, inputs: torch.Tensor, device: torch.device, use_mixed_precision: bool) -> dict:
    with torch.no_grad():
        with _autocast_context(device, use_mixed_precision):
            _ = model(inputs)
    _sync_if_needed(device)
    return {}


def _run_train_step(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_mixed_precision: bool,
) -> dict:
    stage_allocated_bytes: dict[str, int] = {}
    stage_reserved_bytes: dict[str, int] = {}

    with _autocast_context(device, use_mixed_precision):
        logits = model(inputs)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
    _sync_if_needed(device)
    stage_allocated_bytes["after_forward"] = int(torch.cuda.memory_allocated(device))
    stage_reserved_bytes["after_forward"] = int(torch.cuda.memory_reserved(device))

    loss.backward()
    _sync_if_needed(device)
    stage_allocated_bytes["after_backward"] = int(torch.cuda.memory_allocated(device))
    stage_reserved_bytes["after_backward"] = int(torch.cuda.memory_reserved(device))

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    _sync_if_needed(device)
    stage_allocated_bytes["after_optimizer_step"] = int(torch.cuda.memory_allocated(device))
    stage_reserved_bytes["after_optimizer_step"] = int(torch.cuda.memory_reserved(device))

    return {
        "stage_allocated_bytes": stage_allocated_bytes,
        "stage_reserved_bytes": stage_reserved_bytes,
    }


def run_once(args: argparse.Namespace, context_length: int, mode: str, precision: str) -> dict:
    if args.device != "cuda":
        raise RuntimeError("memory_profiling only supports CUDA runs.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")

    device = torch.device("cuda")
    use_mixed_precision = precision == "mixed_bf16"

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if hasattr(torch.cuda.memory, "_record_memory_history"):
        if args.capture_history_from_init:
            _start_memory_history(max_entries=args.memory_history_max_entries)
    else:
        raise RuntimeError("torch.cuda.memory._record_memory_history is unavailable in this torch build.")

    model_args = _make_model_args(args, context_length)
    model = _build_model(model_args, device=device, dtype=torch.float32)
    if mode == "forward":
        model.eval()
    else:
        model.train()

    optimizer = None
    if mode == "train_step":
        if args.optimizer == "torch_adamw":
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=args.learning_rate,
                betas=(args.beta1, args.beta2),
                eps=args.eps,
                weight_decay=args.weight_decay,
            )
        elif args.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=args.learning_rate,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
            )
        else:
            adamw_cls = _load_adamw_cls()
            optimizer = adamw_cls(
                model.parameters(),
                lr=args.learning_rate,
                betas=(args.beta1, args.beta2),
                eps=args.eps,
                weight_decay=args.weight_decay,
            )

    inputs = torch.randint(
        low=0,
        high=args.vocab_size,
        size=(args.batch_size, context_length),
        device=device,
        dtype=torch.long,
    )
    targets = torch.randint(
        low=0,
        high=args.vocab_size,
        size=(args.batch_size, context_length),
        device=device,
        dtype=torch.long,
    )

    for _ in range(args.warmup_steps):
        if mode == "forward":
            _run_forward(model=model, inputs=inputs, device=device, use_mixed_precision=use_mixed_precision)
        else:
            assert optimizer is not None
            _run_train_step(
                model=model,
                inputs=inputs,
                targets=targets,
                optimizer=optimizer,
                device=device,
                use_mixed_precision=use_mixed_precision,
            )

    torch.cuda.reset_peak_memory_stats(device)
    if not args.capture_history_from_init:
        _start_memory_history(max_entries=args.memory_history_max_entries)

    stage_metrics: dict = {}
    start = default_timer()
    if mode == "forward":
        stage_metrics = _run_forward(model=model, inputs=inputs, device=device, use_mixed_precision=use_mixed_precision)
    else:
        assert optimizer is not None
        stage_metrics = _run_train_step(
            model=model,
            inputs=inputs,
            targets=targets,
            optimizer=optimizer,
            device=device,
            use_mixed_precision=use_mixed_precision,
        )
    elapsed = default_timer() - start
    _sync_if_needed(device)

    snapshot_name = f"size-{args.size}_ctx-{context_length}_mode-{mode}_precision-{precision}.pickle"
    snapshot_path = args.output_dir / "snapshots" / snapshot_name
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    torch.cuda.memory._dump_snapshot(str(snapshot_path))
    torch.cuda.memory._record_memory_history(enabled=None)

    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))

    result = {
        "size": args.size,
        "context_length": context_length,
        "mode": mode,
        "precision": precision,
        "status": "ok",
        "elapsed_seconds": elapsed,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "peak_allocated_mb": _mb(peak_allocated),
        "peak_reserved_mb": _mb(peak_reserved),
        "snapshot_path": str(snapshot_path),
        **stage_metrics,
    }

    del model, inputs, targets, optimizer
    torch.cuda.empty_cache()
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Memory profiling for CS336 Transformer model.")
    parser.add_argument("--size", choices=sorted(MODEL_SIZE_PRESETS.keys()), default="2.7b")
    parser.add_argument("--context-lengths", nargs="+", type=int, default=[128, 256, 512])
    parser.add_argument("--modes", nargs="+", default=["forward", "train_step"])
    parser.add_argument("--precisions", nargs="+", default=["full_fp32", "mixed_bf16"])
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--rope-theta", type=float, default=10_000.0)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--memory-history-max-entries", type=int, default=1_000_000)
    parser.add_argument("--capture-history-from-init", action="store_true")
    parser.add_argument("--device", choices=["cuda"], default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--optimizer", choices=["cs336_adamw", "torch_adamw", "sgd"], default="cs336_adamw")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--output-dir", type=Path, default=Path("results/memory_profiling"))
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    for context_length in args.context_lengths:
        for mode in args.modes:
            for precision in args.precisions:
                record = {
                    "size": args.size,
                    "context_length": context_length,
                    "mode": mode,
                    "precision": precision,
                    "status": "unknown",
                }
                try:
                    record = run_once(args, context_length=context_length, mode=mode, precision=precision)
                except torch.cuda.OutOfMemoryError as exc:
                    record["status"] = "oom"
                    record["error"] = str(exc)
                    if hasattr(torch.cuda.memory, "_record_memory_history"):
                        torch.cuda.memory._record_memory_history(enabled=None)
                    torch.cuda.empty_cache()
                except RuntimeError as exc:
                    record["status"] = "runtime_error"
                    record["error"] = str(exc)
                    if hasattr(torch.cuda.memory, "_record_memory_history"):
                        torch.cuda.memory._record_memory_history(enabled=None)
                    torch.cuda.empty_cache()
                results.append(record)

                text = json.dumps(record, ensure_ascii=False)
                with (args.output_dir / "summary.jsonl").open("a", encoding="utf-8") as f:
                    f.write(text + "\n")

    config = dict(vars(args))
    config["output_dir"] = str(args.output_dir)
    payload = {"config": config, "results": results}
    out_json = args.output_dir / "results.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
