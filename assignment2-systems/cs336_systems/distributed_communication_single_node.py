from __future__ import annotations

import argparse
import json
import os
import socket
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, stdev
from timeit import default_timer

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


BYTES_PER_FLOAT32 = 4


@dataclass
class BenchConfig:
    backend: str
    device_type: str
    world_size: int
    size_mb: int
    warmup_steps: int
    measure_steps: int
    master_addr: str
    master_port: int


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        return int(s.getsockname()[1])


def _sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _setup_process_group(rank: int, cfg: BenchConfig) -> torch.device:
    os.environ["MASTER_ADDR"] = cfg.master_addr
    os.environ["MASTER_PORT"] = str(cfg.master_port)

    if cfg.backend == "nccl":
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cpu")

    dist.init_process_group(backend=cfg.backend, rank=rank, world_size=cfg.world_size)
    return device


def _bench_all_reduce(rank: int, cfg: BenchConfig, out_jsonl: str) -> None:
    device = _setup_process_group(rank=rank, cfg=cfg)

    numel = (cfg.size_mb * 1024 * 1024) // BYTES_PER_FLOAT32
    tensor = torch.randn(numel, dtype=torch.float32, device=device)

    for _ in range(cfg.warmup_steps):
        dist.all_reduce(tensor, async_op=False)
        _sync_if_needed(device)

    per_rank_timings_ms: list[float] = []
    for _ in range(cfg.measure_steps):
        _sync_if_needed(device)
        start = default_timer()
        dist.all_reduce(tensor, async_op=False)
        _sync_if_needed(device)
        per_rank_timings_ms.append((default_timer() - start) * 1000.0)

    per_rank_mean = mean(per_rank_timings_ms)
    all_rank_means: list[float] = [0.0 for _ in range(cfg.world_size)]
    dist.all_gather_object(all_rank_means, per_rank_mean)

    if rank == 0:
        record = {
            **asdict(cfg),
            "status": "ok",
            "rank_mean_ms": all_rank_means,
            "global_mean_ms": mean(all_rank_means),
            "global_std_ms": stdev(all_rank_means) if len(all_rank_means) > 1 else 0.0,
            "global_max_ms": max(all_rank_means),
            "global_min_ms": min(all_rank_means),
        }
        with Path(out_jsonl).open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    dist.destroy_process_group()


def _to_markdown(records: list[dict]) -> str:
    lines = [
        "| backend | device | world_size | size_mb | status | mean_ms | std_ms | min_ms | max_ms |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|",
    ]
    for r in records:
        if r.get("status") != "ok":
            lines.append(
                "| {backend} | {device_type} | {world_size} | {size_mb} | {status} | - | - | - | - |".format(
                    backend=r.get("backend", "-"),
                    device_type=r.get("device_type", "-"),
                    world_size=r.get("world_size", -1),
                    size_mb=r.get("size_mb", -1),
                    status=r.get("status", "unknown"),
                )
            )
            continue

        lines.append(
            "| {backend} | {device_type} | {world_size} | {size_mb} | ok | {mean:.3f} | {std:.3f} | {minv:.3f} | {maxv:.3f} |".format(
                backend=r["backend"],
                device_type=r["device_type"],
                world_size=r["world_size"],
                size_mb=r["size_mb"],
                mean=r["global_mean_ms"],
                std=r["global_std_ms"],
                minv=r["global_min_ms"],
                maxv=r["global_max_ms"],
            )
        )

    return "\n".join(lines) + "\n"


def _run_one(cfg: BenchConfig, out_jsonl: Path) -> dict:
    base = {
        **asdict(cfg),
        "status": "unknown",
    }

    if cfg.backend == "nccl":
        if not torch.cuda.is_available():
            return {**base, "status": "skipped", "error": "CUDA unavailable for NCCL."}
        gpu_count = torch.cuda.device_count()
        if cfg.world_size > gpu_count:
            return {
                **base,
                "status": "skipped",
                "error": f"Requested world_size={cfg.world_size}, but only {gpu_count} GPUs available.",
            }

    try:
        mp.spawn(
            _bench_all_reduce,
            args=(cfg, str(out_jsonl)),
            nprocs=cfg.world_size,
            join=True,
        )

        lines = out_jsonl.read_text(encoding="utf-8").splitlines()
        return json.loads(lines[-1])
    except torch.cuda.OutOfMemoryError as exc:
        return {**base, "status": "oom", "error": str(exc)}
    except RuntimeError as exc:
        err = str(exc)
        if "out of memory" in err.lower():
            return {**base, "status": "oom", "error": err}
        return {**base, "status": "runtime_error", "error": err}
    except Exception as exc:
        return {**base, "status": "runtime_error", "error": str(exc)}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark single-node all-reduce across backends/devices/sizes/process counts.")
    parser.add_argument("--backends", nargs="+", default=["gloo", "nccl"])
    parser.add_argument("--devices", nargs="+", default=["cpu", "cuda"])
    parser.add_argument("--world-sizes", nargs="+", type=int, default=[2, 4, 6])
    parser.add_argument("--sizes-mb", nargs="+", type=int, default=[1, 10, 100, 1024])
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measure-steps", type=int, default=20)
    parser.add_argument("--master-addr", type=str, default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/benchmarks/distributed_communication_single_node"),
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    out_jsonl = args.output_dir / "summary.jsonl"
    out_jsonl.write_text("", encoding="utf-8")

    records: list[dict] = []
    for backend in args.backends:
        for device_type in args.devices:
            if backend == "gloo" and device_type != "cpu":
                continue
            if backend == "nccl" and device_type != "cuda":
                continue

            for world_size in args.world_sizes:
                for size_mb in args.sizes_mb:
                    cfg = BenchConfig(
                        backend=backend,
                        device_type=device_type,
                        world_size=world_size,
                        size_mb=size_mb,
                        warmup_steps=args.warmup_steps,
                        measure_steps=args.measure_steps,
                        master_addr=args.master_addr,
                        master_port=args.master_port if args.master_port > 0 else _find_free_port(),
                    )

                    rec = _run_one(cfg=cfg, out_jsonl=out_jsonl)
                    records.append(rec)

    payload = {
        "config": {**vars(args), "output_dir": str(args.output_dir)},
        "results": records,
    }
    out_json = args.output_dir / "results.json"
    out_md = args.output_dir / "table.md"

    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    out_md.write_text(_to_markdown(records), encoding="utf-8")

    ok_count = sum(1 for r in records if r.get("status") == "ok")
    skip_count = sum(1 for r in records if r.get("status") == "skipped")
    oom_count = sum(1 for r in records if r.get("status") == "oom")
    err_count = sum(1 for r in records if r.get("status") == "runtime_error")

    print(
        "Completed {n} configs: ok={ok}, skipped={sk}, oom={oom}, runtime_error={err}".format(
            n=len(records), ok=ok_count, sk=skip_count, oom=oom_count, err=err_count
        )
    )
    print(f"JSON results: {out_json}")
    print(f"Markdown table: {out_md}")


if __name__ == "__main__":
    main()
