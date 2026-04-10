from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path


def _run(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=str(cwd), text=True, capture_output=True)


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run nsys profiling sweep for CS336 benchmarking script.")
    parser.add_argument("--sizes", nargs="+", default=["small", "medium", "large", "xl", "2.7b"])
    parser.add_argument("--context-lengths", nargs="+", type=int, default=[128, 256, 512, 1024])
    parser.add_argument("--modes", nargs="+", default=["forward", "forward_backward", "train_step"])
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measure-steps", type=int, default=10)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--optimizer", choices=["cs336_adamw", "torch_adamw"], default="cs336_adamw")
    parser.add_argument("--output-dir", type=Path, default=Path("results/nsys_profile"))
    args = parser.parse_args()

    nsys_bin = shutil.which("nsys")
    if nsys_bin is None:
        raise RuntimeError("nsys not found in PATH. Install Nsight Systems or run inside an environment with nsys.")

    repo_root = Path(__file__).resolve().parents[1]
    benchmark_script = repo_root / "cs336_systems" / "benchmarking_script.py"

    raw_dir = args.output_dir / "raw"
    logs_dir = args.output_dir / "logs"
    stats_dir = args.output_dir / "stats"
    summary_path = args.output_dir / "summary.jsonl"
    raw_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    run_count = 0
    with summary_path.open("w", encoding="utf-8") as summary_f:
        for size in args.sizes:
            for ctx in args.context_lengths:
                for mode in args.modes:
                    tag = f"size-{size}_ctx-{ctx}_mode-{mode}"
                    run_count += 1

                    profile_base = raw_dir / tag
                    benchmark_json = logs_dir / f"{tag}.benchmark.json"
                    profile_stdout = logs_dir / f"{tag}.profile.stdout.log"
                    profile_stderr = logs_dir / f"{tag}.profile.stderr.log"

                    profile_cmd = [
                        nsys_bin,
                        "profile",
                        "-o",
                        str(profile_base),
                        "--force-overwrite=true",
                        "--trace=cuda,nvtx,osrt",
                        "--sample=none",
                        sys.executable,
                        str(benchmark_script),
                        "--size",
                        size,
                        "--context-length",
                        str(ctx),
                        "--mode",
                        mode,
                        "--warmup-steps",
                        str(args.warmup_steps),
                        "--measure-steps",
                        str(args.measure_steps),
                        "--dtype",
                        args.dtype,
                        "--device",
                        args.device,
                        "--optimizer",
                        args.optimizer,
                        "--enable-nvtx",
                        "--output",
                        str(benchmark_json),
                    ]

                    profile_proc = _run(profile_cmd, cwd=repo_root)
                    _write_text(profile_stdout, profile_proc.stdout)
                    _write_text(profile_stderr, profile_proc.stderr)

                    rep_file = profile_base.with_suffix(".nsys-rep")
                    if not rep_file.exists():
                        match = re.search(r"Generated:\s*\n\s*(\S+\.nsys-rep)", profile_proc.stdout)
                        if match is not None:
                            generated_rep = Path(match.group(1))
                            if generated_rep.exists():
                                generated_rep.replace(rep_file)

                    stats_text = ""
                    stats_cmd = [
                        nsys_bin,
                        "stats",
                        "--force-export=true",
                        "--report",
                        "cuda_gpu_kern_sum,cuda_api_sum",
                        str(rep_file),
                    ]
                    if profile_proc.returncode == 0 and rep_file.exists():
                        stats_proc = _run(stats_cmd, cwd=repo_root)
                        stats_text = stats_proc.stdout + "\n" + stats_proc.stderr
                        _write_text(stats_dir / f"{tag}.stats.txt", stats_text)

                    summary = {
                        "tag": tag,
                        "size": size,
                        "context_length": ctx,
                        "mode": mode,
                        "profile_returncode": profile_proc.returncode,
                        "rep_file": str(rep_file),
                        "benchmark_json": str(benchmark_json),
                    }
                    summary_f.write(json.dumps(summary, ensure_ascii=False) + "\n")

    print(f"Completed {run_count} profiling runs.")
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
