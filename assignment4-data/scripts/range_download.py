from __future__ import annotations

import argparse
import gzip
import os
import sys
import time
import urllib.request
from pathlib import Path


def get_remote_size(url: str) -> int:
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request, timeout=60) as response:
        return int(response.headers["Content-Length"])


def download_range(url: str, path: Path, log_path: Path) -> None:
    remote_size = get_remote_size(url)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    path.parent.mkdir(parents=True, exist_ok=True)

    while True:
        local_size = path.stat().st_size if path.exists() else 0
        if local_size == remote_size:
            with gzip.open(path, "rb") as f:
                while f.read(1024 * 1024):
                    pass
            print(f"[{time.strftime('%Y-%m-%dT%H:%M:%S%z')}] complete size={local_size}", flush=True)
            return

        if local_size > remote_size:
            print(f"local file larger than remote ({local_size}>{remote_size}); restarting", flush=True)
            path.unlink()
            local_size = 0

        headers = {}
        mode = "wb"
        if local_size > 0:
            headers["Range"] = f"bytes={local_size}-"
            mode = "ab"

        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                status = response.status
                if local_size > 0 and status != 206:
                    print(f"server ignored Range with status={status}; restarting", flush=True)
                    path.unlink(missing_ok=True)
                    continue
                with path.open(mode) as out:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        out.write(chunk)
            new_size = path.stat().st_size if path.exists() else 0
            print(f"[{time.strftime('%Y-%m-%dT%H:%M:%S%z')}] progress {new_size}/{remote_size}", flush=True)
        except Exception as exc:
            current = path.stat().st_size if path.exists() else 0
            print(f"[{time.strftime('%Y-%m-%dT%H:%M:%S%z')}] retry after {type(exc).__name__}: {exc}; size={current}/{remote_size}", flush=True)
            time.sleep(5)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    args = parser.parse_args()

    with args.log.open("a", encoding="utf-8") as log:
        os.dup2(log.fileno(), sys.stdout.fileno())
        os.dup2(log.fileno(), sys.stderr.fileno())
        download_range(args.url, args.output, args.log)


if __name__ == "__main__":
    main()
