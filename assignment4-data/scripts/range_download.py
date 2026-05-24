"""健壮的断点续传 HTTP(S) 下载器。

支持通过 HTTP ``Range`` 头续传大文件。下载完成后用 ``gzip -t`` 校验完整性。
自动检测服务器是否忽略 Range 请求（返回 200 而非 206），若是则从头重下。
出现网络错误时等待 5 秒后自动重试。
"""

from __future__ import annotations

import argparse
import gzip
import os
import sys
import time
import urllib.request
from pathlib import Path


def get_remote_size(url: str) -> int:
    """通过 HEAD 请求获取远程文件的 ``Content-Length`` 字节数。"""
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request, timeout=60) as response:
        return int(response.headers["Content-Length"])


def download_range(url: str, path: Path, log_path: Path) -> None:
    """带断点续传下载文件到 ``path``。

    1. 获取远程文件大小。
    2. 若本地已存在且大小相同，通过 ``gzip`` 流读取校验完整性。
    3. 否则通过 ``Range: bytes=<local_size>-`` 续传。
    4. 若服务器忽略 Range（返回非 206），删除本地文件重新下载。
    5. 任何异常等 5 秒后重试。
    """
    remote_size = get_remote_size(url)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    path.parent.mkdir(parents=True, exist_ok=True)

    while True:
        local_size = path.stat().st_size if path.exists() else 0
        if local_size == remote_size:
            # 大小匹配时用 gzip 流读取验证（相当于 gzip -t）
            with gzip.open(path, "rb") as f:
                while f.read(1024 * 1024):
                    pass
            print(f"[{time.strftime('%Y-%m-%dT%H:%M:%S%z')}] complete size={local_size}", flush=True)
            return

        if local_size > remote_size:
            # 本地文件比远程大（异常情况），删除重下
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
                # 服务器应返回 206 Partial Content，若返回其他码说明不支持 Range
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
    parser = argparse.ArgumentParser(description="健壮的断点续传 HTTP 下载器。")
    parser.add_argument("--url", required=True, help="下载 URL")
    parser.add_argument("--output", type=Path, required=True, help="输出文件路径")
    parser.add_argument("--log", type=Path, required=True, help="日志文件路径")
    args = parser.parse_args()

    # 将 stdout/stderr 重定向到日志文件
    with args.log.open("a", encoding="utf-8") as log:
        os.dup2(log.fileno(), sys.stdout.fileno())
        os.dup2(log.fileno(), sys.stderr.fileno())
        download_range(args.url, args.output, args.log)


if __name__ == "__main__":
    main()
