from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import requests

from cs336_scaling.scaling_config import TrainingConfig


DEFAULT_BASE_URL = "http://hyperturing.stanford.edu:8000"
API_KEY_ENV = "CS336_SCALING_API_KEY"


class ScalingApiError(RuntimeError):
    pass


class ScalingApiClient:
    def __init__(self, api_key: str, base_url: str = DEFAULT_BASE_URL, timeout: float = 30.0):
        if not api_key:
            raise ValueError("API key is empty")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    @classmethod
    def from_env(cls, base_url: str = DEFAULT_BASE_URL, timeout: float = 30.0) -> "ScalingApiClient":
        api_key = os.environ.get(API_KEY_ENV, "")
        if not api_key:
            raise ScalingApiError(
                f"Set {API_KEY_ENV} to your SSH public key string before querying the API."
            )
        return cls(api_key=api_key, base_url=base_url, timeout=timeout)

    def total_flops_used(self) -> Any:
        return self._get("/total_flops_used", {"api_key": self.api_key})

    def previous_runs(self) -> list[dict[str, Any]]:
        response = self._get("/previous_runs", {"api_key": self.api_key})
        if isinstance(response, dict) and "previous_runs" in response:
            return list(response["previous_runs"])
        raise ScalingApiError(f"Unexpected /previous_runs response: {response}")

    def loss(self, config: TrainingConfig) -> dict[str, Any]:
        response = self._get("/loss", config.to_api_params(self.api_key))
        if isinstance(response, dict) and "loss" in response:
            record = config.to_record()
            record["loss"] = float(response["loss"])
            record["total_flops_used"] = float(response["total_flops_used"])
            return record
        raise ScalingApiError(f"Unexpected /loss response for {config}: {response}")

    def _get(self, path: str, params: dict[str, Any]) -> Any:
        url = f"{self.base_url}{path}"
        response = requests.get(url, params=params, timeout=self.timeout)
        try:
            payload = response.json()
        except json.JSONDecodeError as exc:
            raise ScalingApiError(f"Non-JSON response from {url}: {response.text[:500]}") from exc
        if response.status_code >= 400:
            raise ScalingApiError(f"HTTP {response.status_code} from {url}: {payload}")
        return payload


def config_key(record: dict[str, Any]) -> tuple[int, int, int, int, str, int]:
    return (
        int(record["d_model"]),
        int(record["num_layers"]),
        int(record["num_heads"]),
        int(record["batch_size"]),
        f"{float(record['learning_rate']):.12g}",
        int(record["train_flops"]),
    )


def load_runs(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open() as f:
        payload = json.load(f)
    if isinstance(payload, dict) and "runs" in payload:
        return list(payload["runs"])
    if isinstance(payload, dict) and "previous_runs" in payload:
        return list(payload["previous_runs"])
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Could not parse runs from {path}")


def save_runs(path: Path, runs: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    deduped = merge_runs([], runs)
    with path.open("w") as f:
        json.dump({"runs": deduped}, f, indent=2, sort_keys=True)


def merge_runs(*run_lists: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[tuple[int, int, int, int, str, int], dict[str, Any]] = {}
    for runs in run_lists:
        for run in runs:
            if _is_run_record(run):
                merged[config_key(run)] = run
    return list(merged.values())


def _is_run_record(record: dict[str, Any]) -> bool:
    required = {"d_model", "num_layers", "num_heads", "batch_size", "learning_rate", "train_flops"}
    return required.issubset(record)
