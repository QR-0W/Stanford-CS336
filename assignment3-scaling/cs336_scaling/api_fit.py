from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from cs336_scaling.isoflops import PowerLawFit, fit_power_law
from cs336_scaling.scaling_config import estimate_non_embedding_params


@dataclass(frozen=True)
class ApiIsoFlopsOptimum:
    train_flops: float
    non_embedding_params: float
    loss: float
    d_model: int
    num_layers: int
    num_heads: int
    batch_size: int
    learning_rate: float


@dataclass(frozen=True)
class LossFit:
    irreducible_loss: float
    coefficient: float
    exponent: float
    sse: float

    def predict(self, compute: float | np.ndarray) -> float | np.ndarray:
        return self.irreducible_loss + self.coefficient * np.asarray(compute) ** self.exponent

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def normalize_run(record: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(record)
    normalized["d_model"] = int(normalized["d_model"])
    normalized["num_layers"] = int(normalized["num_layers"])
    normalized["num_heads"] = int(normalized["num_heads"])
    normalized["batch_size"] = int(normalized["batch_size"])
    normalized["learning_rate"] = float(normalized["learning_rate"])
    normalized["train_flops"] = int(float(normalized["train_flops"]))
    normalized["loss"] = float(normalized["loss"])
    normalized["non_embedding_params"] = int(
        normalized.get(
            "non_embedding_params",
            estimate_non_embedding_params(normalized["num_layers"], normalized["d_model"]),
        )
    )
    return normalized


def select_best_by_compute(runs: list[dict[str, Any]]) -> list[ApiIsoFlopsOptimum]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for raw_run in runs:
        if "loss" not in raw_run:
            continue
        run = normalize_run(raw_run)
        grouped.setdefault(run["train_flops"], []).append(run)

    optima: list[ApiIsoFlopsOptimum] = []
    for train_flops, compute_runs in sorted(grouped.items()):
        best = min(compute_runs, key=lambda run: run["loss"])
        optima.append(
            ApiIsoFlopsOptimum(
                train_flops=float(train_flops),
                non_embedding_params=float(best["non_embedding_params"]),
                loss=float(best["loss"]),
                d_model=int(best["d_model"]),
                num_layers=int(best["num_layers"]),
                num_heads=int(best["num_heads"]),
                batch_size=int(best["batch_size"]),
                learning_rate=float(best["learning_rate"]),
            )
        )
    return optima


def fit_model_size_scaling(optima: list[ApiIsoFlopsOptimum]) -> PowerLawFit:
    compute = np.array([point.train_flops for point in optima], dtype=np.float64)
    params = np.array([point.non_embedding_params for point in optima], dtype=np.float64)
    return fit_power_law(compute, params)


def fit_loss_scaling(optima: list[ApiIsoFlopsOptimum]) -> LossFit:
    if len(optima) < 3:
        raise ValueError("At least three IsoFLOPs optima are needed to fit loss scaling.")

    compute = np.array([point.train_flops for point in optima], dtype=np.float64)
    losses = np.array([point.loss for point in optima], dtype=np.float64)
    min_loss = float(losses.min())

    lower = max(0.0, min_loss - 4.0)
    upper = min_loss - 1e-4
    candidates = np.linspace(lower, upper, 5000)

    best_fit: LossFit | None = None
    log_compute = np.log(compute)
    for irreducible_loss in candidates:
        residual = losses - irreducible_loss
        if np.any(residual <= 0):
            continue
        exponent, log_coefficient = np.polyfit(log_compute, np.log(residual), deg=1)
        coefficient = float(np.exp(log_coefficient))
        predicted = irreducible_loss + coefficient * compute**exponent
        sse = float(np.sum((losses - predicted) ** 2))
        candidate_fit = LossFit(
            irreducible_loss=float(irreducible_loss),
            coefficient=coefficient,
            exponent=float(exponent),
            sse=sse,
        )
        if best_fit is None or candidate_fit.sse < best_fit.sse:
            best_fit = candidate_fit

    if best_fit is None:
        raise ValueError("Could not fit loss scaling curve.")
    return best_fit
