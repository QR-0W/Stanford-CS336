from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class IsoFlopsOptimum:
    compute_budget: float
    parameters: float
    dataset_tokens: float
    final_loss: float


@dataclass(frozen=True)
class PowerLawFit:
    coefficient: float
    exponent: float

    def predict(self, x: float | np.ndarray) -> float | np.ndarray:
        return self.coefficient * np.asarray(x, dtype=np.float64) ** self.exponent

    def to_dict(self) -> dict[str, float]:
        return {"coefficient": self.coefficient, "exponent": self.exponent}


def load_isoflops_runs(path: str | Path) -> list[dict[str, float]]:
    with Path(path).open() as f:
        runs = json.load(f)
    return [
        {
            "parameters": float(run["parameters"]),
            "compute_budget": float(run["compute_budget"]),
            "final_loss": float(run["final_loss"]),
        }
        for run in runs
    ]


def select_isoflops_optima(runs: list[dict[str, float]]) -> list[IsoFlopsOptimum]:
    by_compute: dict[float, list[dict[str, float]]] = {}
    for run in runs:
        by_compute.setdefault(run["compute_budget"], []).append(run)

    optima: list[IsoFlopsOptimum] = []
    for compute_budget, compute_runs in sorted(by_compute.items()):
        best_run = min(compute_runs, key=lambda run: run["final_loss"])
        parameters = best_run["parameters"]
        optima.append(
            IsoFlopsOptimum(
                compute_budget=compute_budget,
                parameters=parameters,
                dataset_tokens=compute_budget / (6.0 * parameters),
                final_loss=best_run["final_loss"],
            )
        )
    return optima


def fit_power_law(xs: np.ndarray, ys: np.ndarray) -> PowerLawFit:
    if np.any(xs <= 0) or np.any(ys <= 0):
        raise ValueError("Power-law fitting requires positive x and y values.")

    exponent, log_coefficient = np.polyfit(np.log(xs), np.log(ys), deg=1)
    return PowerLawFit(coefficient=float(np.exp(log_coefficient)), exponent=float(exponent))


def fit_isoflops_scaling_laws(optima: list[IsoFlopsOptimum]) -> tuple[PowerLawFit, PowerLawFit]:
    compute = np.array([point.compute_budget for point in optima], dtype=np.float64)
    parameters = np.array([point.parameters for point in optima], dtype=np.float64)
    dataset_tokens = np.array([point.dataset_tokens for point in optima], dtype=np.float64)
    return fit_power_law(compute, parameters), fit_power_law(compute, dataset_tokens)


def make_summary(
    optima: list[IsoFlopsOptimum],
    parameter_fit: PowerLawFit,
    dataset_fit: PowerLawFit,
    target_budgets: list[float],
) -> dict[str, object]:
    return {
        "optima": [asdict(point) for point in optima],
        "parameter_fit": parameter_fit.to_dict(),
        "dataset_fit": dataset_fit.to_dict(),
        "predictions": [
            {
                "compute_budget": budget,
                "parameters": float(parameter_fit.predict(budget)),
                "dataset_tokens": float(dataset_fit.predict(budget)),
            }
            for budget in target_budgets
        ],
    }
