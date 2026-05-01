"""Regression metric helpers.

Kept dependency-light (numpy only). ``r2`` matches scikit-learn's definition
(1 − SS_res / SS_tot using the test mean) so numbers are directly comparable
to the paper's Table 1.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RegressionMetrics:
    mae: float
    rmse: float
    r2: float

    def to_dict(self) -> dict[str, float]:
        return {"mae": self.mae, "rmse": self.rmse, "r2": self.r2}


def regression_metrics(pred: np.ndarray, target: np.ndarray) -> RegressionMetrics:
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs target {target.shape}")
    diff = pred - target
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff**2)))
    ss_res = float(np.sum(diff**2))
    ss_tot = float(np.sum((target - target.mean()) ** 2))
    r2 = float("nan") if ss_tot == 0 else float(1.0 - ss_res / ss_tot)
    return RegressionMetrics(mae=mae, rmse=rmse, r2=r2)
