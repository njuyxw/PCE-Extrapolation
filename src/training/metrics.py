"""Regression + ranking metric helpers.

Numpy/scipy only. ``r2`` matches scikit-learn's definition (1 − SS_res / SS_tot
using the test mean). Ranking metrics matter for material-discovery use cases
where the goal is to *select* the top-K candidates by predicted PCE — the
absolute MSE may be misleading when the model under-/over-estimates uniformly.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import kendalltau, spearmanr


@dataclass(frozen=True)
class RegressionMetrics:
    mae: float
    rmse: float
    r2: float

    def to_dict(self) -> dict[str, float]:
        return {"mae": self.mae, "rmse": self.rmse, "r2": self.r2}


@dataclass(frozen=True)
class RankingMetrics:
    spearman: float        # rank correlation, [-1, 1]
    kendall: float         # rank correlation (concordant pairs), [-1, 1]
    top10_precision: float # |topK_pred ∩ topK_true| / K, K=min(10, n)
    top20_precision: float # K=min(20, n)
    ndcg_at_10: float      # gain-weighted ranking, [0, 1]; relevance = actual PCE

    def to_dict(self) -> dict[str, float]:
        return {
            "spearman": self.spearman, "kendall": self.kendall,
            "top10_precision": self.top10_precision,
            "top20_precision": self.top20_precision,
            "ndcg_at_10": self.ndcg_at_10,
        }


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


def _topk_precision(pred: np.ndarray, target: np.ndarray, k: int) -> float:
    """Fraction of items in pred-top-K that are also in target-top-K."""
    n = len(pred)
    k_eff = min(k, n)
    if k_eff <= 0:
        return float("nan")
    pred_top = np.argpartition(-pred, k_eff - 1)[:k_eff]
    true_top = np.argpartition(-target, k_eff - 1)[:k_eff]
    return float(len(set(pred_top.tolist()) & set(true_top.tolist())) / k_eff)


def _ndcg_at_k(pred: np.ndarray, target: np.ndarray, k: int) -> float:
    """NDCG@k with relevance = actual target value (PCE).

    Relevance is shifted to be non-negative so the gain is well-defined even
    when target has negative values; this preserves the ranking by relevance.
    """
    n = len(pred)
    k_eff = min(k, n)
    if k_eff <= 0:
        return float("nan")
    rel = target - target.min()  # non-negative
    pred_order = np.argsort(-pred)[:k_eff]
    ideal_order = np.argsort(-target)[:k_eff]
    discount = 1.0 / np.log2(np.arange(2, k_eff + 2))  # log2(rank+1), rank starts at 1
    dcg = float(np.sum(rel[pred_order] * discount))
    idcg = float(np.sum(rel[ideal_order] * discount))
    return float("nan") if idcg == 0 else dcg / idcg


def ranking_metrics(pred: np.ndarray, target: np.ndarray) -> RankingMetrics:
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs target {target.shape}")
    if len(pred) < 2:
        nan = float("nan")
        return RankingMetrics(nan, nan, nan, nan, nan)
    sp = float(spearmanr(pred, target).statistic)
    kt = float(kendalltau(pred, target).statistic)
    return RankingMetrics(
        spearman=sp,
        kendall=kt,
        top10_precision=_topk_precision(pred, target, 10),
        top20_precision=_topk_precision(pred, target, 20),
        ndcg_at_10=_ndcg_at_k(pred, target, 10),
    )
