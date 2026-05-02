"""Composite loss for the phys_rank algorithm.

Three components, each addressing a known failure mode of the P3 baseline:

1. **Regression MSE on PCE** — keeps the absolute prediction reasonable.
2. **Multi-task auxiliary** — Voc / Jsc / FF MSEs on the same head outputs
   that compose PCE = Voc * Jsc * FF. Adds 3x supervision signal density,
   most useful in the small-data tail.
3. **Listwise ranking loss (ListMLE)** — Plackett-Luce style; directly
   optimizes the ordering of predicted PCE within each batch. Improves
   Spearman / NDCG@k where MSE alone is insufficient (especially when the
   tail predictions are biased uniformly downward, as we observed under
   ``high_pce_holdout``).
4. **Physics consistency** — ensures the PCE output ≈ Voc * Jsc * FF, so
   the multi-task heads remain coupled to the final metric.

All terms are computed *on standardized PCE space* where applicable, so the
loss magnitudes are comparable across folds.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class CompositeLossWeights:
    pce: float = 1.0
    aux: float = 0.5
    rank: float = 0.5
    phys: float = 0.1

    def to_dict(self) -> dict[str, float]:
        return {"pce": self.pce, "aux": self.aux, "rank": self.rank, "phys": self.phys}


def listmle_loss(scores: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """ListMLE (Plackett-Luce) loss for a single list (batch).

    Sorts ``targets`` descending, then computes the negative log likelihood
    of the implied permutation under a Plackett-Luce model with logits
    ``scores``. Formally: ``L = - Σ_i [s_{π(i)} - logsumexp(s_{π(i):})]``.

    Args:
        scores: predicted scores, shape ``[B]``.
        targets: ground-truth values, shape ``[B]``.

    Returns:
        Scalar loss, mean across batch (scaled by 1/B for stability).
    """
    if scores.numel() < 2:
        return scores.new_zeros(())
    # Order by target descending
    order = torch.argsort(targets, descending=True)
    s_sorted = scores[order]
    # log-cumsum-exp from the right: for position i, denominator = sum_{j>=i} exp(s_j)
    # Implementation via flip + logcumsumexp + flip.
    rev = torch.flip(s_sorted, dims=[0])
    log_cumsum = torch.logcumsumexp(rev, dim=0)
    log_denoms = torch.flip(log_cumsum, dims=[0])  # [B], log(Σ_{j>=i} exp(s_j))
    return -(s_sorted - log_denoms).mean()


def position_weighted_listmle(scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """ListMLE with NDCG-style position weighting (1/log2(rank+2)).

    Top-1 is weighted 1.0, top-2 is 0.63, top-10 is 0.29 — so the optimizer
    spends most of its capacity on the highest-PCE positions, which is what
    we care about for material discovery (top-K precision and NDCG@K).
    """
    if scores.numel() < 2:
        return scores.new_zeros(())
    n = scores.numel()
    order = torch.argsort(targets, descending=True)
    s_sorted = scores[order]
    rev = torch.flip(s_sorted, dims=[0])
    log_cumsum = torch.logcumsumexp(rev, dim=0)
    log_denoms = torch.flip(log_cumsum, dims=[0])
    pos = torch.arange(n, device=scores.device, dtype=scores.dtype)
    pos_w = 1.0 / torch.log2(pos + 2.0)              # [B], top → 1.0
    pos_w = pos_w / pos_w.sum() * n                  # renormalize so mean is 1
    return -((s_sorted - log_denoms) * pos_w).mean()


def top_quantile_margin_loss(
    scores: torch.Tensor, targets: torch.Tensor,
    quantile: float = 0.7, target_diff_min: float = 0.5, margin: float = 0.3,
) -> torch.Tensor:
    """Squared hinge margin loss restricted to high-target pairs.

    For every pair (i, j) where target_i > target_j by at least
    ``target_diff_min`` AND target_i is in the top ``1-quantile`` of the
    batch, require ``pred_i - pred_j >= margin``. Returns mean squared
    hinge violation. Zero if no qualifying pairs exist.
    """
    n = scores.numel()
    if n < 2:
        return scores.new_zeros(())
    threshold = torch.quantile(targets, q=float(quantile))
    is_top = targets >= threshold                                              # [B]
    # Pair matrices.
    diff_t = targets.unsqueeze(1) - targets.unsqueeze(0)                       # [B, B], i - j
    diff_s = scores.unsqueeze(1) - scores.unsqueeze(0)
    qualifying = (diff_t > target_diff_min) & is_top.unsqueeze(1)              # only i in top
    if qualifying.sum() == 0:
        return scores.new_zeros(())
    hinge = torch.clamp(margin - diff_s, min=0.0) ** 2
    return hinge[qualifying].mean()


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """MSE that ignores rows where ``mask`` == 0."""
    if mask.sum() == 0:
        return pred.new_zeros(())
    diff = (pred - target) ** 2
    return (diff * mask).sum() / mask.sum().clamp_min(1.0)


@dataclass(frozen=True)
class RankFocalWeights:
    """Weights for ``RankFocalLoss``."""
    # Loss-component weights
    pce: float = 0.5
    aux: float = 0.5
    rank: float = 1.0
    margin_w: float = 0.5
    phys: float = 0.05

    # Hyper-parameters of the top-quantile margin loss
    margin_quantile: float = 0.7
    margin_target_diff_min: float = 0.5
    margin_value: float = 0.3

    def to_dict(self) -> dict[str, float]:
        return {
            "pce": self.pce, "aux": self.aux, "rank": self.rank,
            "margin_w": self.margin_w, "phys": self.phys,
            "margin_quantile": self.margin_quantile,
            "margin_target_diff_min": self.margin_target_diff_min,
            "margin_value": self.margin_value,
        }


class RankFocalLoss(nn.Module):
    """Tail-focused composite loss for the rank_focal algorithm.

    Differs from ``CompositeLoss``:
      - **position-weighted ListMLE** (top-of-list dominates) instead of plain ListMLE
      - additional **top-quantile pairwise margin** that pushes high-PCE
        predictions away from lower-PCE predictions
      - lighter PCE-MSE and physics weights (the rank loss is the primary signal)

    Sample weighting (oversampling high-PCE) is applied in the trainer via a
    ``WeightedRandomSampler``, not here.
    """

    def __init__(self, w: RankFocalWeights | None = None) -> None:
        super().__init__()
        self.w = w or RankFocalWeights()

    def forward(
        self, pred: dict[str, torch.Tensor], target: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        l_pce = F.mse_loss(pred["pce_z"], target["pce_z"])

        aux_pred = torch.stack([pred["voc"], pred["jsc"], pred["ff"]], dim=-1)
        aux_tgt = target["aux"]; aux_mask = target["aux_mask"]
        with torch.no_grad():
            scale = aux_tgt.std(dim=0, unbiased=False).clamp_min(0.1)
        l_aux = masked_mse(aux_pred / scale, aux_tgt / scale, aux_mask)

        l_rank = position_weighted_listmle(pred["pce_z"], target["pce_z"])
        l_margin = top_quantile_margin_loss(
            pred["pce_z"], target["pce_z"],
            quantile=self.w.margin_quantile,
            target_diff_min=self.w.margin_target_diff_min,
            margin=self.w.margin_value,
        )

        pce_phys = pred["voc"] * pred["jsc"] * pred["ff"]
        l_phys = F.mse_loss(pce_phys, pred["pce"])

        total = (
            self.w.pce * l_pce + self.w.aux * l_aux
            + self.w.rank * l_rank + self.w.margin_w * l_margin
            + self.w.phys * l_phys
        )
        log = {
            "total": float(total.detach().item()),
            "pce_mse": float(l_pce.detach().item()),
            "aux_mse": float(l_aux.detach().item()),
            "rank": float(l_rank.detach().item()),
            "margin": float(l_margin.detach().item()),
            "phys": float(l_phys.detach().item()),
        }
        return total, log


class CompositeLoss(nn.Module):
    """Sum of MSE(PCE) + multi-task aux + ListMLE + physics consistency.

    Expects ``pred`` to be a dict with at least keys ``{pce, voc, jsc, ff}``
    and ``target`` to provide ``{pce, pce_z, aux, aux_mask}``. ``pce_z`` is
    the standardized PCE used for the regression and ranking terms; ``aux``
    is ``[B, 3]`` raw [Voc, Jsc, FF] with ``aux_mask`` ``[B, 3]`` boolean.
    """

    def __init__(self, weights: CompositeLossWeights | None = None) -> None:
        super().__init__()
        self.w = weights or CompositeLossWeights()

    def forward(
        self, pred: dict[str, torch.Tensor], target: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        # Regression on standardized PCE for stable scale.
        l_pce = F.mse_loss(pred["pce_z"], target["pce_z"])

        # Auxiliary multi-task on RAW Voc / Jsc / FF (units differ; rescale per channel).
        # Use channel-wise std normalization to keep the auxiliary loss well-scaled.
        aux_pred = torch.stack([pred["voc"], pred["jsc"], pred["ff"]], dim=-1)   # [B, 3]
        aux_tgt = target["aux"]                                                  # [B, 3]
        aux_mask = target["aux_mask"]                                            # [B, 3]
        # Per-channel scale = std over the batch (clamped) for unit normalization.
        with torch.no_grad():
            scale = aux_tgt.std(dim=0, unbiased=False).clamp_min(0.1)            # [3]
        l_aux = masked_mse(aux_pred / scale, aux_tgt / scale, aux_mask)

        # Listwise ranking on PCE (standardized so targets and preds use the same scale).
        l_rank = listmle_loss(pred["pce_z"], target["pce_z"])

        # Physics consistency: pce_pred ≈ voc * jsc * ff (in raw units).
        pce_phys = pred["voc"] * pred["jsc"] * pred["ff"]
        l_phys = F.mse_loss(pce_phys, pred["pce"])

        total = (
            self.w.pce * l_pce
            + self.w.aux * l_aux
            + self.w.rank * l_rank
            + self.w.phys * l_phys
        )
        log = {
            "total": float(total.detach().item()),
            "pce_mse": float(l_pce.detach().item()),
            "aux_mse": float(l_aux.detach().item()),
            "rank": float(l_rank.detach().item()),
            "phys": float(l_phys.detach().item()),
        }
        return total, log
