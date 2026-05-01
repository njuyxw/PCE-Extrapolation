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


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """MSE that ignores rows where ``mask`` == 0."""
    if mask.sum() == 0:
        return pred.new_zeros(())
    diff = (pred - target) ** 2
    return (diff * mask).sum() / mask.sum().clamp_min(1.0)


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
