"""Physics committee — Scharber / Imamura / Alharbi PCE formulas.

Three empirical PCE estimators, each with different validity regimes:

  Scharber (Adv. Mater. 2006)            — bulk OPV, conservative
    Voc = max(0, |HOMO_D| - |LUMO_A| - 0.3)
    Jsc = 0.65 * J_max(Eg)                                        # FF=0.65

  Imamura (refined Scharber)             — tighter Jsc envelope, FF=0.70
    Voc = max(0, |HOMO_D| - |LUMO_A| - 0.3)
    Jsc = 0.65 * 75 * exp(-0.95*(Eg - 0.7))                       # FF=0.70

  Alharbi (J. Mater. Chem. 2014)         — Shockley-Queisser-style
    Voc = Eg - 0.5 - 0.0114 * |LUMO_A|^1.8617 - 0.057 * Eg
    Jsc = 0.65 * J_max(Eg)
    FF  = Voc / (Voc + 12 kT/q)         (12 kT/q ≈ 0.31 V at 300 K)

The three formulas have *different sensitivities to the same inputs*, so
they disagree most in physics-non-ideal regimes (very deep HOMO_D, very
shallow LUMO_A, exotic gaps). The disagreement (`σ_phys`) is therefore a
useful *epistemic uncertainty* signal — without Bayesian / MC-dropout
machinery — that we use to gate the ensemble blend with a learned model.

The reference review for these formulas:
  Jiang, Yao, Yang, Wang. *Solar RRL* 8 (2024) 2400567 — Table 1.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _jmax_envelope_a(eg: np.ndarray) -> np.ndarray:
    """AM1.5G photon-flux envelope used by the Scharber/Alharbi formulas."""
    return 70.0 * np.exp(-1.05 * (np.clip(eg, 0.5, 4.0) - 0.7))


def _jmax_envelope_b(eg: np.ndarray) -> np.ndarray:
    """Slightly different envelope used by Imamura — same shape, different
    pre-factors so the two predictions disagree where the bandgap drifts."""
    return 75.0 * np.exp(-0.95 * (np.clip(eg, 0.5, 4.0) - 0.7))


def scharber_pce(homo_d: np.ndarray, lumo_a: np.ndarray) -> dict[str, np.ndarray]:
    voc = np.clip(np.abs(homo_d) - np.abs(lumo_a) - 0.30, 0.0, 2.5)
    eg = lumo_a - homo_d
    jsc = 0.65 * _jmax_envelope_a(eg)
    ff = np.full_like(voc, 0.65)
    return {"pce": voc * jsc * ff, "voc": voc, "jsc": jsc, "ff": ff, "eg": eg}


def imamura_pce(homo_d: np.ndarray, lumo_a: np.ndarray) -> dict[str, np.ndarray]:
    voc = np.clip(np.abs(homo_d) - np.abs(lumo_a) - 0.30, 0.0, 2.5)
    eg = lumo_a - homo_d
    jsc = 0.65 * _jmax_envelope_b(eg)
    ff = np.full_like(voc, 0.70)
    return {"pce": voc * jsc * ff, "voc": voc, "jsc": jsc, "ff": ff, "eg": eg}


def alharbi_pce(homo_d: np.ndarray, lumo_a: np.ndarray) -> dict[str, np.ndarray]:
    eg = lumo_a - homo_d
    eg = np.clip(eg, 1e-3, 4.0)
    voc_raw = eg - 0.5 - 0.0114 * np.abs(lumo_a) ** 1.8617 - 0.057 * eg
    voc = np.clip(voc_raw, 0.0, 2.5)
    jsc = 0.65 * _jmax_envelope_a(eg)
    ff = voc / (voc + 0.31 + 1e-6)         # SQ-style FF; 12 kT/q at 300 K
    return {"pce": voc * jsc * ff, "voc": voc, "jsc": jsc, "ff": ff, "eg": eg}


@dataclass(frozen=True)
class CommitteeResult:
    members: dict[str, np.ndarray]      # {"scharber": [...], "imamura": [...], "alharbi": [...]}
    mean: np.ndarray                    # μ across members
    std: np.ndarray                     # σ across members (epistemic uncertainty)

    def to_dataframe(self):
        import pandas as pd
        df = pd.DataFrame(self.members)
        df["physics_mean"] = self.mean
        df["physics_std"] = self.std
        return df


def physics_committee(homo_d: np.ndarray, lumo_a: np.ndarray) -> CommitteeResult:
    """Run all three physics formulas and return per-pair mean + std."""
    members = {
        "scharber": scharber_pce(homo_d, lumo_a)["pce"],
        "imamura": imamura_pce(homo_d, lumo_a)["pce"],
        "alharbi": alharbi_pce(homo_d, lumo_a)["pce"],
    }
    stack = np.stack(list(members.values()), axis=0)            # [3, N]
    return CommitteeResult(
        members=members,
        mean=stack.mean(axis=0),
        std=stack.std(axis=0),
    )


# ---------------------------------------------------------------------------
# Blending strategies
# ---------------------------------------------------------------------------


def disagreement_gated_blend(
    physics_mean: np.ndarray, physics_std: np.ndarray,
    learned_pred: np.ndarray, tau: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Blend physics-mean with learned-pred, gated by physics committee disagreement.

    ``w_phys = exp(-σ_phys / τ)`` ∈ (0, 1]. High committee agreement (small
    σ) → physics dominates. High disagreement → learned prediction is
    given more weight.

    Returns:
        (final_pred, w_phys) — the second is per-pair so it can be diagnosed.
    """
    w = np.exp(-np.asarray(physics_std) / max(tau, 1e-6))
    final = w * physics_mean + (1.0 - w) * learned_pred
    return final, w


def reciprocal_rank_fusion(scores_list: list[np.ndarray], k: float = 60.0) -> np.ndarray:
    """Reciprocal Rank Fusion: combine multiple ranked lists into one.

    ``score_RRF(x) = Σ_i 1 / (k + rank_i(x))``  where rank=1 is the top.

    Output is monotonic in the fused rank so downstream rank metrics
    (Spearman, NDCG, top-K) treat it as a continuous predictor. Magnitude
    is meaningless. ``k`` is the standard RRF damping (60 in IR
    literature; the result is robust to k ∈ [10, 100]).
    """
    fused = np.zeros_like(scores_list[0], dtype=np.float64)
    for s in scores_list:
        order = np.argsort(-np.asarray(s))                     # descending
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, len(order) + 1)            # rank 1 = best
        fused += 1.0 / (k + ranks)
    return fused
