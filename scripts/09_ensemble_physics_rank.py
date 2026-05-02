"""Physics-committee × learned-model ensemble.

Combines:
  1. Three empirical PCE formulas (Scharber / Imamura / Alharbi) computed
     from MOE2-predicted HOMO_D and LUMO_A (no extra inputs needed).
  2. The trained learned model (e.g. rank_focal) — loaded from a fold
     checkpoint produced by ``07_train_rank_focal.py``.

Blending modes compared on the same test set:

  M0  scharber           — single Scharber prediction (reference floor)
  M1  physics_mean       — naive mean of the three physics formulas
  M2  learned_only       — the learned model's PCE prediction
  M3  fixed_blend        — α·physics_mean + (1-α)·learned, α=0.5
  M4  disagreement_gated — w_phys = exp(-σ_phys/τ), then blend
  M5  rrf                — Reciprocal Rank Fusion of (physics_mean, learned)
  M6  rrf_full           — RRF of all four (scharber, imamura, alharbi, learned)

The point of running ALL modes side-by-side is to test which integration
pattern is meaningful: if M3 (naive blend) wins by the same margin as M4
(disagreement gate), the "uncertainty" signal is not adding value.
Conversely if M5/M6 wins, the magnitudes themselves are not informative —
only the orderings are — and the ensemble should live in rank space.

Run:
    python scripts/09_ensemble_physics_rank.py \\
        --config configs/rank_focal.yaml \\
        ensemble.fold_ckpt=outputs/rank_focal_high_pce_q85/fold1_best.pt
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Subset
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import OPVPairDataset, build_split  # noqa: E402
from src.models import build_predictor  # noqa: E402
from src.models.encoders.moe2 import MOE2  # noqa: E402
from src.training.metrics import ranking_metrics, regression_metrics  # noqa: E402
from src.training.physics_committee import (  # noqa: E402
    disagreement_gated_blend,
    physics_committee,
    reciprocal_rank_fusion,
    scharber_pce,
)
from src.utils import load_config, set_seed  # noqa: E402


# --------------------------------------------------------------------- inference helpers


@torch.no_grad()
def predict_branch_homolumo(
    encoder: MOE2, dataset: OPVPairDataset, idx: np.ndarray, branch: str,
    device: torch.device, batch_size: int, num_workers: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run MOE2 in 'homo_lumo' mode on every donor (branch='donor') or
    acceptor (branch='acceptor') graph in ``idx``."""
    loader = DataLoader(
        Subset(dataset, idx.tolist()),
        batch_size=batch_size, shuffle=False, drop_last=False,
        num_workers=num_workers, pin_memory=True,
    )
    homo_list, lumo_list, ids = [], [], []
    encoder.eval()
    for batch in loader:
        batch = batch.to(device)
        graphs = Batch.from_data_list(getattr(batch, branch)).to(device)
        out = encoder(graphs.x, graphs.edge_index, graphs.edge_attr,
                      graphs.batch, "homo_lumo")          # [B, 2]
        homo_list.append(out[:, 0].cpu().numpy())
        lumo_list.append(out[:, 1].cpu().numpy())
        ids.append(batch.mol_id.view(-1).cpu().numpy())
    return (np.concatenate(homo_list), np.concatenate(lumo_list),
            np.concatenate(ids))


@torch.no_grad()
def predict_learned_pce(
    model: torch.nn.Module, dataset: OPVPairDataset, idx: np.ndarray,
    device: torch.device, batch_size: int, num_workers: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the learned predictor (P3 / P3Physics / etc.) on the test split."""
    # Auto-include aux only if the dataset supports it (P3Physics expects it
    # but doesn't read it during inference; we always pass include_aux=True
    # at construction-site so this is safe).
    loader = DataLoader(
        Subset(dataset, idx.tolist()),
        batch_size=batch_size, shuffle=False, drop_last=False,
        num_workers=num_workers, pin_memory=True,
    )
    preds, actuals, ids = [], [], []
    model.eval()
    for batch in loader:
        batch = batch.to(device)
        donor = Batch.from_data_list(batch.donor).to(device)
        acceptor = Batch.from_data_list(batch.acceptor).to(device)
        out = model(donor, acceptor)
        # Predictor may return a tensor (P3) or a dict (P3Physics).
        pce = out["pce"] if isinstance(out, dict) else out[:, 0]
        preds.append(pce.cpu().numpy())
        actuals.append(batch.y.view(-1).cpu().numpy())
        ids.append(batch.mol_id.view(-1).cpu().numpy())
    return (np.concatenate(preds), np.concatenate(actuals),
            np.concatenate(ids))


# --------------------------------------------------------------------- main


def evaluate_modes(
    actual: np.ndarray, learned_pred: np.ndarray,
    homo_d: np.ndarray, lumo_a: np.ndarray,
    tau: float, alpha: float, rrf_k: float,
) -> tuple[dict[str, dict], pd.DataFrame]:
    committee = physics_committee(homo_d, lumo_a)
    scharber_only = scharber_pce(homo_d, lumo_a)["pce"]

    # Disagreement-gated blend
    gated, w_phys = disagreement_gated_blend(
        committee.mean, committee.std, learned_pred, tau=tau)

    # Naive fixed blend
    fixed = alpha * committee.mean + (1.0 - alpha) * learned_pred

    # RRF — physics_mean + learned (2 lists)
    rrf2 = reciprocal_rank_fusion([committee.mean, learned_pred], k=rrf_k)
    # RRF — all four predictions
    rrf4 = reciprocal_rank_fusion(
        [committee.members["scharber"], committee.members["imamura"],
         committee.members["alharbi"], learned_pred], k=rrf_k)

    modes = {
        "M0_scharber":     scharber_only,
        "M1_physics_mean": committee.mean,
        "M2_learned_only": learned_pred,
        "M3_fixed_blend":  fixed,
        "M4_gated_blend":  gated,
        "M5_rrf_2":        rrf2,           # rank-only (not on PCE scale!)
        "M6_rrf_4":        rrf4,           # rank-only (not on PCE scale!)
    }

    rows: dict[str, dict] = {}
    for name, pred in modes.items():
        m = regression_metrics(pred, actual) if not name.startswith("M5_") and not name.startswith("M6_") else None
        r = ranking_metrics(pred, actual)
        # For RRF modes, R²/MAE/RMSE are meaningless (different units), so we
        # report only ranking metrics there.
        rows[name] = {
            "r2": (m.r2 if m else float("nan")),
            "mae": (m.mae if m else float("nan")),
            "rmse": (m.rmse if m else float("nan")),
            "spearman": r.spearman, "kendall": r.kendall,
            "top10": r.top10_precision, "top20": r.top20_precision,
            "ndcg10": r.ndcg_at_10,
        }

    diagnostics = pd.DataFrame({
        "Actual": actual, "Learned": learned_pred,
        "Scharber": committee.members["scharber"],
        "Imamura": committee.members["imamura"],
        "Alharbi": committee.members["alharbi"],
        "PhysMean": committee.mean, "PhysStd": committee.std,
        "WPhys": w_phys, "Gated": gated, "RRF2": rrf2, "RRF4": rrf4,
    })
    return rows, diagnostics


def main() -> None:
    cfg = load_config(default="configs/rank_focal.yaml")
    set_seed(int(cfg.seed))

    e_cfg = cfg.get("ensemble") or {}
    fold_ckpt = e_cfg.get("fold_ckpt")
    if not fold_ckpt:
        raise SystemExit("ensemble.fold_ckpt is required (a fold*_best.pt from a learned-model run)")
    moe2_ckpt = e_cfg.get("moe2_ckpt", "checkpoints/moe2_calc.pt")
    tau = float(e_cfg.get("tau", 1.0))
    alpha = float(e_cfg.get("alpha", 0.5))
    rrf_k = float(e_cfg.get("rrf_k", 60.0))
    out_dir = Path(e_cfg.get("out_dir", Path(fold_ckpt).parent / "ensemble"))
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    dataset = OPVPairDataset(
        cfg.data.opv_csv, max_smiles_len=int(cfg.data.max_smiles_len),
        include_aux=True,
    )
    print(f"OPV2D: {len(dataset)} pairs from {cfg.data.opv_csv}")

    # MOE2 encoder for Scharber / Imamura / Alharbi inputs.
    moe2 = MOE2(**dict(cfg.predictor.kwargs.get("encoder_kwargs", {}))) if False else MOE2(
        in_channels=int(cfg.predictor.kwargs.in_channels),
        edge_dim=int(cfg.predictor.kwargs.edge_dim),
        hidden_channels=int(cfg.predictor.kwargs.hidden_channels),
        out_channels=int(cfg.predictor.kwargs.out_channels),
        regression_targets=int(cfg.predictor.kwargs.homolumo_targets),
        heads=int(cfg.predictor.kwargs.num_heads),
        dropout_rate=float(cfg.predictor.kwargs.dropout_rate),
    )
    moe2.load_state_dict(torch.load(moe2_ckpt, map_location="cpu", weights_only=False))
    moe2.to(device)
    print(f"Loaded MOE2 (HOMO/LUMO) from {moe2_ckpt}")

    # Learned predictor (for the *learned* PCE prediction).
    learned = build_predictor(str(cfg.predictor.kind), **dict(cfg.predictor.kwargs)).to(device)
    ckpt = torch.load(fold_ckpt, map_location=device, weights_only=False)
    learned.load_state_dict(ckpt["state_dict"])
    y_mean = float(ckpt.get("y_mean", 0.0))
    y_std = float(ckpt.get("y_std", 1.0))
    print(f"Loaded learned predictor from {fold_ckpt} "
          f"(y_mean={y_mean:.3f}, y_std={y_std:.3f})")

    # Build the same split the learned model trained under.
    split_kind = str(cfg.split.kind)
    split_kwargs = dict(cfg.split.kwargs)
    folds = build_split(split_kind, dataset.df, **split_kwargs)
    print(f"Split: {split_kind} → {len(folds)} fold(s)")

    fold_results: list[dict] = []
    for i, (_, _, te_idx) in enumerate(folds):
        print(f"\n=== Fold {i+1}/{len(folds)} (n_test={len(te_idx)}) ===")

        # Run inference for each piece.
        homo_d, _, ids_d = predict_branch_homolumo(
            moe2, dataset, te_idx, "donor", device,
            int(cfg.trainer.batch_size), int(cfg.trainer.num_workers),
        )
        _, lumo_a, ids_a = predict_branch_homolumo(
            moe2, dataset, te_idx, "acceptor", device,
            int(cfg.trainer.batch_size), int(cfg.trainer.num_workers),
        )
        learned_pred, actual, ids_l = predict_learned_pce(
            learned, dataset, te_idx, device,
            int(cfg.trainer.batch_size), int(cfg.trainer.num_workers),
        )
        # Predictor outputs are already de-normalized in P3Physics (returns
        # PCE in physical units). For tensors-only P3, the stored y_mean/y_std
        # would have to be applied — but the trainer for P3Physics emits raw
        # PCE so no rescaling is needed here.
        assert (ids_d == ids_a).all() and (ids_d == ids_l).all(), "id mismatch"

        rows, diagnostics = evaluate_modes(
            actual=actual, learned_pred=learned_pred,
            homo_d=homo_d, lumo_a=lumo_a,
            tau=tau, alpha=alpha, rrf_k=rrf_k,
        )
        diagnostics.insert(0, "Mol_ID", ids_d)
        diagnostics.to_csv(out_dir / f"fold{i+1}_diagnostics.csv", index=False)
        fold_results.append({"fold": i + 1, **rows})

    # Aggregate across folds.
    mode_names = list(fold_results[0].keys()); mode_names.remove("fold")
    summary: dict[str, dict] = {}
    for m in mode_names:
        agg = pd.DataFrame([fr[m] for fr in fold_results])
        summary[m] = {col: float(agg[col].mean()) for col in agg.columns}
    with open(out_dir / "summary.json", "w") as f:
        json.dump({"split": split_kind, "tau": tau, "alpha": alpha, "rrf_k": rrf_k,
                   "modes": summary}, f, indent=2)

    # Pretty-print.
    print("\n" + "=" * 110)
    print(f"{'Mode':<22} {'R²':>9} {'MAE':>7} {'Spearman':>10} {'top10':>7} {'top20':>7} {'NDCG@10':>9}")
    print("-" * 110)
    for m, s in summary.items():
        r2 = "      —" if np.isnan(s["r2"]) else f"{s['r2']:>+9.4f}"
        mae = "    —" if np.isnan(s["mae"]) else f"{s['mae']:>7.3f}"
        print(f"{m:<22} {r2:>9} {mae:>7} {s['spearman']:>+10.4f} "
              f"{s['top10']:>7.2f} {s['top20']:>7.2f} {s['ndcg10']:>9.4f}")


if __name__ == "__main__":
    main()
