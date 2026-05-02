"""Ensemble M3 (α-sweep) on random_kfold — uses one ckpt per fold.

Identical math to ``09_ensemble_physics_rank.py`` but loops over the 5
random_kfold folds, using each fold's specific ``fold{i}_best.pt`` (so
no leakage of test data across folds).

For each fold the script also caches a per-fold diagnostics CSV with
columns ``[Actual, Learned, Scharber, Imamura, Alharbi, PhysMean,
PhysStd]`` so that ``scripts/10_alpha_sweep_multiseed.py`` (or a hand
analysis) can re-sweep α post-hoc.

Usage:
    python scripts/11_ensemble_random_kfold.py \\
        --learned_run outputs/rank_focal_random_kfold \\
        --moe2_ckpt   checkpoints/moe2_calc.pt
"""
from __future__ import annotations

import argparse
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

from omegaconf import OmegaConf

from src.data import OPVPairDataset, build_split  # noqa: E402
from src.models import build_predictor  # noqa: E402
from src.models.encoders.moe2 import MOE2  # noqa: E402
from src.training.metrics import ranking_metrics, regression_metrics  # noqa: E402
from src.training.physics_committee import physics_committee  # noqa: E402
from src.utils import set_seed  # noqa: E402


# ----------------------------------------------------------- inference helpers


@torch.no_grad()
def _predict_branch_homolumo(
    encoder: MOE2, dataset: OPVPairDataset, idx: np.ndarray, branch: str,
    device: torch.device, batch_size: int, num_workers: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
                      graphs.batch, "homo_lumo")
        homo_list.append(out[:, 0].cpu().numpy())
        lumo_list.append(out[:, 1].cpu().numpy())
        ids.append(batch.mol_id.view(-1).cpu().numpy())
    return (np.concatenate(homo_list), np.concatenate(lumo_list),
            np.concatenate(ids))


@torch.no_grad()
def _predict_learned_pce(
    model: torch.nn.Module, dataset: OPVPairDataset, idx: np.ndarray,
    device: torch.device, batch_size: int, num_workers: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        pce = out["pce"] if isinstance(out, dict) else out[:, 0]
        preds.append(pce.cpu().numpy())
        actuals.append(batch.y.view(-1).cpu().numpy())
        ids.append(batch.mol_id.view(-1).cpu().numpy())
    return (np.concatenate(preds), np.concatenate(actuals),
            np.concatenate(ids))


def _metrics(pred: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    m = regression_metrics(pred, actual)
    r = ranking_metrics(pred, actual)
    return {**m.to_dict(), **r.to_dict()}


# ----------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/rank_focal.yaml")
    parser.add_argument("--learned_run", type=str, default="outputs/rank_focal_random_kfold")
    parser.add_argument("--moe2_ckpt", type=str, default="checkpoints/moe2_calc.pt")
    parser.add_argument("--alphas", type=str, default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--split_seed", type=int, default=3407)
    parser.add_argument("--out_dir", type=str, default="outputs/ensemble_random_kfold")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    set_seed(int(cfg.seed))
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    learned_run = Path(args.learned_run)
    alphas = [float(a) for a in args.alphas.split(",")]

    # Dataset.
    dataset = OPVPairDataset(
        cfg.data.opv_csv, max_smiles_len=int(cfg.data.max_smiles_len),
        include_aux=True,
    )
    print(f"OPV2D: {len(dataset)} pairs")

    # MOE2 encoder for HOMO_D / LUMO_A.
    moe2 = MOE2(
        in_channels=int(cfg.predictor.kwargs.in_channels),
        edge_dim=int(cfg.predictor.kwargs.edge_dim),
        hidden_channels=int(cfg.predictor.kwargs.hidden_channels),
        out_channels=int(cfg.predictor.kwargs.out_channels),
        regression_targets=int(cfg.predictor.kwargs.homolumo_targets),
        heads=int(cfg.predictor.kwargs.num_heads),
        dropout_rate=float(cfg.predictor.kwargs.dropout_rate),
    )
    moe2.load_state_dict(torch.load(args.moe2_ckpt, map_location="cpu", weights_only=False))
    moe2.to(device)
    print(f"Loaded MOE2 (HOMO/LUMO) from {args.moe2_ckpt}")

    # Build the same 5 random_kfold splits the learned model used.
    folds = build_split("random_kfold", dataset.df,
                        n_splits=args.n_splits, seed=args.split_seed)

    fold_diag_paths: list[Path] = []
    fold_summaries: list[dict] = []                       # list of {alpha: metrics dict}

    for i, (_, _, te_idx) in enumerate(folds):
        fold_id = i + 1
        ckpt_path = learned_run / f"fold{fold_id}_best.pt"
        if not ckpt_path.exists():
            raise SystemExit(f"missing fold ckpt: {ckpt_path}")

        print(f"\n=== Fold {fold_id}/{len(folds)} (n_test={len(te_idx)}) ===")
        learned = build_predictor(str(cfg.predictor.kind),
                                  **dict(cfg.predictor.kwargs)).to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        learned.load_state_dict(ckpt["state_dict"])

        homo_d, _, ids_d = _predict_branch_homolumo(
            moe2, dataset, te_idx, "donor", device,
            int(cfg.trainer.batch_size), int(cfg.trainer.num_workers),
        )
        _, lumo_a, ids_a = _predict_branch_homolumo(
            moe2, dataset, te_idx, "acceptor", device,
            int(cfg.trainer.batch_size), int(cfg.trainer.num_workers),
        )
        learned_pred, actual, ids_l = _predict_learned_pce(
            learned, dataset, te_idx, device,
            int(cfg.trainer.batch_size), int(cfg.trainer.num_workers),
        )
        assert (ids_d == ids_a).all() and (ids_d == ids_l).all(), "id mismatch"

        committee = physics_committee(homo_d, lumo_a)
        diag = pd.DataFrame({
            "Mol_ID": ids_d, "Actual": actual, "Learned": learned_pred,
            "Scharber": committee.members["scharber"],
            "Imamura": committee.members["imamura"],
            "Alharbi": committee.members["alharbi"],
            "PhysMean": committee.mean, "PhysStd": committee.std,
        })
        diag_path = out_dir / f"fold{fold_id}_diagnostics.csv"
        diag.to_csv(diag_path, index=False)
        fold_diag_paths.append(diag_path)

        # α sweep for this fold.
        per_alpha = {}
        for a in alphas:
            blended = a * committee.mean + (1.0 - a) * learned_pred
            per_alpha[a] = _metrics(blended, actual)
        fold_summaries.append(per_alpha)
        # Print summary line per fold.
        ndcg = ", ".join(f"α={a:.1f}:{per_alpha[a]['ndcg_at_10']:.2f}"
                         for a in (0.0, 0.3, 0.5, 0.7, 1.0))
        print(f"  NDCG@10 by α: {ndcg}")

    # Aggregate across folds.
    print(f"\n{'='*100}")
    print(f"Aggregate (random_kfold n={len(folds)} folds, single seed):")
    print(f"{'α':>4}  {'R² mean±std':>14}  {'MAE':>14}  "
          f"{'Spear':>9}  {'top10':>7}  {'NDCG@10':>14}")
    print("-" * 100)
    rows = []
    for a in alphas:
        vals = pd.DataFrame([fs[a] for fs in fold_summaries])
        agg = {"alpha": a}
        for col in ("r2", "mae", "rmse", "spearman", "kendall",
                    "top10_precision", "top20_precision", "ndcg_at_10"):
            agg[f"mean_{col}"] = float(vals[col].mean())
            agg[f"std_{col}"] = float(vals[col].std())
            agg[f"median_{col}"] = float(vals[col].median())
        rows.append(agg)
        print(f"{a:>4.2f}  "
              f"{agg['mean_r2']:>+8.3f}±{agg['std_r2']:<5.3f}  "
              f"{agg['mean_mae']:>6.3f}±{agg['std_mae']:<5.3f}  "
              f"{agg['mean_spearman']:>+9.3f}  "
              f"{agg['mean_top10_precision']:>7.2f}  "
              f"{agg['mean_ndcg_at_10']:>5.3f}±{agg['std_ndcg_at_10']:<5.3f}")

    with open(out_dir / "alpha_sweep_summary.json", "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nWrote alpha-sweep summary to {out_dir / 'alpha_sweep_summary.json'}")
    print(f"Per-fold diagnostics: {[str(p) for p in fold_diag_paths]}")


if __name__ == "__main__":
    main()
