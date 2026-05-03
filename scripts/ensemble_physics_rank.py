"""Physics-committee × learned-model ensemble (M3 fixed blend).

Combines:
  1. Empirical PCE formulas from MOE2-predicted HOMO_D / LUMO_A — see
     ``src/training/physics_committee.py``. Default ``physics_subset`` is
     the 3-formula mean ``{Scharber, Imamura, Alharbi}``; pass a comma-list
     (e.g. ``ensemble.physics_subset='Imamura,Alharbi'``) to drop one.
  2. The learned ``rank_focal`` predictor — loaded from a fold checkpoint
     produced by ``train_rank_focal.py``.

Final prediction = ``α · physics_subset_mean + (1-α) · learned``. Reports
regression + ranking metrics on each fold's test set and writes a
diagnostics CSV with per-pair Actual / Learned / Scharber / Imamura /
Alharbi columns (used by ``evaluate_discovery_mix.py``).

Run:
    python scripts/ensemble_physics_rank.py --config configs/rank_focal.yaml \\
        ensemble.fold_ckpt=outputs/<run>/fold1_best.pt \\
        ensemble.alpha=0.3
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
from src.training.physics_committee import physics_committee  # noqa: E402
from src.utils import load_config, set_seed  # noqa: E402


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
    homo, lumo, ids = [], [], []
    encoder.eval()
    for batch in loader:
        batch = batch.to(device)
        graphs = Batch.from_data_list(getattr(batch, branch)).to(device)
        out = encoder(graphs.x, graphs.edge_index, graphs.edge_attr,
                      graphs.batch, "homo_lumo")
        homo.append(out[:, 0].cpu().numpy())
        lumo.append(out[:, 1].cpu().numpy())
        ids.append(batch.mol_id.view(-1).cpu().numpy())
    return np.concatenate(homo), np.concatenate(lumo), np.concatenate(ids)


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
    return np.concatenate(preds), np.concatenate(actuals), np.concatenate(ids)


# ----------------------------------------------------------- main


def main() -> None:
    cfg = load_config(default="configs/rank_focal.yaml")
    set_seed(int(cfg.seed))

    e_cfg = cfg.get("ensemble") or {}
    fold_ckpt = e_cfg.get("fold_ckpt")
    if not fold_ckpt:
        raise SystemExit("ensemble.fold_ckpt is required")
    moe2_ckpt = e_cfg.get("moe2_ckpt", "checkpoints/moe2_calc.pt")
    alpha = float(e_cfg.get("alpha", 0.3))
    subset_str = e_cfg.get("physics_subset", "Scharber,Imamura,Alharbi")
    subset = tuple(s.strip() for s in subset_str.split(","))
    out_dir = Path(e_cfg.get("out_dir", Path(fold_ckpt).parent / "ensemble"))
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    dataset = OPVPairDataset(
        cfg.data.opv_csv, max_smiles_len=int(cfg.data.max_smiles_len),
        include_aux=True,
    )
    print(f"OPV2D: {len(dataset)} pairs from {cfg.data.opv_csv}")

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
    moe2.load_state_dict(torch.load(moe2_ckpt, map_location="cpu", weights_only=False))
    moe2.to(device)

    # Learned predictor.
    learned = build_predictor(str(cfg.predictor.kind), **dict(cfg.predictor.kwargs)).to(device)
    ckpt = torch.load(fold_ckpt, map_location=device, weights_only=False)
    learned.load_state_dict(ckpt["state_dict"])
    print(f"loaded MOE2={moe2_ckpt}, learned={fold_ckpt}, "
          f"physics_subset={subset}, alpha={alpha}")

    folds = build_split(str(cfg.split.kind), dataset.df, **dict(cfg.split.kwargs))
    print(f"Split: {cfg.split.kind} → {len(folds)} fold(s)")

    fold_metrics = []
    for i, (_, _, te_idx) in enumerate(folds):
        print(f"\n=== Fold {i+1}/{len(folds)} (n_test={len(te_idx)}) ===")
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
        physics_subset_mean = np.mean(np.stack(
            [committee.members[m.lower()] for m in subset], axis=0), axis=0)
        blended = alpha * physics_subset_mean + (1.0 - alpha) * learned_pred

        m = regression_metrics(blended, actual)
        r = ranking_metrics(blended, actual)
        diag = pd.DataFrame({
            "Mol_ID": ids_d, "Actual": actual, "Learned": learned_pred,
            "Scharber": committee.members["scharber"],
            "Imamura": committee.members["imamura"],
            "Alharbi": committee.members["alharbi"],
            "PhysMean": committee.mean, "PhysStd": committee.std,
            "Predicted": blended,
        })
        diag.to_csv(out_dir / f"fold{i+1}_diagnostics.csv", index=False)

        fold_metrics.append({"fold": i + 1, "n_test": int(len(te_idx)),
                             **m.to_dict(), **r.to_dict()})
        print(f"  R²={m.r2:+.4f}  MAE={m.mae:.3f}  "
              f"Spearman={r.spearman:+.3f}  top10={r.top10_precision:.2f}  "
              f"NDCG@10={r.ndcg_at_10:.3f}")

    summary = {
        "split": str(cfg.split.kind),
        "alpha": alpha, "physics_subset": list(subset),
        "moe2_ckpt": moe2_ckpt, "learned_ckpt": str(fold_ckpt),
        "per_fold": fold_metrics,
    }
    if len(fold_metrics) > 1:
        df = pd.DataFrame(fold_metrics)
        for col in ("r2", "mae", "spearman", "top10_precision", "ndcg_at_10"):
            summary[f"mean_{col}"] = float(df[col].mean())
            summary[f"std_{col}"] = float(df[col].std())
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
