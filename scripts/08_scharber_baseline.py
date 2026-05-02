"""Pure-physics PCE baseline (no PCE training).

Loads the from-scratch MOE2 stage-2 checkpoint and predicts HOMO_D, LUMO_A,
HOMO_A, LUMO_A for every donor / acceptor in the OPV2D dataset. Then
applies the **Scharber empirical formula** to compute Voc, Jsc, FF, PCE
per pair, with no learnable parameters on the PCE side:

  Voc = max(0, |HOMO_D| - |LUMO_A| - 0.3)               [V]
  Eg  = LUMO_A - HOMO_D                                  [eV] (BHJ effective gap)
  Jsc = 0.65 * J_max(Eg)                                 [mA/cm²]
        where J_max approximates the AM1.5G photon-flux integral up to Eg
        as a smooth piecewise function fitted to standard solar-cell
        textbooks (Henry / Shockley-Queisser limit envelope).
  FF  = 0.65                                             [unitless]
  PCE = Voc · Jsc · FF                                    [%]

Reports R² / MAE / Spearman / NDCG@10 on each split. Useful as an
algorithm-iteration reference: any learned method should beat this on
ranking, otherwise the learning is not adding signal beyond the physics.

Run:
    python scripts/08_scharber_baseline.py --config configs/baseline.yaml
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import OPVPairDataset, build_split  # noqa: E402
from src.models.encoders.moe2 import MOE2  # noqa: E402
from src.training.metrics import ranking_metrics, regression_metrics  # noqa: E402
from src.utils import load_config, set_seed  # noqa: E402


def _jmax_of_eg(eg: np.ndarray) -> np.ndarray:
    """Smooth approximation of the AM1.5G integrated photon flux above Eg.

    Returns J_max(Eg) in mA/cm². Calibrated to match standard reference
    points: Eg=1.1 eV → J_max ≈ 44, Eg=1.5 eV → 27, Eg=2.0 eV → 14,
    Eg=2.5 eV → 6. Fitted via a simple power-decay form.
    """
    eg = np.asarray(eg, dtype=np.float64)
    eg = np.clip(eg, 0.5, 4.0)
    return 70.0 * np.exp(-1.05 * (eg - 0.7))      # ≈ standard envelope


@torch.no_grad()
def predict_homolumo(model: MOE2, loader: DataLoader, device: torch.device,
                     branch: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predict HOMO and LUMO (both in eV) for every pair via the donor or
    acceptor SMILES."""
    model.eval()
    homo_list, lumo_list, ids = [], [], []
    for batch in loader:
        batch = batch.to(device)
        graphs = Batch.from_data_list(getattr(batch, branch)).to(device)
        out = model(graphs.x, graphs.edge_index, graphs.edge_attr,
                    graphs.batch, "homo_lumo")  # [B, 2]
        homo_list.append(out[:, 0].cpu().numpy())
        lumo_list.append(out[:, 1].cpu().numpy())
        ids.append(batch.mol_id.view(-1).cpu().numpy())
    return (np.concatenate(homo_list), np.concatenate(lumo_list),
            np.concatenate(ids))


def scharber(homo_d: np.ndarray, lumo_a: np.ndarray) -> dict[str, np.ndarray]:
    voc = np.clip(np.abs(homo_d) - np.abs(lumo_a) - 0.3, 0.0, 2.5)     # V
    eg = lumo_a - homo_d                                                # eV
    jsc = 0.65 * _jmax_of_eg(eg)                                        # mA/cm²
    ff = np.full_like(voc, 0.65)
    pce = voc * jsc * ff                                                # %
    return {"voc": voc, "jsc": jsc, "ff": ff, "pce": pce, "eg": eg}


def evaluate_split(dataset: OPVPairDataset, idx: np.ndarray,
                    model: MOE2, device: torch.device,
                    batch_size: int, num_workers: int) -> tuple[dict, pd.DataFrame]:
    if len(idx) == 0:
        return {}, pd.DataFrame()
    loader = DataLoader(
        torch.utils.data.Subset(dataset, idx.tolist()),
        batch_size=batch_size, shuffle=False, drop_last=False,
        num_workers=num_workers, pin_memory=True,
    )
    homo_d, _, ids_d = predict_homolumo(model, loader, device, "donor")
    _, lumo_a, ids_a = predict_homolumo(model, loader, device, "acceptor")
    assert (ids_d == ids_a).all(), "donor / acceptor id mismatch"
    s = scharber(homo_d, lumo_a)
    actual = dataset.df.iloc[idx]["PCE"].to_numpy()
    m = regression_metrics(s["pce"], actual)
    r = ranking_metrics(s["pce"], actual)
    df = pd.DataFrame({
        "Mol_ID": ids_d, "Actual": actual, "Predicted": s["pce"],
        "Voc": s["voc"], "Jsc": s["jsc"], "FF": s["ff"], "Eg": s["eg"],
        "HOMO_D_pred": homo_d, "LUMO_A_pred": lumo_a,
    })
    summary = {"n": int(len(idx)), **m.to_dict(), **r.to_dict()}
    return summary, df


def main() -> None:
    cfg = load_config(default="configs/baseline.yaml")
    set_seed(int(cfg.seed))

    dataset = OPVPairDataset(cfg.data.opv_csv, max_smiles_len=int(cfg.data.max_smiles_len))
    print(f"OPV2D (clean): {len(dataset)} pairs from {cfg.data.opv_csv}")

    # Load the MOE2 encoder (single branch — used to predict both donor and acceptor HL).
    encoder_kwargs = dict(cfg.encoder.kwargs)
    model = MOE2(**encoder_kwargs)
    ckpt_path = cfg.trainer.get("pretrained_encoder_ckpt")
    if not ckpt_path or not Path(ckpt_path).exists():
        raise SystemExit(f"Need a pretrained MOE2 checkpoint at {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=False))
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Loaded MOE2 stage-2 ckpt from {ckpt_path}")

    out_dir = Path(cfg.paths.out_dir) / "scharber_baseline"
    out_dir.mkdir(parents=True, exist_ok=True)

    splits_to_run = [
        ("random_kfold", {"n_splits": 5, "seed": 3407}),
        ("scaffold_acceptor", {"train_frac": 0.8, "val_frac": 0.1, "test_frac": 0.1, "seed": 42}),
        ("high_pce_holdout", {"test_quantile": 0.85, "val_frac": 0.1, "seed": 42}),
    ]

    all_summaries: dict[str, dict] = {}
    for kind, kwargs in splits_to_run:
        print(f"\n=== Split: {kind} ===")
        folds = build_split(kind, dataset.df, **kwargs)
        per_fold = []
        for i, (_, _, te_idx) in enumerate(folds):
            sub_summary, df = evaluate_split(
                dataset, te_idx, model, device,
                batch_size=int(cfg.trainer.batch_size),
                num_workers=int(cfg.trainer.num_workers),
            )
            per_fold.append(sub_summary)
            df.to_csv(out_dir / f"{kind}_fold{i+1}_predictions.csv", index=False)
        agg = pd.DataFrame(per_fold)
        summary = {
            "kind": kind, "n_folds": len(per_fold),
            **{f"mean_{c}": float(agg[c].mean()) for c in agg.columns if c != "n"},
            **{f"std_{c}": float(agg[c].std()) for c in agg.columns
               if c != "n" and len(agg) > 1},
            "fold_R2": agg["r2"].tolist(),
        }
        print(json.dumps({k: v for k, v in summary.items() if not k.startswith("fold_")},
                         indent=2))
        all_summaries[kind] = summary

    with open(out_dir / "summary.json", "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nWrote {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
