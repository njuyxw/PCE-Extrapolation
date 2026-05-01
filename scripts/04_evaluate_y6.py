"""Stage-2 evaluation on the Y6 holdout (or any external CSV).

Loads a trained predictor checkpoint from a previous train_pce run and reports
{R², MAE, RMSE} on the Y6 acceptor pairs. Also writes per-row predictions CSV.

The checkpoint dict written by ``PCETrainer`` contains the fold's y_mean and
y_std, so de-normalization happens automatically.

Run:
    python scripts/04_evaluate_y6.py --config configs/baseline.yaml \\
        eval.fold_ckpt=outputs/pce_run/fold1_best.pt
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import OPVPairDataset  # noqa: E402
from src.models import build_predictor  # noqa: E402
from src.training.metrics import ranking_metrics, regression_metrics  # noqa: E402
from src.utils import load_config, set_seed  # noqa: E402


def main() -> None:
    cfg = load_config(default="configs/baseline.yaml")
    set_seed(int(cfg.seed))

    eval_cfg = cfg.get("eval") or {}
    csv_path = eval_cfg.get("csv") or cfg.data.y6_csv
    fold_ckpt = eval_cfg.get("fold_ckpt")
    if not fold_ckpt:
        raise SystemExit("eval.fold_ckpt is required (e.g. outputs/pce_run/fold1_best.pt)")
    out_path = Path(eval_cfg.get("out_dir", Path(fold_ckpt).parent / "y6_eval"))
    out_path.mkdir(parents=True, exist_ok=True)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    model = build_predictor(str(cfg.predictor.kind), **dict(cfg.predictor.kwargs)).to(device)

    ckpt = torch.load(fold_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    y_mean = torch.tensor(float(ckpt.get("y_mean", 0.0)), dtype=torch.float32, device=device)
    y_std = torch.tensor(float(ckpt.get("y_std", 1.0)), dtype=torch.float32, device=device)
    print(f"loaded predictor checkpoint from {fold_ckpt} "
          f"(y_mean={y_mean.item():.3f}, y_std={y_std.item():.3f})")

    dataset = OPVPairDataset(csv_path, max_smiles_len=int(cfg.data.max_smiles_len))
    print(f"Eval set: {len(dataset)} pairs from {csv_path}")
    if len(dataset) == 0:
        raise SystemExit(f"Empty CSV: {csv_path}")

    loader = DataLoader(dataset, batch_size=int(cfg.trainer.batch_size),
                        shuffle=False, num_workers=int(cfg.trainer.num_workers),
                        pin_memory=True)

    preds, actuals, ids = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            d_b = Batch.from_data_list(batch.donor).to(device)
            a_b = Batch.from_data_list(batch.acceptor).to(device)
            out = model(d_b, a_b)[:, 0]
            preds.append((out * y_std + y_mean).cpu().numpy())
            actuals.append(batch.y.view(-1).cpu().numpy())
            ids.append(batch.mol_id.view(-1).cpu().numpy())

    preds = np.concatenate(preds)
    actuals = np.concatenate(actuals)
    ids = np.concatenate(ids)
    m = regression_metrics(preds, actuals)
    r = ranking_metrics(preds, actuals)

    pd.DataFrame({"Mol_ID": ids, "Actual": actuals, "Predicted": preds}).to_csv(
        out_path / "predictions.csv", index=False)
    summary = {
        "csv": str(csv_path), "n": int(len(preds)), "checkpoint": str(fold_ckpt),
        **m.to_dict(), **r.to_dict(),
    }
    with open(out_path / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
