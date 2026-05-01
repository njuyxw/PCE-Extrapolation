"""Post-hoc ranking metrics from saved fold prediction CSVs.

Use this to add Spearman / Kendall / top-K precision / NDCG@10 to runs that
finished before ranking metrics were wired into the trainer. For new runs,
the trainer already writes these into ``fold_summary.csv``.

Usage:
    python scripts/05_rank_from_predictions.py outputs/repro_random_kfold \\
                                                outputs/repro_scaffold_acceptor \\
                                                outputs/repro_high_pce_q85
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.metrics import ranking_metrics, regression_metrics  # noqa: E402


def evaluate_run(run_dir: Path) -> dict:
    """Compute regression + ranking metrics for every fold*_predictions.csv."""
    rows: list[dict] = []
    fold_files = sorted(run_dir.glob("fold*_predictions.csv"))
    if not fold_files:
        raise FileNotFoundError(f"No fold*_predictions.csv under {run_dir}")
    for fp in fold_files:
        df = pd.read_csv(fp)
        pred = df["Predicted"].to_numpy()
        actual = df["Actual"].to_numpy()
        m = regression_metrics(pred, actual)
        r = ranking_metrics(pred, actual)
        rows.append({"fold": fp.stem.replace("_predictions", ""),
                     "n": len(df), **m.to_dict(), **r.to_dict()})
    df = pd.DataFrame(rows)
    summary = {"run": str(run_dir), "n_folds": len(df)}
    for col in ("r2", "mae", "rmse", "spearman", "kendall",
                "top10_precision", "top20_precision", "ndcg_at_10"):
        summary[f"mean_{col}"] = float(df[col].mean())
        summary[f"std_{col}"] = float(df[col].std()) if len(df) > 1 else 0.0
    return {"per_fold": rows, "summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+", help="output directories of past runs")
    args = parser.parse_args()

    fmt = ("{run:30s} folds={folds:>2}  "
           "R²={r2:>+7.4f}  MAE={mae:>5.3f}  "
           "Spearman={sp:>+7.4f}  Kendall={kt:>+7.4f}  "
           "top10={t10:>4.2f}  top20={t20:>4.2f}  NDCG@10={ndcg:>5.3f}")
    print(f"{'run':30s} {'N':>5}  {'R²':>9}  {'MAE':>8}  "
          f"{'Spearman':>11}  {'Kendall':>10}  {'top10':>8}  {'top20':>8}  {'NDCG@10':>10}")
    print("-" * 130)

    for d in args.run_dirs:
        run = Path(d)
        if not run.is_dir():
            print(f"skip (not a dir): {d}"); continue
        res = evaluate_run(run)
        out_path = run / "ranking_summary.json"
        with open(out_path, "w") as f:
            json.dump(res, f, indent=2)
        s = res["summary"]
        print(fmt.format(
            run=run.name, folds=s["n_folds"],
            r2=s["mean_r2"], mae=s["mean_mae"],
            sp=s["mean_spearman"], kt=s["mean_kendall"],
            t10=s["mean_top10_precision"], t20=s["mean_top20_precision"],
            ndcg=s["mean_ndcg_at_10"],
        ))


if __name__ == "__main__":
    main()
