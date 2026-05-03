"""Decompose the physics committee — evaluate every non-empty subset.

Reuses the per-fold ``fold{i}_diagnostics.csv`` files written by the
ensemble scripts. For each subset of {Scharber, Imamura, Alharbi}:

  1. **Standalone** (α=1.0): physics-only prediction = mean of the
     subset members.
  2. **Blended with learned** at one or more α values: prediction =
     α · physics_subset_mean + (1 − α) · learned.

Reports regression + ranking metrics, aggregated across folds /
seeds. Produces JSON + a printed table per split so you can see which
*subset* (and which α) actually wins, rather than assuming the
3-formula mean is the best physics anchor.

Inputs are the diagnostic CSVs that the ensemble scripts already write
— no model loading, no retraining; runs in seconds.

Usage:
    # high_pce_holdout — 3 seeds, single fold each
    python scripts/12_physics_subset_sweep.py \\
        --diagnostics outputs/rank_focal_high_pce_q85/ensemble_a0.6/fold1_diagnostics.csv \\
                       outputs/rank_focal_high_pce_q85_seed1/ensemble_a0.6/fold1_diagnostics.csv \\
                       outputs/rank_focal_high_pce_q85_seed2/ensemble_a0.6/fold1_diagnostics.csv \\
        --label high_pce_q85 --alphas 0.0,0.5,1.0

    # random_kfold — 5 folds, single seed
    python scripts/12_physics_subset_sweep.py \\
        --diagnostics outputs/ensemble_random_kfold/fold1_diagnostics.csv \\
                       outputs/ensemble_random_kfold/fold2_diagnostics.csv \\
                       outputs/ensemble_random_kfold/fold3_diagnostics.csv \\
                       outputs/ensemble_random_kfold/fold4_diagnostics.csv \\
                       outputs/ensemble_random_kfold/fold5_diagnostics.csv \\
        --label random_kfold --alphas 0.0,0.2,0.5
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.metrics import ranking_metrics, regression_metrics  # noqa: E402

MEMBERS = ("Scharber", "Imamura", "Alharbi")


def all_subsets() -> list[tuple[str, ...]]:
    out: list[tuple[str, ...]] = []
    for k in range(1, len(MEMBERS) + 1):
        out.extend(combinations(MEMBERS, k))
    return out


def metrics_row(pred: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    m = regression_metrics(pred, actual)
    r = ranking_metrics(pred, actual)
    return {**m.to_dict(), **r.to_dict()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostics", nargs="+", required=True,
                        help="paths to fold*_diagnostics.csv (one per fold or seed)")
    parser.add_argument("--label", type=str, default="run")
    parser.add_argument("--alphas", type=str, default="0.0,0.5,1.0")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    alphas = [float(a) for a in args.alphas.split(",")]

    # Load all diagnostics CSVs.
    dfs = [pd.read_csv(p) for p in args.diagnostics]
    print(f"Loaded {len(dfs)} diagnostic CSVs (label={args.label!r}); "
          f"sizes {[len(d) for d in dfs]}")

    rows: list[dict] = []
    for subset in all_subsets():
        for alpha in alphas:
            per_fold = []
            for df in dfs:
                actual = df["Actual"].to_numpy()
                learned = df["Learned"].to_numpy()
                physics = np.mean(np.stack([df[m].to_numpy() for m in subset], axis=0),
                                  axis=0)
                blended = alpha * physics + (1.0 - alpha) * learned
                per_fold.append(metrics_row(blended, actual))
            agg = pd.DataFrame(per_fold)
            row = {
                "subset": "+".join(s[0] for s in subset),         # e.g. "S+I+A"
                "subset_full": list(subset),
                "alpha": alpha,
                "n_folds": len(dfs),
            }
            for col in ("r2", "mae", "rmse", "spearman", "kendall",
                        "top10_precision", "top20_precision", "ndcg_at_10"):
                row[f"mean_{col}"] = float(agg[col].mean())
                row[f"std_{col}"] = float(agg[col].std()) if len(agg) > 1 else 0.0
                row[f"median_{col}"] = float(agg[col].median())
            rows.append(row)

    # Print.
    print(f"\n{'subset':>8}  {'α':>4}  "
          f"{'R² mean ± std':>17}  {'MAE':>13}  "
          f"{'Spear':>9}  {'top10':>7}  {'NDCG@10 (med)':>15}")
    print("-" * 100)
    rows.sort(key=lambda r: (r["alpha"], r["subset"]))
    for r in rows:
        print(
            f"{r['subset']:>8}  {r['alpha']:>4.2f}  "
            f"{r['mean_r2']:>+8.3f}±{r['std_r2']:<6.3f}  "
            f"{r['mean_mae']:>5.3f}±{r['std_mae']:<5.3f}  "
            f"{r['mean_spearman']:>+9.3f}  "
            f"{r['mean_top10_precision']:>7.2f}  "
            f"{r['mean_ndcg_at_10']:>5.3f} ({r['median_ndcg_at_10']:.2f})"
        )

    # Highlight winners by metric.
    metrics_to_rank = [("mean_r2", True), ("mean_mae", False),
                       ("mean_spearman", True), ("mean_top10_precision", True),
                       ("mean_ndcg_at_10", True), ("median_ndcg_at_10", True)]
    print(f"\nWinners on {args.label}:")
    for metric, higher_is_better in metrics_to_rank:
        ranked = sorted(rows, key=lambda r: r[metric], reverse=higher_is_better)
        best = ranked[0]
        print(f"  best {metric:>26s}: subset={best['subset']:>8} "
              f"α={best['alpha']:.2f} value={best[metric]:+.4f}")

    out_path = Path(args.out or f"outputs/physics_subset_sweep_{args.label}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nFull JSON: {out_path}")


if __name__ == "__main__":
    main()
