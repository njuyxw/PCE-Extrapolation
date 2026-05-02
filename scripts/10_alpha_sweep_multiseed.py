"""Multi-seed × α sweep, computed post-hoc from saved diagnostics CSVs.

Reuses the per-seed ``fold1_diagnostics.csv`` files written by
``scripts/09_ensemble_physics_rank.py`` (columns: Actual, Learned,
Scharber, Imamura, Alharbi, PhysMean, PhysStd). For each (seed, α)
combination, computes the M3 fixed-blend prediction
``α·PhysMean + (1-α)·Learned`` and reports regression + ranking metrics.

No model loading, no retraining — pure numpy on saved CSVs. Takes seconds.

Usage:
    python scripts/10_alpha_sweep_multiseed.py
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


DEFAULT_RUN_DIRS = {
    1:  Path("outputs/rank_focal_high_pce_q85_seed1/ensemble_a0.6/fold1_diagnostics.csv"),
    2:  Path("outputs/rank_focal_high_pce_q85_seed2/ensemble_a0.6/fold1_diagnostics.csv"),
    42: Path("outputs/rank_focal_high_pce_q85/ensemble_a0.6/fold1_diagnostics.csv"),
}


def metrics_row(pred: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    m = regression_metrics(pred, actual)
    r = ranking_metrics(pred, actual)
    return {**m.to_dict(), **r.to_dict()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alphas", type=str,
                        default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    parser.add_argument("--out", type=str,
                        default="outputs/alpha_sweep_multiseed.json")
    args = parser.parse_args()

    alphas = [float(a) for a in args.alphas.split(",")]

    # Load each seed's diagnostics.
    per_seed: dict[int, pd.DataFrame] = {}
    for seed, csv in DEFAULT_RUN_DIRS.items():
        if not csv.exists():
            raise SystemExit(f"missing diagnostics for seed {seed}: {csv}")
        per_seed[seed] = pd.read_csv(csv)

    # For each α, collect per-seed metrics, then aggregate.
    results: list[dict] = []
    for alpha in alphas:
        per_seed_rows = []
        for seed, df in per_seed.items():
            actual = df["Actual"].to_numpy()
            phys = df["PhysMean"].to_numpy()
            learned = df["Learned"].to_numpy()
            blended = alpha * phys + (1.0 - alpha) * learned
            row = metrics_row(blended, actual)
            row["seed"] = seed
            per_seed_rows.append(row)
        per_seed_df = pd.DataFrame(per_seed_rows)
        agg = {"alpha": alpha}
        for col in ("r2", "mae", "rmse", "spearman", "kendall",
                    "top10_precision", "top20_precision", "ndcg_at_10"):
            agg[f"mean_{col}"] = float(per_seed_df[col].mean())
            agg[f"std_{col}"] = float(per_seed_df[col].std())
            agg[f"median_{col}"] = float(per_seed_df[col].median())
        agg["per_seed"] = per_seed_rows
        results.append(agg)

    # Pretty-print.
    print(f"{'alpha':>5}  "
          f"{'R² mean±std':>16}  {'MAE mean±std':>14}  "
          f"{'Spear mean':>10}  {'top10':>7}  {'NDCG@10 (med)':>14}")
    print("-" * 100)
    for r in results:
        print(
            f"{r['alpha']:>5.2f}  "
            f"{r['mean_r2']:>+8.3f}±{r['std_r2']:<6.3f}  "
            f"{r['mean_mae']:>6.3f}±{r['std_mae']:<5.3f}  "
            f"{r['mean_spearman']:>+10.3f}  "
            f"{r['mean_top10_precision']:>7.2f}  "
            f"{r['mean_ndcg_at_10']:>6.3f} ({r['median_ndcg_at_10']:.2f})"
        )

    # Pick the recommended α: highest mean NDCG@10 with best secondary on
    # mean R². (User can re-rank using a different criterion if desired.)
    best = max(results,
               key=lambda r: (r["mean_ndcg_at_10"], -abs(r["mean_r2"])))
    print("\nRecommended α (max mean NDCG@10, tie-break on |R²|):")
    print(f"  α = {best['alpha']:.2f}")
    print(f"  R² mean ± std = {best['mean_r2']:+.3f} ± {best['std_r2']:.3f}")
    print(f"  MAE mean ± std = {best['mean_mae']:.3f} ± {best['std_mae']:.3f}")
    print(f"  Spearman mean ± std = {best['mean_spearman']:+.3f} ± {best['std_spearman']:.3f}")
    print(f"  top10 mean ± std = {best['mean_top10_precision']:.2f} ± {best['std_top10_precision']:.2f}")
    print(f"  NDCG@10 mean ± std = {best['mean_ndcg_at_10']:.3f} ± {best['std_ndcg_at_10']:.3f}")
    print(f"  NDCG@10 median = {best['median_ndcg_at_10']:.3f}")

    # Persist.
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull JSON: {args.out}")


if __name__ == "__main__":
    main()
