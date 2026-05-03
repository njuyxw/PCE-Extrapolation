"""Discovery-mix evaluator — overall + bulk-only + tail-only metrics.

Loads each seed's diagnostics CSV (with Actual / Learned / Scharber /
Imamura / Alharbi / PhysMean), runs the M3 ensemble blend at given α
and physics subset, and reports metrics three ways:

  - Overall: full mixed test (bulk + tail)
  - Bulk-only: just rows with PCE < high_pce_quantile cutoff
  - Tail-only: rows with PCE >= cutoff

This isolates the "ranks ordinary candidates correctly" question from
the "ranks exceptional candidates correctly" question while running a
single mixed test set.

Usage:
    python scripts/evaluate_discovery_mix.py \\
        --diagnostics outputs/rank_focal_discovery_mix/ensemble/fold1_diagnostics.csv \\
                       outputs/rank_focal_discovery_mix_seed1/ensemble/fold1_diagnostics.csv \\
        --high_pce_cutoff 11.87 \\
        --subsets "S+I+A,I+A,A" \\
        --alphas 0.0,0.5,0.7
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

MEMBER_FULL = {"S": "Scharber", "I": "Imamura", "A": "Alharbi"}


def parse_subset(s: str) -> list[str]:
    return [MEMBER_FULL[c] for c in s.split("+")]


def metrics_row(pred: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    if len(pred) < 2:
        return {k: float("nan") for k in
                ("mae", "rmse", "r2", "spearman", "kendall",
                 "top10_precision", "top20_precision", "ndcg_at_10")}
    m = regression_metrics(pred, actual)
    r = ranking_metrics(pred, actual)
    return {**m.to_dict(), **r.to_dict()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostics", nargs="+", required=True)
    parser.add_argument("--high_pce_cutoff", type=float, default=11.87,
                        help="cutoff PCE — rows with Actual >= this are 'tail'")
    parser.add_argument("--subsets", type=str, default="S+I+A,I+A,A",
                        help="comma-separated subset codes (S, I, A and their unions)")
    parser.add_argument("--alphas", type=str, default="0.0,0.5,0.7")
    parser.add_argument("--out", type=str,
                        default="outputs/discovery_mix_eval.json")
    args = parser.parse_args()

    subsets = [parse_subset(s) for s in args.subsets.split(",")]
    alphas = [float(a) for a in args.alphas.split(",")]
    dfs = [pd.read_csv(p) for p in args.diagnostics]
    print(f"Loaded {len(dfs)} diagnostics CSVs (sizes {[len(d) for d in dfs]})")
    print(f"Tail cutoff = {args.high_pce_cutoff} PCE\n")

    rows: list[dict] = []
    for subset in subsets:
        subset_label = "+".join(s[0] for s in subset)
        for alpha in alphas:
            # Per-seed metrics in three regions.
            region_rows = {"overall": [], "bulk": [], "tail": []}
            for df in dfs:
                actual = df["Actual"].to_numpy()
                learned = df["Learned"].to_numpy()
                physics = np.mean(np.stack([df[m].to_numpy() for m in subset], axis=0),
                                  axis=0)
                blended = alpha * physics + (1.0 - alpha) * learned

                tail_mask = actual >= args.high_pce_cutoff
                region_rows["overall"].append(metrics_row(blended, actual))
                region_rows["bulk"].append(
                    metrics_row(blended[~tail_mask], actual[~tail_mask]))
                region_rows["tail"].append(
                    metrics_row(blended[tail_mask], actual[tail_mask]))

            agg = {"subset": subset_label, "alpha": alpha}
            for region, per_seed in region_rows.items():
                vals = pd.DataFrame(per_seed)
                for col in ("r2", "mae", "spearman", "top10_precision",
                            "top20_precision", "ndcg_at_10"):
                    agg[f"{region}_mean_{col}"] = float(vals[col].mean())
                    agg[f"{region}_std_{col}"] = (float(vals[col].std())
                                                   if len(vals) > 1 else 0.0)
            rows.append(agg)

    # Print: overall first.
    def _fmt(r, region):
        return (
            f"{r[f'{region}_mean_r2']:>+8.3f}±{r[f'{region}_std_r2']:<5.3f}  "
            f"{r[f'{region}_mean_mae']:>5.3f}  "
            f"{r[f'{region}_mean_spearman']:>+7.3f}  "
            f"{r[f'{region}_mean_top10_precision']:>5.2f}  "
            f"{r[f'{region}_mean_ndcg_at_10']:>5.3f}"
        )

    for region in ("overall", "bulk", "tail"):
        print(f"\n=== {region.upper()} (cutoff at {args.high_pce_cutoff} PCE) ===")
        print(f"{'subset':>8}  {'α':>4}  "
              f"{'R² mean±std':>16}  {'MAE':>5}  "
              f"{'Spear':>7}  {'top10':>5}  {'NDCG':>5}")
        print("-" * 90)
        for r in sorted(rows, key=lambda x: (x["alpha"], x["subset"])):
            print(f"{r['subset']:>8}  {r['alpha']:>4.2f}  {_fmt(r, region)}")

    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nFull JSON: {out_path}")


if __name__ == "__main__":
    main()
