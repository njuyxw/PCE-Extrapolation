# dev branch — algorithm experiments

Branch goal: design ML algorithms that improve **ranking quality on
`high_pce_holdout`** while staying competitive on `random_kfold`. Master
remains the paper-faithful baseline.

## Algorithm 1: `phys_rank` — physics-anchored multi-task ranking

### Motivation

The from-scratch baseline P³ collapses on `high_pce_holdout` (Spearman 0.24,
NDCG@10 0.21) because:

1. **Predictions cap near 10 PCE** while truth spans 11.9–17.8.
   The MSE-trained head has no upper structure; outputs cluster around the
   train mean.
2. **MSE is not ranking.** Even when predictions are correctly ordered up
   to a uniform under-bias, MSE penalizes the offset and ranks degrade.

Three orthogonal interventions, each addressing a different failure:

| # | Intervention | Targets | How |
|---|---|---|---|
| 1 | Physics-anchored Voc | regression-to-mean cap | Voc = clamp(LUMO_A − HOMO_D − 0.3, 0, 2.5) + 0.2·δ_Voc, with HOMO_D / LUMO_A from the MOE² regression heads (already supervised in stage 2). Inspired by the Scharber model surveyed in Jiang et al. *Solar RRL* 2024. |
| 2 | Multi-task decomposition | weak supervision | Predict Voc, Jsc, FF separately; PCE_pred = Voc·Jsc·FF + δ_PCE. OPV²D already has Voc/Jsc/FF labels — 3× supervision density at no data cost. |
| 3 | Listwise ranking loss | rank ≠ MSE | Add ListMLE (Plackett-Luce) on each batch's predicted PCE; directly optimizes the ranking distribution. |

Combined loss:

```
L = λ_pce · MSE(pce_z)
  + λ_aux · MSE(voc, jsc, ff)              # masked, per-channel std-normalized
  + λ_rank · ListMLE(pce_z, pce_target)
  + λ_phys · MSE(voc·jsc·ff, pce_pred)     # physics consistency
```

Default weights `(λ_pce, λ_aux, λ_rank, λ_phys) = (1.0, 0.5, 0.5, 0.1)`.

### Files

- `src/models/predictors/p3_physics.py`  — multi-output head, Voc anchored to MOE² HOMO/LUMO
- `src/training/losses.py`               — ListMLE + CompositeLoss
- `src/training/multitask_trainer.py`    — sibling of PCETrainer that handles dict outputs and aux labels
- `src/data/datasets.py`                  — `OPVPairDataset(include_aux=True)` exposes Voc/Jsc/FF with NaN-mask
- `configs/phys_rank.yaml`                — default config
- `scripts/06_train_phys_rank.py`         — entry point

### Reproduction

Pretrained encoder: `checkpoints/moe2_calc.pt` (from `master` branch's
from-scratch Stage 2 pretrain — same weights as the baseline).

```bash
# high_pce_holdout (q=0.85) — the algorithm target
python scripts/06_train_phys_rank.py --config configs/phys_rank.yaml

# random_kfold cross-check — make sure we don't regress on in-distribution
python scripts/06_train_phys_rank.py --config configs/phys_rank.yaml \
    split.kind=random_kfold split.kwargs.n_splits=5 split.kwargs.seed=3407 \
    trainer.out_subdir=phys_rank_random_kfold
```

### Results (single seed, same pretrain ckpt as master baseline)

| Split | Method | R² | MAE | Spearman | top10 | top20 | NDCG@10 |
|---|---|---|---|---|---|---|---|
| `high_pce_holdout` (q=0.85) | baseline P³ | -8.97 | 3.71 | +0.24 | 0.10 | 0.05 | 0.21 |
| `high_pce_holdout` (q=0.85) | **phys_rank** | **-6.92** | **3.24** | **+0.32** | 0.10 | 0.05\* | **0.44** |
| `random_kfold` (5-fold mean) | baseline P³ | 0.661 ± 0.052 | 1.66 | +0.79 | 0.56 | 0.53 | 0.94 |
| `random_kfold` (5-fold mean) | **phys_rank** | **0.671 ± 0.055** | 1.66 | +0.78 | 0.52 | 0.51 | 0.93 |

\* top20 on q=0.85 is dominated by the same handful of ~17.8-PCE molecules
that the model still cannot reach; it remains a hard target for future
algorithms.

### Headline

- **`high_pce_holdout` NDCG@10: 0.21 → 0.44 (+110%).**
- `high_pce_holdout` R² also improves (-8.97 → -6.92, +22%) and MAE (3.71 → 3.24, -13%) — better but still net negative because predictions still under-shoot the 17.8-PCE tail.
- `random_kfold` performance is unchanged within noise — no in-distribution sacrifice.

### Per-fold (random_kfold)

| Fold | R² | MAE | Spearman | top10 | NDCG@10 |
|---|---|---|---|---|---|
| 1 | 0.6887 | 1.65 | 0.82 | 0.60 | 0.93 |
| 2 | 0.6764 | 1.65 | 0.77 | 0.50 | 0.88 |
| 3 | 0.6346 | 1.65 | 0.78 | 0.50 | 0.95 |
| 4 | 0.6045 | 1.68 | 0.73 | 0.30 | 0.91 |
| 5 | 0.7493 | 1.40 | 0.80 | 0.70 | 0.95 |

### What's left for further iterations

- **top10 still 0.10 on q=0.85** — the model picks the right ranking
  *direction* (NDCG@10 doubled) but cannot identify the very-top molecules.
  Likely needs a tail-aware sample weighting or a pairwise loss focused on
  the highest quantile.
- **R² still negative on q=0.85** because predictions max out near 11.
  Loosening the Voc clamp or adding a Jsc bandgap-integral term
  (Alharbi-style) might extend the predicted range.
- **Single seed** — should average over ≥3 seeds before claiming the
  high_pce delta is robust; under noise, fold-level Spearman variance is
  ~0.05 from the random_kfold table.
