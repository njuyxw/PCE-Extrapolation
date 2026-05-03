# Reproduction Report: P³ on OPV²D

End-to-end **from-scratch** reproduction of the P³ PCE-prediction pipeline
from CycleChemist (arXiv:2511.19500v2). All weights — MOE² encoder and P³
head — are trained inside this repo from random initialization. **No
external checkpoints are used.**

## Setup

- Hardware: NVIDIA RTX 3090 (24 GB), CUDA 12.1
- Pretraining: `scripts/pretrain_moe2.py` (`configs/baseline.yaml`)
  - **Stage 1** — MLM atom-type prediction on Lopez 51k, 100 epochs
    (best val_acc = 0.9719 @ ep 92, ~42 min)
  - **Stage 2** — calc HOMO/LUMO regression on Lopez 51k, GAT layers frozen,
    head-only, 150 epochs
    (best val: HOMO R² = 0.849, LUMO R² = 0.770 @ ep 141, ~62 min)
  - Stage 3 (exp HOMO/LUMO on OPV²D union) is *not* used by the paper PCE
    protocol — skipped.
- PCE training: `scripts/train_pce.py` loading the stage-2 ckpt
  (this is the path the paper's released `train_pce.py` actually executes)
  - Random 5-fold KFold, seed = 3407, batch = 32
  - 100 epochs (warmup 20 with encoders frozen + finetune 80 unfrozen at lr×0.1)
  - lr = 1e-4, AdamW weight_decay = 5e-4, grad_clip = 1.0
  - Target standardization with **train-fold (μ, σ) only**

## Results

All numbers below are computed by `scripts/rank_from_predictions.py`
from the saved per-fold prediction CSVs.

### Random 5-fold (paper protocol)

`split.kind=random_kfold split.kwargs.n_splits=5 split.kwargs.seed=3407`

| Source | F1 | F2 | F3 | F4 | F5 | **Mean ± Std** | MAE |
|---|---|---|---|---|---|---|---|
| Paper Table 1 (P³ — GAT embedding) | — | — | — | — | — | **0.736 ± 0.033** | — |
| This repo (from-scratch)           | 0.6293 | 0.6736 | 0.6011 | 0.6608 | 0.7387 | **0.6607 ± 0.0519** | 1.66 |

Δ = -0.075 R² absolute. Most plausible cause is the omitted Stage 3
experimental HOMO/LUMO finetune on OPV²D plus a slightly weaker Stage 1
(val_acc 0.972 here vs paper 0.99997).

### Ranking quality (all from-scratch, same trained P³)

For material discovery, getting the *order* of candidates right matters as
much as the absolute MAE. Spearman ρ measures monotonic rank correlation;
top-K precision = |topK_pred ∩ topK_true| / K; NDCG@10 is gain-weighted
ranking with the true PCE as relevance score.

| Split | R² | MAE | Spearman ρ | Kendall τ | top10 | top20 | NDCG@10 |
|---|---|---|---|---|---|---|---|
| `random_kfold` (5-fold mean)    | **+0.66** | 1.66 | **+0.79** | +0.61 | **0.56** | 0.53 | **0.94** |
| `scaffold_acceptor` (n_test=23) | **+0.69** | 1.75 | **+0.80** | +0.63 | **0.70** | 0.95\* | **0.94** |
| `high_pce_holdout` q=0.85 (n_test=229) | **-8.97** | 3.71 | **+0.24** | +0.16 | **0.10** | 0.05 | **0.21** |

\* `scaffold_acceptor` top20 is high-noise because n_test=23 — top20 is
basically "rank the entire test set", so it's saturated when the rough
order is right.

### What this tells us

1. **In-distribution and scaffold-extrapolation ranking is strong.** With
   Spearman ρ ≈ 0.8 and NDCG@10 ≈ 0.94 on both `random_kfold` and
   `scaffold_acceptor`, P³ does pick out genuinely high-PCE candidates
   even when the absolute prediction is off by ~1.7 PCE on average.

2. **High-PCE extrapolation breaks both regression and ranking.** R² ≪ 0
   was already known, but Spearman drops from 0.80 to 0.24 and top10
   precision from 0.7 to 0.10 — i.e., when the test set is *only* high-PCE
   pairs, P³ also can't tell which one is best. Predictions cap at ~10
   while truth spans 11.9-17.8, so within the held-out tail the predicted
   ordering is barely better than random.

3. **Ranking is the right metric for material discovery.** For
   `high_pce_holdout`, MAE/RMSE alone hide that the *actual* failure is in
   distinguishing which extrapolated candidate is best. NDCG@10=0.21 is
   the headline number an algorithm aiming at OPV discovery should beat.

## Recommended algorithm — rank_focal + physics-committee ensemble

Built on top of the from-scratch baseline. Two complementary parts:

- **Learned**: `p3_physics` predictor (Voc anchored to Scharber
  `LUMO_A − HOMO_D − 0.3` from MOE² heads, multi-output Voc/Jsc/FF/δPCE)
  trained with `RankFocalLoss` (position-weighted ListMLE +
  top-quantile pairwise margin) and `WeightedRandomSampler` oversampling
  the high-PCE tail (α=2).
- **Physics committee**: `mean(Scharber, Imamura, Alharbi)` computed
  from MOE² HOMO_D / LUMO_A. Final prediction =
  `α · physics_mean + (1 − α) · learned`.

Headlines on `discovery_mix` (train on bulk only, test = bulk sample +
ALL high-PCE; 3 seeds):

| Method | Overall R² | NDCG@10 | top10 | Bulk R² | Tail R² |
|---|---|---|---|---|---|
| Baseline P³ (paper protocol) | +0.41 ± 0.06 | 0.86 | 0.27 | +0.56 | -4.31 |
| **rank_focal + ensemble (α=0.3)** | **+0.52 ± 0.04** | **0.92** | **0.43** | +0.39 | -0.33 |

Improvements: NDCG@10 +7 %, top10 **+59 %**, R² preserved, tail R²
jumps from -4.3 to -0.3. **α = 0.3 is the Pareto knee** — gives up
0.17 R² on bulk to gain 4.0 R² on the tail. Higher α (0.5–0.7) shifts
further toward the tail at greater bulk cost.

```bash
# Recommended pipeline
python scripts/train_rank_focal.py --config configs/rank_focal.yaml
python scripts/ensemble_physics_rank.py --config configs/rank_focal.yaml \
    ensemble.fold_ckpt=outputs/rank_focal_discovery_mix/fold1_best.pt
python scripts/evaluate_discovery_mix.py \
    --diagnostics outputs/rank_focal_discovery_mix/ensemble/fold1_diagnostics.csv
```

## Provenance

| File | Origin |
|---|---|
| `data/raw/lopez51k.csv`           | Lopez NFA 51k candidate database (Joule 2017) |
| `data/raw/opv2d.csv`              | OPV²D — 1567 D-A pairs from CycleChemist `exp_dataset.csv` |
| `data/processed/opv2d_clean.csv`  | OPV²D after `prepare_data.py` (1525 rows; Y6 acceptors held out) |
| `data/processed/opv2d_y6.csv`     | Y6 holdout (33 pairs) reserved for stage-2 evaluation |
| `checkpoints/moe2_mlm.pt`         | trained from scratch by `pretrain_moe2.py` (Stage 1) |
| `checkpoints/moe2_calc.pt`        | trained from scratch by `pretrain_moe2.py` (Stage 2) |
| `outputs/repro_*/`                | all 5-fold runs by `train_pce.py` |
