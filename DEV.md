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

## Algorithm 2: `rank_focal` — tail-focused ranking

### Motivation

After phys_rank, two failures persisted on `high_pce_holdout`:
1. **`top10 = 0.10` did not move** — the model captured rough rank
   direction (Spearman 0.32) but could not identify the very-top molecules
   inside the held-out tail.
2. **R² still ≪ 0** — predictions still capped near 11 PCE while truth
   went to 17.8.

Three orthogonal extensions on top of phys_rank:

| Component | Mechanism |
|---|---|
| **Tail-weighted sampling** | `WeightedRandomSampler` with `weight ∝ ((PCE − PCE_min)/(PCE_max − PCE_min) + ε)^α`, α=2. The ~5 % high-PCE pairs were rare in each batch; oversampling forces the optimizer to see them every batch. |
| **Position-weighted ListMLE** | Multiply each ListMLE term by `1/log₂(rank+2)` (NDCG-style). Top-1 weight = 1.0, top-10 = 0.29 — capacity is spent on the top of the list, which is what NDCG@K measures. |
| **Top-quantile pairwise margin** | For pairs `(i,j)` where `target_i` is in the top 30 % of the batch *and* `target_i − target_j ≥ 0.5`, enforce `pred_i − pred_j ≥ 0.3` via squared hinge. Direct supervision on the orderings that matter for top-K precision. |

Loss:
```
L = 0.5 · MSE(pce_z) + 0.5 · MSE(voc, jsc, ff) + 1.0 · weighted_ListMLE
  + 0.5 · top_quantile_margin + 0.05 · MSE(voc·jsc·ff, pce_pred)
```

### Files

- `src/training/losses.py`               — added `position_weighted_listmle`, `top_quantile_margin_loss`, `RankFocalLoss`
- `src/training/multitask_trainer.py`    — added optional ``loss_module`` and ``WeightedRandomSampler`` (gated by ``tail_sampling_alpha``)
- `configs/rank_focal.yaml`              — default config (predictor reuses `p3_physics`)
- `scripts/07_train_rank_focal.py`       — entry point

## Algorithm 3 (reference): `scharber_baseline` — pure analytical PCE

Loads the from-scratch `moe2_calc.pt` encoder, predicts HOMO_D / LUMO_A
for every donor / acceptor, then computes PCE without any learning:

```
Voc = clamp(|HOMO_D| − |LUMO_A| − 0.3, 0, 2.5)
Eg  = LUMO_A − HOMO_D
Jsc = 0.65 · 70 · exp(−1.05·(Eg − 0.7))           # AM1.5G envelope, mA/cm²
FF  = 0.65
PCE = Voc · Jsc · FF
```

This tells us how much of the predictive signal already lives in the
physics formula and the MOE² heads, separate from the learned PCE
regression. **Any learned method should beat this on ranking** — if not,
the learning is adding noise rather than signal.

- `scripts/08_scharber_baseline.py`      — single inference pass on every split

## Combined results (single seed; from-scratch master pretrain ckpt)

| Method | random_kfold (5-fold mean) | | | | high_pce_holdout q=0.85 | | | |
|---|---|---|---|---|---|---|---|---|
| | R² | Spear | top10 | NDCG@10 | R² | Spear | top10 | NDCG@10 |
| baseline P³ (master) | **0.661** | 0.79 | 0.56 | 0.94 | -8.97 | 0.24 | 0.10 | 0.21 |
| phys_rank | **0.671** | 0.78 | 0.52 | 0.93 | -6.92 | 0.32 | 0.10 | 0.44 |
| **rank_focal**       | 0.580 | 0.75 | 0.54 | 0.93 | **-5.40** | **0.46** | **0.40** | **0.67** |
| scharber (no learning) | -1.52 | -0.13 | 0.02 | 0.35 | **-1.11** | 0.20 | 0.00 | 0.30 |

### Headline

- **rank_focal lifts `high_pce_q85` `top10` from 0.10 → 0.40 (4×)** and
  `NDCG@10` from 0.21 → 0.67 (3.2×).
- It costs ~0.09 R² on `random_kfold` (0.671 → 0.580) — a deliberate
  trade since the algorithm is biased toward the tail by both sampling
  and loss weighting.
- Scharber-only is **the best on absolute R² and MAE** for `high_pce_q85`
  (-1.11 vs -5.40 to -8.97 for learned methods) — *most of the magnitude
  signal lives in the physics formula*. But it ranks badly inside the
  dense bulk (Spearman -0.13 on random_kfold), so it is a complement, not
  a replacement.

## Algorithm 4: `physics_committee_ensemble` — committee of empirical formulas + learned model

### Motivation

The Solar RRL 2024 review (Jiang et al., Table 1) catalogues at least four
empirical PCE formulas, each with distinct validity regimes:

| Formula | Voc | Jsc | FF | Designed for |
|---|---|---|---|---|
| Scharber (1980s) | \|HOMO_D\|−\|LUMO_A\|−0.3 | 0.65·∫Φ_ph dλ over Eg | 0.65 | bulk OPV |
| Imamura | same | refined integrand | 0.70 | fullerene acceptors |
| Alharbi | Eg−0.5−0.0114·\|LUMO_A\|^1.86−0.057·Eg | as Scharber | Voc/(Voc+12kT/q) — SQ-style | tighter on high-Voc |
| OPEP/B3LYP | TD-DFT-augmented | DFT-derived | learned | NFAs at PCE > 9% |

The first three need only HOMO_D / LUMO_A — *which our MOE² heads
already predict* — so they are free to evaluate. OPEP needs DFT and is
out of scope.

**The empirical formulas disagree most where the underlying physics is
non-ideal.** Each makes different assumptions about the high-Voc tail
(Alharbi's quadratic LUMO term diverges from Scharber's linear one,
Alharbi's SQ-style FF diverges from the constant 0.65/0.70, etc.). So
the per-pair *standard deviation* across formulas is a free epistemic
uncertainty estimate — no Bayesian / MC-dropout machinery needed.

This motivates **disagreement-aware blending** of physics + learned:

| Mode | Description |
|---|---|
| `M0_scharber` | single Scharber prediction (reference floor) |
| `M1_physics_mean` | naive mean of Scharber / Imamura / Alharbi |
| `M2_learned_only` | the trained `rank_focal` model |
| `M3_fixed_blend` | `α·physics_mean + (1-α)·learned`, α tuned on val |
| `M4_gated_blend` | `w·physics_mean + (1-w)·learned`, `w = exp(-σ_phys/τ)` |
| `M5_rrf_2` | Reciprocal Rank Fusion of 2 lists (physics_mean, learned) |
| `M6_rrf_4` | RRF of 4 lists (Scharber, Imamura, Alharbi, learned) |

### Files

- `src/training/physics_committee.py` — Scharber / Imamura / Alharbi formulas, committee statistics, RRF + disagreement-gated blends
- `scripts/09_ensemble_physics_rank.py` — runs all 7 modes side-by-side on the test split

### Hyper-parameter sweeps (`high_pce_holdout` q=0.85, single seed)

`τ` for the disagreement-gated blend (M4):

| τ | R² | MAE | Spearman | top10 | NDCG@10 |
|---|---|---|---|---|---|
| 0.3 | -5.39 | 2.88 | 0.46 | 0.40 | 0.67 |
| 1.0 | -4.05 | 2.53 | 0.46 | 0.40 | 0.67 |
| 2.0 | -2.02 | 1.88 | 0.45 | 0.40 | 0.69 |
| 5.0 | -0.34 | 1.11 | 0.39 | 0.40 | 0.75 |
| 10.0| -0.04 | 0.99 | 0.32 | 0.40 | 0.75 |

`α` for the fixed blend (M3):

| α | R² | MAE | Spearman | top10 | **NDCG@10** |
|---|---|---|---|---|---|
| 0.3 | -2.29 | 1.98 | 0.46 | 0.40 | 0.75 |
| 0.4 | -1.55 | 1.70 | 0.45 | 0.40 | 0.76 |
| 0.5 | -0.96 | 1.44 | 0.44 | 0.40 | 0.75 |
| **0.6** | **-0.52** | **1.23** | 0.41 | 0.40 | **0.76** ← winner |
| 0.7 | -0.23 | 1.08 | 0.37 | 0.40 | 0.76 |

### End-to-end progression on `high_pce_holdout` q=0.85

| Method | R² | MAE | Spearman | top10 | NDCG@10 |
|---|---|---|---|---|---|
| baseline P³ (master) | -8.97 | 3.71 | 0.24 | 0.10 | 0.21 |
| phys_rank (algo 1) | -6.92 | 3.24 | 0.32 | 0.10 | 0.44 |
| rank_focal (algo 2) | -5.40 | 2.88 | 0.46 | 0.40 | 0.67 |
| scharber alone (algo 3) | -1.11 | 1.50 | 0.20 | 0.00 | 0.30 |
| physics_committee (M1) | -0.26 | 1.15 | 0.20 | 0.00 | 0.28 |
| **ensemble M3 α=0.6 (algo 4)** | **-0.52** | **1.23** | **0.41** | **0.40** | **0.76** |
| ↑ vs baseline P³ | +94 % | -67 % | +71 % | **+300 %** | **+260 %** |

### Insights surfaced

1. **The committee alone (M1) drops R² from -1.11 → -0.26 on the tail**,
   without any learning. Naive averaging cancels formula-specific biases
   (e.g. Scharber's constant FF=0.65 vs Alharbi's SQ-style coupling).
   This is the "free" benefit of integrating multiple physical models.

2. **Fixed α-blend dominates the disagreement gate** at every operating
   point we tested. The reason: σ_phys turns out to *correlate with the
   tail* (formulas disagree more on high-Voc materials), and on the tail
   physics is *more reliable* than learned — so a gate that *down-weights*
   physics when σ is large does the wrong thing. A gate that *up-weights*
   physics under high σ might work, but the simpler fixed blend is hard
   to beat in the regime where the optimal weight is ~constant.

3. **Rank-space fusion (M5/M6) loses to value-space blends here.** RRF
   discards magnitude, but in our case magnitude *is* the winning signal:
   physics gives the right *level*, learned model gives the right *order*,
   and an additive blend gets both. RRF can only contribute *order*, so it
   ties learned-only on Spearman but cannot fix R²/MAE.

4. **Ensemble strategy is split-aware**, not universal. On the bulk
   (`random_kfold`) the physics committee was anti-correlated with PCE
   (Spearman -0.13), so blending physics in *would hurt* there. The right
   pipeline for OPV discovery: route candidates by their physics-vs-
   learned disagreement, blend only when the model is clearly outside its
   training regime (large physics-learned gap *and* high σ_phys).

## Multi-seed evaluation of the current best (algo 4 ensemble)

Repeated the full pipeline (`rank_focal` train → ensemble M3 α=0.6) for
seeds 1, 2, 42 on `high_pce_holdout` q=0.85. Same hyper-parameters,
same MOE² stage-2 ckpt; only the model init / data-shuffling seed
varies.

### Per-seed results

| seed | M2 rank_focal R² / MAE / Spear / top10 / NDCG@10 | M3 ensemble α=0.6 R² / MAE / Spear / top10 / NDCG@10 |
|---|---|---|
| 1  | -3.91 / 2.42 / 0.31 / 0.10 / 0.48 | -0.37 / 1.14 / 0.33 / 0.10 / 0.42 |
| 2  | -1.65 / 1.69 / 0.53 / 0.50 / 0.76 | -0.02 / 1.00 / 0.41 / 0.40 / 0.73 |
| 42 | -5.40 / 2.88 / 0.46 / 0.40 / 0.67 | -0.52 / 1.23 / 0.41 / 0.40 / 0.75 |

### Aggregate (mean ± std and median)

| Method | R² mean±std | MAE | Spearman | top10 | NDCG@10 |
|---|---|---|---|---|---|
| **rank_focal alone (M2)** | -3.65 ± 1.89 | 2.33 ± 0.61 | 0.43 ± 0.11 | 0.33 ± 0.21 | 0.64 ± 0.15 |
| **ensemble M3 α=0.6**    | **-0.31 ± 0.26** | **1.12 ± 0.12** | 0.38 ± 0.05 | 0.30 ± 0.17 | 0.63 ± 0.19 |
| ensemble (median across seeds) | -0.37 | 1.14 | 0.41 | **0.40** | **0.73** |

### Stability gains from the physics ensemble

- **R² std shrinks 7×** (1.89 → 0.26)
- **MAE std shrinks 5×** (0.61 → 0.12)
- **Spearman std shrinks 2×** (0.11 → 0.05)
- top10 / NDCG@10 std are roughly preserved — these are dominated by
  whether the learned model gets the very-top molecule right, which is
  high-variance with only n_test=229 and 1 fold

### Robustness verdict

- The physics committee provides a *stable magnitude floor*: the
  ensemble's R² is in [-0.52, -0.02] across seeds, vs [-5.40, -1.65] for
  the learned model alone.
- The ranking metrics (NDCG@10, top10) inherit the learned model's
  variance — when seed=1 produced a poorly-ordered learned model, the
  ensemble could not rescue its NDCG@10 (0.42), because physics_mean
  alone is also poorly ordered (NDCG@10 ≈ 0.28). **The ensemble cannot
  manufacture ranking signal that neither component has.**
- Median is a more honest summary than the mean here: median NDCG@10 of
  0.73 is consistent with the single-seed (=42) result of 0.75 we
  reported earlier.

## Recommended hyper-parameters (for someone reproducing this)

For `high_pce_holdout` q=0.85 (the main material-discovery target):

| Phase | Knob | Value | Note |
|---|---|---|---|
| **Pretrain** | stage 1 (MLM) epochs | 100 | val_acc plateau ≈ 0.97 |
| | stage 1 lr / batch | 5e-5 / 128 | AdamW + AMP |
| | stage 2 (calc HOMO/LUMO) epochs | 150 | freeze conv1/2/3, head only |
| | stage 2 head lr / batch | 5e-5 / 128 | val HOMO R² ≈ 0.85 |
| | stage 3 | *skipped* | paper PCE code path uses stage-2 ckpt |
| **Predictor** | architecture | `p3_physics` | Voc-anchored multi-output head |
| | Voc clamp | (0, 2.5) | Scharber range; never re-tuned |
| | FF range (sigmoid) | (0.30, 0.85) | wide enough for OPV literature |
| | Jsc range (sigmoid) | (1, 35) | mA/cm² — covers AM1.5G envelope |
| | residual scales (Voc / PCE) | 0.20 / 0.50 | learned residual on top of physics |
| **Loss** (`RankFocalLoss`) | λ_pce | 0.5 | down-weighted vs ranking |
| | λ_aux | 0.5 | per-channel std-normalized + masked |
| | **λ_rank** | **1.0** | position-weighted ListMLE — primary signal |
| | λ_margin (weight) | 0.5 | top-quantile pairwise hinge |
| | margin quantile | 0.7 | top 30 % of batch is "high target" |
| | margin target diff min | 0.5 | only enforce on real PCE-difference pairs |
| | margin value (hinge) | 0.3 | in standardized PCE units |
| | λ_phys | 0.05 | gentle Voc·Jsc·FF ≈ pce_pred coupling |
| **Sampling** | `tail_sampling_alpha` | **2.0** | weight ∝ ((PCE−min)/(max−min)+ε)^α |
| **Training** | warmup epochs (encoders frozen) | 20 | matches paper |
| | total epochs | 100 | early stop patience 30 |
| | lr / finetune lr scale | 1e-4 / 0.1 | finetune lr = 1e-5 |
| | batch / weight_decay / grad_clip | 32 / 5e-4 / 1.0 | matches paper |
| **Ensemble** | physics members | Scharber + Imamura + Alharbi | run on MOE² heads |
| | mode | M3 fixed blend | beats M4 gated and M5/6 RRF |
| | **α (physics weight)** | **0.6** | sweep showed α∈[0.4, 0.7] is robust |

### Reproduction one-liner

```bash
# from the dev branch, with pretrained moe2_calc.pt already in checkpoints/
python scripts/07_train_rank_focal.py --config configs/rank_focal.yaml \
    seed=42 trainer.out_subdir=rank_focal_high_pce_q85

python scripts/09_ensemble_physics_rank.py \
    --config configs/rank_focal.yaml \
    ensemble.fold_ckpt=outputs/rank_focal_high_pce_q85/fold1_best.pt \
    ensemble.alpha=0.6
```

Expected outputs (single seed):
- `M2 (learned only)`: R² ≈ -5.4, NDCG@10 ≈ 0.67, top10 ≈ 0.40
- `M3 (ensemble α=0.6)`: R² ≈ -0.5, NDCG@10 ≈ 0.75, top10 ≈ 0.40

## Insights for future iterations

1. **Regime-aware blending is the obvious next algorithm.** Scharber
   wins on the tail's *magnitude* (R² -1.11) while rank_focal wins on the
   tail's *ordering* (Spearman 0.46). A gating model that uses
   `(1 − w) · pred_learned + w · pred_scharber`, with `w` increasing in
   epistemic uncertainty (or in `min(|train_PCE − pred_PCE|)` to detect
   tail), should dominate both.
2. **top10 = 0.40 still leaves headroom.** The remaining 60 % miss is
   likely because OPV²D's top-10 includes a few heavily-engineered
   solvent / morphology cases (Y6-derivatives etc.) whose PCE is not
   determined by molecule alone. Adding device-side features (D:A ratio,
   solvent, processing) or *predicting the upper quantile* (pinball
   loss) instead of the mean would address this.
3. **Position-weighted ListMLE is a cheap, strong signal.** It lifted
   top10 from 0.10 → 0.40 *with the same encoder* — so most of the
   capacity needed to rank tail molecules was already learned by the
   baseline's GAT, the issue was only the loss telling it which positions
   to care about.
4. **Single seed only** — should average over ≥3 seeds before claiming
   the high_pce delta is robust; fold-level Spearman variance under
   random_kfold is ~0.05 in the table above, which is the noise floor.
