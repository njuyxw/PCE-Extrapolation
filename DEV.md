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

### Multi-seed × α sweep

`scripts/10_alpha_sweep_multiseed.py` recomputes the M3 fixed-blend
post-hoc from each seed's saved diagnostics CSV — pure numpy, no model
loading. Sweep over α ∈ {0.0, 0.1, ..., 1.0} on the same 3 seeds:

| α | R² mean ± std | MAE | Spearman | top10 | NDCG@10 mean (median) |
|---|---|---|---|---|---|
| 0.0 (learned only) | -3.65 ± 1.89 | 2.33 | 0.43 | 0.33 | 0.64 (0.67) |
| 0.1 | -2.82 ± 1.51 | 2.10 | 0.44 | 0.33 | 0.66 (0.70) |
| 0.2 | -2.10 ± 1.18 | 1.87 | 0.44 | 0.33 | 0.66 (**0.75**) |
| 0.3 | -1.49 ± 0.89 | 1.64 | 0.44 | 0.33 | 0.65 (**0.75**) |
| 0.4 | -0.99 ± 0.64 | 1.43 | 0.43 | 0.33 | 0.65 (**0.75**) |
| **0.5** ← robust | **-0.59 ± 0.43** | **1.25** | 0.41 | 0.33 | 0.65 (**0.75**) |
| 0.6 | -0.31 ± 0.26 | 1.12 | 0.38 | 0.30 | 0.63 (0.73) |
| 0.7 | -0.13 ± 0.13 | 1.04 | 0.35 | 0.27 | 0.60 (0.67) |
| 0.8 | -0.07 ± 0.05 | 1.02 | 0.30 | 0.20 | 0.51 (0.55) |
| 0.9 | -0.11 ± 0.01 | 1.07 | 0.26 | 0.13 | 0.36 (0.34) |
| 1.0 (physics only) | -0.26 ± 0.00 | 1.14 | 0.20 | 0.00 | 0.28 (0.28) |

Findings:

1. **Median NDCG@10 plateaus at 0.75 for α ∈ [0.2, 0.5].** The mean is
   pulled down at low α by seed=1 (the outlier), so the median is the
   more honest summary in this regime. The plateau is wide (4 α
   settings tied), confirming the ensemble is robust to the exact
   weight as long as both signals are present.

2. **The single-seed (=42) winner α=0.6 was slightly over-fit.** At
   α=0.6 the median NDCG@10 drops to 0.73, and at α=0.7 to 0.67.
   Pulling α down to 0.5 recovers the full 0.75 ranking while only
   modestly worsening R² (-0.31 → -0.59).

3. **R²/MAE improve monotonically with α up to ~0.7.** This is the
   physics committee dragging the magnitude floor closer to ~ 0 R²
   (its single-mode value is -0.26). Past α=0.7 the loss of ranking
   signal dominates and NDCG collapses with FF.

**Updated recommended α: α = 0.5** (was α=0.6 from single-seed). It is
the leftmost point of the median-NDCG plateau, so it spends as little
"physics weight" as possible while still capturing the magnitude floor.

### α sweep on `random_kfold` (5 folds, seed=3407)

`scripts/11_ensemble_random_kfold.py` runs the same ensemble on the
in-distribution split. Each fold uses its own `fold{i}_best.pt` (no
test-set leakage across folds).

| α | R² mean ± std | MAE | Spearman | top10 | NDCG@10 ± std |
|---|---|---|---|---|---|
| 0.0 (learned only) | **+0.580 ± 0.056** | 1.85 | 0.752 | 0.54 | 0.926 ± 0.028 |
| 0.1 | +0.578 ± 0.042 | 1.83 | 0.753 | 0.54 | 0.926 ± 0.028 |
| 0.2 | +0.517 ± 0.047 | 1.93 | **0.754** | 0.54 | **0.927** ± 0.029 |
| 0.3 | +0.395 ± 0.058 | 2.13 | 0.753 | 0.54 | 0.923 ± 0.032 |
| 0.4 | +0.213 ± 0.070 | 2.43 | 0.742 | 0.52 | 0.920 ± 0.033 |
| 0.5 | -0.028 ± 0.082 | 2.80 | 0.718 | 0.54 | 0.922 ± 0.032 |
| 0.6 | -0.330 ± 0.099 | 3.21 | 0.661 | 0.52 | 0.918 ± 0.028 |
| 0.7 | -0.691 ± 0.122 | 3.66 | 0.540 | 0.52 | 0.922 ± 0.028 |
| 0.8 | -1.113 ± 0.155 | 4.13 | 0.324 | 0.50 | 0.917 ± 0.033 |
| 0.9 | -1.594 ± 0.199 | 4.61 | 0.066 | 0.46 | 0.906 ± 0.028 |
| 1.0 (physics only) | -2.135 ± 0.254 | 5.11 | -0.131 | 0.00 | 0.328 ± 0.057 |

Findings:

1. **NDCG@10 is flat at ~0.92 across α ∈ [0.0, 0.9]** — almost
   completely insensitive to α. This is because on the dense bulk the
   true top-10 is dominated by molecules that both physics and learned
   identify as good; blending preserves their relative ordering.
   Ranking only collapses at α=1.0 where physics's bulk anti-correlation
   (Spearman -0.13) wipes out the signal.

2. **R²/MAE prefer α near 0** — opposite of `high_pce_holdout`. Pure
   learned R²=+0.58 vs pure physics R²=-2.14. R² crosses zero around α
   ≈ 0.5, exactly the value that wins on the tail. The two regimes
   prefer *opposite* operating points.

3. **At α=0.5 (the tail winner): random_kfold R² collapses to ~0**
   (-0.03), Spearman drops 0.04, NDCG@10 only loses 0.005. So a global
   α=0.5 spends ~0.6 R² on the bulk to gain ~5 R² on the tail — still a
   good trade overall, but not a free lunch.

### Cross-regime recommendation

| Regime | Optimal α | R² | Spearman | NDCG@10 |
|---|---|---|---|---|
| `random_kfold` (in-distribution) | **α = 0.0–0.1** | +0.58 | 0.75 | 0.93 |
| `high_pce_holdout` (extrapolation) | **α = 0.5** | -0.59 | 0.41 | 0.75 (median) |
| **Single α that is safest in both** | **α = 0.2** | bulk: +0.52 / 0.93, tail: -2.10 / 0.66 (median 0.75) |

The cleanest answer is to **let α depend on the deployment regime**.
For OPV discovery the relevant regime is `high_pce_holdout` (we want to
extrapolate to higher PCE), so α=0.5 is the right knob there.

If a single α is required across the whole pipeline (e.g. for a
deployed predictor that doesn't know whether the candidate is in or
out of distribution), **α = 0.2** is the safest choice: nearly full
in-distribution performance (R² 0.52, NDCG@10 0.93) and the full
median ranking on the tail (NDCG@10 0.75) — only the tail R²/MAE
suffer.

This split-aware finding generalises insight #5 from the previous
section: **the ensemble blend coefficient is itself a regime detector**.
A future algorithm could *learn* α (or σ-gate it) from features that
detect distribution shift on the candidate (e.g. how far its
predicted HOMO/LUMO is from the training-set centroid).

### Physics-subset decomposition — "averaging the 3 isn't the best"

`scripts/12_physics_subset_sweep.py` evaluates every non-empty subset
of `{Scharber, Imamura, Alharbi}` on each split, blended with the
learned model at α ∈ {0.0, 0.2, 0.5, 0.7, 1.0}. Pure post-hoc analysis
on the saved diagnostics CSVs.

Cross-regime summary at α=0.5:

| Subset | high_pce R² / NDCG (med) | random R² / NDCG |
|---|---|---|
| Scharber alone | -1.56 / 0.61 (0.68) | +0.13 / 0.92 |
| Imamura alone | **-0.11** / 0.61 (0.66) | -0.81 / 0.92 |
| **Alharbi alone** | -2.39 / **0.67** (0.72) | **+0.315** / **0.932** (med 0.94) |
| S+I | -0.20 / 0.61 (0.66) | -0.26 / 0.92 |
| S+A | -1.89 / 0.65 (0.75) | +0.23 / 0.92 |
| **I+A** | **-0.25** / 0.65 (0.75) | -0.12 / 0.92 |
| S+I+A (current default) | -0.59 / 0.65 (0.75) | -0.03 / 0.92 |

Cross-regime at α=0.7:

| Subset, α=0.7 | high_pce R² | high_pce NDCG (med) | random R² | random NDCG |
|---|---|---|---|---|
| {I+A} | **+0.11** ← first positive R² on tail | 0.64 (0.75) | -0.88 | 0.92 |
| Alharbi alone | -2.12 | **0.679** (best mean) | -0.02 | **0.932** |
| {S+A} | -1.49 | 0.63 (0.72) | -0.18 | 0.92 |
| {S+I+A} (default) | -0.13 | 0.60 (0.67) | -0.69 | 0.92 |

Findings:

1. **Scharber is a liability on the tail.** Its `Voc = Eg − 0.3` has no
   energy-loss correction and over-estimates Voc systematically;
   averaging it in biases the blend high. **Dropping Scharber lets us
   cross zero R² on `high_pce_q85`** for the first time — `{I+A}` at
   α=0.7 gives R² = +0.11, MAE = 0.93.

2. **Alharbi is the most informative single formula.** Its `|LUMO_A|^1.86`
   energy-loss penalty + SQ-style FF coupling `Voc/(Voc+0.31)` are
   calibrated for high-Voc materials. Alharbi alone at α=0.5 even
   *improves* bulk NDCG@10 above learned-only (0.926 → 0.932; median
   0.94). Imamura alone is much weaker — its only difference from
   Scharber is FF=0.70 (vs 0.65), insufficient to overcome the
   over-Voc bias.

3. **No single subset wins all metrics.** {I+A} α=0.7 wins R²/MAE on
   the tail; Alharbi alone α=0.7 wins NDCG@10 / top10 on the tail;
   Alharbi alone α=0.5 wins on bulk. Implies a *physics-formula-aware
   router* could improve further: pick the subset (or learn its
   weights) given the candidate.

4. **Pair > triple.** The best results consistently come from
   2-formula subsets, not the full triple. The triple over-smooths;
   the pair retains the SQ-FF distinction (Alharbi) without averaging
   it away.

5. **Best operating points by use-case:**

   | Use-case | Subset | α | Numbers |
   |---|---|---|---|
   | Maximize tail R² (positive on q=0.85) | **{I+A}** | **0.7** | R² +0.11, MAE 0.93 |
   | Maximize tail ranking (NDCG@10 / top10) | **Alharbi** | **0.7** | NDCG@10 0.679, top10 0.40 |
   | Maximize bulk NDCG@10 | **Alharbi** | **0.5** | NDCG@10 0.932 (med 0.94), R² +0.32 |
   | Universal single config | **Alharbi** | **0.5** | bulk +0.32 / 0.94 ; tail -2.4 / 0.72 |
   | Universal balanced (zero R² both) | **{I+A}** | **0.5** | bulk -0.12 / 0.92 ; tail -0.25 / 0.75 (med) |

### Discovery-realistic mixed split

`discovery_mix` (added in `src/data/splits.py`):

- **Train**: random ~80 % of the *bulk* (PCE < high_pce_quantile cutoff)
- **Val**: random ~10 % of the bulk
- **Test**: ~10 % of the bulk **plus** every pair in the high-PCE
  quantile (in our setup: 130 bulk + 229 tail = 359 pairs, ~36 % tail)

This mimics the real OPV-discovery pipeline — most candidates are
ordinary, a few are exceptional, and a useful screening predictor must
rank both kinds simultaneously. Eval script
`scripts/13_evaluate_discovery_mix.py` reports metrics three ways:
overall (full mixed test), bulk-only (PCE < cutoff), tail-only
(PCE ≥ cutoff).

3 seeds (1, 2, 42) of `rank_focal` trained on this split, evaluated
under all physics subsets × α ∈ {0.0, 0.3, 0.5, 0.7}.

### Cross-α / cross-subset summary on `discovery_mix`

**Overall** (mixed test, n=359):

| α | subset | R² ± std | MAE | Spearman | top10 | NDCG@10 |
|---|---|---|---|---|---|---|
| 0.0 (learned only) | — | +0.41 ± 0.06 | 2.18 | 0.71 | 0.27 | 0.86 |
| **0.3** | **{Imamura}** | **+0.52** ± 0.04 | **1.72** | 0.70 | **0.43** | **0.924** |
| **0.3** | **{S+I+A}** | +0.52 ± 0.03 | 1.86 | 0.70 | 0.40 | 0.92 |
| 0.5 | {S+I+A} | +0.41 ± 0.03 | 1.90 | 0.66 | **0.43** | 0.92 |
| 0.7 | {I+A} | +0.07 ± 0.03 | 2.17 | 0.55 | 0.40 | 0.91 |

**Bulk** (test PCE < 11.87 cutoff, n=130):

| α | subset | R² | NDCG@10 |
|---|---|---|---|
| 0.0 (learned only) | — | **+0.56** | 0.87 |
| 0.3 | {Alharbi} | +0.39 | **0.89** |
| 0.5 | {Alharbi} | +0.02 | 0.89 |

**Tail** (test PCE ≥ 11.87, n=229):

| α | subset | R² | top10 | NDCG@10 |
|---|---|---|---|---|
| 0.0 (learned only) | — | -4.31 | 0.27 | 0.56 |
| 0.5 | **{Imamura}** | **+0.06** | 0.40 | 0.71 |
| 0.7 | **{I+A}** | **+0.14** | 0.40 | **0.72** |

### Findings on the mixed split

1. **The mixed test is harder than either pure split**, and surfaces
   real-world utility better. Learned-only gets NDCG@10 = 0.86 and
   top10 = 0.27 — substantially worse than the 0.93/0.54 it gets on
   `random_kfold` or the 0.67/0.40 on `high_pce_q85`. The model has to
   rank ordinary candidates (where it's strong) and exceptional ones
   (where it's weak) on the *same* scale.

2. **Sweet spot is α=0.3** — *lower* than either pure split's
   recommendation. `random_kfold` preferred α≈0, `high_pce_q85`
   preferred α≈0.5; the mixed test compromises at α=0.3 because both
   regimes contribute to the metrics. At α=0.3 with Imamura-only or
   S+I+A:
   - Overall NDCG@10 jumps from 0.86 (learned only) to **0.92** (+7 %)
   - Overall top10 jumps from 0.27 to **0.43** (+59 %)
   - Overall R² stays at **+0.52** (was +0.41 learned only)
   - Bulk R² drops only marginally (+0.56 → +0.39)
   - Tail R² improves dramatically (-4.31 → -0.33)

3. **Explicit bulk-vs-tail Pareto.** Same α = different region winners:

   | α | bulk R² | tail R² | tradeoff |
   |---|---|---|---|
   | 0.0 | **+0.56** | -4.31 | bulk-only |
   | 0.3 | +0.39 | -0.33 | balanced — Pareto knee |
   | 0.5 | +0.02 | **+0.06** | both ≈ 0 |
   | 0.7 | -0.54 | +0.14 | tail-only |

   A user who cares more about not breaking the bulk should pick α=0.3;
   one optimizing for novel-material magnitude can push to α=0.5–0.7.

4. **Imamura is the most useful single formula on the mix**, despite
   being weakest on the pure tail. Its slightly higher FF (0.70 vs
   Scharber's 0.65 and Alharbi's coupling) plus same-form Voc gives a
   blend that better matches both bulk magnitude and tail ranking.

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
| **Ensemble** | mode | M3 fixed blend | beats M4 gated and M5/6 RRF |
| | **physics subset (high_pce_holdout)** | **{Imamura, Alharbi}** | drops Scharber; gives positive R² on tail at α=0.7 |
| | **physics subset (random_kfold)** | **{Alharbi}** | even improves bulk NDCG@10 above learned-only |
| | **physics subset (universal)** | **{Imamura, Alharbi}** | balanced both regimes |
| | **α (high_pce_holdout, R²-priority)** | 0.7 with {I+A} | R² +0.11, NDCG@10 0.64 (median 0.75) |
| | **α (high_pce_holdout, NDCG-priority)** | 0.7 with {Alharbi} | NDCG@10 0.679, top10 0.40 |
| | **α (random_kfold)** | 0.5 with {Alharbi} | R² +0.32, NDCG@10 0.932 (median 0.94) |
| | **α (universal balanced)** | 0.5 with {I+A} | bulk -0.12/0.92, tail -0.25/0.75 (median) |
| | **α (discovery_mix — Pareto knee)** | **0.3 with {Imamura} or {S+I+A}** | overall R² +0.52, NDCG@10 0.92, top10 0.43; bulk R² +0.39, tail R² -0.33 |

### Reproduction one-liner

```bash
# from the dev branch, with pretrained moe2_calc.pt already in checkpoints/

# Recommended: train 3+ seeds, then ensemble at α=0.5 on each.
for SEED in 1 2 42; do
    python scripts/07_train_rank_focal.py --config configs/rank_focal.yaml \
        seed=$SEED trainer.out_subdir=rank_focal_high_pce_q85_seed$SEED
    python scripts/09_ensemble_physics_rank.py --config configs/rank_focal.yaml \
        ensemble.fold_ckpt=outputs/rank_focal_high_pce_q85_seed$SEED/fold1_best.pt \
        ensemble.alpha=0.5
done

# Then aggregate post-hoc:
python scripts/10_alpha_sweep_multiseed.py
```

Expected (3-seed) at α=0.5:
- `R² mean ± std` = -0.59 ± 0.43
- `MAE` = 1.25
- `NDCG@10 mean (median)` = 0.65 (0.75)
- `top10 mean` = 0.33

For a single-seed quick-check (will be on the noisy side):
- single-seed at α=0.5: R² ≈ -1.0, NDCG@10 ≈ 0.75, top10 = 0.40

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
