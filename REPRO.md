# Reproduction Report: P³ on OPV²D

End-to-end **from-scratch** reproduction of the P³ PCE-prediction pipeline
from CycleChemist (arXiv:2511.19500v2). All weights — MOE² encoder and P³
head — are trained inside this repo from random initialization. **No
external checkpoints are used.**

## Setup

- Hardware: NVIDIA RTX 3090 (24 GB), CUDA 12.1
- Pretraining: `scripts/02_pretrain_moe2.py` (`configs/baseline.yaml`)
  - **Stage 1** — MLM atom-type prediction on Lopez 51k, 100 epochs
    (best val_acc = 0.9719 @ ep 92, ~42 min)
  - **Stage 2** — calc HOMO/LUMO regression on Lopez 51k, GAT layers frozen,
    head-only, 150 epochs
    (best val: HOMO R² = 0.849, LUMO R² = 0.770 @ ep 141, ~62 min)
  - Stage 3 (exp HOMO/LUMO on OPV²D union) is *not* used by the paper PCE
    protocol — skipped.
- PCE training: `scripts/03_train_pce.py` loading the stage-2 ckpt
  (this is the path the paper's released `train_pce.py` actually executes)
  - Random 5-fold KFold, seed = 3407, batch = 32
  - 100 epochs (warmup 20 with encoders frozen + finetune 80 unfrozen at lr×0.1)
  - lr = 1e-4, AdamW weight_decay = 5e-4, grad_clip = 1.0
  - Target standardization with **train-fold (μ, σ) only**

## Results

### Random 5-fold (paper protocol)

`split.kind=random_kfold split.kwargs.n_splits=5 split.kwargs.seed=3407`

| Source | F1 | F2 | F3 | F4 | F5 | **Mean ± Std** | MAE |
|---|---|---|---|---|---|---|---|
| Paper Table 1 (P³ — GAT embedding) | — | — | — | — | — | **0.736 ± 0.033** | — |
| This repo (from-scratch)           | 0.6293 | 0.6736 | 0.6011 | 0.6608 | 0.7387 | **0.6607 ± 0.0519** | 1.66 |

Δ = -0.075 R² absolute. Most plausible cause is the omitted Stage 3
experimental HOMO/LUMO finetune on OPV²D plus a slightly weaker Stage 1
(val_acc 0.972 here vs paper 0.99997).

### Extrapolation splits (same trained P³)

| Split | R² | MAE | RMSE | Notes |
|---|---|---|---|---|
| `scaffold_acceptor` | **0.6855** | 1.75 | 2.15 | unseen acceptor Bemis-Murcko scaffolds (n_test=23) |
| `high_pce_holdout` (q=0.85) | **-8.97** | 3.71 | 3.92 | top 15 % PCE held out (n_test=229) |

The high-PCE holdout R² is far below 0 — i.e., **worse than predicting the
train-set mean.** Diagnostic on this run: actual test PCEs span 11.9-17.8
while predictions cap at ~10. P³ cannot emit values above its training
range, a textbook regression-to-the-mean failure on a held-out
distribution tail.

## Provenance

| File | Origin |
|---|---|
| `data/raw/lopez51k.csv`           | Lopez NFA 51k candidate database (Joule 2017) |
| `data/raw/opv2d.csv`              | OPV²D — 1567 D-A pairs from CycleChemist `exp_dataset.csv` |
| `data/processed/opv2d_clean.csv`  | OPV²D after `01_prepare_data.py` (1525 rows; Y6 acceptors held out) |
| `data/processed/opv2d_y6.csv`     | Y6 holdout (33 pairs) reserved for stage-2 evaluation |
| `checkpoints/moe2_mlm.pt`         | trained from scratch by `02_pretrain_moe2.py` (Stage 1) |
| `checkpoints/moe2_calc.pt`        | trained from scratch by `02_pretrain_moe2.py` (Stage 2) |
| `outputs/repro_*/`                | all 5-fold runs by `03_train_pce.py` |
