# Reproduction Report: P³ on OPV²D

End-to-end from-scratch reproduction of the P³ PCE-prediction pipeline from
CycleChemist (arXiv:2511.19500v2). All weights — MOE² encoder and P³ head —
are trained inside this repo from random initialization. No external
checkpoints are loaded.

## Setup

- Hardware: NVIDIA RTX 3090 (24 GB), CUDA 12.1
- Pretraining: `scripts/02_pretrain_moe2.py` with `configs/baseline.yaml`
  - Stage 1 — MLM atom-type prediction on Lopez 51k (100 epochs)
  - Stage 2 — calc HOMO/LUMO regression on Lopez 51k, GAT layers frozen, head only (150 epochs)
  - (Stage 3 — exp HOMO/LUMO on OPV²D union — not used by the paper PCE protocol)
- PCE training: `scripts/03_train_pce.py` loading the stage-2 ckpt (paper code path)
  - Random 5-fold KFold, seed=3407, batch=32, 100 epochs (warmup 20 + finetune 80)
  - lr=1e-4, finetune lr=1e-5, AdamW weight_decay=5e-4, grad_clip=1.0
  - Target standardization with train-fold (μ, σ) only

## Results

### Random 5-fold (paper protocol)

`split.kind=random_kfold split.kwargs.n_splits=5 split.kwargs.seed=3407`

| Source | F1 | F2 | F3 | F4 | F5 | Mean ± Std |
|---|---|---|---|---|---|---|
| Paper Table 1 (P³ — GAT embedding) | — | — | — | — | — | **0.736 ± 0.033** |
| This repo (from-scratch)           | _ | _ | _ | _ | _ | _pending_ |

_(Numbers will be filled in after the from-scratch pretrain + 5-fold PCE run completes; see `outputs/repro_random_kfold/summary.json`.)_

### Extrapolation splits

The same trained P³ model evaluated under harder splits (where holding out
unseen scaffolds or the high-PCE tail tests true generalization rather than
within-paper memorization):

| Split | Mean R² | MAE |
|---|---|---|
| `scaffold_acceptor`           | _pending_ | _pending_ |
| `high_pce_holdout` (q=0.85)   | _pending_ | _pending_ |

## Provenance

| File | Origin |
|---|---|
| `data/raw/lopez51k.csv`           | Lopez NFA 51k candidate database (Joule 2017) |
| `data/raw/opv2d.csv`              | OPV²D — 1567 D-A pairs from CycleChemist `exp_dataset.csv` |
| `data/processed/opv2d_clean.csv`  | OPV²D after `01_prepare_data.py` (1525 rows; Y6 acceptors held out) |
| `data/processed/opv2d_y6.csv`     | Y6 holdout (33 pairs) reserved for stage-2 evaluation |
| `checkpoints/moe2_*.pt`           | Trained from scratch by `02_pretrain_moe2.py` |
| `outputs/repro_*/`                | All 5-fold runs by `03_train_pce.py` |
