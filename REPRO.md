# Reproduction Report: P³ on OPV²D

End-to-end reproduction of the CycleChemist (arXiv:2511.19500v2) PCE
prediction pipeline using this baseline framework. The pretrained MOE²
encoder checkpoint comes from the original paper repo
(`cyclechemist/property_predictors/moe2_p3/ckpt/moe2/homolumo_*model.pth`),
so the only training step is the 5-fold P³ PCE head + finetuning.

## Setup

- Hardware: NVIDIA RTX 3090 (24 GB), CUDA 12.1
- Dataset: `data/processed/opv2d_clean.csv` (1525 D-A pairs after Y6 holdout)
- Pretrained encoder: byte-compatible with our `MOE2` (33/33 keys, 0 shape mismatches)
- All hyper-parameters from `configs/baseline.yaml` (paper-default values)
  - batch=32, epochs=100 (warmup 20 + finetune 80), lr=1e-4, finetune lr=1e-5
  - AdamW weight_decay=5e-4, ReduceLROnPlateau patience=3 factor=0.5
  - early_stop_patience=30, grad_clip=1.0
  - target standardization on train fold only

## Results

### Random 5-fold (paper protocol)

`split.kind=random_kfold split.kwargs.n_splits=5 split.kwargs.seed=3407`
Encoder ckpt: stage 3 (`homolumo_exp_model.pth`)

| Fold | R² | MAE | RMSE |
|---|---|---|---|
| 1 | 0.7229 | 1.554 | 1.956 |
| 2 | 0.7009 | 1.651 | 2.012 |
| 3 | 0.7319 | 1.393 | 1.746 |
| 4 | 0.6326 | 1.556 | 2.006 |
| 5 | 0.7359 | 1.455 | 1.883 |
| **Mean** | **0.7048 ± 0.0426** | **1.522** | **1.921** |

Paper Table 1 ("P³ — GAT embedding"): **0.736 ± 0.033**

Δ = 0.031 R² (≈ 0.96 σ paper). Within the paper's reported error band.

### Scaffold-acceptor extrapolation (this framework's default)

`split.kind=scaffold_acceptor`
Encoder ckpt: stage 3 (`homolumo_exp_model.pth`)

| Fold | R² | MAE | RMSE | n_train | n_val | n_test |
|---|---|---|---|---|---|---|
| 1 | 0.6078 | 1.981 | 2.397 | 1446 | 56 | 23 |

**Extrapolation gap = 0.097 R² absolute** (random 0.7048 → scaffold 0.6078).

## Interpretation

1. **Reproduction validates the framework.** The paper's exact protocol gives
   0.7048 ± 0.0426 here vs the paper's 0.736 ± 0.033 — well within 1 σ.
   Residual gap is most likely seed/init noise; one fold (#4) drives most of
   the variance.

2. **Extrapolation is materially harder.** Holding out unseen acceptor
   scaffolds drops R² by 0.097 absolute and inflates MAE by 30 %. The
   "easy" 0.736 number under random K-fold reflects the high within-paper
   structural redundancy of OPV²D — many fold pairs share core scaffolds
   between train and test, so memorization helps.

3. **The default split now matches the algorithm-iteration goal.** New
   algorithms tried in this repo will be measured on the harder, more
   honest scaffold-acceptor split by default (`scaffold_acceptor`); the
   paper's number remains reproducible by flipping `split.kind=random_kfold`.

## Provenance

| File | Origin |
|---|---|
| `checkpoints/moe2_exp.pt` | copy of `cyclechemist/.../homolumo_exp_model.pth` |
| `data/raw/opv2d.csv` | copy of `cyclechemist/data/exp_dataset.csv` (1567 rows) |
| `data/processed/opv2d_clean.csv` | OPV²D after `01_prepare_data.py` (1525 rows; Y6 acceptors held out) |
| `outputs/repro_random_kfold/` | 5-fold paper-protocol run, ~1 h on RTX 3090 |
| `outputs/repro_scaffold_acceptor/` | scaffold extrapolation run, ~10 min |
| `outputs/repro_random_kfold_stage2/` | optional cross-check using stage-2 ckpt (the original paper code path) |
