"""Train the phys_rank algorithm — physics-anchored multi-task ranking.

Wires together:
  - OPVPairDataset(include_aux=True) so [Voc, Jsc, FF] labels are exposed
  - P3Physics predictor (Scharber-anchored Voc, multi-output head)
  - CompositeLoss (PCE MSE + multi-task aux + ListMLE + physics consistency)
  - MultiTaskPCETrainer (per-fold CV with the same train/val/test split API)

Run:
    python scripts/06_train_phys_rank.py --config configs/phys_rank.yaml \\
        split.kind=high_pce_holdout split.kwargs.test_quantile=0.85
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import OPVPairDataset  # noqa: E402
from src.training import CompositeLossWeights, MultiTaskPCETrainer  # noqa: E402
from src.training.pce_trainer import PCETrainerConfig  # noqa: E402
from src.utils import load_config, save_config, set_seed  # noqa: E402


def main() -> None:
    cfg = load_config(default="configs/phys_rank.yaml")
    set_seed(int(cfg.seed))

    out_dir = Path(cfg.paths.out_dir) / cfg.trainer.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out_dir / "resolved_config.yaml")

    dataset = OPVPairDataset(
        cfg.data.opv_csv,
        max_smiles_len=int(cfg.data.max_smiles_len),
        include_aux=True,
    )
    print(f"OPV2D (clean, +aux): {len(dataset)} pairs from {cfg.data.opv_csv}")

    trainer_cfg = PCETrainerConfig(
        batch_size=int(cfg.trainer.batch_size),
        epochs=int(cfg.trainer.epochs),
        warmup_epochs=int(cfg.trainer.warmup_epochs),
        lr=float(cfg.trainer.lr),
        unfreeze_lr_scale=float(cfg.trainer.unfreeze_lr_scale),
        weight_decay=float(cfg.trainer.weight_decay),
        grad_clip=float(cfg.trainer.grad_clip),
        early_stop_patience=int(cfg.trainer.early_stop_patience),
        min_delta=float(cfg.trainer.min_delta),
        lr_patience=int(cfg.trainer.lr_patience),
        lr_factor=float(cfg.trainer.lr_factor),
        num_workers=int(cfg.trainer.num_workers),
        device=str(cfg.trainer.device),
    )
    weights = CompositeLossWeights(
        pce=float(cfg.loss.pce), aux=float(cfg.loss.aux),
        rank=float(cfg.loss.rank), phys=float(cfg.loss.phys),
    )

    trainer = MultiTaskPCETrainer(
        cfg=trainer_cfg,
        predictor_kind=str(cfg.predictor.kind),
        predictor_kwargs=dict(cfg.predictor.kwargs),
        loss_weights=weights,
        pretrained_encoder_ckpt=cfg.trainer.get("pretrained_encoder_ckpt"),
        out_dir=out_dir,
    )
    results = trainer.cross_validate(
        dataset=dataset,
        split_kind=str(cfg.split.kind),
        split_kwargs=dict(cfg.split.kwargs),
    )

    summary = {
        "split": str(cfg.split.kind), "n_folds": int(len(results)),
        "loss_weights": weights.to_dict(),
        "mean_r2": float(results["test_r2"].mean()),
        "mean_mae": float(results["test_mae"].mean()),
        "mean_rmse": float(results["test_rmse"].mean()),
        "mean_spearman": float(results["test_spearman"].mean()),
        "mean_kendall": float(results["test_kendall"].mean()),
        "mean_top10": float(results["test_top10"].mean()),
        "mean_top20": float(results["test_top20"].mean()),
        "mean_ndcg10": float(results["test_ndcg10"].mean()),
        "fold_r2": results["test_r2"].tolist(),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n" + json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
