"""3-stage MOE2 pretraining driver.

  Stage 1: MLM on Lopez 51k (atom-type masked prediction).
  Stage 2: HOMO/LUMO regression on Lopez 51k (DFT-calibrated labels).
  Stage 3: HOMO/LUMO regression on OPV2D donor+acceptor union, with layerwise LR.

Each stage can be skipped via the config (``pretrain.stageN.enabled: false``)
or the CLI:
    python scripts/pretrain_moe2.py --config configs/baseline.yaml \\
        pretrain.stage1_mlm.enabled=false
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training import pretrain_homolumo, pretrain_mlm  # noqa: E402
from src.utils import load_config, save_config, set_seed  # noqa: E402


def main() -> None:
    cfg = load_config(default="configs/baseline.yaml")
    set_seed(int(cfg.seed))

    encoder_kwargs = dict(cfg.encoder.kwargs)
    pre = cfg.pretrain
    out_dir = Path(cfg.paths.out_dir) / "pretrain"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out_dir / "resolved_config.yaml")

    # ---- Stage 1: MLM ----
    if pre.stage1_mlm.enabled:
        print("\n" + "=" * 70 + "\n[Stage 1] MLM pretraining\n" + "=" * 70)
        pretrain_mlm(
            csv_path=cfg.data.lopez_csv,
            ckpt_path=pre.stage1_mlm.ckpt,
            encoder_kwargs=encoder_kwargs,
            epochs=int(pre.stage1_mlm.epochs),
            batch_size=int(pre.stage1_mlm.batch_size),
            lr=float(pre.stage1_mlm.lr),
            val_ratio=float(pre.stage1_mlm.val_ratio),
            seed=int(pre.stage1_mlm.seed),
            num_workers=int(pre.num_workers),
            amp=bool(pre.amp),
            device=str(cfg.device),
        )

    # ---- Stage 2: HOMO/LUMO on Lopez 51k ----
    if pre.stage2_calc.enabled:
        print("\n" + "=" * 70 + "\n[Stage 2] HOMO/LUMO on Lopez 51k\n" + "=" * 70)
        layerwise = (dict(pre.stage2_calc.layerwise_lr)
                     if pre.stage2_calc.get("layerwise_lr") is not None else None)
        pretrain_homolumo(
            csv_path=cfg.data.lopez_csv,
            ckpt_path=pre.stage2_calc.ckpt,
            encoder_kwargs=encoder_kwargs,
            epochs=int(pre.stage2_calc.epochs),
            batch_size=int(pre.stage2_calc.batch_size),
            lr=float(pre.stage2_calc.lr),
            val_ratio=float(pre.stage2_calc.val_ratio),
            seed=int(pre.stage2_calc.seed),
            num_workers=int(pre.num_workers),
            amp=bool(pre.amp),
            device=str(cfg.device),
            init_from=pre.stage2_calc.get("init_from"),
            layerwise_lr=layerwise,
        )

    # ---- Stage 3: HOMO/LUMO on OPV2D union ----
    if pre.stage3_exp.enabled:
        print("\n" + "=" * 70 + "\n[Stage 3] HOMO/LUMO on OPV2D union\n" + "=" * 70)
        layerwise = (dict(pre.stage3_exp.layerwise_lr)
                     if pre.stage3_exp.get("layerwise_lr") is not None else None)
        pretrain_homolumo(
            csv_path=cfg.data.opv_full_csv,
            ckpt_path=pre.stage3_exp.ckpt,
            encoder_kwargs=encoder_kwargs,
            epochs=int(pre.stage3_exp.epochs),
            batch_size=int(pre.stage3_exp.batch_size),
            lr=float(pre.stage3_exp.lr),
            val_ratio=float(pre.stage3_exp.val_ratio),
            seed=int(pre.stage3_exp.seed),
            num_workers=int(pre.num_workers),
            amp=bool(pre.amp),
            device=str(cfg.device),
            init_from=pre.stage3_exp.get("init_from"),
            layerwise_lr=layerwise,
        )


if __name__ == "__main__":
    main()
