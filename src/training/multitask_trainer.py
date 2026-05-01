"""MultiTaskPCETrainer — trainer for the phys_rank algorithm.

Differences from ``PCETrainer``:
  - Predictor returns a ``dict``; trainer composes the multi-component
    ``CompositeLoss`` (PCE MSE + multi-task aux + ListMLE + physics).
  - Dataset must be ``OPVPairDataset(include_aux=True)`` so [Voc, Jsc, FF]
    labels are available per item (with NaN-mask).
  - Standard regression / ranking metrics are reported the same way as the
    baseline trainer, so the new model is directly comparable.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Subset
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from ..data import OPVPairDataset, build_split
from ..models import build_predictor
from .losses import CompositeLoss, CompositeLossWeights
from .metrics import ranking_metrics, regression_metrics
from .pce_trainer import PCETrainerConfig


class MultiTaskPCETrainer:
    """Per-fold training loop with composite loss + multi-task heads."""

    def __init__(
        self,
        cfg: PCETrainerConfig,
        predictor_kind: str,
        predictor_kwargs: dict,
        loss_weights: CompositeLossWeights,
        pretrained_encoder_ckpt: str | Path | None = None,
        out_dir: str | Path = "outputs/phys_rank",
    ) -> None:
        self.cfg = cfg
        self.predictor_kind = predictor_kind
        self.predictor_kwargs = dict(predictor_kwargs)
        self.loss_weights = loss_weights
        self.pretrained_ckpt = Path(pretrained_encoder_ckpt) if pretrained_encoder_ckpt else None
        self.out_dir = Path(out_dir); self.out_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # ---------------------------------------------------------- helpers

    def _build_model(self) -> nn.Module:
        model = build_predictor(self.predictor_kind, **self.predictor_kwargs).to(self.device)
        if self.pretrained_ckpt and self.pretrained_ckpt.exists():
            sd = torch.load(self.pretrained_ckpt, map_location=self.device, weights_only=False)
            if hasattr(model, "load_pretrained_encoders"):
                model.load_pretrained_encoders(sd)
                print(f"[MultiTaskTrainer] loaded pretrained encoders from {self.pretrained_ckpt}")
        return model

    def _make_loader(self, dataset, indices: np.ndarray, shuffle: bool, drop_last: bool) -> DataLoader:
        return DataLoader(
            Subset(dataset, indices.tolist()),
            batch_size=self.cfg.batch_size, shuffle=shuffle, drop_last=drop_last,
            num_workers=self.cfg.num_workers, pin_memory=True,
        )

    @staticmethod
    def _to_branch_batches(batch, device) -> tuple:
        donor_batch = Batch.from_data_list(batch.donor).to(device)
        acceptor_batch = Batch.from_data_list(batch.acceptor).to(device)
        return donor_batch, acceptor_batch

    def _build_target_dict(self, batch, y_mean: torch.Tensor, y_std: torch.Tensor) -> dict:
        y = batch.y.view(-1)
        return {
            "pce": y,
            "pce_z": (y - y_mean) / y_std,
            "aux": batch.aux.view(-1, 3),                              # [B, 3]
            "aux_mask": batch.aux_mask.view(-1, 3),                    # [B, 3]
        }

    @staticmethod
    def _attach_pce_z(pred: dict, y_mean: torch.Tensor, y_std: torch.Tensor) -> dict:
        pred = dict(pred)
        pred["pce_z"] = (pred["pce"] - y_mean) / y_std
        return pred

    # ---------------------------------------------------------- fold loop

    def run_fold(
        self, dataset: OPVPairDataset, fold_id: int,
        train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray,
    ) -> dict:
        cfg = self.cfg
        model = self._build_model()
        criterion = CompositeLoss(self.loss_weights).to(self.device)

        y_train = dataset.df.iloc[train_idx]["PCE"].to_numpy()
        y_mean = torch.tensor(float(np.mean(y_train)), dtype=torch.float32, device=self.device)
        y_std = torch.tensor(float(np.std(y_train) or 1.0), dtype=torch.float32, device=self.device)

        train_loader = self._make_loader(dataset, train_idx, shuffle=True, drop_last=True)
        test_loader = self._make_loader(dataset, test_idx, shuffle=False, drop_last=False)
        val_loader = (
            self._make_loader(dataset, val_idx, shuffle=False, drop_last=False)
            if len(val_idx) > 0 else test_loader
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=cfg.lr_factor, patience=cfg.lr_patience,
        )
        best_path = self.out_dir / f"fold{fold_id+1}_best.pt"

        # ------------- phase 1: warmup with encoders frozen ----------
        if hasattr(model, "freeze_encoders") and cfg.warmup_epochs > 0:
            print(f"[Fold {fold_id+1}] Phase 1: encoders FROZEN ({cfg.warmup_epochs} ep)")
            model.freeze_encoders(True)
        self._train_phase(
            model, train_loader, val_loader, criterion, optimizer, scheduler,
            cfg.warmup_epochs, fold_id, best_path, y_mean, y_std, phase="warmup",
        )

        # ------------- phase 2: unfreeze + finetune ------------------
        if hasattr(model, "freeze_encoders"):
            print(f"[Fold {fold_id+1}] Phase 2: encoders UNFROZEN")
            model.freeze_encoders(False)
        for g in optimizer.param_groups:
            g["lr"] = cfg.lr * cfg.unfreeze_lr_scale
        remaining = max(cfg.epochs - cfg.warmup_epochs, 0)
        self._train_phase(
            model, train_loader, val_loader, criterion, optimizer, scheduler,
            remaining, fold_id, best_path, y_mean, y_std, phase="finetune",
        )

        if best_path.exists():
            ckpt = torch.load(best_path, map_location=self.device, weights_only=False)
            model.load_state_dict(ckpt["state_dict"])
        preds, actuals, ids = self._predict(model, test_loader)
        m = regression_metrics(preds, actuals)
        r = ranking_metrics(preds, actuals)
        result = {
            "fold": fold_id + 1,
            "n_train": int(len(train_idx)),
            "n_val": int(len(val_idx)),
            "n_test": int(len(test_idx)),
            "test_mae": m.mae, "test_rmse": m.rmse, "test_r2": m.r2,
            "test_spearman": r.spearman, "test_kendall": r.kendall,
            "test_top10": r.top10_precision, "test_top20": r.top20_precision,
            "test_ndcg10": r.ndcg_at_10,
        }
        pd.DataFrame({"Mol_ID": ids, "Actual": actuals, "Predicted": preds}).to_csv(
            self.out_dir / f"fold{fold_id+1}_predictions.csv", index=False)
        print(f"[Fold {fold_id+1}] R²={m.r2:.4f} MAE={m.mae:.4f} | "
              f"Spearman={r.spearman:.4f} top10={r.top10_precision:.2f} "
              f"NDCG@10={r.ndcg_at_10:.4f}")
        return result

    # ---------------------------------------------------------- phases

    def _train_phase(
        self, model, train_loader, val_loader, criterion, optimizer, scheduler,
        epochs: int, fold_id: int, best_path: Path,
        y_mean: torch.Tensor, y_std: torch.Tensor, phase: str,
    ) -> None:
        best_val = float("inf"); patience = 0
        for ep in range(1, epochs + 1):
            model.train()
            running: dict[str, float] = {}
            for batch in tqdm(train_loader, desc=f"F{fold_id+1} {phase} ep{ep}", leave=False):
                batch = batch.to(self.device)
                d_b, a_b = self._to_branch_batches(batch, self.device)
                tgt = self._build_target_dict(batch, y_mean, y_std)
                optimizer.zero_grad(set_to_none=True)
                pred = model(d_b, a_b)
                pred = self._attach_pce_z(pred, y_mean, y_std)
                loss, log = criterion(pred, tgt)
                loss.backward()
                if self.cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.cfg.grad_clip)
                optimizer.step()
                for k, v in log.items():
                    running[k] = running.get(k, 0.0) + v
            n = max(len(train_loader), 1)
            tr = {k: v / n for k, v in running.items()}

            val_loss = self._eval_loss(model, val_loader, criterion, y_mean, y_std)
            scheduler.step(val_loss)
            print(f"  ep{ep:3d} tr_total={tr['total']:.4f} (pce={tr['pce_mse']:.3f} "
                  f"aux={tr['aux_mse']:.3f} rank={tr['rank']:.3f} phys={tr['phys']:.3f}) | "
                  f"val={val_loss:.4f} lr={optimizer.param_groups[0]['lr']:.2e}")

            if val_loss + self.cfg.min_delta < best_val:
                best_val = val_loss; patience = 0
                torch.save({
                    "state_dict": model.state_dict(),
                    "y_mean": float(y_mean.item()), "y_std": float(y_std.item()),
                    "epoch": ep, "val_loss": val_loss, "phase": phase,
                }, best_path)
            else:
                patience += 1
                if patience >= self.cfg.early_stop_patience:
                    print(f"  early stop at ep{ep}"); return

    def _eval_loss(self, model, loader, criterion, y_mean, y_std) -> float:
        model.eval(); total = 0.0
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                d_b, a_b = self._to_branch_batches(batch, self.device)
                tgt = self._build_target_dict(batch, y_mean, y_std)
                pred = model(d_b, a_b)
                pred = self._attach_pce_z(pred, y_mean, y_std)
                loss, _ = criterion(pred, tgt)
                total += float(loss.item())
        return total / max(len(loader), 1)

    def _predict(self, model, loader) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        model.eval()
        preds, actuals, ids = [], [], []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                d_b, a_b = self._to_branch_batches(batch, self.device)
                pred = model(d_b, a_b)
                preds.append(pred["pce"].cpu().numpy())
                actuals.append(batch.y.view(-1).cpu().numpy())
                ids.append(batch.mol_id.view(-1).cpu().numpy())
        return np.concatenate(preds), np.concatenate(actuals), np.concatenate(ids)

    # ---------------------------------------------------------- public

    def cross_validate(
        self, dataset: OPVPairDataset, split_kind: str, split_kwargs: dict,
    ) -> pd.DataFrame:
        folds = build_split(split_kind, dataset.df, **split_kwargs)
        rows = []
        for i, (tr, va, te) in enumerate(folds):
            print(f"\n=== Fold {i+1}/{len(folds)} (split={split_kind}) "
                  f"train={len(tr)} val={len(va)} test={len(te)} ===")
            rows.append(self.run_fold(dataset, i, tr, va, te))
        results = pd.DataFrame(rows)
        results.to_csv(self.out_dir / "fold_summary.csv", index=False)
        if len(results) > 1:
            print(f"\n>>> Mean R²={results['test_r2'].mean():.4f} ± {results['test_r2'].std():.4f}"
                  f" | Spearman={results['test_spearman'].mean():.4f}"
                  f" | top10={results['test_top10'].mean():.2f}"
                  f" | NDCG@10={results['test_ndcg10'].mean():.4f}")
        return results
