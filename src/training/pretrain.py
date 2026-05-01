"""3-stage MOE2 pretraining (mirrors paper).

Stages:
  1. MLM masked-atom-type pretraining on Lopez 51k (computed-property dataset).
  2. HOMO/LUMO regression on Lopez 51k (DFT-calibrated labels).
  3. HOMO/LUMO regression on OPV2D donor+acceptor union (literature labels).

Stage 3 uses layer-wise LR per the published recipe: conv1 frozen,
conv2/3 with smaller LR, head with full LR.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from ..data import HomoLumoDataset, MLMDataset
from ..models.encoders.moe2 import MOE2
from .metrics import regression_metrics


# ---------------------------------------------------------------------------
# Common helpers
# ---------------------------------------------------------------------------


def _make_loaders(dataset, batch_size: int, val_ratio: float, seed: int,
                  num_workers: int) -> tuple[DataLoader, DataLoader]:
    n_val = int(len(dataset) * val_ratio)
    n_train = len(dataset) - n_val
    tr, va = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )
    persistent = num_workers > 0
    return (
        DataLoader(tr, batch_size, shuffle=True, num_workers=num_workers,
                   persistent_workers=persistent),
        DataLoader(va, batch_size, shuffle=False, num_workers=num_workers,
                   persistent_workers=persistent),
    )


def _save_summary(out_dir: Path, name: str, history: list[dict], best: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{name}_history.json", "w") as f:
        json.dump(history, f, indent=2)
    with open(out_dir / f"{name}_best.json", "w") as f:
        json.dump(best, f, indent=2)


# ---------------------------------------------------------------------------
# Stage 1: MLM
# ---------------------------------------------------------------------------


def pretrain_mlm(
    csv_path: str | Path,
    ckpt_path: str | Path,
    *,
    encoder_kwargs: dict,
    epochs: int = 100,
    batch_size: int = 128,
    lr: float = 5e-5,
    val_ratio: float = 0.15,
    seed: int = 42,
    num_workers: int = 4,
    amp: bool = True,
    device: str | torch.device = "cuda",
) -> dict:
    """Stage 1: masked-atom-type pretraining."""
    device = torch.device(device)
    ds = MLMDataset(csv_path)
    tr_loader, va_loader = _make_loaders(ds, batch_size, val_ratio, seed, num_workers)

    model = MOE2(**encoder_kwargs).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = GradScaler(device="cuda", enabled=amp and device.type == "cuda")

    best = {"val_loss": float("inf")}
    history: list[dict] = []
    ckpt_path = Path(ckpt_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    for ep in range(1, epochs + 1):
        t0 = time.time()
        # train
        model.train()
        tr_loss = 0.0; tr_correct = 0; tr_total = 0
        for data in tqdm(tr_loader, desc=f"S1 train ep{ep}", leave=False):
            data = data.to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", enabled=amp and device.type == "cuda"):
                pred = model(data.x, data.edge_index, data.edge_attr,
                             data.batch, "mlm")
                loss = criterion(pred.view(-1, pred.size(-1)),
                                 data.mask_labels.view(-1))
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
            mask = data.mask_labels != -100
            tr_correct += (pred.argmax(-1).view(-1)[mask] == data.mask_labels.view(-1)[mask]).sum().item()
            tr_total += mask.sum().item()
            tr_loss += loss.item()
        tr_loss /= max(len(tr_loader), 1)
        tr_acc = tr_correct / max(tr_total, 1)

        # eval
        model.eval()
        va_loss = 0.0; va_correct = 0; va_total = 0
        with torch.no_grad():
            for data in va_loader:
                data = data.to(device)
                pred = model(data.x, data.edge_index, data.edge_attr,
                             data.batch, "mlm")
                loss = criterion(pred.view(-1, pred.size(-1)),
                                 data.mask_labels.view(-1))
                mask = data.mask_labels != -100
                va_correct += (pred.argmax(-1).view(-1)[mask] == data.mask_labels.view(-1)[mask]).sum().item()
                va_total += mask.sum().item()
                va_loss += loss.item()
        va_loss /= max(len(va_loader), 1)
        va_acc = va_correct / max(va_total, 1)

        row = {"epoch": ep, "train_loss": tr_loss, "train_acc": tr_acc,
               "val_loss": va_loss, "val_acc": va_acc, "time": time.time() - t0}
        history.append(row)
        print(f"[S1 ep {ep:3d}] tr_loss={tr_loss:.4f} tr_acc={tr_acc:.4f} | "
              f"va_loss={va_loss:.4f} va_acc={va_acc:.4f} | {row['time']:.1f}s")
        if va_loss < best["val_loss"]:
            best = row
            torch.save(model.state_dict(), ckpt_path)

    _save_summary(ckpt_path.parent, ckpt_path.stem, history, best)
    return best


# ---------------------------------------------------------------------------
# Stage 2 / 3: HOMO/LUMO regression
# ---------------------------------------------------------------------------


def _eval_homolumo(model: MOE2, loader: DataLoader, criterion: nn.Module,
                   device: torch.device) -> dict:
    model.eval()
    total_loss = 0.0
    homo_p, homo_t, lumo_p, lumo_t = [], [], [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            pred = model(data.x, data.edge_index, data.edge_attr, data.batch, "homo_lumo")
            loss = criterion(pred.view(-1), data.y)
            total_loss += loss.item()
            p = pred.detach().cpu().numpy().reshape(-1, 2)
            t = data.y.detach().cpu().numpy().reshape(-1, 2)
            homo_p.append(p[:, 0]); homo_t.append(t[:, 0])
            lumo_p.append(p[:, 1]); lumo_t.append(t[:, 1])
    h = regression_metrics(np.concatenate(homo_p), np.concatenate(homo_t))
    l = regression_metrics(np.concatenate(lumo_p), np.concatenate(lumo_t))
    return {
        "loss": total_loss / max(len(loader), 1),
        "HOMO_mae": h.mae, "HOMO_rmse": h.rmse, "HOMO_r2": h.r2,
        "LUMO_mae": l.mae, "LUMO_rmse": l.rmse, "LUMO_r2": l.r2,
        "mean_MAE": (h.mae + l.mae) / 2.0,
    }


def pretrain_homolumo(
    csv_path: str | Path,
    ckpt_path: str | Path,
    *,
    encoder_kwargs: dict,
    epochs: int = 150,
    batch_size: int = 128,
    seed: int = 43,
    val_ratio: float = 0.15,
    num_workers: int = 4,
    amp: bool = True,
    device: str | torch.device = "cuda",
    init_from: str | Path | None = None,
    layerwise_lr: dict | None = None,
    lr: float = 5e-5,
) -> dict:
    """Stage 2/3: HOMO/LUMO regression.

    Args:
        layerwise_lr: if provided, dict of ``{"conv1_lr": float, "conv2_lr": float,
            "conv3_lr": float, "head_lr": float}``. ``conv1_lr=0`` freezes conv1.
            When ``None`` the entire model is trained at ``lr``.
    """
    device = torch.device(device)
    ds = HomoLumoDataset(csv_path)
    tr_loader, va_loader = _make_loaders(ds, batch_size, val_ratio, seed, num_workers)

    model = MOE2(**encoder_kwargs).to(device)
    if init_from is not None and Path(init_from).exists():
        model.load_state_dict(torch.load(init_from, map_location=device))
        print(f"[Stage] loaded init weights from {init_from}")

    criterion = nn.MSELoss()
    if layerwise_lr is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    else:
        if layerwise_lr.get("conv1_lr", 0.0) == 0.0:
            for p in model.conv1.parameters():
                p.requires_grad = False
        param_groups = []
        if layerwise_lr.get("conv1_lr", 0.0) > 0:
            param_groups.append({"params": list(model.conv1.parameters()),
                                 "lr": layerwise_lr["conv1_lr"], "name": "conv1"})
        param_groups += [
            {"params": list(model.conv2.parameters()),
             "lr": layerwise_lr.get("conv2_lr", lr), "name": "conv2"},
            {"params": list(model.conv3.parameters()),
             "lr": layerwise_lr.get("conv3_lr", lr), "name": "conv3"},
            {"params": list(model.regression_head.parameters()) + list(model.pool.parameters()),
             "lr": layerwise_lr.get("head_lr", lr), "name": "head+pool"},
        ]
        optimizer = torch.optim.AdamW(param_groups)

    scaler = GradScaler(device="cuda", enabled=amp and device.type == "cuda")
    best = {"loss": float("inf")}
    history: list[dict] = []
    ckpt_path = Path(ckpt_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        tr_loss = 0.0
        for data in tqdm(tr_loader, desc=f"HL train ep{ep}", leave=False):
            data = data.to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", enabled=amp and device.type == "cuda"):
                pred = model(data.x, data.edge_index, data.edge_attr,
                             data.batch, "homo_lumo")
                loss = criterion(pred.view(-1), data.y)
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
            tr_loss += loss.item()
        tr_loss /= max(len(tr_loader), 1)

        va = _eval_homolumo(model, va_loader, criterion, device)
        row = {"epoch": ep, "train_loss": tr_loss, "time": time.time() - t0,
               **{f"val_{k}": v for k, v in va.items()}}
        history.append(row)
        print(f"[HL ep {ep:3d}] tr={tr_loss:.4f} | va_loss={va['loss']:.4f} "
              f"H_MAE={va['HOMO_mae']:.4f} H_R²={va['HOMO_r2']:.4f} "
              f"L_MAE={va['LUMO_mae']:.4f} L_R²={va['LUMO_r2']:.4f} "
              f"| {row['time']:.1f}s")
        if va["loss"] < best.get("val_loss", float("inf")):
            best = row
            torch.save(model.state_dict(), ckpt_path)

    _save_summary(ckpt_path.parent, ckpt_path.stem, history, best)
    return best
