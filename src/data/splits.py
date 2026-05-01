"""Split strategies for PCE evaluation.

Each strategy returns a list of folds; each fold is ``(train_idx, val_idx, test_idx)``.
Random K-fold returns K folds with empty val (legacy paper baseline).
Extrapolation strategies typically return a single fold.

Add a new strategy:

    @register_split("my_split")
    def my_split(df: pd.DataFrame, **cfg) -> list[Fold]:
        ...
        return [(train_idx, val_idx, test_idx)]

then reference it from a yaml config: ``split.kind: my_split``.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import KFold

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

Fold = tuple[np.ndarray, np.ndarray, np.ndarray]  # (train, val, test) row-index arrays
SplitFn = Callable[..., list[Fold]]
SPLIT_REGISTRY: dict[str, SplitFn] = {}


def register_split(name: str) -> Callable[[SplitFn], SplitFn]:
    def deco(fn: SplitFn) -> SplitFn:
        if name in SPLIT_REGISTRY:
            raise ValueError(f"Split `{name}` already registered.")
        SPLIT_REGISTRY[name] = fn
        return fn
    return deco


def build_split(kind: str, df: pd.DataFrame, **kwargs) -> list[Fold]:
    if kind not in SPLIT_REGISTRY:
        raise KeyError(f"Unknown split `{kind}`. Available: {sorted(SPLIT_REGISTRY)}")
    return SPLIT_REGISTRY[kind](df, **kwargs)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def bemis_murcko(smiles: str, include_chirality: bool = False) -> str:
    """Return canonical Bemis–Murcko scaffold SMILES, or '' on failure."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ""
        scaff = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaff, isomericSmiles=include_chirality)
    except Exception:
        return ""


@dataclass(frozen=True)
class _SplitFractions:
    train: float
    val: float
    test: float

    def check(self) -> None:
        if not np.isclose(self.train + self.val + self.test, 1.0):
            raise ValueError(f"Fractions must sum to 1.0, got {self.train + self.val + self.test}")


def _scaffold_group_assign(
    groups: dict[str, list[int]], n: int, fr: _SplitFractions, seed: int
) -> Fold:
    """Assign whole scaffold groups largest-first to train, then val, then test."""
    rng = np.random.default_rng(seed)
    n_val = int(round(fr.val * n))
    n_test = int(round(fr.test * n))
    n_train = n - n_val - n_test
    train_idx, val_idx, test_idx = [], [], []
    sorted_groups = sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    for _, members in sorted_groups:
        if len(members) > 1:
            target = (
                "train" if len(train_idx) + len(members) <= n_train
                else "val" if len(val_idx) + len(members) <= n_val
                else "test"
            )
        else:
            r = rng.random()
            target = ("train" if r < fr.train
                      else "val" if r < fr.train + fr.val
                      else "test")
        {"train": train_idx, "val": val_idx, "test": test_idx}[target].extend(members)
    return (
        np.array(sorted(train_idx), dtype=np.int64),
        np.array(sorted(val_idx), dtype=np.int64),
        np.array(sorted(test_idx), dtype=np.int64),
    )


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


@register_split("random_kfold")
def random_kfold(df: pd.DataFrame, *, n_splits: int = 5, seed: int = 3407) -> list[Fold]:
    """Paper baseline: 5-fold random K-fold. ``val`` is empty per fold (test = val)."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds: list[Fold] = []
    for tr, te in kf.split(df):
        folds.append((np.asarray(tr, dtype=np.int64),
                      np.empty(0, dtype=np.int64),
                      np.asarray(te, dtype=np.int64)))
    return folds


@register_split("scaffold_acceptor")
def scaffold_acceptor(
    df: pd.DataFrame, *, train_frac: float = 0.8, val_frac: float = 0.1,
    test_frac: float = 0.1, seed: int = 42,
) -> list[Fold]:
    """Extrapolation by acceptor Bemis–Murcko scaffold. Largest scaffold groups → train."""
    fr = _SplitFractions(train_frac, val_frac, test_frac); fr.check()
    groups: dict[str, list[int]] = defaultdict(list)
    for i, smi in enumerate(df["Acceptor SMILES"].tolist()):
        s = bemis_murcko(smi)
        groups[s if s else f"__INVALID__{i}"].append(i)
    return [_scaffold_group_assign(groups, len(df), fr, seed)]


@register_split("scaffold_pair")
def scaffold_pair(
    df: pd.DataFrame, *, train_frac: float = 0.8, val_frac: float = 0.1,
    test_frac: float = 0.1, seed: int = 42,
) -> list[Fold]:
    """Group by (donor_scaffold, acceptor_scaffold) tuple — strictest scaffold split."""
    fr = _SplitFractions(train_frac, val_frac, test_frac); fr.check()
    groups: dict[str, list[int]] = defaultdict(list)
    for i, (d, a) in enumerate(zip(df["Donor SMILES"], df["Acceptor SMILES"])):
        ds, as_ = bemis_murcko(d), bemis_murcko(a)
        key = f"{ds}||{as_}" if (ds and as_) else f"__INVALID__{i}"
        groups[key].append(i)
    return [_scaffold_group_assign(groups, len(df), fr, seed)]


@register_split("acceptor_disjoint")
def acceptor_disjoint(
    df: pd.DataFrame, *, train_frac: float = 0.8, val_frac: float = 0.1,
    test_frac: float = 0.1, seed: int = 42,
) -> list[Fold]:
    """Test set acceptors never appear in train (canonical SMILES match)."""
    fr = _SplitFractions(train_frac, val_frac, test_frac); fr.check()
    groups: dict[str, list[int]] = defaultdict(list)
    for i, smi in enumerate(df["Acceptor SMILES"].tolist()):
        try:
            mol = Chem.MolFromSmiles(smi)
            key = Chem.MolToSmiles(mol) if mol is not None else f"__INVALID__{i}"
        except Exception:
            key = f"__INVALID__{i}"
        groups[key].append(i)
    return [_scaffold_group_assign(groups, len(df), fr, seed)]


@register_split("high_pce_holdout")
def high_pce_holdout(
    df: pd.DataFrame, *, test_quantile: float = 0.85, val_frac: float = 0.1, seed: int = 42,
) -> list[Fold]:
    """Hold out top-PCE quantile as test (extrapolation to higher efficiency).

    Train comes from the remaining low/mid-PCE pairs; val is a random slice of
    train so the model can early-stop without leaking high-PCE info.
    """
    rng = np.random.default_rng(seed)
    pce = df["PCE"].to_numpy()
    cutoff = np.quantile(pce, test_quantile)
    test_mask = pce >= cutoff
    test_idx = np.where(test_mask)[0]
    rest_idx = np.where(~test_mask)[0]
    rng.shuffle(rest_idx)
    n_val = int(round(val_frac * len(df)))
    val_idx = rest_idx[:n_val]
    train_idx = rest_idx[n_val:]
    return [(np.sort(train_idx), np.sort(val_idx), np.sort(test_idx))]


@register_split("leave_one_doi_out")
def leave_one_doi_out(
    df: pd.DataFrame, *, min_doi_size: int = 5, val_frac: float = 0.1, seed: int = 42,
) -> list[Fold]:
    """One fold per DOI (paper) with ≥``min_doi_size`` rows; that DOI's pairs become test.

    Models the realistic deployment of "predict efficiency for a paper we
    haven't seen". Useful for diagnosing dataset-coupling; computationally
    heavier than scaffold split.
    """
    if "DOI" not in df.columns:
        raise ValueError("leave_one_doi_out requires a DOI column.")
    rng = np.random.default_rng(seed)
    folds: list[Fold] = []
    doi_groups = df.groupby("DOI").indices
    for doi, idx in doi_groups.items():
        if not isinstance(doi, str) or len(idx) < min_doi_size:
            continue
        test_idx = np.asarray(idx, dtype=np.int64)
        rest_idx = np.setdiff1d(np.arange(len(df), dtype=np.int64), test_idx, assume_unique=False)
        rng.shuffle(rest_idx)
        n_val = int(round(val_frac * len(rest_idx)))
        val_idx = np.sort(rest_idx[:n_val])
        train_idx = np.sort(rest_idx[n_val:])
        folds.append((train_idx, val_idx, test_idx))
    if not folds:
        raise ValueError(f"No DOI group with ≥{min_doi_size} rows found.")
    return folds
