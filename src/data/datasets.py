"""PyG datasets used by the three training tasks.

Three datasets, each thin and single-purpose:

- ``MLMDataset``       — masked-atom pretraining on Lopez 51k SMILES.
- ``HomoLumoDataset``  — HOMO/LUMO regression on Lopez 51k or OPV2D donor/acceptor union.
- ``OPVPairDataset``   — donor+acceptor pairs with PCE label, for P3 training.

All three accept a CSV path so they can be retargeted by a config flip.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from rdkit import Chem
from torch_geometric.data import Data, Dataset

from .featurizer import mol_to_graph

# ----------------------- MLM (atom-type prediction) -----------------------


class MLMDataset(Dataset):
    """Random-atom-mask pretraining. ``mask_labels`` field carries the target.

    Args:
        csv_path: must contain a ``smiles`` column.
        mask_ratio: fraction of atoms to mask per molecule.
        smiles_col: column name (default ``smiles``).
    """

    ATOM_VOCAB: tuple[str, ...] = ("C", "N", "O", "S", "F", "P", "Cl", "Br", "I")

    def __init__(self, csv_path: str | Path, mask_ratio: float = 0.15,
                 smiles_col: str = "smiles") -> None:
        super().__init__()
        self.df = pd.read_csv(csv_path)
        if smiles_col not in self.df.columns:
            raise ValueError(f"`{smiles_col}` column not found in {csv_path}")
        self.smiles_col = smiles_col
        self.mask_ratio = mask_ratio

    def len(self) -> int:  # PyG API
        return len(self.df)

    def get(self, idx: int) -> Data:  # PyG API
        smiles = self.df.iloc[idx][self.smiles_col]
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES at row {idx}: {smiles}")
        graph = mol_to_graph(mol)
        return self._apply_mask(graph)

    def _apply_mask(self, data: Data) -> Data:
        n = data.x.size(0)
        n_mask = max(1, int(n * self.mask_ratio))
        mask_idx = torch.randperm(n)[:n_mask]
        labels = torch.full((n,), -100, dtype=torch.long)
        # First |ATOM_VOCAB| dims are the one-hot atom type
        labels[mask_idx] = data.x[mask_idx, : len(self.ATOM_VOCAB)].argmax(dim=1)
        x = data.x.clone()
        x[mask_idx] = 0.0  # zero the entire feature vector to prevent leakage
        data.x = x
        data.mask_labels = labels
        return data


# ----------------------- HOMO / LUMO regression -----------------------


class HomoLumoDataset(Dataset):
    """HOMO/LUMO regression on either Lopez 51k or OPV2D (donor+acceptor union).

    Auto-detects format:
      - simple: columns include ``smiles`` + (``HOMO_calib``/``LUMO_calib``)
                                            or (``HOMO``/``LUMO``).
      - opv:    columns include ``Donor SMILES``/``Acceptor SMILES`` and the
                paired ``HOMO_D``/``LUMO_D``/``HOMO_A``/``LUMO_A``. The OPV
                file is automatically expanded to a 2N-row simple table.
    """

    def __init__(self, csv_path: str | Path) -> None:
        super().__init__()
        df = pd.read_csv(csv_path)
        if "smiles" in df.columns:
            self.df = df
        elif {"Donor SMILES", "Acceptor SMILES"}.issubset(df.columns):
            self.df = self._expand_opv(df)
        else:
            raise ValueError(
                f"{csv_path} must have either `smiles` or both `Donor SMILES`"
                f" and `Acceptor SMILES` columns."
            )

        if {"HOMO_calib", "LUMO_calib"}.issubset(self.df.columns):
            self.homo_col, self.lumo_col = "HOMO_calib", "LUMO_calib"
        elif {"HOMO", "LUMO"}.issubset(self.df.columns):
            self.homo_col, self.lumo_col = "HOMO", "LUMO"
        else:
            raise ValueError("Dataset must have HOMO_calib/LUMO_calib or HOMO/LUMO.")

    @staticmethod
    def _expand_opv(df: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict] = []
        for _, row in df.iterrows():
            if (pd.notna(row.get("Donor SMILES")) and pd.notna(row.get("HOMO_D"))
                    and pd.notna(row.get("LUMO_D"))):
                rows.append({"smiles": row["Donor SMILES"],
                             "HOMO": row["HOMO_D"], "LUMO": row["LUMO_D"]})
            if (pd.notna(row.get("Acceptor SMILES")) and pd.notna(row.get("HOMO_A"))
                    and pd.notna(row.get("LUMO_A"))):
                rows.append({"smiles": row["Acceptor SMILES"],
                             "HOMO": row["HOMO_A"], "LUMO": row["LUMO_A"]})
        return pd.DataFrame(rows)

    def len(self) -> int:
        return len(self.df)

    def get(self, idx: int) -> Data:
        row = self.df.iloc[idx]
        mol = Chem.MolFromSmiles(row["smiles"])
        if mol is None:
            raise ValueError(f"Invalid SMILES at row {idx}: {row['smiles']}")
        graph = mol_to_graph(mol)
        graph.y = torch.tensor([float(row[self.homo_col]), float(row[self.lumo_col])],
                               dtype=torch.float)
        return graph


# ----------------------- Donor / Acceptor pair (PCE) -----------------------


class OPVPairDataset(Dataset):
    """Donor + Acceptor pair dataset for PCE regression.

    Each item carries:
      - ``donor``    : ``Data`` graph
      - ``acceptor`` : ``Data`` graph
      - ``y``        : 1-d PCE tensor (shape [1])
      - ``mol_id``   : pair identifier (int) for prediction tracking
      - ``aux``      : 1-d tensor [Voc, Jsc, FF] (only if ``include_aux=True``).
                       NaN-protected: missing values are replaced by 0 and a
                       boolean mask field ``aux_mask`` indicates validity.

    Long SMILES (>``max_smiles_len``) are filtered for memory safety.
    """

    AUX_COLS: tuple[str, ...] = ("Voc", "Jsc", "FF")

    def __init__(self, csv_path: str | Path, max_smiles_len: int = 600,
                 include_aux: bool = False) -> None:
        super().__init__()
        df = pd.read_csv(csv_path)
        required = {"Donor SMILES", "Acceptor SMILES", "PCE"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"OPV2D CSV missing columns: {missing}")
        df = df[
            (df["Donor SMILES"].str.len() <= max_smiles_len)
            & (df["Acceptor SMILES"].str.len() <= max_smiles_len)
        ].reset_index(drop=True)
        if "Mol_ID" not in df.columns:
            df = df.copy()
            df["Mol_ID"] = range(len(df))
        self.df = df
        self.include_aux = include_aux
        if include_aux:
            for c in self.AUX_COLS:
                if c not in df.columns:
                    raise ValueError(f"include_aux=True but column `{c}` missing.")

    def len(self) -> int:
        return len(self.df)

    def get(self, idx: int) -> Data:
        row = self.df.iloc[idx]
        donor_mol = Chem.MolFromSmiles(row["Donor SMILES"])
        acceptor_mol = Chem.MolFromSmiles(row["Acceptor SMILES"])
        if donor_mol is None or acceptor_mol is None:
            raise ValueError(f"Invalid donor/acceptor SMILES at row {idx}")
        donor = mol_to_graph(donor_mol)
        acceptor = mol_to_graph(acceptor_mol)
        data = Data(
            donor=donor,
            acceptor=acceptor,
            y=torch.tensor([float(row["PCE"])], dtype=torch.float),
            mol_id=torch.tensor([int(row["Mol_ID"])], dtype=torch.long),
        )
        if self.include_aux:
            vals = [row.get(c) for c in self.AUX_COLS]
            mask = [1.0 if (v is not None and pd.notna(v)) else 0.0 for v in vals]
            vals = [float(v) if (v is not None and pd.notna(v)) else 0.0 for v in vals]
            data.aux = torch.tensor(vals, dtype=torch.float).unsqueeze(0)        # [1, 3]
            data.aux_mask = torch.tensor(mask, dtype=torch.float).unsqueeze(0)   # [1, 3]
        return data

    @property
    def smiles_table(self) -> pd.DataFrame:
        """Donor/Acceptor SMILES + PCE columns; used by split builders."""
        return self.df[["Donor SMILES", "Acceptor SMILES", "PCE"]].copy()
