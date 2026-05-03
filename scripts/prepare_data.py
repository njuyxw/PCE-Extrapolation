"""Clean raw OPV2D into trainable form, with Y6 acceptors held out separately.

Inputs (default):
  data/raw/opv2d.csv        — 1567 donor-acceptor rows (CycleChemist exp dataset)
  data/raw/lopez51k.csv     — 51k Lopez NFA candidates (used by pretraining)

Outputs:
  data/processed/opv2d_clean.csv  — OPV2D minus Y6 (training set for stage 1)
  data/processed/opv2d_y6.csv     — Y6 acceptor pairs (held out for stage 2)
  data/processed/prepare_report.json

Y6 detection: by molecular formula match (default ``C82H86F4N8O2S5``) plus an
optional list of canonical SMILES. Override either via CLI flags.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

DEFAULT_Y6_FORMULA = "C82H86F4N8O2S5"
# Optional manual SMILES list for additional Y6-family hits (kept empty by
# default — stage-2 evaluation can use this to widen the holdout to BTP-eC9 etc.)
DEFAULT_Y6_EXTRA_SMILES: list[str] = []


def canonical(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def molecular_formula(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return rdMolDescriptors.CalcMolFormula(mol)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--opv_csv", default="data/raw/opv2d.csv")
    parser.add_argument("--out_dir", default="data/processed")
    parser.add_argument("--y6_formula", default=DEFAULT_Y6_FORMULA,
                        help="Comma-separated formulas to treat as Y6 acceptors.")
    parser.add_argument("--y6_smiles_file", default=None,
                        help="Optional file with one Y6-family SMILES per line.")
    parser.add_argument("--max_smiles_len", type=int, default=600)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.opv_csv)
    n_raw = len(df)

    # Required columns
    required = {"Donor SMILES", "Acceptor SMILES", "PCE"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"OPV2D CSV missing columns: {missing}")

    # Length filter
    df = df[
        (df["Donor SMILES"].str.len() <= args.max_smiles_len)
        & (df["Acceptor SMILES"].str.len() <= args.max_smiles_len)
    ].reset_index(drop=True)
    n_after_len = len(df)

    # Drop NaN PCE
    df = df.dropna(subset=["PCE"]).reset_index(drop=True)
    n_after_pce = len(df)

    # Add Mol_ID if absent
    if "Mol_ID" not in df.columns:
        df["Mol_ID"] = range(len(df))

    # Y6 detection
    y6_formulas = {f.strip() for f in args.y6_formula.split(",") if f.strip()}
    extra_smiles = list(DEFAULT_Y6_EXTRA_SMILES)
    if args.y6_smiles_file:
        for line in Path(args.y6_smiles_file).read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                extra_smiles.append(line)
    extra_canonical = {canonical(s) for s in extra_smiles}
    extra_canonical.discard(None)

    formulas = df["Acceptor SMILES"].map(molecular_formula)
    canon = df["Acceptor SMILES"].map(canonical)
    y6_mask = formulas.isin(y6_formulas) | canon.isin(extra_canonical)
    df_y6 = df[y6_mask].reset_index(drop=True)
    df_clean = df[~y6_mask].reset_index(drop=True)

    clean_path = out_dir / "opv2d_clean.csv"
    y6_path = out_dir / "opv2d_y6.csv"
    df_clean.to_csv(clean_path, index=False)
    df_y6.to_csv(y6_path, index=False)

    report = {
        "input": str(Path(args.opv_csv).resolve()),
        "n_raw": int(n_raw),
        "n_after_smiles_len_filter": int(n_after_len),
        "n_after_drop_nan_pce": int(n_after_pce),
        "y6_formulas": sorted(y6_formulas),
        "y6_extra_smiles_count": len(extra_canonical),
        "n_y6_holdout": int(len(df_y6)),
        "n_train_pool": int(len(df_clean)),
        "outputs": {"clean": str(clean_path.resolve()), "y6": str(y6_path.resolve())},
    }
    with open(out_dir / "prepare_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
