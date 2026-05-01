"""SMILES → PyG ``Data`` graph featurization.

Atom feature dim = 31, bond feature dim = 15. These dimensions are part of the
public API: the encoder constructors read them from the constants below so a
featurization tweak only needs to update one file.

The feature set mirrors the published MOE2 implementation (CycleChemist) so
that pretrained checkpoints transfer 1:1.
"""
from __future__ import annotations

import torch
from rdkit import Chem
from torch_geometric.data import Data

# ----------------------- public constants -----------------------

ATOM_TYPES: tuple[str, ...] = ("C", "N", "O", "S", "F", "P", "Cl", "Br", "I")
HYBRIDIZATIONS: tuple = (
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
)
BOND_TYPES: tuple = (
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
)
BOND_STEREO: tuple = (
    Chem.rdchem.BondStereo.STEREONONE,
    Chem.rdchem.BondStereo.STEREOANY,
    Chem.rdchem.BondStereo.STEREOZ,
    Chem.rdchem.BondStereo.STEREOE,
)
ATOMIC_RADIUS: dict[str, float] = {
    "H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "F": 1.47,
    "P": 1.80, "S": 1.80, "Cl": 1.75, "Br": 1.85, "I": 1.98,
}
BOND_ORDER: dict = {
    Chem.rdchem.BondType.SINGLE: 1.0,
    Chem.rdchem.BondType.DOUBLE: 2.0,
    Chem.rdchem.BondType.TRIPLE: 3.0,
    Chem.rdchem.BondType.AROMATIC: 1.5,
}

ATOM_FEAT_DIM: int = 31  # see _atom_features for the breakdown
BOND_FEAT_DIM: int = 15


# ----------------------- atom / bond features -----------------------


def _atom_features(atom: Chem.Atom) -> list[float]:
    """31-d atom feature vector. Order is part of the API."""
    sym = atom.GetSymbol()
    atom_type_oh = [int(sym == a) for a in ATOM_TYPES]                             # 9
    formal_charge = [atom.GetFormalCharge()]                                       # 1
    hyb_oh = [int(atom.GetHybridization() == h) for h in HYBRIDIZATIONS]           # 5
    aromatic = [int(atom.GetIsAromatic())]                                         # 1
    degree = [atom.GetDegree()]                                                    # 1
    num_h = [atom.GetTotalNumHs()]                                                 # 1
    in_ring = [int(atom.IsInRing())]                                               # 1
    try:
        partial = [float(atom.GetProp("_GasteigerCharge"))]
    except KeyError:
        partial = [0.0]                                                            # 1
    is_hetero = [int(sym not in ("C",))]                                           # 1
    is_terminal = [int(atom.GetDegree() == 1)]                                     # 1
    is_pos = [int(atom.GetFormalCharge() > 0)]                                     # 1
    is_neg = [int(atom.GetFormalCharge() < 0)]                                     # 1
    radius = [ATOMIC_RADIUS.get(sym, 1.70)]                                        # 1
    lone_pairs = [
        atom.GetTotalValence()
        - atom.GetExplicitValence()
        - atom.GetTotalNumHs()
        - atom.GetFormalCharge()
    ]                                                                              # 1
    radicals = [atom.GetNumRadicalElectrons()]                                     # 1
    is_db_c = [int(sym == "C" and atom.GetTotalNumHs() < 4 and
                   atom.GetHybridization() == Chem.rdchem.HybridizationType.SP2)]  # 1
    is_tb_c = [int(sym == "C" and atom.GetTotalNumHs() < 4 and
                   atom.GetHybridization() == Chem.rdchem.HybridizationType.SP)]   # 1
    is_arom_c = [int(sym == "C" and atom.GetIsAromatic())]                         # 1
    is_charged = [int(atom.GetFormalCharge() != 0)]                                # 1
    return (
        atom_type_oh + formal_charge + hyb_oh + aromatic + degree + num_h
        + in_ring + partial + is_hetero + is_terminal + is_pos + is_neg
        + radius + lone_pairs + radicals + is_db_c + is_tb_c + is_arom_c
        + is_charged
    )


def _bond_features(bond: Chem.Bond) -> list[float]:
    """15-d bond feature vector. Order is part of the API."""
    bt = bond.GetBondType()
    bt_oh = [int(bt == b) for b in BOND_TYPES]                                     # 4
    st_oh = [int(bond.GetStereo() == s) for s in BOND_STEREO]                      # 4
    is_conj = [int(bond.GetIsConjugated())]                                        # 1
    is_in_ring = [int(bond.IsInRing())]                                            # 1
    num_rings = [bond.GetOwningMol().GetRingInfo().NumBondRings(bond.GetIdx())]    # 1
    direction = [int(bond.GetBondDir() != Chem.rdchem.BondDir.NONE)]               # 1
    is_bridge = [int(bond.IsInRing() and num_rings[0] > 1)]                        # 1
    bond_order = [BOND_ORDER.get(bt, 1.0)]                                         # 1
    is_conj2 = [int(bond.GetIsConjugated())]                                       # 1 (kept to match published dim)
    return (
        bt_oh + st_oh + is_conj + is_in_ring + num_rings + direction
        + is_bridge + bond_order + is_conj2
    )


# ----------------------- public API -----------------------


def mol_to_graph(mol: Chem.Mol) -> Data:
    """Convert an RDKit Mol to a PyG ``Data`` (no labels attached)."""
    if mol is None:
        raise ValueError("mol is None — invalid SMILES upstream.")

    x = torch.tensor([_atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)

    edge_index_list: list[list[int]] = []
    edge_attr_list: list[list[float]] = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feats = _bond_features(bond)
        edge_index_list += [[i, j], [j, i]]
        edge_attr_list += [feats, feats]

    if not edge_index_list:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, BOND_FEAT_DIM), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def smiles_to_graph(smiles: str) -> Data:
    """Convenience wrapper. Raises ``ValueError`` on invalid SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return mol_to_graph(mol)
