from functools import lru_cache
from typing import Dict, Iterable, Optional, Tuple

import torch

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, rdFingerprintGenerator, rdMolDescriptors
except Exception:
    Chem = None
    AllChem = None
    Descriptors = None
    rdFingerprintGenerator = None
    rdMolDescriptors = None


NODE_FEAT_DIM = 74
DRUG_EDGE_FEAT_DIM = 12
DEFAULT_MORGAN_RADIUS = 2
DEFAULT_MORGAN_NBITS = 1024

ATOM_TYPES = [
    "C", "N", "O", "S", "F", "Si", "P", "Cl", "Br", "Mg", "Na", "Ca",
    "Fe", "As", "Al", "I", "B", "V", "K", "Tl", "Yb", "Sb", "Sn", "Ag",
    "Pd", "Co", "Se", "Ti", "Zn", "H", "Li", "Ge", "Cu", "Au", "Ni", "Cd",
    "In", "Mn", "Zr", "Cr", "Pt", "Hg", "Pb",
]
DEGREES = list(range(11))
IMPLICIT_VALENCE = list(range(7))
TOTAL_H = list(range(5))


BOND_STEREO_TYPES = [
    Chem.rdchem.BondStereo.STEREONONE,
    Chem.rdchem.BondStereo.STEREOANY,
    Chem.rdchem.BondStereo.STEREOZ,
    Chem.rdchem.BondStereo.STEREOE,
    Chem.rdchem.BondStereo.STEREOCIS,
    Chem.rdchem.BondStereo.STEREOTRANS,
] if Chem is not None else []


DRUG_DESC_NAMES = [
    "conf_valid",
    "num_atoms",
    "num_heavy_atoms",
    "mol_wt",
    "heavy_atom_mol_wt",
    "exact_mol_wt",
    "num_valence_electrons",
    "num_hetero_atoms",
    "num_h_acceptors",
    "num_h_donors",
    "num_rotatable_bonds",
    "ring_count",
    "num_aromatic_rings",
    "num_saturated_rings",
    "num_aliphatic_rings",
    "tpsa",
    "mol_logp",
    "fraction_csp3",
    "labute_asa",
    "radius_of_gyration",
    "asphericity",
    "eccentricity",
    "inertial_shape_factor",
    "npr1",
    "npr2",
    "pmi1",
    "pmi2",
    "pmi3",
    "spherocity_index",
    "chiral_centers_total",
    "chiral_centers_assigned",
    "chiral_centers_unassigned",
    "cip_r_count",
    "cip_s_count",
    "atom_chiral_possible_count",
    "bond_stereo_count",
    "bond_stereo_z_count",
    "bond_stereo_e_count",
    "bond_stereo_cis_count",
    "bond_stereo_trans_count",
]
DRUG_DESC_DIM = len(DRUG_DESC_NAMES)


def _require_rdkit():
    if Chem is None or AllChem is None or Descriptors is None or rdMolDescriptors is None:
        raise ImportError("RDKit is required for canonical drug featurization.")


def _one_hot(value, allowable_set: Iterable) -> list:
    return [value == x for x in allowable_set]


def _get_implicit_valence(atom) -> int:
    if hasattr(Chem.rdchem, "ValenceType"):
        return int(atom.GetValence(Chem.rdchem.ValenceType.IMPLICIT))
    return int(atom.GetImplicitValence())


def _atom_feat_74(atom) -> torch.Tensor:
    hyb_set = [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ]
    feat = []
    feat += _one_hot(atom.GetSymbol(), ATOM_TYPES)
    feat += _one_hot(atom.GetDegree(), DEGREES)
    feat += _one_hot(_get_implicit_valence(atom), IMPLICIT_VALENCE)
    feat += [atom.GetFormalCharge()]
    feat += [atom.GetNumRadicalElectrons()]
    feat += _one_hot(atom.GetHybridization(), hyb_set)
    feat += [atom.GetIsAromatic()]
    feat += _one_hot(atom.GetTotalNumHs(), TOTAL_H)
    out = torch.tensor(feat, dtype=torch.float32)
    if int(out.numel()) != NODE_FEAT_DIM:
        raise RuntimeError(
            f"DrugBAN canonical feature dim mismatch: got {int(out.numel())}, expected {NODE_FEAT_DIM}."
        )
    return out


def _canonical_atom_features(mol) -> torch.Tensor:
    feats = [_atom_feat_74(atom) for atom in mol.GetAtoms()]
    if not feats:
        return torch.zeros((1, NODE_FEAT_DIM), dtype=torch.float32)
    return torch.stack(feats, dim=0)


def _bond_feat(edge) -> torch.Tensor:
    bt = edge.GetBondType()
    feat = [
        bt == Chem.rdchem.BondType.SINGLE,
        bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE,
        bt == Chem.rdchem.BondType.AROMATIC,
        bool(edge.GetIsConjugated()),
        bool(edge.IsInRing()),
    ]
    stereo = edge.GetStereo()
    feat += _one_hot(stereo, BOND_STEREO_TYPES)
    out = torch.tensor(feat, dtype=torch.float32)
    if int(out.numel()) != DRUG_EDGE_FEAT_DIM:
        raise RuntimeError(
            f"Drug bond feature dim mismatch: got {int(out.numel())}, expected {DRUG_EDGE_FEAT_DIM}."
        )
    return out


def smiles_to_mol(smiles: str):
    _require_rdkit()
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    try:
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    except Exception:
        pass
    return mol


def _coords_from_conformer(mol) -> torch.Tensor:
    conf = mol.GetConformer()
    coords = []
    for i in range(mol.GetNumAtoms()):
        pos = conf.GetAtomPosition(i)
        coords.append([float(pos.x), float(pos.y), float(pos.z)])
    return torch.tensor(coords, dtype=torch.float32)


def _normalize_coords(pos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
    if pos.numel() == 0:
        return pos, torch.zeros((1, 3), dtype=pos.dtype), 1.0
    center = pos.mean(dim=0, keepdim=True)
    pos0 = pos - center
    scale = float(torch.sqrt(pos0.pow(2).sum(dim=-1).mean().clamp_min(1e-12)).item())
    return pos0 / max(scale, 1e-6), center, max(scale, 1e-6)


@lru_cache(maxsize=8)
def _get_morgan_generator(radius: int, nbits: int):
    if rdFingerprintGenerator is None:
        return None
    return rdFingerprintGenerator.GetMorganGenerator(
        radius=int(radius),
        fpSize=int(nbits),
        includeChirality=True,
    )


def mol_to_conformer_data(mol, return_mol3d: bool = False) -> Dict[str, object]:
    _require_rdkit()
    n = int(mol.GetNumAtoms()) if mol is not None else 0
    zero_pos = torch.zeros((max(n, 1), 3), dtype=torch.float32)
    out: Dict[str, object] = {
        "node_pos": zero_pos[:n] if n > 0 else zero_pos,
        "conf_valid": 0.0,
        "coord_scale": 1.0,
    }
    if return_mol3d:
        out["mol_3d"] = None

    if mol is None or n <= 0:
        return out

    mol_3d = None
    try:
        mol_h = Chem.AddHs(Chem.Mol(mol))
        params = AllChem.ETKDGv3()
        params.randomSeed = 0xF00D
        try:
            params.useSmallRingTorsions = True
        except Exception:
            pass
        try:
            params.useRandomCoords = True
        except Exception:
            pass
        status = int(AllChem.EmbedMolecule(mol_h, params))
        if status == 0:
            try:
                if AllChem.MMFFHasAllMoleculeParams(mol_h):
                    AllChem.MMFFOptimizeMolecule(mol_h, maxIters=200)
                    mol_3d = Chem.RemoveHs(mol_h)
            except Exception:
                mol_3d = None
    except Exception:
        mol_3d = None

    if mol_3d is None or mol_3d.GetNumConformers() == 0 or int(mol_3d.GetNumAtoms()) != n:
        return out

    pos_raw = _coords_from_conformer(mol_3d)
    pos_norm, _, scale = _normalize_coords(pos_raw)
    out["node_pos"] = pos_norm.float()
    out["conf_valid"] = 1.0
    out["coord_scale"] = float(scale)
    if return_mol3d:
        out["mol_3d"] = mol_3d
    return out


def _count_bond_stereo(mol) -> Tuple[int, int, int, int, int]:
    total = 0
    z_count = 0
    e_count = 0
    cis_count = 0
    trans_count = 0
    for bond in mol.GetBonds():
        stereo = bond.GetStereo()
        if stereo in (
            Chem.rdchem.BondStereo.STEREOZ,
            Chem.rdchem.BondStereo.STEREOE,
            Chem.rdchem.BondStereo.STEREOCIS,
            Chem.rdchem.BondStereo.STEREOTRANS,
        ):
            total += 1
        if stereo == Chem.rdchem.BondStereo.STEREOZ:
            z_count += 1
        elif stereo == Chem.rdchem.BondStereo.STEREOE:
            e_count += 1
        elif stereo == Chem.rdchem.BondStereo.STEREOCIS:
            cis_count += 1
        elif stereo == Chem.rdchem.BondStereo.STEREOTRANS:
            trans_count += 1
    return total, z_count, e_count, cis_count, trans_count


def _count_cip_labels(mol) -> Tuple[int, int, int]:
    total = 0
    r_count = 0
    s_count = 0
    for atom in mol.GetAtoms():
        if atom.HasProp("_ChiralityPossible"):
            total += 1
        if atom.HasProp("_CIPCode"):
            code = str(atom.GetProp("_CIPCode")).strip().upper()
            if code == "R":
                r_count += 1
            elif code == "S":
                s_count += 1
    return total, r_count, s_count


def mol_to_descriptor_vector(mol, mol_3d=None, conf_valid: float = 0.0) -> torch.Tensor:
    _require_rdkit()
    if mol is None:
        return torch.zeros((DRUG_DESC_DIM,), dtype=torch.float32)

    chiral_centers = Chem.FindMolChiralCenters(
        mol,
        includeUnassigned=True,
        useLegacyImplementation=False,
    )
    chiral_total = int(len(chiral_centers))
    chiral_assigned = int(sum(1 for _, tag in chiral_centers if str(tag) not in ("?", "")))
    chiral_unassigned = int(chiral_total - chiral_assigned)
    chiral_possible, cip_r_count, cip_s_count = _count_cip_labels(mol)
    stereo_total, stereo_z, stereo_e, stereo_cis, stereo_trans = _count_bond_stereo(mol)

    desc = [
        float(conf_valid),
        float(mol.GetNumAtoms()),
        float(mol.GetNumHeavyAtoms()),
        float(Descriptors.MolWt(mol)),
        float(Descriptors.HeavyAtomMolWt(mol)),
        float(Descriptors.ExactMolWt(mol)),
        float(Descriptors.NumValenceElectrons(mol)),
        float(Descriptors.NumHeteroatoms(mol)),
        float(Descriptors.NumHAcceptors(mol)),
        float(Descriptors.NumHDonors(mol)),
        float(Descriptors.NumRotatableBonds(mol)),
        float(Descriptors.RingCount(mol)),
        float(Descriptors.NumAromaticRings(mol)),
        float(Descriptors.NumSaturatedRings(mol)),
        float(Descriptors.NumAliphaticRings(mol)),
        float(Descriptors.TPSA(mol)),
        float(Descriptors.MolLogP(mol)),
        float(Descriptors.FractionCSP3(mol)),
        float(rdMolDescriptors.CalcLabuteASA(mol)),
    ]

    if mol_3d is not None and float(conf_valid) > 0.5:
        desc.extend(
            [
                float(rdMolDescriptors.CalcRadiusOfGyration(mol_3d)),
                float(rdMolDescriptors.CalcAsphericity(mol_3d)),
                float(rdMolDescriptors.CalcEccentricity(mol_3d)),
                float(rdMolDescriptors.CalcInertialShapeFactor(mol_3d)),
                float(rdMolDescriptors.CalcNPR1(mol_3d)),
                float(rdMolDescriptors.CalcNPR2(mol_3d)),
                float(rdMolDescriptors.CalcPMI1(mol_3d)),
                float(rdMolDescriptors.CalcPMI2(mol_3d)),
                float(rdMolDescriptors.CalcPMI3(mol_3d)),
                float(rdMolDescriptors.CalcSpherocityIndex(mol_3d)),
            ]
        )
    else:
        desc.extend([0.0] * 10)

    desc.extend(
        [
            float(chiral_total),
            float(chiral_assigned),
            float(chiral_unassigned),
            float(cip_r_count),
            float(cip_s_count),
            float(chiral_possible),
            float(stereo_total),
            float(stereo_z),
            float(stereo_e),
            float(stereo_cis),
            float(stereo_trans),
        ]
    )
    out = torch.tensor(desc, dtype=torch.float32)
    if int(out.numel()) != DRUG_DESC_DIM:
        raise RuntimeError(
            f"Drug descriptor dim mismatch: got {int(out.numel())}, expected {DRUG_DESC_DIM}."
        )
    return out


def mol_to_morgan_fp(mol, radius: int = DEFAULT_MORGAN_RADIUS, nbits: int = DEFAULT_MORGAN_NBITS) -> torch.Tensor:
    _require_rdkit()
    if mol is None or int(nbits) <= 0:
        return torch.zeros((max(int(nbits), 0),), dtype=torch.float32)
    fp = None
    gen = _get_morgan_generator(int(radius), int(nbits))
    if gen is not None:
        fp = gen.GetFingerprint(mol)
    else:
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol,
            int(radius),
            nBits=int(nbits),
            useChirality=True,
        )
    bitstr = fp.ToBitString()
    return torch.tensor([1.0 if ch == "1" else 0.0 for ch in bitstr], dtype=torch.float32)


def mol_to_graph_tensors(mol) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if mol is None or mol.GetNumAtoms() <= 0:
        node_feat = torch.zeros((1, NODE_FEAT_DIM), dtype=torch.float32)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, DRUG_EDGE_FEAT_DIM), dtype=torch.float32)
        return node_feat, edge_index, edge_attr

    node_feat = _canonical_atom_features(mol)
    edges = []
    edge_feats = []
    for bond in mol.GetBonds():
        i = int(bond.GetBeginAtomIdx())
        j = int(bond.GetEndAtomIdx())
        bf = _bond_feat(bond)
        edges.append((i, j))
        edge_feats.append(bf)
        edges.append((j, i))
        edge_feats.append(bf.clone())

    edge_index = (
        torch.tensor(edges, dtype=torch.long).t().contiguous()
        if edges
        else torch.zeros((2, 0), dtype=torch.long)
    )
    edge_attr = (
        torch.stack(edge_feats, dim=0)
        if edge_feats
        else torch.zeros((0, DRUG_EDGE_FEAT_DIM), dtype=torch.float32)
    )
    return node_feat, edge_index, edge_attr


def smiles_to_graph_canon(smiles: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _require_rdkit()
    mol = smiles_to_mol(smiles)
    return mol_to_graph_tensors(mol)


def smiles_to_drug_data(
    smiles: str,
    morgan_radius: int = DEFAULT_MORGAN_RADIUS,
    morgan_nbits: int = DEFAULT_MORGAN_NBITS,
) -> Dict[str, torch.Tensor]:
    _require_rdkit()

    mol = smiles_to_mol(smiles)
    node_feat, edge_index, edge_attr = mol_to_graph_tensors(mol)
    conf = mol_to_conformer_data(mol, return_mol3d=True)
    node_pos = conf["node_pos"].float()
    conf_valid = float(conf.get("conf_valid", 0.0))
    mol_3d = conf.get("mol_3d", None)
    if int(node_pos.shape[0]) != int(node_feat.shape[0]):
        node_pos = torch.zeros((int(node_feat.shape[0]), 3), dtype=torch.float32)
        conf_valid = 0.0
        mol_3d = None

    drug_fp = mol_to_morgan_fp(mol, radius=morgan_radius, nbits=morgan_nbits)
    drug_desc = mol_to_descriptor_vector(mol, mol_3d=mol_3d, conf_valid=conf_valid)

    return {
        "node_feat": node_feat,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "node_pos": node_pos,
        "drug_fp": drug_fp,
        "drug_desc": drug_desc,
    }
