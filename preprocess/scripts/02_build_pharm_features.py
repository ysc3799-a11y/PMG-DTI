import argparse
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import os
import sys

import pandas as pd
import torch

try:
    from rdkit import Chem
    from rdkit.Chem import ChemicalFeatures
    from rdkit import RDConfig
except Exception:
    Chem = None
    ChemicalFeatures = None
    RDConfig = None

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


_THIS = os.path.abspath(__file__)
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_THIS)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from dataset.featurize_canon import mol_to_conformer_data


FAMILY_PRIORITY = {
    "PosIonizable": 0,
    "NegIonizable": 0,
    "Donor": 1,
    "Acceptor": 1,
    "Aromatic": 2,
    "Hydrophobe": 3,
    "LumpedHydrophobe": 3,
}

ELEM_BUCKETS = ("C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "B", "Si", "OTHER")
ELEM_BUCKET_SET = set(ELEM_BUCKETS[:-1])
LEGACY_ATOM_BUCKETS_9 = ("C", "N", "O", "S", "P", "F", "Cl", "Br", "OTHER")
LEGACY_ATOM_BUCKET_SET = set(LEGACY_ATOM_BUCKETS_9[:-1])
PHARM_ELEM_BUCKETS = ("C", "N", "O", "S", "P", "HALOGEN", "OTHER")
HALOGEN_SET = {"F", "Cl", "Br", "I"}

PHARM_GEOM_NAMES = [
    "conf_valid",
    "group_size_norm",
    "mean_center_dist",
    "max_center_dist",
    "std_center_dist",
    "mean_pair_dist",
    "max_pair_dist",
    "min_pair_dist",
    "center_norm",
]
PHARM_CONTEXT_NAMES = [
    "group_size_norm",
    "feature_arom_frac",
    "feature_ring_frac",
    "feature_pos_frac",
    "feature_neg_frac",
    "feature_chiral_frac",
    "feature_cip_r_frac",
    "feature_cip_s_frac",
    "feat_elem_c",
    "feat_elem_n",
    "feat_elem_o",
    "feat_elem_s",
    "feat_elem_p",
    "feat_elem_halogen",
    "feat_elem_other",
    "nbr_elem_c",
    "nbr_elem_n",
    "nbr_elem_o",
    "nbr_elem_s",
    "nbr_elem_p",
    "nbr_elem_halogen",
    "nbr_elem_other",
    "nbr_arom_frac",
    "nbr_ring_frac",
]


def load_factory():
    if ChemicalFeatures is None or RDConfig is None:
        return None
    fdef = Path(RDConfig.RDDataDir) / "BaseFeatures.fdef"
    if not fdef.is_file():
        return None
    return ChemicalFeatures.BuildFeatureFactory(str(fdef))


def _charge_sign(v: int) -> str:
    if v > 0:
        return "+"
    if v < 0:
        return "-"
    return "0"


def _safe_frac(num: float, den: float) -> float:
    return float(num / max(float(den), 1e-6))


def _anchor_bucket(mol, atom_idx: int, bucket_mode: str) -> str:
    try:
        atom = mol.GetAtomWithIdx(int(atom_idx))
        sym = str(atom.GetSymbol())
    except Exception:
        return "OTHER|arom0|chg0" if bucket_mode == "elem_arom_charge" else "OTHER"
    if bucket_mode == "legacy9_element":
        return sym if sym in LEGACY_ATOM_BUCKET_SET else "OTHER"
    elem = sym if sym in ELEM_BUCKET_SET else "OTHER"
    if bucket_mode == "element":
        return elem
    arom = 1 if bool(atom.GetIsAromatic()) else 0
    chg = _charge_sign(int(atom.GetFormalCharge()))
    return f"{elem}|arom{arom}|chg{chg}"


def _feature_elem_bucket(sym: str) -> str:
    s = str(sym)
    if s in HALOGEN_SET:
        return "HALOGEN"
    if s in {"C", "N", "O", "S", "P"}:
        return s
    return "OTHER"


def _rank_anchor_atoms(mol, atom_ids: List[int]) -> List[int]:
    scored = []
    for aid in atom_ids:
        try:
            atom = mol.GetAtomWithIdx(int(aid))
        except Exception:
            continue
        sym = str(atom.GetSymbol())
        score = (
            1 if int(atom.GetFormalCharge()) != 0 else 0,
            1 if bool(atom.GetIsAromatic()) else 0,
            1 if atom.HasProp("_CIPCode") else 0,
            int(atom.GetDegree()),
            1 if sym not in {"C", "H"} else 0,
            -int(aid),
        )
        scored.append((score, int(aid)))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [int(aid) for _, aid in scored] or [int(atom_ids[0])]


def mirror_processed_path(path: Path) -> Optional[Path]:
    parts = list(path.parts)
    if "interim" not in parts:
        return None
    i = parts.index("interim")
    parts[i] = "processed"
    return Path(*parts)


def _progress(iterable, desc: str, total: Optional[int] = None):
    if tqdm is None:
        return iterable
    return tqdm(iterable, desc=desc, total=total)


def build_vocab(
    token_lists: List[List[Tuple[str, int]]],
    min_token_freq: int,
    max_vocab: int,
) -> Tuple[Dict[str, int], Dict[str, int]]:
    cnt = Counter()
    for cur in token_lists:
        for tok, _ in cur:
            cnt[str(tok)] += 1
    items = [(k, int(v)) for k, v in cnt.items() if int(v) >= int(max(1, min_token_freq))]
    items.sort(key=lambda x: (-int(x[1]), str(x[0])))
    if int(max_vocab) > 0:
        items = items[: int(max_vocab)]
    token2id: Dict[str, int] = {"__PAD__": 0, "__OTHER__": 1}
    for tok, _ in items:
        if tok not in token2id:
            token2id[tok] = int(len(token2id))
    return token2id, {k: int(v) for k, v in cnt.items()}


def _stable_unique_records(records: List[Dict[str, object]]) -> List[Dict[str, object]]:
    seen = set()
    out: List[Dict[str, object]] = []
    for rec in records:
        key = (
            str(rec["token_key"]),
            int(rec["anchor"]),
            tuple(int(x) for x in rec["atom_sig"]),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(rec)
    return out


def _pairwise_distance_stats(pos: torch.Tensor) -> Tuple[float, float, float]:
    if pos.numel() == 0 or int(pos.shape[0]) <= 1:
        return 0.0, 0.0, 0.0
    dist = torch.cdist(pos, pos, p=2)
    idx = torch.triu_indices(int(pos.shape[0]), int(pos.shape[0]), offset=1)
    vals = dist[idx[0], idx[1]]
    if vals.numel() == 0:
        return 0.0, 0.0, 0.0
    return float(vals.mean().item()), float(vals.max().item()), float(vals.min().item())


def _compute_feature_geometry(atom_ids: List[int], node_pos: torch.Tensor, conf_valid: float) -> Tuple[torch.Tensor, torch.Tensor]:
    if float(conf_valid) > 0.5 and node_pos.numel() > 0:
        idx = torch.tensor(atom_ids, dtype=torch.long)
        pos = node_pos.index_select(0, idx)
        center = pos.mean(dim=0)
        center_dist = torch.norm(pos - center.view(1, 3), dim=-1)
        mean_pair, max_pair, min_pair = _pairwise_distance_stats(pos)
        geom = torch.tensor(
            [
                float(conf_valid),
                _safe_frac(len(atom_ids), 4.0),
                float(center_dist.mean().item()) if center_dist.numel() > 0 else 0.0,
                float(center_dist.max().item()) if center_dist.numel() > 0 else 0.0,
                float(center_dist.std(unbiased=False).item()) if center_dist.numel() > 0 else 0.0,
                mean_pair,
                max_pair,
                min_pair,
                float(torch.norm(center, p=2).item()),
            ],
            dtype=torch.float32,
        )
        return center.float(), geom

    center = torch.zeros((3,), dtype=torch.float32)
    geom = torch.tensor(
        [
            0.0,
            _safe_frac(len(atom_ids), 4.0),
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        dtype=torch.float32,
    )
    return center, geom


def _compute_feature_context(mol, atom_ids: List[int]) -> torch.Tensor:
    feature_set = {int(aid) for aid in atom_ids}
    feat_elem_counter = Counter()
    nbr_elem_counter = Counter()
    feature_arom = 0
    feature_ring = 0
    feature_pos = 0
    feature_neg = 0
    feature_chiral = 0
    feature_cip_r = 0
    feature_cip_s = 0
    neighbors = set()

    for aid in atom_ids:
        atom = mol.GetAtomWithIdx(int(aid))
        feat_elem_counter[_feature_elem_bucket(atom.GetSymbol())] += 1
        feature_arom += int(bool(atom.GetIsAromatic()))
        feature_ring += int(bool(atom.IsInRing()))
        feature_pos += int(int(atom.GetFormalCharge()) > 0)
        feature_neg += int(int(atom.GetFormalCharge()) < 0)
        feature_chiral += int(atom.HasProp("_ChiralityPossible"))
        if atom.HasProp("_CIPCode"):
            code = str(atom.GetProp("_CIPCode")).strip().upper()
            feature_cip_r += int(code == "R")
            feature_cip_s += int(code == "S")
        for nbr in atom.GetNeighbors():
            nid = int(nbr.GetIdx())
            if nid not in feature_set:
                neighbors.add(nid)

    nbr_arom = 0
    nbr_ring = 0
    for nid in neighbors:
        nbr = mol.GetAtomWithIdx(int(nid))
        nbr_elem_counter[_feature_elem_bucket(nbr.GetSymbol())] += 1
        nbr_arom += int(bool(nbr.GetIsAromatic()))
        nbr_ring += int(bool(nbr.IsInRing()))

    n_feat = float(max(len(atom_ids), 1))
    n_nbr = float(max(len(neighbors), 1))
    vals = [
        _safe_frac(len(atom_ids), 4.0),
        _safe_frac(feature_arom, n_feat),
        _safe_frac(feature_ring, n_feat),
        _safe_frac(feature_pos, n_feat),
        _safe_frac(feature_neg, n_feat),
        _safe_frac(feature_chiral, n_feat),
        _safe_frac(feature_cip_r, n_feat),
        _safe_frac(feature_cip_s, n_feat),
    ]
    for bucket in PHARM_ELEM_BUCKETS:
        vals.append(_safe_frac(feat_elem_counter.get(bucket, 0), n_feat))
    for bucket in PHARM_ELEM_BUCKETS:
        vals.append(_safe_frac(nbr_elem_counter.get(bucket, 0), n_nbr))
    vals.append(_safe_frac(nbr_arom, n_nbr))
    vals.append(_safe_frac(nbr_ring, n_nbr))
    return torch.tensor(vals, dtype=torch.float32)


def _charge_tag(mol, atom_ids: List[int]) -> str:
    pos = 0
    neg = 0
    for aid in atom_ids:
        atom = mol.GetAtomWithIdx(int(aid))
        pos += int(int(atom.GetFormalCharge()) > 0)
        neg += int(int(atom.GetFormalCharge()) < 0)
    if pos > 0 and neg > 0:
        return "mix"
    if pos > 0:
        return "pos"
    if neg > 0:
        return "neg"
    return "neu"


def _chirality_tag(mol, atom_ids: List[int]) -> str:
    r = 0
    s = 0
    any_chiral = 0
    for aid in atom_ids:
        atom = mol.GetAtomWithIdx(int(aid))
        any_chiral += int(atom.HasProp("_ChiralityPossible"))
        if atom.HasProp("_CIPCode"):
            code = str(atom.GetProp("_CIPCode")).strip().upper()
            r += int(code == "R")
            s += int(code == "S")
    if any_chiral <= 0:
        return "chi0"
    if r > 0 and s > 0:
        return "chimix"
    if r > 0:
        return "chiR"
    if s > 0:
        return "chiS"
    return "chi?"


def _span_tag(max_pair_dist: float, n_atoms: int) -> str:
    if int(n_atoms) <= 1:
        return "point"
    if float(max_pair_dist) < 0.5:
        return "tight"
    if float(max_pair_dist) < 1.25:
        return "short"
    if float(max_pair_dist) < 2.5:
        return "mid"
    return "wide"


def _dominant_bucket(counter: Counter) -> str:
    if not counter:
        return "none"
    items = sorted(counter.items(), key=lambda x: (-int(x[1]), str(x[0])))
    return str(items[0][0]).lower()


def _build_token_key(
    mol,
    fam: str,
    atom_ids: List[int],
    anchor_ids: List[int],
    bucket_mode: str,
    token_mode: str,
    context: torch.Tensor,
    geom: torch.Tensor,
) -> str:
    if token_mode == "first_anchor":
        bucket = _anchor_bucket(mol, int(anchor_ids[0]), bucket_mode=bucket_mode)
        return f"{fam}|{bucket}"

    anchor_buckets = sorted({_anchor_bucket(mol, int(aid), bucket_mode=bucket_mode) for aid in anchor_ids})
    if token_mode == "all_anchor":
        return f"{fam}|{';'.join(anchor_buckets)}"

    feat_counter = Counter()
    nbr_counter = Counter()
    for aid in atom_ids:
        atom = mol.GetAtomWithIdx(int(aid))
        feat_counter[_feature_elem_bucket(atom.GetSymbol())] += 1
        for nbr in atom.GetNeighbors():
            if int(nbr.GetIdx()) not in set(int(x) for x in atom_ids):
                nbr_counter[_feature_elem_bucket(nbr.GetSymbol())] += 1

    size_tag = "1" if len(atom_ids) <= 1 else ("2" if len(atom_ids) == 2 else "3p")
    ring_tag = "r1" if float(context[2].item()) >= 0.5 else "r0"
    charge_tag = _charge_tag(mol, atom_ids)
    chirality_tag = _chirality_tag(mol, atom_ids)
    span = _span_tag(float(geom[6].item()), len(atom_ids))
    feat_dom = _dominant_bucket(feat_counter)
    nbr_dom = _dominant_bucket(nbr_counter)
    return (
        f"{fam}|sz{size_tag}|anch:{';'.join(anchor_buckets)}|"
        f"feat:{feat_dom}|nbr:{nbr_dom}|chg:{charge_tag}|{ring_tag}|sp:{span}|{chirality_tag}"
    )


def extract_pharm_records(
    smi: str,
    used_rdkit: bool,
    factory,
    family2id,
    max_tokens: int,
    token_mode: str,
    bucket_mode: str,
    anchor_topk: int,
) -> List[Dict[str, object]]:
    if not used_rdkit:
        return []
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return []
    try:
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    except Exception:
        pass

    conf = mol_to_conformer_data(mol)
    node_pos = conf["node_pos"].float()
    conf_valid = float(conf.get("conf_valid", 0.0))

    feats = factory.GetFeaturesForMol(mol)
    records = []
    for ft in feats:
        fam = str(ft.GetFamily())
        if fam not in family2id:
            continue
        atom_ids = sorted(int(aid) for aid in ft.GetAtomIds())
        if not atom_ids:
            continue
        ranked_anchor_ids = _rank_anchor_atoms(mol, atom_ids)
        anchor_ids = ranked_anchor_ids[: int(max(1, anchor_topk))]
        repr_anchor = int(anchor_ids[0])
        center, geom = _compute_feature_geometry(atom_ids, node_pos=node_pos, conf_valid=conf_valid)
        context = _compute_feature_context(mol, atom_ids)
        token_key = _build_token_key(
            mol=mol,
            fam=fam,
            atom_ids=atom_ids,
            anchor_ids=anchor_ids,
            bucket_mode=bucket_mode,
            token_mode=token_mode,
            context=context,
            geom=geom,
        )
        records.append(
            {
                "fam": fam,
                "family_id": int(family2id[fam]),
                "priority": int(FAMILY_PRIORITY.get(fam, 9)),
                "token_key": token_key,
                "anchor": repr_anchor,
                "anchor_ids": [int(x) for x in anchor_ids],
                "atom_sig": tuple(atom_ids),
                "group_size": int(len(atom_ids)),
                "center": center.float(),
                "geom": geom.float(),
                "context": context.float(),
            }
        )

    deduped = _stable_unique_records(records)
    deduped.sort(
        key=lambda r: (
            int(r["priority"]),
            -int(r["group_size"]),
            int(r["anchor"]),
            str(r["token_key"]),
        )
    )
    return deduped[: int(max_tokens)]


def main():
    ap = argparse.ArgumentParser(description="Build pharmacophore token tensors.")
    ap.add_argument("--in_csv", default="data/interim/human/pairs_with_pid/full_with_pid.csv")
    ap.add_argument("--smiles_col", default="SMILES")
    ap.add_argument("--max_tokens", type=int, default=64)
    ap.add_argument(
        "--families",
        default="Acceptor,Donor,Aromatic,Hydrophobe,LumpedHydrophobe,NegIonizable,PosIonizable",
        help="comma-separated pharm feature families",
    )
    ap.add_argument("--out_pt", default="data/interim/human/pharm_features/pharm_rich3d_functional.pt")
    ap.add_argument(
        "--token_mode",
        default="rich_context",
        choices=["rich_context", "all_anchor", "first_anchor"],
        help="pharm token key mode",
    )
    ap.add_argument(
        "--bucket_mode",
        default="elem_arom_charge",
        choices=["elem_arom_charge", "element", "legacy9_element"],
        help="anchor atom bucket mode",
    )
    ap.add_argument(
        "--vocab_mode",
        default="dynamic",
        choices=["dynamic", "fixed64"],
        help="dynamic vocab (new) or fixed 64 vocab (legacy)",
    )
    ap.add_argument("--min_token_freq", type=int, default=3, help="tokens below this freq are mapped to OTHER")
    ap.add_argument("--max_vocab", type=int, default=0, help="cap kept tokens before OTHER merge (0 = no cap)")
    ap.add_argument("--anchor_topk", type=int, default=4, help="keep up to K anchor atoms per pharm token")
    ap.add_argument(
        "--dedup",
        dest="dedup",
        action="store_true",
        help="Deduplicate SMILES before featurization, then map features back to all rows.",
    )
    ap.add_argument(
        "--no_dedup",
        dest="dedup",
        action="store_false",
        help="Disable SMILES-level deduplication.",
    )
    ap.set_defaults(dedup=True)
    ap.add_argument("--no_mirror_processed", action="store_true", help="Disable auto mirror to data/processed/*")
    args = ap.parse_args()

    in_csv = Path(args.in_csv)
    if not in_csv.is_file():
        raise FileNotFoundError(in_csv)
    df = pd.read_csv(in_csv)
    if args.smiles_col not in df.columns:
        raise ValueError(f"missing smiles col: {args.smiles_col}")

    families = [x.strip() for x in args.families.split(",") if x.strip()]
    family2id = {f: i for i, f in enumerate(families)}
    n = len(df)
    p = int(args.max_tokens)
    anchor_topk = int(max(1, args.anchor_topk))

    pharm_type = torch.zeros((n, p), dtype=torch.long)
    pharm_mask = torch.zeros((n, p), dtype=torch.long)
    pharm_anchor = torch.full((n, p), -1, dtype=torch.long)
    pharm_family = torch.zeros((n, p), dtype=torch.long)
    pharm_center = torch.zeros((n, p, 3), dtype=torch.float32)
    pharm_geom = torch.zeros((n, p, len(PHARM_GEOM_NAMES)), dtype=torch.float32)
    pharm_context = torch.zeros((n, p, len(PHARM_CONTEXT_NAMES)), dtype=torch.float32)
    pharm_multi_anchor = torch.full((n, p, anchor_topk), -1, dtype=torch.long)
    pharm_multi_anchor_mask = torch.zeros((n, p, anchor_topk), dtype=torch.long)

    factory = load_factory()
    used_rdkit = Chem is not None and factory is not None
    n_with_feature = 0
    token_id_counter: Counter = Counter()
    row_unique_types: List[int] = []
    smiles_list = df[args.smiles_col].astype(str).tolist()
    token_mode_eff = str(args.token_mode)
    bucket_mode_eff = str(args.bucket_mode)

    def _encode_records(record_lists: List[List[Dict[str, object]]]):
        token_lists = [[(str(rec["token_key"]), int(rec["anchor"])) for rec in recs] for recs in record_lists]
        if str(args.vocab_mode) == "fixed64":
            token2id = {"__PAD__": 0}
            for fi, fam in enumerate(families):
                for bi, bucket in enumerate(LEGACY_ATOM_BUCKETS_9):
                    token2id[f"{fam}|{bucket}"] = fi * len(LEGACY_ATOM_BUCKETS_9) + bi + 1
            token_freq = {}
            num_types = int(1 + len(families) * len(LEGACY_ATOM_BUCKETS_9))
            other_id = 0
        else:
            token2id, token_freq = build_vocab(
                token_lists,
                min_token_freq=int(args.min_token_freq),
                max_vocab=int(args.max_vocab),
            )
            num_types = int(len(token2id))
            other_id = int(token2id.get("__OTHER__", 1))
        return token2id, token_freq, num_types, other_id

    if args.dedup:
        smi2u = {}
        uniq_smiles = []
        row2u = []
        for smi in smiles_list:
            uid = smi2u.get(smi, None)
            if uid is None:
                uid = len(uniq_smiles)
                smi2u[smi] = uid
                uniq_smiles.append(smi)
            row2u.append(uid)

        uniq_records = []
        for smi in _progress(uniq_smiles, desc="extract_pharm_records", total=len(uniq_smiles)):
            uniq_records.append(
                extract_pharm_records(
                    smi,
                    used_rdkit,
                    factory,
                    family2id,
                    p,
                    token_mode=token_mode_eff,
                    bucket_mode=bucket_mode_eff,
                    anchor_topk=anchor_topk,
                )
            )
        token2id, token_freq, num_types, other_id = _encode_records(uniq_records)
        uniq_has_feature = [1 if recs else 0 for recs in uniq_records]

        for i, uid in _progress(enumerate(row2u), desc="pack_rows", total=len(row2u)):
            cur = uniq_records[uid]
            if cur:
                n_with_feature += int(uniq_has_feature[uid])
            row_unique_types.append(int(len(set(int(token2id.get(str(rec["token_key"]), other_id)) for rec in cur))))
            for j, rec in enumerate(cur[:p]):
                if str(args.vocab_mode) == "fixed64":
                    tid = int(token2id.get(str(rec["token_key"]), 0))
                else:
                    tid = int(token2id.get(str(rec["token_key"]), other_id))
                pharm_type[i, j] = int(tid)
                pharm_mask[i, j] = 1
                pharm_anchor[i, j] = int(rec["anchor"])
                pharm_family[i, j] = int(rec["family_id"])
                pharm_center[i, j] = rec["center"]
                pharm_geom[i, j] = rec["geom"]
                pharm_context[i, j] = rec["context"]
                anchor_ids = [int(x) for x in rec["anchor_ids"][:anchor_topk]]
                if anchor_ids:
                    pharm_multi_anchor[i, j, : len(anchor_ids)] = torch.tensor(anchor_ids, dtype=torch.long)
                    pharm_multi_anchor_mask[i, j, : len(anchor_ids)] = 1
                token_id_counter[int(tid)] += 1
        n_unique_smiles = len(uniq_smiles)
    else:
        all_records = []
        for smi in _progress(smiles_list, desc="extract_pharm_records", total=len(smiles_list)):
            all_records.append(
                extract_pharm_records(
                    smi,
                    used_rdkit,
                    factory,
                    family2id,
                    p,
                    token_mode=token_mode_eff,
                    bucket_mode=bucket_mode_eff,
                    anchor_topk=anchor_topk,
                )
            )
        token2id, token_freq, num_types, other_id = _encode_records(all_records)
        for i, cur in _progress(enumerate(all_records), desc="pack_rows", total=len(all_records)):
            if cur:
                n_with_feature += 1
            row_unique_types.append(int(len(set(int(token2id.get(str(rec["token_key"]), other_id)) for rec in cur))))
            for j, rec in enumerate(cur[:p]):
                if str(args.vocab_mode) == "fixed64":
                    tid = int(token2id.get(str(rec["token_key"]), 0))
                else:
                    tid = int(token2id.get(str(rec["token_key"]), other_id))
                pharm_type[i, j] = int(tid)
                pharm_mask[i, j] = 1
                pharm_anchor[i, j] = int(rec["anchor"])
                pharm_family[i, j] = int(rec["family_id"])
                pharm_center[i, j] = rec["center"]
                pharm_geom[i, j] = rec["geom"]
                pharm_context[i, j] = rec["context"]
                anchor_ids = [int(x) for x in rec["anchor_ids"][:anchor_topk]]
                if anchor_ids:
                    pharm_multi_anchor[i, j, : len(anchor_ids)] = torch.tensor(anchor_ids, dtype=torch.long)
                    pharm_multi_anchor_mask[i, j, : len(anchor_ids)] = 1
                token_id_counter[int(tid)] += 1
        n_unique_smiles = len(set(smiles_list))

    valid_total = int(sum(int(v) for _, v in token_id_counter.items()))
    non_special = [(int(k), int(v)) for k, v in token_id_counter.items() if int(k) > 1]
    non_special.sort(key=lambda x: (-x[1], x[0]))
    top5 = non_special[:5]
    top5_cnt = int(sum(v for _, v in top5))
    top5_share = float(top5_cnt / valid_total) if valid_total > 0 else 0.0
    median_unique = float(torch.tensor(row_unique_types, dtype=torch.float32).median().item()) if row_unique_types else 0.0

    out = {
        "pharm_type": pharm_type,
        "pharm_mask": pharm_mask,
        "pharm_anchor": pharm_anchor,
        "pharm_family": pharm_family,
        "pharm_center": pharm_center,
        "pharm_geom": pharm_geom,
        "pharm_context": pharm_context,
        "pharm_multi_anchor": pharm_multi_anchor,
        "pharm_multi_anchor_mask": pharm_multi_anchor_mask,
        "family2id": family2id,
        "token2id": token2id,
        "num_types": int(num_types),
        "num_families": int(len(family2id)),
        "max_tokens": p,
        "num_rows": n,
        "dedup": bool(args.dedup),
        "n_unique_smiles": int(n_unique_smiles),
        "used_rdkit": used_rdkit,
        "rows_with_feature": n_with_feature,
        "token_mode": str(token_mode_eff),
        "bucket_mode": str(bucket_mode_eff),
        "vocab_mode": str(args.vocab_mode),
        "min_token_freq": int(args.min_token_freq),
        "max_vocab": int(args.max_vocab),
        "anchor_topk": int(anchor_topk),
        "geom_dim": int(len(PHARM_GEOM_NAMES)),
        "context_dim": int(len(PHARM_CONTEXT_NAMES)),
        "pharm_geom_names": list(PHARM_GEOM_NAMES),
        "pharm_context_names": list(PHARM_CONTEXT_NAMES),
        "token_freq": token_freq,
        "token_counts_after_merge": {int(k): int(v) for k, v in token_id_counter.items()},
        "median_unique_types_per_row": float(median_unique),
        "top5_token_share": float(top5_share),
        "family_priority": dict(FAMILY_PRIORITY),
    }
    out_pt = Path(args.out_pt)
    out_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_pt)
    print(f"[OK] saved: {out_pt}")

    if not args.no_mirror_processed:
        m_pt = mirror_processed_path(out_pt)
        if m_pt is not None and m_pt != out_pt:
            m_pt.parent.mkdir(parents=True, exist_ok=True)
            torch.save(out, m_pt)
            print(f"[OK] mirrored: {m_pt}")

    print(
        f"[INFO] used_rdkit={used_rdkit} rows_with_feature={n_with_feature}/{n} "
        f"dedup={bool(args.dedup)} unique_smiles={n_unique_smiles}"
    )
    print(
        f"[INFO] vocab(num_types)={num_types} vocab_mode={args.vocab_mode} "
        f"token_mode={token_mode_eff} bucket_mode={bucket_mode_eff} "
        f"min_token_freq={int(args.min_token_freq)} max_vocab={int(args.max_vocab)} "
        f"anchor_topk={anchor_topk}"
    )
    print(f"[INFO] median_unique_types_per_row={median_unique:.2f} top5_token_share={top5_share:.4f}")


if __name__ == "__main__":
    main()
