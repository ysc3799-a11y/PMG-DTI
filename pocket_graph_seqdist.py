#解析 P2Rank pocket、读取 PDB CA 坐标、kNN 构图与截断。

import csv
import os
from typing import Dict, List, Optional, Tuple, Any

import torch


def _read_csv_dicts(path: str) -> List[dict]:
    if not path or not os.path.isfile(path):
        return []
    rows: List[dict] = []
    with open(path, newline="", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        for row in reader:
            if not row:
                continue
            rows.append(row)
    return rows


def _parse_int(value: Any) -> Optional[int]:
    try:
        return int(str(value).strip())
    except Exception:
        return None


def _parse_float(value: Any) -> Optional[float]:
    try:
        return float(str(value).strip())
    except Exception:
        return None


def parse_residue_ids(residue_ids: str) -> List[Tuple[str, int]]:
    out: List[Tuple[str, int]] = []
    if not residue_ids:
        return out
    for tok in str(residue_ids).strip().split():
        if "_" not in tok:
            continue
        chain, idx = tok.rsplit("_", 1)
        rid = _parse_int(idx)
        if rid is None:
            continue
        chain = chain.strip() if chain.strip() else "A"
        out.append((chain, rid))
    return out


# Residue type features:
# - 20-d canonical amino-acid one-hot
# - 5-d physchem flags [hydrophobic, aromatic, positive, negative, polar_or_hbond_like]
AA3_LIST_20 = [
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
]
AA3_TO_IDX = {aa: i for i, aa in enumerate(AA3_LIST_20)}
AA_HYDROPHOBIC = {"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "PRO", "TYR", "CYS"}
AA_AROMATIC = {"PHE", "TRP", "TYR", "HIS"}
AA_POSITIVE = {"LYS", "ARG", "HIS"}
AA_NEGATIVE = {"ASP", "GLU"}
AA_POLAR_OR_HBOND = {"SER", "THR", "ASN", "GLN", "HIS", "LYS", "ARG", "ASP", "GLU", "TYR", "TRP", "CYS"}
POCKET_NODE_STRUCT_DIM = 28  # 20 aa one-hot + 1 pLDDT + 1 dist2center + 1 dist2raw + 1 local_density + 5 flags
POCKET_RBF_BINS_DEFAULT = 16
POCKET_RADIUS_CUTOFF_DEFAULT = 10.0
POCKET_RADIUS_MAX_NEIGHBORS_DEFAULT = 32


def _aa_onehot_20(resname: str) -> List[float]:
    aa = str(resname or "").strip().upper()
    out = [0.0] * 20
    idx = AA3_TO_IDX.get(aa, None)
    if idx is not None:
        out[int(idx)] = 1.0
    return out


def _aa_physchem_5(resname: str) -> List[float]:
    aa = str(resname or "").strip().upper()
    return [
        1.0 if aa in AA_HYDROPHOBIC else 0.0,
        1.0 if aa in AA_AROMATIC else 0.0,
        1.0 if aa in AA_POSITIVE else 0.0,
        1.0 if aa in AA_NEGATIVE else 0.0,
        1.0 if aa in AA_POLAR_OR_HBOND else 0.0,
    ]


def build_distance_rbf(
    edge_dist: torch.Tensor,
    num_bins: int = POCKET_RBF_BINS_DEFAULT,
    d_min: float = 0.0,
    d_max: float = 20.0,
) -> torch.Tensor:
    """
    Expand scalar distances into Gaussian RBF features.
    edge_dist: [E] or [E,1] distances in Angstrom.
    returns: [E, num_bins]
    """
    if edge_dist.dim() == 1:
        d = edge_dist.view(-1, 1).float()
    else:
        d = edge_dist[:, :1].float()
    e = int(d.shape[0])
    b = int(max(1, num_bins))
    if e == 0:
        return torch.zeros((0, b), dtype=torch.float32)
    lo = float(min(d_min, d_max))
    hi = float(max(d_min, d_max))
    if hi <= lo + 1e-6:
        hi = lo + 1.0
    mu = torch.linspace(lo, hi, steps=b, dtype=torch.float32, device=d.device).view(1, b)
    delta = float((hi - lo) / max(1, b - 1))
    gamma = 1.0 / max((delta ** 2), 1e-6)
    return torch.exp(-gamma * (d - mu) ** 2).float()


def build_edge_type_flags(edge_index: torch.Tensor, node_is_raw: torch.Tensor) -> torch.Tensor:
    """
    Build 3-way edge type one-hot:
      [raw-raw, raw-expanded, expanded-expanded]
    """
    if edge_index.numel() == 0:
        return torch.zeros((0, 3), dtype=torch.float32)
    if node_is_raw.dim() > 1:
        raw = node_is_raw.view(-1).bool()
    else:
        raw = node_is_raw.bool()
    src = edge_index[0].long()
    dst = edge_index[1].long()
    src_raw = raw.index_select(0, src)
    dst_raw = raw.index_select(0, dst)
    rr = (src_raw & dst_raw).float().view(-1, 1)
    re = (src_raw ^ dst_raw).float().view(-1, 1)
    ee = ((~src_raw) & (~dst_raw)).float().view(-1, 1)
    return torch.cat([rr, re, ee], dim=1).float()


def load_p2rank_pockets(
    predictions_csv: Optional[str],
    residues_csv: Optional[str],
    *,
    top_k: int = 5,
    prob_threshold: float = 0.0,
    min_residues: int = 5,
) -> List[Dict[str, Any]]:
    pockets: List[Dict[str, Any]] = []

    if predictions_csv and os.path.isfile(predictions_csv):
        rows = _read_csv_dicts(predictions_csv)
        if rows:
            header = {k.strip().lower(): k for k in rows[0].keys()}
            rank_key = header.get("rank")
            score_key = header.get("score")
            prob_key = header.get("probability") or header.get("prob")
            res_key = header.get("residue_ids")
            cx_key = header.get("center_x")
            cy_key = header.get("center_y")
            cz_key = header.get("center_z")

            for row in rows:
                res_ids = parse_residue_ids(row.get(res_key, "")) if res_key else []
                if len(res_ids) < int(min_residues):
                    continue
                prob = _parse_float(row.get(prob_key, "")) if prob_key else None
                if prob_threshold > 0 and prob is not None and prob < float(prob_threshold):
                    continue
                rank = _parse_int(row.get(rank_key, "")) if rank_key else None
                score = _parse_float(row.get(score_key, "")) if score_key else None
                center = None
                if cx_key and cy_key and cz_key:
                    cx = _parse_float(row.get(cx_key, ""))
                    cy = _parse_float(row.get(cy_key, ""))
                    cz = _parse_float(row.get(cz_key, ""))
                    if cx is not None and cy is not None and cz is not None:
                        center = (cx, cy, cz)
                pockets.append(
                    {
                        "rank": rank if rank is not None else len(pockets) + 1,
                        "score": score if score is not None else 0.0,
                        "prob": prob if prob is not None else 0.0,
                        "center": center,
                        "residues": res_ids,
                    }
                )
            pockets.sort(key=lambda x: int(x.get("rank", 1)))

    if not pockets and residues_csv and os.path.isfile(residues_csv):
        rows = _read_csv_dicts(residues_csv)
        if rows:
            header = {k.strip().lower(): k for k in rows[0].keys()}
            pocket_key = header.get("pocket") or header.get("pocket_id") or header.get("site")
            res_key = (
                header.get("residue_label")
                or header.get("residue")
                or header.get("residue_id")
                or header.get("residue_index")
            )
            prob_key = header.get("probability") or header.get("prob") or header.get("score")
            if pocket_key and res_key:
                grouped: Dict[int, List[int]] = {}
                for row in rows:
                    pid = _parse_int(row.get(pocket_key, ""))
                    rid = _parse_int(row.get(res_key, ""))
                    # Some P2Rank residue CSVs use 0-based pocket ids.
                    # Keep pid == 0 valid; only reject negative/invalid ids.
                    if pid is None or rid is None or pid < 0:
                        continue
                    if prob_key and prob_threshold > 0:
                        prob = _parse_float(row.get(prob_key, ""))
                        if prob is None or prob < float(prob_threshold):
                            continue
                    grouped.setdefault(pid, []).append(rid)
                for pid in sorted(grouped.keys()):
                    rlist = grouped[pid]
                    if len(rlist) < int(min_residues):
                        continue
                    residues = [("A", int(r)) for r in rlist]
                    rank = int(pid) + 1  # normalize to 1-based rank for downstream sorting
                    pockets.append(
                        {
                            "rank": rank,
                            "score": 0.0,
                            "prob": 0.0,
                            "center": None,
                            "residues": residues,
                        }
                    )

    if top_k is not None:
        pockets = pockets[: int(max(1, top_k))]
    return pockets


def parse_pdb_ca_records(pdb_path: str) -> Dict[Tuple[str, int], Dict[str, Any]]:
    """
    Parse CA atom records from pdb and keep residue-level metadata.
    Returns:
      {(chain, rid): {"coord": (x,y,z), "resname": "ALA", "plddt": 83.2}}
    """
    rec: Dict[Tuple[str, int], Dict[str, Any]] = {}
    if not pdb_path or not os.path.isfile(pdb_path):
        return rec
    try:
        with open(pdb_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if not line.startswith("ATOM"):
                    continue
                atom = line[12:16].strip()
                if atom != "CA":
                    continue
                chain = line[21].strip() or "A"
                resseq = line[22:26].strip()
                if not resseq:
                    continue
                try:
                    rid = int(resseq)
                except Exception:
                    continue
                x = _parse_float(line[30:38])
                y = _parse_float(line[38:46])
                z = _parse_float(line[46:54])
                if x is None or y is None or z is None:
                    continue
                resname = str(line[17:20].strip() or "UNK").upper()
                bfac = _parse_float(line[60:66])
                plddt = float(bfac) if bfac is not None else 0.0
                key = (chain, rid)
                if key not in rec:
                    rec[key] = {
                        "coord": (float(x), float(y), float(z)),
                        "resname": resname,
                        "plddt": plddt,
                    }
    except Exception:
        return rec
    return rec


def parse_pdb_ca_coords(pdb_path: str) -> Dict[Tuple[str, int], Tuple[float, float, float]]:
    coords: Dict[Tuple[str, int], Tuple[float, float, float]] = {}
    rec = parse_pdb_ca_records(pdb_path)
    for k, v in rec.items():
        xyz = v.get("coord", None)
        if xyz is None:
            continue
        coords[k] = (float(xyz[0]), float(xyz[1]), float(xyz[2]))
    return coords


def build_knn_edges(xyz: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    n = int(xyz.shape[0])
    if n <= 1:
        return torch.zeros((2, 0), dtype=torch.long), torch.zeros((0, 1), dtype=torch.float32)
    k = int(max(1, k))
    k = min(k, n - 1)
    dist = torch.cdist(xyz, xyz, p=2)  # [N, N]
    src_list: List[torch.Tensor] = []
    dst_list: List[torch.Tensor] = []
    dist_list: List[torch.Tensor] = []
    for i in range(n):
        di = dist[i]
        idx = torch.topk(di, k=min(k + 1, n), largest=False).indices
        idx = idx[idx != i][:k]
        if idx.numel() == 0:
            continue
        src_list.append(torch.full((idx.numel(),), i, dtype=torch.long))
        dst_list.append(idx.long())
        dist_list.append(di[idx].view(-1, 1))
    if not src_list:
        return torch.zeros((2, 0), dtype=torch.long), torch.zeros((0, 1), dtype=torch.float32)
    src = torch.cat(src_list, dim=0)
    dst = torch.cat(dst_list, dim=0)
    edge_index = torch.stack([src, dst], dim=0)
    edge_dist = torch.cat(dist_list, dim=0).float()
    return edge_index, edge_dist


def build_radius_edges(
    xyz: torch.Tensor,
    radius: float = POCKET_RADIUS_CUTOFF_DEFAULT,
    max_neighbors: int = POCKET_RADIUS_MAX_NEIGHBORS_DEFAULT,
) -> Tuple[torch.Tensor, torch.Tensor]:
    n = int(xyz.shape[0])
    if n <= 1:
        return torch.zeros((2, 0), dtype=torch.long), torch.zeros((0, 1), dtype=torch.float32)
    radius = float(max(1e-3, radius))
    max_neighbors = int(max(1, max_neighbors))
    dist = torch.cdist(xyz, xyz, p=2)  # [N, N]
    src_list: List[torch.Tensor] = []
    dst_list: List[torch.Tensor] = []
    dist_list: List[torch.Tensor] = []
    for i in range(n):
        di = dist[i]
        mask = (di <= radius)
        mask[i] = False
        idx = torch.nonzero(mask, as_tuple=False).view(-1)
        if idx.numel() == 0:
            continue
        dsel = di.index_select(0, idx)
        order = torch.argsort(dsel, dim=0)
        idx = idx.index_select(0, order[:max_neighbors])
        dsel = dsel.index_select(0, order[:max_neighbors])
        src_list.append(torch.full((idx.numel(),), i, dtype=torch.long))
        dst_list.append(idx.long())
        dist_list.append(dsel.view(-1, 1))
    if not src_list:
        return build_knn_edges(xyz, k=min(16, max(1, n - 1)))
    src = torch.cat(src_list, dim=0)
    dst = torch.cat(dst_list, dim=0)
    edge_index = torch.stack([src, dst], dim=0)
    edge_dist = torch.cat(dist_list, dim=0).float()
    return edge_index, edge_dist


def build_pocket_templates(
    pockets: List[Dict[str, Any]],
    coord_map: Dict[Tuple[str, int], Tuple[float, float, float]],
    seq_len: int,
    *,
    top_k: int = 5,
    max_nodes: int = 128,
    knn_k: int = 16,
    expand_min_nodes: int = 0,
    seqdist_norm: str = "log1p",
    seqdist_norm_denom: float = 128.0,
    resname_map: Optional[Dict[Tuple[str, int], str]] = None,
    plddt_map: Optional[Dict[Tuple[str, int], float]] = None,
    edge_rbf_bins: int = POCKET_RBF_BINS_DEFAULT,
    edge_rbf_dmin: float = 0.0,
    edge_rbf_dmax: float = 20.0,
    raw_only: bool = False,
    edge_mode: str = "radius",
    radius_cutoff: float = POCKET_RADIUS_CUTOFF_DEFAULT,
    radius_max_neighbors: int = POCKET_RADIUS_MAX_NEIGHBORS_DEFAULT,
) -> List[Optional[Dict[str, Any]]]:
    templates: List[Optional[Dict[str, Any]]] = []
    seq_len = int(max(0, seq_len))
    raw_only = bool(raw_only)
    edge_mode = str(edge_mode or "radius").lower()
    for pocket in pockets[: int(max(1, top_k))]:
        residues = pocket.get("residues", [])
        # rid -> (chain, coord, key, is_raw) with chain-A priority
        selected: Dict[int, Tuple[str, Tuple[float, float, float], Tuple[str, int], bool]] = {}
        seen = set()
        for chain, rid in residues:
            if rid in seen:
                continue
            if seq_len > 0 and (rid < 1 or rid > seq_len):
                continue
            key = (chain, rid)
            coord = coord_map.get(key)
            if coord is None and chain != "A":
                key = ("A", rid)
                coord = coord_map.get(key)
            if coord is None:
                continue
            seen.add(rid)
            selected[int(rid)] = (str(key[0]), coord, (str(key[0]), int(rid)), True)

        # Expand small pockets to at least expand_min_nodes using nearest 3D neighbors
        # around pocket center.
        expand_n = 0 if raw_only else int(max(0, expand_min_nodes))
        if expand_n > 0 and len(selected) < expand_n:
            avail: Dict[int, Tuple[str, Tuple[float, float, float], Tuple[str, int], bool]] = {}
            for (ch, rid), xyz in coord_map.items():
                rid_i = int(rid)
                if seq_len > 0 and (rid_i < 1 or rid_i > seq_len):
                    continue
                prev = avail.get(rid_i, None)
                if prev is None or (ch == "A" and prev[0] != "A"):
                    avail[rid_i] = (str(ch), xyz, (str(ch), rid_i), False)

            if selected:
                center = pocket.get("center", None)
                if center is not None:
                    center_t = torch.tensor(center, dtype=torch.float32)
                else:
                    center_t = torch.tensor([selected[r][1] for r in selected.keys()], dtype=torch.float32).mean(dim=0)

                cand: List[Tuple[float, int]] = []
                for rid, (_, xyz, _, _) in avail.items():
                    if rid in selected:
                        continue
                    xyz_t = torch.tensor(xyz, dtype=torch.float32)
                    d2 = float(((xyz_t - center_t) ** 2).sum().item())
                    cand.append((d2, rid))
                cand.sort(key=lambda x: (x[0], x[1]))

                for _, rid in cand:
                    selected[rid] = avail[rid]
                    if len(selected) >= expand_n:
                        break

        if not selected:
            templates.append(None)
            continue

        res_seq = sorted(selected.keys())
        coords: List[Tuple[float, float, float]] = [selected[r][1] for r in res_seq]
        node_keys: List[Tuple[str, int]] = [selected[r][2] for r in res_seq]
        node_is_raw_list: List[bool] = [bool(selected[r][3]) for r in res_seq]
        xyz = torch.tensor(coords, dtype=torch.float32)
        center = pocket.get("center", None)
        if center is None:
            center_t = xyz.mean(dim=0)
        else:
            center_t = torch.tensor(center, dtype=torch.float32)
        if max_nodes and int(xyz.shape[0]) > int(max_nodes):
            d = ((xyz - center_t) ** 2).sum(dim=1)
            idx = torch.topk(d, k=int(max_nodes), largest=False).indices
            xyz = xyz.index_select(0, idx)
            res_seq = [res_seq[int(i)] for i in idx.cpu().tolist()]
            node_keys = [node_keys[int(i)] for i in idx.cpu().tolist()]
            node_is_raw_list = [node_is_raw_list[int(i)] for i in idx.cpu().tolist()]

        # Build node-level structural attributes.
        n_nodes = int(xyz.shape[0])
        aa_onehot = torch.zeros((n_nodes, 20), dtype=torch.float32)
        aa_phys = torch.zeros((n_nodes, 5), dtype=torch.float32)
        plddt = torch.zeros((n_nodes, 1), dtype=torch.float32)
        for i, key in enumerate(node_keys):
            resname = ""
            if resname_map is not None:
                resname = str(resname_map.get(key, ""))
            aa_onehot[i] = torch.tensor(_aa_onehot_20(resname), dtype=torch.float32)
            aa_phys[i] = torch.tensor(_aa_physchem_5(resname), dtype=torch.float32)
            if plddt_map is not None:
                pv = float(plddt_map.get(key, 0.0))
                plddt[i, 0] = float(max(0.0, min(100.0, pv)) / 100.0)

        dist2center = torch.norm(xyz - center_t.view(1, 3), p=2, dim=1, keepdim=True)
        dist2center = (dist2center / 20.0).clamp(min=0.0, max=5.0)

        raw_coords: List[Tuple[float, float, float]] = []
        raw_seen = set()
        for ch, rid in residues:
            rid_i = int(rid)
            if seq_len > 0 and (rid_i < 1 or rid_i > seq_len):
                continue
            key = (str(ch), rid_i)
            xyz_i = coord_map.get(key, None)
            if xyz_i is None and str(ch) != "A":
                key = ("A", rid_i)
                xyz_i = coord_map.get(key, None)
            if xyz_i is None or key in raw_seen:
                continue
            raw_seen.add(key)
            raw_coords.append((float(xyz_i[0]), float(xyz_i[1]), float(xyz_i[2])))
        if raw_coords:
            raw_xyz = torch.tensor(raw_coords, dtype=torch.float32)
            dist2raw = torch.cdist(xyz, raw_xyz, p=2).min(dim=1).values.view(-1, 1)
            dist2raw = (dist2raw / 20.0).clamp(min=0.0, max=5.0)
        else:
            dist2raw = torch.zeros((n_nodes, 1), dtype=torch.float32)

        if n_nodes > 0:
            dmat = torch.cdist(xyz, xyz, p=2)
            neigh = (dmat <= 8.0).float().sum(dim=1, keepdim=True) - 1.0
            local_density = neigh / float(max(1, n_nodes - 1))
            local_density = local_density.clamp(min=0.0, max=1.0)
        else:
            local_density = torch.zeros((0, 1), dtype=torch.float32)

        node_attr = torch.cat([aa_onehot, plddt, dist2center, dist2raw, local_density, aa_phys], dim=1).float()
        if int(node_attr.shape[1]) != int(POCKET_NODE_STRUCT_DIM):
            # Keep shape contract stable for downstream model code.
            pad = int(POCKET_NODE_STRUCT_DIM) - int(node_attr.shape[1])
            if pad > 0:
                node_attr = torch.cat([node_attr, torch.zeros((n_nodes, pad), dtype=torch.float32)], dim=1)
            else:
                node_attr = node_attr[:, : int(POCKET_NODE_STRUCT_DIM)]

        node_is_raw = torch.tensor(node_is_raw_list, dtype=torch.bool)
        if edge_mode == "radius":
            edge_index, edge_dist = build_radius_edges(
                xyz,
                radius=float(radius_cutoff),
                max_neighbors=int(radius_max_neighbors),
            )
        else:
            edge_index, edge_dist = build_knn_edges(xyz, k=knn_k)
        edge_rbf = build_distance_rbf(
            edge_dist,
            num_bins=int(edge_rbf_bins),
            d_min=float(edge_rbf_dmin),
            d_max=float(edge_rbf_dmax),
        )
        edge_type = build_edge_type_flags(edge_index, node_is_raw)
        templates.append(
            {
                "res_seq": res_seq,
                "edge_index": edge_index,
                "edge_dist": edge_dist,
                "edge_rbf": edge_rbf,
                "edge_type": edge_type,
                "node_is_raw": node_is_raw.float().view(-1, 1),
                "node_attr": node_attr,
                "score": float(pocket.get("prob", pocket.get("score", 0.0)) or 0.0),
                "raw_count": int(node_is_raw.sum().item()),
            }
        )
    while len(templates) < int(max(1, top_k)):
        templates.append(None)
    return templates
