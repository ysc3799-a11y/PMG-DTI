import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

try:
    from rdkit import Chem
    from rdkit.Chem import rdDepictor
except Exception:
    Chem = None
    rdDepictor = None

_THIS = os.path.abspath(__file__)
_ROOT = os.path.dirname(_THIS)
for _p in (_ROOT, os.path.join(_ROOT, "train")):
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)

from dataset.dataset_canon_real_pocket_seqdist import (  # noqa: E402
    HumanDTIDatasetCanonRealPocket,
    collate_fn_real_pocket,
)
from pocket_graph_seqdist import parse_pdb_ca_records  # noqa: E402
from train.train_gated_canon_real_pocket_seqdist_3view_soft_pocket_bias_emaes import (  # noqa: E402
    build_model,
    infer_num_pharm_types,
    load_yaml,
    seed_all,
)


def safe_torch_load(path: str, map_location: str = "cpu"):
    try:
        return torch.load(path, weights_only=True, map_location=map_location)
    except TypeError:
        return torch.load(path, map_location=map_location)
    except Exception:
        return torch.load(path, map_location=map_location)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def abs_project_path(path: Optional[str]) -> Optional[str]:
    if not path:
        return path
    if os.path.isabs(path):
        return os.path.normpath(path)
    return os.path.normpath(os.path.join(_ROOT, path))


def move_batch_to_device(batch: Dict[str, Any], device: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.to(device)
        else:
            out[key] = value
    return out


def to_python(obj: Any):
    if isinstance(obj, dict):
        return {str(k): to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_python(v) for v in obj]
    if isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return obj.item()
        return obj.detach().cpu().tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (float, int, str, bool)) or obj is None:
        return obj
    try:
        if hasattr(obj, "item"):
            return obj.item()
    except Exception:
        pass
    return str(obj)


def write_rows_csv(path: str, rows: List[Dict[str, Any]]):
    if not rows:
        with open(path, "w", newline="", encoding="utf-8") as f:
            f.write("")
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: to_python(v) for k, v in row.items()})


def write_matrix_csv(path: str, matrix: torch.Tensor, row_labels: List[str], col_labels: List[str]):
    matrix = matrix.detach().cpu()
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([""] + list(col_labels))
        n_rows = int(matrix.shape[0])
        n_cols = int(matrix.shape[1]) if matrix.dim() >= 2 else 0
        for i in range(n_rows):
            values = [float(matrix[i, j].item()) for j in range(n_cols)]
            label = row_labels[i] if i < len(row_labels) else f"row_{i}"
            writer.writerow([label] + values)


def sigmoid_scalar(x: float) -> float:
    x = max(min(float(x), 50.0), -50.0)
    return 1.0 / (1.0 + math.exp(-x))


def attn_pool_weights(pool, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if mask.dtype != torch.bool:
        mask = mask > 0
    score = pool.scorer(x).squeeze(-1)
    score = score.masked_fill(~mask, float("-inf"))
    alpha = torch.softmax(score, dim=1)
    alpha = torch.where(torch.isfinite(alpha), alpha, torch.zeros_like(alpha))
    alpha = alpha * mask.float()
    alpha = alpha / alpha.sum(dim=1, keepdim=True).clamp(min=1e-8)
    return score, alpha


def mean_pool_weights(mask: torch.Tensor) -> torch.Tensor:
    if mask.dtype != torch.bool:
        mask = mask > 0
    alpha = mask.float()
    alpha = alpha / alpha.sum(dim=1, keepdim=True).clamp(min=1e-8)
    return alpha


def select_top_pairs(
    pair_score: torch.Tensor,
    top_t: int,
    max_per_q: int,
    max_per_r: int,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if pair_score.numel() == 0 or int(top_t) <= 0:
        return out
    _m, n = pair_score.shape
    flat_score = pair_score.reshape(-1)
    order = torch.argsort(flat_score, descending=True)
    selected: List[int] = []
    q_counts: Dict[int, int] = {}
    r_counts: Dict[int, int] = {}
    for oid in order.tolist():
        if len(selected) >= int(top_t):
            break
        i = int(oid // n)
        j = int(oid % n)
        if int(max_per_q) > 0 and q_counts.get(i, 0) >= int(max_per_q):
            continue
        if int(max_per_r) > 0 and r_counts.get(j, 0) >= int(max_per_r):
            continue
        selected.append(int(oid))
        q_counts[i] = q_counts.get(i, 0) + 1
        r_counts[j] = r_counts.get(j, 0) + 1
    for rank, oid in enumerate(selected, start=1):
        i = int(oid // n)
        j = int(oid % n)
        raw_score = float(flat_score[oid].item())
        out.append(
            {
                "local_rank": int(rank),
                "flat_index": int(oid),
                "q_local_index": int(i),
                "r_local_index": int(j),
                "pair_score_logit": raw_score,
                "pair_score_prob": sigmoid_scalar(raw_score),
            }
        )
    return out


def find_residue_record(rec_map: Dict[Tuple[str, int], Dict[str, Any]], rid: int) -> Tuple[str, Dict[str, Any]]:
    key = ("A", int(rid))
    if key in rec_map:
        return "A", rec_map[key]
    for (chain, rid_i), rec in rec_map.items():
        if int(rid_i) == int(rid):
            return str(chain), rec
    return "", {}


def mol_2d_rows(smiles: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Optional[str]]:
    if Chem is None:
        return [], [], "RDKit is unavailable; skipped atom/bond export."
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [], [], f"RDKit failed to parse SMILES: {smiles}"
    mol = Chem.RemoveHs(mol)
    if rdDepictor is not None:
        try:
            rdDepictor.Compute2DCoords(mol)
        except Exception:
            pass
    conf = mol.GetConformer() if mol.GetNumConformers() > 0 else None
    atom_rows: List[Dict[str, Any]] = []
    for atom in mol.GetAtoms():
        idx = int(atom.GetIdx())
        row = {
            "atom_index": idx,
            "symbol": str(atom.GetSymbol()),
            "atomic_num": int(atom.GetAtomicNum()),
            "formal_charge": int(atom.GetFormalCharge()),
            "is_aromatic": int(atom.GetIsAromatic()),
        }
        if conf is not None:
            pos = conf.GetAtomPosition(idx)
            row["x2d"] = float(pos.x)
            row["y2d"] = float(pos.y)
        atom_rows.append(row)
    bond_rows: List[Dict[str, Any]] = []
    for bond in mol.GetBonds():
        bond_rows.append(
            {
                "bond_index": int(bond.GetIdx()),
                "begin_atom_index": int(bond.GetBeginAtomIdx()),
                "end_atom_index": int(bond.GetEndAtomIdx()),
                "bond_type": str(bond.GetBondType()),
                "is_aromatic": int(bond.GetIsAromatic()),
                "is_conjugated": int(bond.GetIsConjugated()),
            }
        )
    return atom_rows, bond_rows, None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="checkpoint path, usually best_auc.pt")
    p.add_argument("--cfg", default=None, help="optional yaml config; defaults to cfg stored in checkpoint")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--sample_index", type=int, default=0, help="0-based index within the chosen split")
    p.add_argument("--row_index", type=int, default=None, help="original CSV row index")
    p.add_argument("--protein_id", default=None, help="select the first sample in split with this Protein_ID")
    p.add_argument("--device", default=None)
    p.add_argument("--out_dir", default=None, help="default: <repo_root>/explanations/<sample_tag>")
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def resolve_cfg(ckpt_payload: Dict[str, Any], cfg_path: Optional[str]) -> Dict[str, Any]:
    if cfg_path:
        return load_yaml(abs_project_path(cfg_path))
    cfg = ckpt_payload.get("cfg", None)
    if isinstance(cfg, dict):
        return cfg
    raise ValueError("Checkpoint does not contain cfg. Please provide --cfg.")


def build_dataset_from_cfg(cfg: Dict[str, Any], split: str):
    data = cfg.get("DATA", {})
    model_cfg = cfg.get("MODEL", {})
    csv_path = abs_project_path(data.get("CSV", "data/human/full_with_pid.csv"))
    split_path = abs_project_path(data.get("SPLIT", "data/human/splits/human_random_seed42.json"))
    pharm_path = abs_project_path(data.get("PHARM", "data/human/precomputed/pharm_max64_seed42.pt"))
    esm2_path = abs_project_path(data.get("ESM2", "data/human/precomputed/esm2_tokens_t6_fp16.pt"))
    p2rank_work_dir = abs_project_path(data.get("P2RANK_WORK_DIR", ""))
    p2rank_csv = abs_project_path(data.get("PROTEIN_POCKETS_CSV", "")) if data.get("PROTEIN_POCKETS_CSV") else None
    pdb_index_csv = abs_project_path(data.get("PDB_INDEX_CSV", "")) if data.get("PDB_INDEX_CSV") else None
    p2rank_out_dir = abs_project_path(data.get("P2RANK_OUT_DIR", "")) if data.get("P2RANK_OUT_DIR") else None
    pocket_cache_dir = abs_project_path(data.get("POCKET_CACHE_DIR", "")) if data.get("POCKET_CACHE_DIR") else None
    drug_graph_cache_path = abs_project_path(data.get("DRUG_GRAPH_CACHE", "")) if data.get("DRUG_GRAPH_CACHE") else None
    use_pocket_model = bool(int(model_cfg.get("USE_POCKET", 1)))
    if use_pocket_model:
        p2rank_work_dir_ds = p2rank_work_dir if p2rank_work_dir and os.path.isdir(p2rank_work_dir) else None
        p2rank_csv_ds = p2rank_csv
        pdb_index_csv_ds = pdb_index_csv
        p2rank_out_dir_ds = p2rank_out_dir
        pocket_cache_dir_ds = pocket_cache_dir
        drop_no_pocket_ds = bool(int(data.get("DROP_NO_POCKET", 0)))
    else:
        p2rank_work_dir_ds = None
        p2rank_csv_ds = None
        pdb_index_csv_ds = None
        p2rank_out_dir_ds = None
        pocket_cache_dir_ds = None
        drop_no_pocket_ds = False

    ds_common = dict(
        csv_path=csv_path,
        split_path=split_path,
        pharm_path=pharm_path,
        esm2_path=esm2_path,
        mask_special_tokens=bool(int(data.get("MASK_SPECIAL_TOKENS", 1))),
        p2rank_work_dir=p2rank_work_dir_ds,
        protein_pockets_csv=p2rank_csv_ds,
        pdb_index_csv=pdb_index_csv_ds,
        p2rank_out_dir=p2rank_out_dir_ds,
        protein_id_col=str(data.get("PROTEIN_ID_COL", "Protein_ID")),
        pocket_cache_dir=pocket_cache_dir_ds,
        pocket_topk=int(data.get("POCKET_TOPK", 3)),
        pocket_max_nodes=int(data.get("POCKET_MAX_NODES", 96)),
        pocket_knn_k=int(data.get("POCKET_KNN_K", 16)),
        pocket_expand_min_nodes=int(data.get("POCKET_EXPAND_MIN_NODES", 0)),
        pocket_seqdist_norm=str(data.get("POCKET_SEQDIST_NORM", "log1p")),
        pocket_seqdist_norm_denom=float(data.get("POCKET_SEQDIST_NORM_DENOM", 128.0)),
        pocket_prob_thres=float(data.get("POCKET_PROB_THRES", 0.0)),
        pocket_min_residues=int(data.get("POCKET_MIN_RESIDUES", 5)),
        drop_no_pocket=drop_no_pocket_ds,
        pocket_use_struct_feat=bool(int(data.get("POCKET_USE_STRUCT_FEAT", data.get("POCKET_STRUCT_FEAT", 1)))),
        pocket_struct_dim=int(data.get("POCKET_STRUCT_DIM", 29)),
        pocket_edge_rbf_bins=int(data.get("POCKET_EDGE_RBF_BINS", 16)),
        pocket_edge_use_seqdist=bool(int(data.get("POCKET_EDGE_USE_SEQDIST", 1))),
        pocket_raw_only=bool(int(data.get("POCKET_RAW_ONLY", 1))),
        pocket_edge_mode=str(data.get("POCKET_EDGE_MODE", "radius")),
        pocket_radius_cutoff=float(data.get("POCKET_RADIUS_CUTOFF", 10.0)),
        pocket_radius_max_neighbors=int(data.get("POCKET_RADIUS_MAX_NEIGHBORS", 32)),
        drug_graph_cache_path=drug_graph_cache_path,
    )
    ds = HumanDTIDatasetCanonRealPocket(split=split, **ds_common)
    return ds, csv_path, pharm_path, use_pocket_model


def resolve_sample_position(
    ds: HumanDTIDatasetCanonRealPocket,
    sample_index: int,
    row_index: Optional[int],
    protein_id: Optional[str],
) -> int:
    if row_index is not None:
        ridx = int(row_index)
        for pos, cur in enumerate(ds.indices):
            if int(cur) == ridx:
                return int(pos)
        raise IndexError(f"row_index={row_index} not found in split.")
    if protein_id:
        target = str(protein_id).strip()
        matches = []
        for pos, ridx in enumerate(ds.indices):
            row = ds.df.iloc[int(ridx)]
            pid = str(row.get(ds.protein_id_col, "")).strip().split("_len=")[0]
            if pid == target:
                matches.append(int(pos))
        if not matches:
            raise IndexError(f"Protein_ID={protein_id} not found in split.")
        return matches[0]
    pos = int(sample_index)
    if pos < 0 or pos >= len(ds):
        raise IndexError(f"sample_index={sample_index} out of range for split size {len(ds)}.")
    return pos


def build_model_from_cfg(cfg: Dict[str, Any], pharm_path: str, ds: HumanDTIDatasetCanonRealPocket):
    num_pharm_types = infer_num_pharm_types(pharm_path)
    node_in = int(ds[0][0]["node_feat"].shape[1]) if len(ds) > 0 else int(cfg.get("MODEL", {}).get("NODE_IN", 74))
    esm_dim = int(ds.esm_dim)
    model = build_model(
        cfg,
        num_pharm_types=num_pharm_types,
        esm_dim=esm_dim,
        node_in=node_in,
        use_multiview=bool(int(cfg.get("MODEL", {}).get("USE_MULTIVIEW", 0))),
    )
    return model


def main():
    args = parse_args()
    ckpt_path = abs_project_path(args.ckpt)
    ckpt = safe_torch_load(ckpt_path, map_location="cpu")
    cfg = resolve_cfg(ckpt, args.cfg)
    solver = cfg.get("SOLVER", {})
    seed = int(args.seed if args.seed is not None else solver.get("SEED", ckpt.get("seed", 42)))
    seed_all(seed)

    ds, csv_path, pharm_path, use_pocket_model = build_dataset_from_cfg(cfg, args.split)
    sample_pos = resolve_sample_position(ds, args.sample_index, args.row_index, args.protein_id)
    ridx = int(ds.indices[sample_pos])
    row = ds.df.iloc[ridx]
    pid = str(row.get(ds.protein_id_col, "")).strip().split("_len=")[0]
    smiles = str(row.get("SMILES", ""))
    sample_tag = f"{args.split}_sample{sample_pos:04d}_row{ridx:06d}_{pid or 'nopid'}"
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else os.path.join(_ROOT, "explanations", sample_tag)
    ensure_dir(out_dir)

    device = str(args.device if args.device is not None else solver.get("DEVICE", "cuda:0"))
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    model = build_model_from_cfg(cfg, pharm_path, ds)
    load_warning = None
    try:
        load_info = model.load_state_dict(ckpt["model"], strict=True)
    except RuntimeError as e:
        load_info = model.load_state_dict(ckpt["model"], strict=False)
        load_warning = str(e)
    model = model.to(device)
    model.eval()

    pharm_payload = safe_torch_load(pharm_path, map_location="cpu")
    token2id = pharm_payload.get("token2id", {}) if isinstance(pharm_payload, dict) else {}
    id2token = {int(v): str(k) for k, v in token2id.items()} if isinstance(token2id, dict) else {}

    item_cpu, y_cpu = ds[sample_pos]
    batch_cpu, y_batch = collate_fn_real_pocket([(item_cpu, y_cpu)], include_pocket=use_pocket_model)
    batch = move_batch_to_device(batch_cpu, device)
    y_true = float(y_batch.view(-1)[0].item())

    with torch.no_grad():
        logit = model(batch).view(-1)[0]
        prob = float(torch.sigmoid(logit).item())

        drug_node_repr, z_graph = model._run_drug(batch)
        pharm_tokens, z_pharm = model._run_pharm(batch, drug_node_repr)
        esm_tok = batch["esm2_tok"] if "esm2_tok" in batch else batch["esm2_tokens"]
        esm_mask = batch["esm2_mask"]
        esm_tok_proj = model.esm_proj(esm_tok) if getattr(model, "esm_proj", None) is not None else esm_tok
        soft_pocket_bias = None
        if hasattr(model, "_build_soft_pocket_bias"):
            try:
                soft_pocket_bias = model._build_soft_pocket_bias(batch, esm_tok_proj, esm_mask)
            except Exception:
                soft_pocket_bias = None
        z_esm = model._run_protein(batch, pharm_tokens=pharm_tokens)
        h_global = model._build_h_global(z_graph, z_pharm, z_esm)

        if str(getattr(model, "pharm_pool_mode", "attn")).lower() == "attn":
            pharm_pool_score, pharm_pool_alpha = attn_pool_weights(model.pharm_pool_attn, pharm_tokens, batch["pharm_mask"])
        else:
            pharm_pool_score = torch.zeros_like(batch["pharm_mask"], dtype=pharm_tokens.dtype)
            pharm_pool_alpha = mean_pool_weights(batch["pharm_mask"]).to(dtype=pharm_tokens.dtype)

        graph_batch = torch.zeros((0,), dtype=torch.long, device=device)
        num_graphs = 0
        pocket_node_repr = torch.zeros((0, model.view_dim), dtype=h_global.dtype, device=device)
        pocket_graph_repr = torch.zeros((0, model.view_dim), dtype=h_global.dtype, device=device)
        if use_pocket_model and "pocket_node_feat" in batch and batch["pocket_node_feat"].numel() > 0 and getattr(model, "pocket_enc", None) is not None:
            px = batch["pocket_node_feat"]
            pei = batch["pocket_edge_index"]
            pef = batch["pocket_edge_feat"]
            if not getattr(model, "use_sequence_branch", True):
                if (not getattr(model, "pocket_use_struct_feat", False)) or int(px.shape[1]) <= int(model.esm_dim):
                    px = None
                else:
                    px = px.clone()
                    px[:, : int(model.esm_dim)] = 0.0
            if px is not None:
                if getattr(model, "pocket_edge_proj", None) is not None and pef.numel() > 0:
                    pef = model.pocket_edge_proj(pef)
                graph_batch, num_graphs = model._get_pocket_batch_index(batch)
                pocket_node_repr, pocket_graph_repr = model.pocket_enc(px, pei, pef, graph_batch, num_graphs=num_graphs)
                pocket_node_repr = model.pocket_node_align(pocket_node_repr)
                pocket_graph_repr = model.pocket_graph_align(pocket_graph_repr)

    valid_token_mask = item_cpu["pharm_mask"] > 0
    valid_token_indices = torch.nonzero(valid_token_mask, as_tuple=False).view(-1).tolist()
    pharm_best_local = torch.zeros((len(valid_token_indices),), dtype=torch.float32)

    seq_len = int((item_cpu["esm2_mask"] > 0).sum().item())
    templates = ds._get_templates(pid, seq_len) if pid else []
    pocket_meta = ds._pocket_meta_cache.get(pid, [])
    pdb_path = ds.pdb_index.get(pid, "")
    rec_map = ds._pdb_record_cache.get(pdb_path, {})
    if not rec_map and pdb_path:
        rec_map = parse_pdb_ca_records(pdb_path)

    tok_pos = torch.nonzero(item_cpu["esm2_mask"] > 0, as_tuple=False).view(-1)
    pocket_rows: List[Dict[str, Any]] = []
    pharm_rows: List[Dict[str, Any]] = []
    residue_rows_all: List[Dict[str, Any]] = []
    evidence_rows: List[Dict[str, Any]] = []
    raw_artifacts: Dict[str, Any] = {"pair_matrices": {}, "pockets": []}

    for order_idx, tok_idx in enumerate(valid_token_indices):
        token_id = int(item_cpu["pharm_type"][tok_idx].item())
        token_label = id2token.get(token_id, str(token_id))
        pool_weight = float(pharm_pool_alpha[0, tok_idx].detach().cpu().item())
        raw_score = float(pharm_pool_score[0, tok_idx].detach().cpu().item())
        anchor = int(item_cpu["pharm_anchor"][tok_idx].item()) if "pharm_anchor" in item_cpu else -1
        pharm_rows.append(
            {
                "split_index": sample_pos,
                "csv_row_index": ridx,
                "pharm_token_index": int(tok_idx),
                "pharm_order_index": int(order_idx),
                "token_id": token_id,
                "token_label": token_label,
                "anchor_atom_index": anchor,
                "pool_score_logit": raw_score,
                "pool_weight": pool_weight,
            }
        )

    if use_pocket_model and int(num_graphs) > 0:
        graph_to_sample = model._graph_to_sample_index(batch, num_graphs)
        for g in range(int(num_graphs)):
            graph_slot = int(g)
            start, end = batch_cpu["pocket_graph_slices"][g]
            node_count = int(end - start)
            is_valid = int(batch_cpu["pocket_graph_valid"].view(-1)[g].item()) if batch_cpu["pocket_graph_valid"].numel() > g else 0
            template = templates[g] if g < len(templates) else None
            meta = pocket_meta[g] if g < len(pocket_meta) else {}
            p2rank_score = float(batch_cpu["pocket_graph_score"].view(-1)[g].item()) if batch_cpu["pocket_graph_score"].numel() > g else 0.0
            base_row: Dict[str, Any] = {
                "pocket_index": graph_slot,
                "pocket_rank": int(meta.get("rank", graph_slot + 1)),
                "is_valid": is_valid,
                "num_nodes": node_count,
                "p2rank_prob": p2rank_score,
                "p2rank_score": float(meta.get("score", 0.0) or 0.0),
                "p2rank_residue_count": int(len(meta.get("residues", []))) if isinstance(meta, dict) else 0,
                "raw_count": int(template.get("raw_count", 0)) if isinstance(template, dict) else 0,
            }
            if int(graph_to_sample[g].item()) != 0 or node_count <= 0:
                pocket_rows.append(base_row)
                raw_artifacts["pockets"].append(base_row)
                continue

            node_mask = graph_batch == g
            pharm_mask_b = batch["pharm_mask"][0].bool()
            pharm_b = pharm_tokens[0][pharm_mask_b]
            if pharm_b.numel() == 0:
                pocket_rows.append(base_row)
                raw_artifacts["pockets"].append(base_row)
                continue
            pharm_b = model.pharm_align(pharm_b).unsqueeze(0)
            pocket_nodes_g = pocket_node_repr[node_mask]
            if pocket_nodes_g.numel() == 0:
                pocket_rows.append(base_row)
                raw_artifacts["pockets"].append(base_row)
                continue

            with torch.no_grad():
                pair_out = model.pair_cross(pharm_b, pocket_nodes_g.unsqueeze(0), return_details=True)
                pair_score = pair_out["pair_score"]
                pair_prob = torch.sigmoid(pair_score)
                graph_g = pocket_graph_repr[g:g + 1]
                if getattr(model, "use_pair_coverage", False) and getattr(model, "coverage_encoder", None) is not None:
                    cov_g, cov_stats = model.coverage_encoder(
                        pair_score=pair_out["pair_score"],
                        q_tokens=pair_out["q_tokens"],
                        r_tokens=pair_out["r_tokens"],
                        graph_g=graph_g,
                    )
                else:
                    cov_g = torch.zeros_like(graph_g)
                    cov_stats = {}
                pair_local_g = model.hpair_pocket_fuse(torch.cat([pair_out["pair_g"], cov_g, graph_g], dim=-1))
                score_in = torch.cat(
                    [
                        pair_local_g,
                        graph_g,
                        h_global[0:1],
                        pair_local_g * h_global[0:1],
                    ],
                    dim=-1,
                )
                score_g = model.pocket_relevance(score_in)
                score_before_prior = float(score_g.view(-1)[0].item())
                if getattr(model, "use_pocket_score_prior", False) and "pocket_graph_score" in batch and batch["pocket_graph_score"].numel() >= g + 1:
                    prior = batch["pocket_graph_score"].view(-1)[g:g + 1].view(1, 1).to(score_g.device, dtype=score_g.dtype)
                    score_g = score_g + model.pocket_score_scale * prior
                learned_logit = float(score_g.view(-1)[0].item())

                tok_scores = torch.zeros((0,), dtype=pair_prob.dtype, device=pair_prob.device)
                selected_pairs = select_top_pairs(
                    pair_score=pair_score,
                    top_t=int(getattr(model, "set_top_pair_per_pocket", 0)),
                    max_per_q=int(getattr(model, "set_max_per_pharm_token", 0)),
                    max_per_r=int(getattr(model, "set_max_per_pocket_node", 0)),
                )
                if getattr(model, "use_set_view", False) and getattr(model, "top_pair_tokenizer", None) is not None and int(getattr(model, "set_top_pair_per_pocket", 0)) > 0:
                    _tok_g, tok_scores = model.top_pair_tokenizer(
                        pair_score=pair_out["pair_score"],
                        pair_feat=pair_out["pair_feat"],
                        q_tokens=pair_out["q_tokens"],
                        r_tokens=pair_out["r_tokens"],
                        graph_g=graph_g,
                        top_t=int(model.set_top_pair_per_pocket),
                        max_per_q=int(model.set_max_per_pharm_token),
                        max_per_r=int(model.set_max_per_pocket_node),
                    )

            pharm_local_best = pair_prob.max(dim=1).values.detach().cpu()
            for local_idx, val in enumerate(pharm_local_best.tolist()):
                if local_idx < int(pharm_best_local.numel()):
                    pharm_best_local[local_idx] = max(float(pharm_best_local[local_idx].item()), float(val))

            residue_pair_best = pair_prob.max(dim=0).values.detach().cpu()
            pair_score_cpu = pair_score.detach().cpu()
            pair_prob_cpu = pair_prob.detach().cpu()

            res_seq_raw = template.get("res_seq", []) if isinstance(template, dict) else []
            filtered_res_seq = [int(r) for r in res_seq_raw if 1 <= int(r) <= int(tok_pos.numel())]
            if len(filtered_res_seq) > node_count:
                filtered_res_seq = filtered_res_seq[:node_count]
            while len(filtered_res_seq) < node_count:
                filtered_res_seq.append(-1)

            node_is_raw = []
            if isinstance(template, dict) and isinstance(template.get("node_is_raw", None), torch.Tensor):
                raw_t = template["node_is_raw"].view(-1).detach().cpu().tolist()
                node_is_raw = [int(bool(v)) for v in raw_t[:node_count]]
            while len(node_is_raw) < node_count:
                node_is_raw.append(0)

            row_labels: List[str] = []
            for tok_idx in valid_token_indices:
                token_id = int(item_cpu["pharm_type"][tok_idx].item())
                token_label = id2token.get(token_id, str(token_id))
                anchor = int(item_cpu["pharm_anchor"][tok_idx].item()) if "pharm_anchor" in item_cpu else -1
                row_labels.append(f"pharm{tok_idx}:{token_label}:atom{anchor}")

            col_labels: List[str] = []
            residue_rows_this: List[Dict[str, Any]] = []
            for node_idx, rid in enumerate(filtered_res_seq):
                chain, rec = find_residue_record(rec_map, rid) if rid > 0 else ("", {})
                resname = str(rec.get("resname", "")) if isinstance(rec, dict) else ""
                coord = rec.get("coord", None) if isinstance(rec, dict) else None
                seq_tok_idx = int(tok_pos[rid - 1].item()) if rid > 0 and rid <= int(tok_pos.numel()) else -1
                soft_score = 0.0
                if soft_pocket_bias is not None and seq_tok_idx >= 0 and soft_pocket_bias.numel() > 0:
                    soft_score = float(soft_pocket_bias[0, seq_tok_idx].detach().cpu().item())
                pair_res_score = float(residue_pair_best[node_idx].item()) if node_idx < int(residue_pair_best.numel()) else 0.0
                residue_proxy = max(pair_res_score, soft_score)
                label = f"{chain or 'A'}:{rid}:{resname or 'UNK'}" if rid > 0 else f"node_{node_idx}"
                col_labels.append(label)
                residue_row = {
                    "pocket_index": graph_slot,
                    "node_index": int(node_idx),
                    "residue_seq": int(rid),
                    "chain_id": chain,
                    "resname": resname,
                    "is_raw_pocket_residue": int(node_is_raw[node_idx]),
                    "pair_residue_score": pair_res_score,
                    "soft_pocket_bias_score": soft_score,
                    "residue_score_proxy": residue_proxy,
                }
                if coord is not None:
                    residue_row["x"] = float(coord[0])
                    residue_row["y"] = float(coord[1])
                    residue_row["z"] = float(coord[2])
                if isinstance(rec, dict) and "plddt" in rec:
                    residue_row["plddt"] = float(rec.get("plddt", 0.0) or 0.0)
                residue_rows_this.append(residue_row)
                residue_rows_all.append(residue_row)

            pair_logit_csv = os.path.join(out_dir, f"pair_score_pocket{graph_slot + 1}_logit.csv")
            pair_prob_csv = os.path.join(out_dir, f"pair_score_pocket{graph_slot + 1}_prob.csv")
            write_matrix_csv(pair_logit_csv, pair_score_cpu, row_labels, col_labels)
            write_matrix_csv(pair_prob_csv, pair_prob_cpu, row_labels, col_labels)
            write_rows_csv(os.path.join(out_dir, f"pocket{graph_slot + 1}_residues.csv"), residue_rows_this)

            pocket_row = dict(base_row)
            pocket_row.update(
                {
                    "learned_pocket_logit_pre_prior": score_before_prior,
                    "learned_pocket_logit": learned_logit,
                    "learned_pocket_prob": sigmoid_scalar(learned_logit),
                    "best_pair_score_logit": float(pair_score_cpu.max().item()) if pair_score_cpu.numel() > 0 else 0.0,
                    "best_pair_score_prob": float(pair_prob_cpu.max().item()) if pair_prob_cpu.numel() > 0 else 0.0,
                    "mean_pair_score_prob": float(pair_prob_cpu.mean().item()) if pair_prob_cpu.numel() > 0 else 0.0,
                    "pharm_coverage_mean": float(cov_stats.get("pharm_cov", torch.zeros((0,))).mean().item()) if isinstance(cov_stats, dict) and cov_stats.get("pharm_cov", None) is not None and cov_stats["pharm_cov"].numel() > 0 else 0.0,
                    "residue_coverage_mean": float(cov_stats.get("pocket_cov", torch.zeros((0,))).mean().item()) if isinstance(cov_stats, dict) and cov_stats.get("pocket_cov", None) is not None and cov_stats["pocket_cov"].numel() > 0 else 0.0,
                }
            )
            pocket_rows.append(pocket_row)

            for ev_idx, ev in enumerate(selected_pairs):
                q_local = int(ev["q_local_index"])
                r_local = int(ev["r_local_index"])
                tok_idx = int(valid_token_indices[q_local]) if q_local < len(valid_token_indices) else -1
                token_id = int(item_cpu["pharm_type"][tok_idx].item()) if tok_idx >= 0 else -1
                token_label = id2token.get(token_id, str(token_id)) if tok_idx >= 0 else ""
                anchor = int(item_cpu["pharm_anchor"][tok_idx].item()) if tok_idx >= 0 and "pharm_anchor" in item_cpu else -1
                rid = int(filtered_res_seq[r_local]) if r_local < len(filtered_res_seq) else -1
                chain, rec = find_residue_record(rec_map, rid) if rid > 0 else ("", {})
                evidence_row = {
                    "evidence_index": len(evidence_rows),
                    "pocket_index": graph_slot,
                    "pocket_rank": int(meta.get("rank", graph_slot + 1)) if isinstance(meta, dict) else graph_slot + 1,
                    "local_rank_in_pocket": int(ev["local_rank"]),
                    "pharm_local_index": q_local,
                    "pharm_token_index": tok_idx,
                    "token_id": token_id,
                    "token_label": token_label,
                    "anchor_atom_index": anchor,
                    "residue_local_index": r_local,
                    "residue_seq": rid,
                    "chain_id": chain,
                    "resname": str(rec.get("resname", "")) if isinstance(rec, dict) else "",
                    "pair_score_logit": float(ev["pair_score_logit"]),
                    "pair_score_prob": float(ev["pair_score_prob"]),
                    "set_token_score": float(tok_scores[ev_idx].detach().cpu().item()) if ev_idx < int(tok_scores.numel()) else 0.0,
                }
                evidence_rows.append(evidence_row)

            raw_artifacts["pair_matrices"][f"pocket_{graph_slot + 1}"] = {
                "pair_score_logit": pair_score_cpu,
                "pair_score_prob": pair_prob_cpu,
                "row_labels": row_labels,
                "col_labels": col_labels,
            }
            raw_artifacts["pockets"].append(
                {
                    **pocket_row,
                    "row_labels": row_labels,
                    "col_labels": col_labels,
                    "residue_rows": residue_rows_this,
                    "selected_pairs": selected_pairs,
                }
            )

    for order_idx, local_best in enumerate(pharm_best_local.tolist()):
        pharm_rows[order_idx]["best_local_pair_prob"] = float(local_best)
        pharm_rows[order_idx]["pharm_score_proxy"] = max(float(pharm_rows[order_idx]["pool_weight"]), float(local_best))

    atom_rows, bond_rows, rdkit_warn = mol_2d_rows(smiles)
    atom_proxy_sum: Dict[int, float] = {}
    atom_proxy_max: Dict[int, float] = {}
    for row_pharm in pharm_rows:
        anch = int(row_pharm.get("anchor_atom_index", -1))
        if anch < 0:
            continue
        score_proxy = float(row_pharm.get("pharm_score_proxy", row_pharm.get("pool_weight", 0.0)))
        atom_proxy_sum[anch] = atom_proxy_sum.get(anch, 0.0) + score_proxy
        atom_proxy_max[anch] = max(atom_proxy_max.get(anch, 0.0), score_proxy)
    for atom_row in atom_rows:
        atom_idx = int(atom_row["atom_index"])
        atom_row["atom_score_proxy_sum"] = float(atom_proxy_sum.get(atom_idx, 0.0))
        atom_row["atom_score_proxy_max"] = float(atom_proxy_max.get(atom_idx, 0.0))

    if evidence_rows:
        order = sorted(range(len(evidence_rows)), key=lambda i: float(evidence_rows[i].get("set_token_score", 0.0)), reverse=True)
        keep_n = int(getattr(model, "set_max_tokens", 0))
        keep_set = set(order[:keep_n]) if keep_n > 0 else set(order)
        for rank, idx in enumerate(order, start=1):
            evidence_rows[idx]["global_rank_by_set_score"] = int(rank)
            evidence_rows[idx]["kept_in_set_encoder"] = int(idx in keep_set)

    write_rows_csv(os.path.join(out_dir, "pharm_tokens.csv"), pharm_rows)
    write_rows_csv(os.path.join(out_dir, "pockets.csv"), pocket_rows)
    write_rows_csv(os.path.join(out_dir, "residues_all.csv"), residue_rows_all)
    write_rows_csv(os.path.join(out_dir, "set_evidence.csv"), evidence_rows)
    if atom_rows:
        write_rows_csv(os.path.join(out_dir, "drug_atoms.csv"), atom_rows)
    if bond_rows:
        write_rows_csv(os.path.join(out_dir, "drug_bonds.csv"), bond_rows)

    summary = {
        "ckpt": os.path.abspath(ckpt_path),
        "cfg_source": os.path.abspath(abs_project_path(args.cfg)) if args.cfg else "checkpoint_payload",
        "csv_path": csv_path,
        "split": args.split,
        "split_index": int(sample_pos),
        "csv_row_index": int(ridx),
        "protein_id": pid,
        "smiles": smiles,
        "label": y_true,
        "logit": float(logit.detach().cpu().item()),
        "prob": prob,
        "best_threshold": float(ckpt.get("best_threshold", 0.5)),
        "pred_positive_at_best_threshold": int(prob >= float(ckpt.get("best_threshold", 0.5))),
        "pdb_path": pdb_path,
        "protein_pocket_note": "Pocket annotations are predicted on single-protein structures, not real ligand-bound complex labels.",
        "rdkit_warning": rdkit_warn,
        "ckpt_load_warning": load_warning,
        "missing_keys": list(getattr(load_info, "missing_keys", [])),
        "unexpected_keys": list(getattr(load_info, "unexpected_keys", [])),
        "out_dir": out_dir,
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(to_python(summary), f, ensure_ascii=False, indent=2)

    readme_lines = [
        "Files exported for one sample.",
        "Directory root is <repo_root>/explanations unless --out_dir is provided.",
        "drug_atoms.csv / drug_bonds.csv: panel (a) drug-side drawing inputs. atom_score_proxy_* are heuristic scores aggregated from pharmacophore tokens via anchor atoms.",
        "pharm_tokens.csv: pharmacophore token labels, pool weights, and best local pair scores.",
        "pockets.csv + pocket*_residues.csv: panel (b) predicted pocket and residue-level evidence. residue_score_proxy=max(pair_residue_score, soft_pocket_bias_score).",
        "pair_score_pocket*_logit.csv / pair_score_pocket*_prob.csv: panel (c) pharmacophore-residue heatmaps.",
        "set_evidence.csv: panel (d) top local evidence pairs. kept_in_set_encoder marks the pairs retained after global set token truncation.",
        "Important: pocket and residue annotations here are model/P2Rank-derived explanation signals, not experimental complex-ground-truth labels.",
    ]
    with open(os.path.join(out_dir, "README.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(readme_lines) + "\n")

    torch.save(
        {
            "summary": summary,
            "pharm_tokens": pharm_rows,
            "pockets": raw_artifacts["pockets"],
            "set_evidence": evidence_rows,
        },
        os.path.join(out_dir, "artifacts.pt"),
    )

    print(json.dumps(to_python(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
