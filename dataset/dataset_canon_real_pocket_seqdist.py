#鐢熸垚鐪熷疄鍙ｈ鍥炬暟鎹苟鎻愪緵 collate_fn_real_pocket


import json
import hashlib
import os
import math
from typing import Any, Dict, List, Tuple, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset

from dataset.featurize_canon import (
    DEFAULT_MORGAN_NBITS,
    DRUG_DESC_DIM,
    DRUG_EDGE_FEAT_DIM,
    NODE_FEAT_DIM,
    smiles_to_drug_data,
)
from p2rank_pockets_strict import build_index_from_protein_pockets_csv
from pocket_graph_seqdist import (
    parse_pdb_ca_records,
    load_p2rank_pockets,
    build_pocket_templates,
)


def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _normalize_seqdist(seq_dist: torch.Tensor, method: str, denom: float) -> torch.Tensor:
    x = seq_dist.float().clamp_min(0.0)
    m = str(method or "log1p").lower()
    d = float(max(1e-6, denom))
    if m in ("none", "raw", "off", "disable", "disabled"):
        return x
    if m in ("linear", "divide", "div"):
        return x / d
    if m == "sqrt":
        return torch.sqrt(x / d)
    # default: log1p normalized by log1p(denom)
    return torch.log1p(x) / float(math.log1p(d))


def build_pdb_index(pdb_index_csv: str) -> Dict[str, str]:
    if not pdb_index_csv or not os.path.isfile(pdb_index_csv):
        return {}
    df = pd.read_csv(pdb_index_csv)
    if "Protein_ID" not in df.columns or "pdb_path" not in df.columns:
        return {}
    out: Dict[str, str] = {}
    for _, row in df.iterrows():
        pid = str(row["Protein_ID"]).strip()
        path = str(row["pdb_path"]).strip()
        if not pid or not path:
            continue
        out[pid] = os.path.normpath(path)
    return out


class HumanDTIDatasetCanonRealPocket(Dataset):
    """
    Canonical graph + pharm tokens + ESM2 tokens + real pocket graphs (P2Rank + PDB).
    """
    def __init__(
        self,
        csv_path: str,
        split_path: str,
        pharm_path: str,
        esm2_path: str,
        split: str = "train",
        mask_special_tokens: bool = True,
        # --- P2Rank / PDB ---
        p2rank_work_dir: Optional[str] = None,
        protein_pockets_csv: Optional[str] = None,
        pdb_index_csv: Optional[str] = None,
        p2rank_out_dir: Optional[str] = None,
        protein_id_col: str = "Protein_ID",
        pocket_cache_dir: Optional[str] = None,
        drug_graph_cache_path: Optional[str] = None,
        # --- pocket graph settings ---
        pocket_topk: int = 3,
        pocket_max_nodes: int = 96,
        pocket_knn_k: int = 16,
        pocket_prob_thres: float = 0.0,
        pocket_min_residues: int = 5,
        pocket_expand_min_nodes: int = 0,
        pocket_seqdist_norm: str = "log1p",
        pocket_seqdist_norm_denom: float = 128.0,
        drop_no_pocket: bool = True,
        pocket_use_struct_feat: bool = True,
        pocket_struct_dim: int = 29,
        pocket_edge_rbf_bins: int = 16,
        pocket_edge_rbf_dmin: float = 0.0,
        pocket_edge_rbf_dmax: float = 20.0,
        pocket_edge_use_seqdist: bool = True,
        pocket_raw_only: bool = True,
        pocket_edge_mode: str = "radius",
        pocket_radius_cutoff: float = 10.0,
        pocket_radius_max_neighbors: int = 32,
    ):
        self.df = pd.read_csv(csv_path)
        with open(split_path, "r", encoding="utf-8") as f:
            split_obj = json.load(f)
        if split not in split_obj:
            raise KeyError(f"split '{split}' not found in {split_path}, available: {list(split_obj.keys())}")
        self.indices = list(split_obj[split])

        pharm = torch.load(pharm_path, weights_only=True, map_location="cpu")
        self.pharm_type_all = pharm["pharm_type"]      # [N, P]
        self.pharm_mask_all = pharm["pharm_mask"]      # [N, P]
        self.pharm_anchor_all = pharm.get("pharm_anchor", None)  # optional [N, P]
        self.pharm_family_all = pharm.get("pharm_family", None)
        self.pharm_center_all = pharm.get("pharm_center", None)
        self.pharm_geom_all = pharm.get("pharm_geom", None)
        self.pharm_context_all = pharm.get("pharm_context", None)
        self.pharm_multi_anchor_all = pharm.get("pharm_multi_anchor", None)
        self.pharm_multi_anchor_mask_all = pharm.get("pharm_multi_anchor_mask", None)
        self.pharm_num_families = int(pharm.get("num_families", len(pharm.get("family2id", {})) if isinstance(pharm.get("family2id", None), dict) else 0))
        self.pharm_geom_dim = int(self.pharm_geom_all.shape[-1]) if isinstance(self.pharm_geom_all, torch.Tensor) and self.pharm_geom_all.dim() == 3 else 0
        self.pharm_context_dim = int(self.pharm_context_all.shape[-1]) if isinstance(self.pharm_context_all, torch.Tensor) and self.pharm_context_all.dim() == 3 else 0
        self.pharm_center_dim = int(self.pharm_center_all.shape[-1]) if isinstance(self.pharm_center_all, torch.Tensor) and self.pharm_center_all.dim() == 3 else 0
        self.pharm_multi_anchor_k = int(self.pharm_multi_anchor_all.shape[-1]) if isinstance(self.pharm_multi_anchor_all, torch.Tensor) and self.pharm_multi_anchor_all.dim() == 3 else 0

        esm = torch.load(esm2_path, weights_only=True, map_location="cpu")
        self.esm_dim = int(esm.get("dim", esm.get("d_model", 0)))
        self.esm_mode = "tokens" if "tokens" in esm else ("mean" if "embeddings" in esm else None)
        if self.esm_mode is None:
            raise ValueError(f"ESM2 file must contain 'tokens' or 'embeddings': {esm2_path}")
        self.esm_table: Dict[str, torch.Tensor] = esm["tokens"] if self.esm_mode == "tokens" else esm["embeddings"]
        self.mask_special_tokens = bool(mask_special_tokens)

        if len(self.df) != int(self.pharm_type_all.shape[0]):
            raise ValueError(
                f"pharm precompute N must match csv rows. csv={len(self.df)} pharm={self.pharm_type_all.shape[0]}"
            )

        # P2Rank/PDB paths
        self.p2rank_work_dir = str(p2rank_work_dir) if p2rank_work_dir else None
        if self.p2rank_work_dir:
            if not protein_pockets_csv:
                protein_pockets_csv = os.path.join(self.p2rank_work_dir, "derived", "protein_pockets.csv")
            if not pdb_index_csv:
                pdb_index_csv = os.path.join(self.p2rank_work_dir, "derived", "pdb_index.csv")
            if not p2rank_out_dir:
                p2rank_out_dir = os.path.join(self.p2rank_work_dir, "p2rank_out")

        self.protein_id_col = str(protein_id_col)
        self.pocket_cache_dir = str(pocket_cache_dir) if pocket_cache_dir else None
        self.drug_graph_cache_path = str(drug_graph_cache_path) if drug_graph_cache_path else None
        self.drug_edge_feat_dim = int(DRUG_EDGE_FEAT_DIM)
        self.drug_fp_dim = int(DEFAULT_MORGAN_NBITS)
        self.drug_desc_dim = int(DRUG_DESC_DIM)
        self.pocket_topk = int(max(1, pocket_topk))
        self.pocket_max_nodes = int(max(0, pocket_max_nodes))
        self.pocket_knn_k = int(max(1, pocket_knn_k))
        self.pocket_prob_thres = float(pocket_prob_thres)
        self.pocket_min_residues = int(max(1, pocket_min_residues))
        self.pocket_expand_min_nodes = int(max(0, pocket_expand_min_nodes))
        self.pocket_seqdist_norm = str(pocket_seqdist_norm or "log1p")
        self.pocket_seqdist_norm_denom = float(pocket_seqdist_norm_denom)
        self.drop_no_pocket = bool(drop_no_pocket)
        self.pocket_use_struct_feat = bool(pocket_use_struct_feat)
        self.pocket_struct_dim = int(max(0, pocket_struct_dim))
        self.pocket_edge_rbf_bins = int(max(1, pocket_edge_rbf_bins))
        self.pocket_edge_rbf_dmin = float(pocket_edge_rbf_dmin)
        self.pocket_edge_rbf_dmax = float(pocket_edge_rbf_dmax)
        self.pocket_edge_use_seqdist = bool(pocket_edge_use_seqdist)
        self.pocket_raw_only = bool(pocket_raw_only)
        self.pocket_edge_mode = str(pocket_edge_mode or "radius").lower()
        self.pocket_radius_cutoff = float(pocket_radius_cutoff)
        self.pocket_radius_max_neighbors = int(max(1, pocket_radius_max_neighbors))
        self.pocket_edge_feat_dim_expected = int(1 + self.pocket_edge_rbf_bins + 3 + (1 if self.pocket_edge_use_seqdist else 0))

        self.p2rank_index: Dict[str, Tuple[str, str]] = {}
        if protein_pockets_csv:
            self.p2rank_index = build_index_from_protein_pockets_csv(
                protein_pockets_csv=protein_pockets_csv,
                p2rank_out_dir=p2rank_out_dir,
                protein_id_col="Protein_ID",
                predictions_col="predictions_csv",
            )

        self.pdb_index: Dict[str, str] = build_pdb_index(pdb_index_csv) if pdb_index_csv else {}

        if self.drop_no_pocket:
            if not self.p2rank_index or not self.pdb_index:
                self.indices = []
            else:
                kept: List[int] = []
                for ridx in self.indices:
                    row = self.df.iloc[int(ridx)]
                    pid = str(row.get(self.protein_id_col, "")).strip()
                    pid = pid.split("_len=")[0]
                    if pid in self.p2rank_index and pid in self.pdb_index:
                        kept.append(int(ridx))
                self.indices = kept

        # caches
        self._coord_cache: Dict[str, Dict[Tuple[str, int], Tuple[float, float, float]]] = {}
        self._pdb_record_cache: Dict[str, Dict[Tuple[str, int], Dict[str, Any]]] = {}
        self._pocket_meta_cache: Dict[str, List[Dict[str, Any]]] = {}
        self._template_cache: Dict[str, List[Optional[Dict[str, Any]]]] = {}
        self._cache_load_warned = False
        self._drug_graph_cache: Dict[str, Dict[str, torch.Tensor]] = {}
        self._drug_cache_miss_warned = False

        if self.drug_graph_cache_path and os.path.isfile(self.drug_graph_cache_path):
            load_err: Optional[Exception] = None
            obj: Optional[Dict[str, Any]] = None
            try:
                obj = torch.load(self.drug_graph_cache_path, weights_only=True, map_location="cpu")
            except Exception as e:
                load_err = e
                try:
                    obj = torch.load(self.drug_graph_cache_path, map_location="cpu")
                except Exception as e2:
                    load_err = e2
                    obj = None
            if isinstance(obj, dict):
                if "drug_edge_feat_dim" in obj:
                    try:
                        self.drug_edge_feat_dim = int(obj["drug_edge_feat_dim"])
                    except Exception:
                        pass
                if "drug_fp_dim" in obj:
                    try:
                        self.drug_fp_dim = int(obj["drug_fp_dim"])
                    except Exception:
                        pass
                if "drug_desc_dim" in obj:
                    try:
                        self.drug_desc_dim = int(obj["drug_desc_dim"])
                    except Exception:
                        pass
                graphs = obj.get("graphs", obj)
                if isinstance(graphs, dict):
                    for k, v in graphs.items():
                        if isinstance(v, dict) and ("node_feat" in v) and ("edge_index" in v):
                            try:
                                edge_attr = v.get("edge_attr", None)
                                if isinstance(edge_attr, torch.Tensor):
                                    edge_attr = edge_attr.float().cpu()
                                    if edge_attr.dim() == 1:
                                        edge_attr = edge_attr.view(-1, 1)
                                else:
                                    n_edges = int(v["edge_index"].shape[1]) if isinstance(v["edge_index"], torch.Tensor) else 0
                                    edge_attr = torch.zeros((n_edges, int(self.drug_edge_feat_dim)), dtype=torch.float32)
                                node_pos = v.get("node_pos", None)
                                if isinstance(node_pos, torch.Tensor):
                                    node_pos = node_pos.float().cpu()
                                    if node_pos.dim() != 2 or int(node_pos.shape[1]) != 3:
                                        node_pos = torch.zeros((int(v["node_feat"].shape[0]), 3), dtype=torch.float32)
                                else:
                                    node_pos = torch.zeros((int(v["node_feat"].shape[0]), 3), dtype=torch.float32)
                                drug_fp = v.get("drug_fp", None)
                                if isinstance(drug_fp, torch.Tensor):
                                    drug_fp = drug_fp.float().cpu().view(-1)
                                else:
                                    drug_fp = torch.zeros((int(self.drug_fp_dim),), dtype=torch.float32)
                                drug_desc = v.get("drug_desc", None)
                                if isinstance(drug_desc, torch.Tensor):
                                    drug_desc = drug_desc.float().cpu().view(-1)
                                else:
                                    drug_desc = torch.zeros((int(self.drug_desc_dim),), dtype=torch.float32)
                                self._drug_graph_cache[str(k)] = {
                                    "node_feat": v["node_feat"].float().cpu(),
                                    "edge_index": v["edge_index"].long().cpu(),
                                    "edge_attr": edge_attr,
                                    "node_pos": node_pos,
                                    "drug_fp": drug_fp,
                                    "drug_desc": drug_desc,
                                }
                            except Exception:
                                continue
            elif load_err is not None:
                print(
                    f"[WARN] drug graph cache load failed; fallback to on-the-fly graph build. "
                    f"path={self.drug_graph_cache_path} err={load_err}"
                )

        # Strict drop: keep only rows whose protein has at least one valid pocket template.
        # This removes "no_valid_pocket" samples when DROP_NO_POCKET=1.
        if self.drop_no_pocket and self.esm_mode == "tokens" and self.indices:
            kept_valid: List[int] = []
            pid_valid_cache: Dict[str, bool] = {}
            for ridx in self.indices:
                row = self.df.iloc[int(ridx)]
                pid = str(row.get(self.protein_id_col, "")).strip().split("_len=")[0]
                if not pid:
                    continue

                is_valid = pid_valid_cache.get(pid, None)
                if is_valid is None:
                    prot = str(row.get("Protein", "")).strip()
                    key = sha1(prot)
                    tok = self.esm_table.get(key, None)
                    if tok is None:
                        is_valid = False
                    else:
                        seq_len = int(tok.shape[0])
                        if self.mask_special_tokens and seq_len >= 2:
                            seq_len -= 2
                        templates = self._get_templates(pid, seq_len)
                        is_valid = any(
                            (t is not None) and (len(t.get("res_seq", [])) > 0)
                            for t in templates
                        )
                    pid_valid_cache[pid] = bool(is_valid)

                if is_valid:
                    kept_valid.append(int(ridx))

            self.indices = kept_valid

    def __len__(self) -> int:
        return len(self.indices)

    def _get_templates(self, pid: str, seq_len: int) -> List[Optional[Dict[str, Any]]]:
        if pid in self._template_cache:
            return self._template_cache[pid]

        if self.pocket_cache_dir:
            cache_path = os.path.join(self.pocket_cache_dir, f"{pid}.pt")
            if os.path.isfile(cache_path):
                load_err: Optional[Exception] = None
                obj: Optional[Dict[str, Any]] = None
                try:
                    obj = torch.load(cache_path, weights_only=True, map_location="cpu")
                except Exception as e:
                    load_err = e
                    # Fallback for environments where cached objects are not
                    # compatible with weights_only-safe unpickling.
                    try:
                        obj = torch.load(cache_path, map_location="cpu")
                    except Exception as e2:
                        load_err = e2
                        obj = None
                try:
                    if not isinstance(obj, dict):
                        raise TypeError(f"cache object is not dict: {type(obj)}")
                    ok = True
                    if "pocket_topk" in obj and int(obj["pocket_topk"]) != int(self.pocket_topk):
                        ok = False
                    if "pocket_max_nodes" in obj and int(obj["pocket_max_nodes"]) != int(self.pocket_max_nodes):
                        ok = False
                    if "pocket_knn_k" in obj and int(obj["pocket_knn_k"]) != int(self.pocket_knn_k):
                        ok = False
                    if "pocket_prob_thres" in obj and float(obj["pocket_prob_thres"]) != float(self.pocket_prob_thres):
                        ok = False
                    if "pocket_min_residues" in obj and int(obj["pocket_min_residues"]) != int(self.pocket_min_residues):
                        ok = False
                    if "pocket_expand_min_nodes" in obj:
                        if int(obj["pocket_expand_min_nodes"]) != int(self.pocket_expand_min_nodes):
                            ok = False
                    elif int(self.pocket_expand_min_nodes) > 0:
                        ok = False
                    if "pocket_edge_feat_dim" in obj and int(obj["pocket_edge_feat_dim"]) != int(self.pocket_edge_feat_dim_expected):
                        ok = False
                    if "pocket_edge_rbf_bins" in obj and int(obj["pocket_edge_rbf_bins"]) != int(self.pocket_edge_rbf_bins):
                        ok = False
                    if "pocket_node_struct_dim" in obj and int(obj["pocket_node_struct_dim"]) != int(self.pocket_struct_dim):
                        ok = False
                    if "pocket_raw_only" in obj and bool(obj["pocket_raw_only"]) != bool(self.pocket_raw_only):
                        ok = False
                    if "pocket_edge_mode" in obj and str(obj["pocket_edge_mode"]) != str(self.pocket_edge_mode):
                        ok = False
                    if "pocket_radius_cutoff" in obj:
                        try:
                            if abs(float(obj["pocket_radius_cutoff"]) - float(self.pocket_radius_cutoff)) > 1e-6:
                                ok = False
                        except Exception:
                            ok = False
                    if "pocket_radius_max_neighbors" in obj and int(obj["pocket_radius_max_neighbors"]) != int(self.pocket_radius_max_neighbors):
                        ok = False
                    if "seq_len" in obj and int(obj["seq_len"]) != int(seq_len):
                        ok = False
                    templates = obj.get("templates", None)
                    if ok and isinstance(templates, list) and templates:
                        if len(templates) != int(self.pocket_topk):
                            if len(templates) > int(self.pocket_topk):
                                templates = templates[: int(self.pocket_topk)]
                            else:
                                templates = templates + [None] * (int(self.pocket_topk) - len(templates))
                        self._template_cache[pid] = templates
                        return templates
                except Exception as e:
                    load_err = e
                if load_err is not None and (not self._cache_load_warned):
                    print(f"[WARN] pocket cache load/validate failed once; will rebuild from P2Rank/PDB. path={cache_path} err={load_err}")
                    self._cache_load_warned = True

        res_csv, pred_csv = self.p2rank_index.get(pid, ("", ""))
        pdb_path = self.pdb_index.get(pid, "")
        if not res_csv or not pdb_path:
            templates = [None for _ in range(self.pocket_topk)]
            self._template_cache[pid] = templates
            return templates

        if pid not in self._pocket_meta_cache:
            pockets = load_p2rank_pockets(
                pred_csv,
                res_csv,
                top_k=self.pocket_topk,
                prob_threshold=self.pocket_prob_thres,
                min_residues=self.pocket_min_residues,
            )
            self._pocket_meta_cache[pid] = pockets
        else:
            pockets = self._pocket_meta_cache[pid]

        if pdb_path in self._pdb_record_cache:
            rec_map = self._pdb_record_cache[pdb_path]
        else:
            rec_map = parse_pdb_ca_records(pdb_path)
            self._pdb_record_cache[pdb_path] = rec_map

        if pdb_path in self._coord_cache:
            coord_map = self._coord_cache[pdb_path]
        else:
            coord_map = {}
            for k, rec in rec_map.items():
                xyz = rec.get("coord", None)
                if xyz is None:
                    continue
                coord_map[k] = (float(xyz[0]), float(xyz[1]), float(xyz[2]))
            self._coord_cache[pdb_path] = coord_map

        resname_map: Dict[Tuple[str, int], str] = {}
        plddt_map: Dict[Tuple[str, int], float] = {}
        for k, rec in rec_map.items():
            resname_map[k] = str(rec.get("resname", "UNK") or "UNK").upper()
            try:
                plddt_map[k] = float(rec.get("plddt", 0.0) or 0.0)
            except Exception:
                plddt_map[k] = 0.0

        templates = build_pocket_templates(
            pockets,
            coord_map,
            seq_len,
            top_k=self.pocket_topk,
            max_nodes=self.pocket_max_nodes,
            knn_k=self.pocket_knn_k,
            expand_min_nodes=self.pocket_expand_min_nodes,
            seqdist_norm=self.pocket_seqdist_norm,
            seqdist_norm_denom=self.pocket_seqdist_norm_denom,
            resname_map=resname_map,
            plddt_map=plddt_map,
            edge_rbf_bins=self.pocket_edge_rbf_bins,
            edge_rbf_dmin=self.pocket_edge_rbf_dmin,
            edge_rbf_dmax=self.pocket_edge_rbf_dmax,
            raw_only=self.pocket_raw_only,
            edge_mode=self.pocket_edge_mode,
            radius_cutoff=self.pocket_radius_cutoff,
            radius_max_neighbors=self.pocket_radius_max_neighbors,
        )
        self._template_cache[pid] = templates
        return templates

    def __getitem__(self, i: int) -> Tuple[Dict[str, Any], torch.Tensor]:
        ridx = int(self.indices[i])
        row = self.df.iloc[ridx]
        smiles = str(row["SMILES"])
        prot = str(row["Protein"]).strip()
        protein_id = str(row.get(self.protein_id_col, "")).strip().split("_len=")[0]
        y = torch.tensor(float(row["Y"]), dtype=torch.float32)

        cached_graph = self._drug_graph_cache.get(smiles, None) if self._drug_graph_cache else None
        use_cached_graph = (
            isinstance(cached_graph, dict)
            and "node_feat" in cached_graph
            and "edge_index" in cached_graph
            and "edge_attr" in cached_graph
            and "node_pos" in cached_graph
            and "drug_fp" in cached_graph
            and "drug_desc" in cached_graph
        )
        if use_cached_graph:
            node_feat = cached_graph["node_feat"]
            edge_index = cached_graph["edge_index"]
            edge_attr = cached_graph.get("edge_attr", None)
            node_pos = cached_graph.get("node_pos", None)
            drug_fp = cached_graph.get("drug_fp", None)
            drug_desc = cached_graph.get("drug_desc", None)
            if not isinstance(edge_attr, torch.Tensor):
                edge_attr = torch.zeros((int(edge_index.shape[1]), int(self.drug_edge_feat_dim)), dtype=torch.float32)
            if not isinstance(node_pos, torch.Tensor):
                node_pos = torch.zeros((int(node_feat.shape[0]), 3), dtype=torch.float32)
            if not isinstance(drug_fp, torch.Tensor):
                drug_fp = torch.zeros((int(self.drug_fp_dim),), dtype=torch.float32)
            if not isinstance(drug_desc, torch.Tensor):
                drug_desc = torch.zeros((int(self.drug_desc_dim),), dtype=torch.float32)
        else:
            if self._drug_graph_cache and (not self._drug_cache_miss_warned):
                print(f"[WARN] drug graph cache key miss or lacks rich fields; fallback to on-the-fly graph. smiles={smiles[:64]}")
                self._drug_cache_miss_warned = True
            drug_data = smiles_to_drug_data(smiles)
            node_feat = drug_data["node_feat"]
            edge_index = drug_data["edge_index"]
            edge_attr = drug_data["edge_attr"]
            node_pos = drug_data["node_pos"]
            drug_fp = drug_data["drug_fp"]
            drug_desc = drug_data["drug_desc"]
        num_nodes = int(node_feat.shape[0])

        pharm_type = self.pharm_type_all[ridx].long()  # [P]
        pharm_mask = self.pharm_mask_all[ridx].long()  # [P]
        pharm_anchor = None
        if self.pharm_anchor_all is not None:
            pharm_anchor = self.pharm_anchor_all[ridx].long()  # [P], 0-based
        pharm_family = self.pharm_family_all[ridx].long() if self.pharm_family_all is not None else None
        pharm_center = self.pharm_center_all[ridx].float() if self.pharm_center_all is not None else None
        pharm_geom = self.pharm_geom_all[ridx].float() if self.pharm_geom_all is not None else None
        pharm_context = self.pharm_context_all[ridx].float() if self.pharm_context_all is not None else None
        pharm_multi_anchor = self.pharm_multi_anchor_all[ridx].long() if self.pharm_multi_anchor_all is not None else None
        pharm_multi_anchor_mask = (
            self.pharm_multi_anchor_mask_all[ridx].long() if self.pharm_multi_anchor_mask_all is not None else None
        )

        # ESM2 tokens/emb
        key = sha1(prot)
        if self.esm_mode == "tokens":
            esm_tok = self.esm_table.get(key, None)
            if esm_tok is None:
                esm_tok = torch.zeros((1, self.esm_dim), dtype=torch.float32)
                esm_mask = torch.zeros((1,), dtype=torch.long)
            else:
                esm_tok = esm_tok.float()
                esm_mask = torch.ones((esm_tok.shape[0],), dtype=torch.long)
                if self.mask_special_tokens and esm_mask.numel() >= 2:
                    esm_mask[0] = 0
                    esm_mask[-1] = 0
        else:
            emb = self.esm_table.get(key, None)
            if emb is None:
                emb = torch.zeros((self.esm_dim,), dtype=torch.float32)
            emb = emb.float().view(1, -1)
            esm_tok = emb
            esm_mask = torch.ones((1,), dtype=torch.long)

        # --- pocket graphs ---
        pocket_node_dim = int(esm_tok.shape[1]) + (int(self.pocket_struct_dim) if self.pocket_use_struct_feat else 0)
        pocket_node_feat = torch.zeros((0, pocket_node_dim), dtype=torch.float32)
        pocket_edge_index = torch.zeros((2, 0), dtype=torch.long)
        pocket_edge_feat = torch.zeros((0, int(self.pocket_edge_feat_dim_expected)), dtype=torch.float32)
        pocket_slices: List[Tuple[int, int]] = []
        pocket_valid = torch.zeros((self.pocket_topk,), dtype=torch.bool)
        pocket_score = torch.zeros((self.pocket_topk,), dtype=torch.float32)

        if self.esm_mode == "tokens" and self.p2rank_index and self.pdb_index:
            pid = protein_id
            if pid:
                seq_len = int((esm_mask > 0).sum().item())
                templates = self._get_templates(pid, seq_len)
                tok_pos = torch.nonzero(esm_mask > 0, as_tuple=False).view(-1)

                node_list: List[torch.Tensor] = []
                edge_list: List[torch.Tensor] = []
                edge_feat_list: List[torch.Tensor] = []
                n_acc = 0
                for gi, tmpl in enumerate(templates):
                    if tmpl is None:
                        pocket_slices.append((n_acc, n_acc))
                        continue
                    res_seq = tmpl.get("res_seq", [])
                    if not res_seq:
                        pocket_slices.append((n_acc, n_acc))
                        continue
                    keep_old = []
                    idx = []
                    for old_i, r in enumerate(res_seq):
                        if 1 <= int(r) <= int(tok_pos.numel()):
                            keep_old.append(int(old_i))
                            idx.append(int(tok_pos[int(r) - 1].item()))
                    if not idx:
                        pocket_slices.append((n_acc, n_acc))
                        continue
                    nf_esm = esm_tok.index_select(0, torch.tensor(idx, dtype=torch.long))
                    node_tok_pos = torch.tensor(idx, dtype=torch.float32)
                    if self.pocket_use_struct_feat:
                        node_attr = tmpl.get("node_attr", None)
                        node_is_raw = tmpl.get("node_is_raw", None)
                        if isinstance(node_attr, torch.Tensor) and node_attr.numel() > 0:
                            node_attr = node_attr.index_select(0, torch.tensor(keep_old, dtype=torch.long)).float()
                        else:
                            node_attr = torch.zeros((len(idx), max(0, self.pocket_struct_dim - 1)), dtype=torch.float32)
                        if isinstance(node_is_raw, torch.Tensor) and node_is_raw.numel() > 0:
                            node_is_raw = node_is_raw.index_select(0, torch.tensor(keep_old, dtype=torch.long)).view(-1, 1).float()
                        else:
                            node_is_raw = torch.ones((len(idx), 1), dtype=torch.float32)
                        struct = torch.cat([node_attr, node_is_raw], dim=1)
                        if int(struct.shape[1]) < int(self.pocket_struct_dim):
                            pad = int(self.pocket_struct_dim) - int(struct.shape[1])
                            struct = torch.cat([struct, torch.zeros((struct.shape[0], pad), dtype=torch.float32)], dim=1)
                        elif int(struct.shape[1]) > int(self.pocket_struct_dim):
                            struct = struct[:, : int(self.pocket_struct_dim)]
                        nf = torch.cat([nf_esm, struct], dim=1)
                    else:
                        nf = nf_esm
                    node_list.append(nf)
                    ei = tmpl.get("edge_index", None)
                    dist = tmpl.get("edge_dist", None)
                    edge_rbf = tmpl.get("edge_rbf", None)
                    edge_type = tmpl.get("edge_type", None)
                    if isinstance(ei, torch.Tensor) and ei.numel() > 0:
                        n_old = int(len(res_seq))
                        mapping = torch.full((n_old,), -1, dtype=torch.long)
                        mapping[torch.tensor(keep_old, dtype=torch.long)] = torch.arange(len(keep_old), dtype=torch.long)
                        src = ei[0].long()
                        dst = ei[1].long()
                        if src.numel() > 0:
                            src_new = mapping[src]
                            dst_new = mapping[dst]
                            mask = (src_new >= 0) & (dst_new >= 0)
                            if mask.any():
                                ei_new = torch.stack([src_new[mask], dst_new[mask]], dim=0)
                                edge_list.append(ei_new + n_acc)
                                if isinstance(dist, torch.Tensor) and dist.numel() > 0:
                                    if dist.dim() == 1:
                                        dist = dist.view(-1, 1)
                                    dist_sel = dist[mask].float()
                                else:
                                    dist_sel = torch.zeros((int(mask.sum().item()), 1), dtype=torch.float32)
                                if isinstance(edge_rbf, torch.Tensor) and edge_rbf.numel() > 0:
                                    rbf_sel = edge_rbf[mask].float()
                                else:
                                    rbf_sel = torch.zeros((int(mask.sum().item()), int(self.pocket_edge_rbf_bins)), dtype=torch.float32)
                                if isinstance(edge_type, torch.Tensor) and edge_type.numel() > 0:
                                    et_sel = edge_type[mask].float()
                                else:
                                    et_sel = torch.zeros((int(mask.sum().item()), 3), dtype=torch.float32)
                                ef_parts = [dist_sel, rbf_sel, et_sel]
                                if self.pocket_edge_use_seqdist:
                                    seq_src = ei_new[0].long()
                                    seq_dst = ei_new[1].long()
                                    seq_dist = (node_tok_pos.index_select(0, seq_src) - node_tok_pos.index_select(0, seq_dst)).abs().view(-1, 1)
                                    seq_dist = _normalize_seqdist(seq_dist, self.pocket_seqdist_norm, self.pocket_seqdist_norm_denom)
                                    ef_parts.append(seq_dist)
                                edge_feat_list.append(torch.cat(ef_parts, dim=1))
                    pocket_slices.append((n_acc, n_acc + int(nf.shape[0])))
                    pocket_valid[gi] = True
                    pocket_score[gi] = float(tmpl.get("score", 0.0))
                    n_acc += int(nf.shape[0])

                if node_list:
                    pocket_node_feat = torch.cat(node_list, dim=0)
                if edge_list:
                    pocket_edge_index = torch.cat(edge_list, dim=1)
                if edge_feat_list:
                    pocket_edge_feat = torch.cat(edge_feat_list, dim=0)
                if not pocket_slices:
                    pocket_slices = [(0, 0) for _ in range(self.pocket_topk)]

        item: Dict[str, Any] = {
            "num_nodes": int(num_nodes),
            "edge_index": edge_index,  # [2, E]
            "edge_attr": edge_attr,    # [E, De]
            "node_feat": node_feat,    # [N, F]
            "node_pos": node_pos,      # [N, 3]
            "drug_fp": drug_fp,        # [Df]
            "drug_desc": drug_desc,    # [Dd]
            "pharm_type": pharm_type,  # [P]
            "pharm_mask": pharm_mask,  # [P]
            "esm2_tok": esm_tok,       # [L, d_esm]
            "esm2_mask": esm_mask,     # [L]
            "pocket_node_feat": pocket_node_feat,      # [Np, d_esm]
            "pocket_edge_index": pocket_edge_index,    # [2, Ep]
            "pocket_edge_feat": pocket_edge_feat,      # [Ep, 2]
            "pocket_slices": pocket_slices,            # list[(s,e)] len=G
            "pocket_valid": pocket_valid,              # [G]
            "pocket_score": pocket_score,              # [G]
            "row_index": int(ridx),
            "protein_id": protein_id,
            "smiles": smiles,
        }
        if pharm_anchor is not None:
            item["pharm_anchor"] = pharm_anchor  # [P]
        if pharm_family is not None:
            item["pharm_family"] = pharm_family
        if pharm_center is not None:
            item["pharm_center"] = pharm_center
        if pharm_geom is not None:
            item["pharm_geom"] = pharm_geom
        if pharm_context is not None:
            item["pharm_context"] = pharm_context
        if pharm_multi_anchor is not None:
            item["pharm_multi_anchor"] = pharm_multi_anchor
        if pharm_multi_anchor_mask is not None:
            item["pharm_multi_anchor_mask"] = pharm_multi_anchor_mask
        return item, y


def collate_fn_real_pocket(
    batch: List[Tuple[Dict[str, Any], torch.Tensor]],
    include_pocket: bool = True,
) -> Tuple[Dict[str, Any], torch.Tensor]:
    items, ys = zip(*batch)
    y = torch.stack(list(ys), dim=0)  # [B]

    # graph batching (drug)
    node_feats: List[torch.Tensor] = []
    node_pos_list: List[torch.Tensor] = []
    edge_indices: List[torch.Tensor] = []
    edge_attrs: List[torch.Tensor] = []
    node_slices: List[Tuple[int, int]] = []
    drug_edge_dim = int(items[0]["edge_attr"].shape[1]) if items else int(DRUG_EDGE_FEAT_DIM)
    n_acc = 0
    for it in items:
        nf = it["node_feat"]
        ei = it["edge_index"]
        ea = it["edge_attr"]
        if ea.dim() == 1:
            ea = ea.view(-1, 1)
        cur_dim = int(ea.shape[1]) if ea.numel() > 0 else int(ea.shape[1]) if ea.dim() == 2 else 0
        if cur_dim < drug_edge_dim:
            pad = torch.zeros((int(ea.shape[0]), drug_edge_dim - cur_dim), dtype=ea.dtype)
            ea = torch.cat([ea, pad], dim=1)
        elif cur_dim > drug_edge_dim:
            ea = ea[:, :drug_edge_dim]
        n = int(it["num_nodes"])
        node_feats.append(nf)
        node_pos_list.append(it["node_pos"].float())
        if ei.numel() > 0:
            edge_indices.append(ei + n_acc)
        edge_attrs.append(ea)
        node_slices.append((n_acc, n_acc + n))
        n_acc += n
    node_feat = torch.cat(node_feats, dim=0) if node_feats else torch.zeros((0, NODE_FEAT_DIM), dtype=torch.float32)
    node_pos = torch.cat(node_pos_list, dim=0) if node_pos_list else torch.zeros((0, 3), dtype=torch.float32)
    edge_index = torch.cat(edge_indices, dim=1) if edge_indices else torch.zeros((2, 0), dtype=torch.long)
    edge_attr = (
        torch.cat(edge_attrs, dim=0)
        if edge_attrs
        else torch.zeros((0, int(drug_edge_dim)), dtype=torch.float32)
    )
    drug_fp = torch.stack([it["drug_fp"].float() for it in items], dim=0) if items else torch.zeros((0, 0), dtype=torch.float32)
    drug_desc = torch.stack([it["drug_desc"].float() for it in items], dim=0) if items else torch.zeros((0, 0), dtype=torch.float32)

    # pharm
    pharm_type = torch.stack([it["pharm_type"] for it in items], dim=0)  # [B,P]
    pharm_mask = torch.stack([it["pharm_mask"] for it in items], dim=0)  # [B,P]
    has_anchor = all(("pharm_anchor" in it) for it in items)
    pharm_anchor = torch.stack([it["pharm_anchor"] for it in items], dim=0) if has_anchor else None
    has_family = all(("pharm_family" in it) for it in items)
    pharm_family = torch.stack([it["pharm_family"] for it in items], dim=0) if has_family else None
    has_center = all(("pharm_center" in it) for it in items)
    pharm_center = torch.stack([it["pharm_center"] for it in items], dim=0) if has_center else None
    has_geom = all(("pharm_geom" in it) for it in items)
    pharm_geom = torch.stack([it["pharm_geom"] for it in items], dim=0) if has_geom else None
    has_context = all(("pharm_context" in it) for it in items)
    pharm_context = torch.stack([it["pharm_context"] for it in items], dim=0) if has_context else None
    has_multi_anchor = all(("pharm_multi_anchor" in it) for it in items)
    pharm_multi_anchor = torch.stack([it["pharm_multi_anchor"] for it in items], dim=0) if has_multi_anchor else None
    has_multi_anchor_mask = all(("pharm_multi_anchor_mask" in it) for it in items)
    pharm_multi_anchor_mask = (
        torch.stack([it["pharm_multi_anchor_mask"] for it in items], dim=0) if has_multi_anchor_mask else None
    )

    # esm tokens (pad)
    Lmax = max(int(it["esm2_tok"].shape[0]) for it in items)
    d = int(items[0]["esm2_tok"].shape[1])
    esm_tok = torch.zeros((len(items), Lmax, d), dtype=torch.float32)
    esm_mask = torch.zeros((len(items), Lmax), dtype=torch.long)
    for b, it in enumerate(items):
        L = int(it["esm2_tok"].shape[0])
        esm_tok[b, :L] = it["esm2_tok"]
        esm_mask[b, :L] = it["esm2_mask"]

    batch_out: Dict[str, Any] = {
        "node_feat": node_feat,
        "node_pos": node_pos,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "node_slices": node_slices,
        "drug_fp": drug_fp,
        "drug_desc": drug_desc,
        "pharm_type": pharm_type,
        "pharm_mask": pharm_mask,
        "esm2_tok": esm_tok,
        "esm2_mask": esm_mask,
        "row_index": [int(it["row_index"]) for it in items],
        "protein_id": [str(it["protein_id"]) for it in items],
        "smiles": [str(it["smiles"]) for it in items],
    }
    if pharm_anchor is not None:
        batch_out["pharm_anchor"] = pharm_anchor
    if pharm_family is not None:
        batch_out["pharm_family"] = pharm_family
    if pharm_center is not None:
        batch_out["pharm_center"] = pharm_center
    if pharm_geom is not None:
        batch_out["pharm_geom"] = pharm_geom
    if pharm_context is not None:
        batch_out["pharm_context"] = pharm_context
    if pharm_multi_anchor is not None:
        batch_out["pharm_multi_anchor"] = pharm_multi_anchor
    if pharm_multi_anchor_mask is not None:
        batch_out["pharm_multi_anchor_mask"] = pharm_multi_anchor_mask

    if include_pocket:
        pocket_node_dim = int(items[0]["pocket_node_feat"].shape[1]) if items else d
        pocket_edge_dim = max([int(it["pocket_edge_feat"].shape[1]) for it in items], default=0)
        pocket_node_feats: List[torch.Tensor] = []
        pocket_edge_indices: List[torch.Tensor] = []
        pocket_edge_feats: List[torch.Tensor] = []
        pocket_graph_slices: List[Tuple[int, int]] = []
        pocket_graph_batch: List[int] = []
        pocket_graph_valid_list: List[torch.Tensor] = []
        pocket_graph_score_list: List[torch.Tensor] = []

        p_acc = 0
        for b, it in enumerate(items):
            pn = it["pocket_node_feat"]
            pe = it["pocket_edge_index"]
            pd = it["pocket_edge_feat"]
            if pd.dim() == 1:
                pd = pd.view(-1, 1)
            cur_pd_dim = int(pd.shape[1]) if pd.dim() == 2 else 0
            if cur_pd_dim < pocket_edge_dim:
                pad = torch.zeros((int(pd.shape[0]), pocket_edge_dim - cur_pd_dim), dtype=pd.dtype)
                pd = torch.cat([pd, pad], dim=1)
            elif cur_pd_dim > pocket_edge_dim:
                pd = pd[:, :pocket_edge_dim]
            if pn.numel() > 0:
                pocket_node_feats.append(pn)
            if pe.numel() > 0:
                pocket_edge_indices.append(pe + p_acc)
            if pd.numel() > 0:
                pocket_edge_feats.append(pd)

            slices = it["pocket_slices"]
            for (s, e) in slices:
                pocket_graph_slices.append((s + p_acc, e + p_acc))
                pocket_graph_batch.append(b)

            pocket_graph_valid_list.append(it["pocket_valid"])
            pocket_graph_score_list.append(it["pocket_score"])
            p_acc += int(pn.shape[0])

        pocket_node_feat = (
            torch.cat(pocket_node_feats, dim=0)
            if pocket_node_feats
            else torch.zeros((0, pocket_node_dim), dtype=torch.float32)
        )
        pocket_edge_index = (
            torch.cat(pocket_edge_indices, dim=1)
            if pocket_edge_indices
            else torch.zeros((2, 0), dtype=torch.long)
        )
        pocket_edge_feat = (
            torch.cat(pocket_edge_feats, dim=0)
            if pocket_edge_feats
            else torch.zeros((0, pocket_edge_dim), dtype=torch.float32)
        )
        if pocket_edge_index.numel() > 0:
            n_total = int(pocket_node_feat.shape[0])
            src = pocket_edge_index[0].long()
            dst = pocket_edge_index[1].long()
            mask = (src >= 0) & (src < n_total) & (dst >= 0) & (dst < n_total)
            if not torch.all(mask):
                pocket_edge_index = pocket_edge_index[:, mask]
                if pocket_edge_feat.numel() == int(mask.numel() * pocket_edge_feat.shape[1]):
                    pocket_edge_feat = pocket_edge_feat[mask]
                elif pocket_edge_feat.numel() == int(mask.sum().item() * pocket_edge_feat.shape[1]):
                    # already aligned
                    pass
                else:
                    # fallback: reset edge features with the current expected dim
                    pocket_edge_feat = torch.zeros((int(pocket_edge_index.shape[1]), pocket_edge_dim), dtype=torch.float32)
        pocket_graph_batch_t = (
            torch.tensor(pocket_graph_batch, dtype=torch.long)
            if pocket_graph_batch
            else torch.zeros((0,), dtype=torch.long)
        )
        pocket_graph_valid = (
            torch.cat(pocket_graph_valid_list, dim=0)
            if pocket_graph_valid_list
            else torch.zeros((0,), dtype=torch.bool)
        )
        pocket_graph_score = (
            torch.cat(pocket_graph_score_list, dim=0)
            if pocket_graph_score_list
            else torch.zeros((0,), dtype=torch.float32)
        )
        batch_out.update({
            "pocket_node_feat": pocket_node_feat,
            "pocket_edge_index": pocket_edge_index,
            "pocket_edge_feat": pocket_edge_feat,
            "pocket_graph_slices": pocket_graph_slices,
            "pocket_graph_batch": pocket_graph_batch_t,
            "pocket_graph_valid": pocket_graph_valid,
            "pocket_graph_score": pocket_graph_score,
        })

    return batch_out, y
