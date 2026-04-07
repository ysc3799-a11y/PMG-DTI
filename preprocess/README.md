# 数据预处理目录框架（基于初稿）

本目录用于承载论文“数据预处理”部分，围绕 MolTrans Human 数据构建如下流程：

1. 原始数据落地（SMILES/序列/标签、结构文件、P2Rank 输出）
2. 主键对齐与清洗（`pid` 对齐、SMILES 规范化）
3. 药物侧功能单元构建（药效团特征）
4. 蛋白侧语义特征构建（ESM2 token）
5. 口袋候选读取与过滤（Top-K、分数阈值、最小残基数）
6. 口袋残基图缓存（空间距离 + 序列距离）
7. 数据划分（random / unseen_protein / unseen_drug）
8. 覆盖率与退化检查（ESM 命中率、pocket/cached 覆盖率）

## 建议目录结构

```text
Pha-DTI/
├─ preprocess/
│  ├─ configs/
│  │  └─ human_preprocess_template.yaml
│  ├─ scripts/
│  │  ├─ 00_fetch_moltrans.py
│  │  ├─ 01_align_pid_and_clean.py
│  │  ├─ 02_build_pharm_features.py
│  │  ├─ 03_extract_esm2_tokens.py
│  │  ├─ 04_parse_p2rank_pockets.py
│  │  ├─ 05_build_pocket_graph_cache.py
│  │  ├─ 06_make_splits.py
│  │  └─ 07_coverage_report.py
│  ├─ logs/
│  └─ reports/
└─ data/
   ├─ raw/
   │  └─ human/
   │     ├─ moltrans/
   │     ├─ alphafold_structures/
   │     └─ p2rank_raw/
   ├─ interim/
   │  └─ human/
   │     ├─ pairs_with_pid/
   │     ├─ drug_canon/
   │     ├─ pharm_features/
   │     ├─ esm2_tokens/
   │     └─ pocket_candidates/
   └─ processed/
      └─ human/
         ├─ pocket_graph_cache/
         ├─ pocket_manifest/
         ├─ splits/
         └─ coverage/
```

## 说明

- `raw`：外部来源的原始文件，不做覆盖式修改。
- `interim`：中间结果，可重算。
- `processed`：训练与评测直接消费的稳定产物（缓存、split、统计）。
- `coverage`：建议统一输出 `esm_coverage.json`、`pocket_coverage.json`、`summary.csv`，用于快速判断是否退化。

## 推荐运行顺序（Human）

```text
00_fetch_moltrans.py
01_align_pid_and_clean.py
02_build_pharm_features.py
02c_build_drug_graph_cache.py          # 离线药物图特征缓存（可选但推荐）
02b_map_uniprot_by_sequence.py        # 可选：按序列匹配 UniProt
03b_fetch_alphafold_structures.py     # 3D 结构下载，生成 pdb_index.csv
03_extract_esm2_tokens.py
04_parse_p2rank_pockets.py
05_build_pocket_graph_cache.py
06_make_splits.py
07_coverage_report.py
```
