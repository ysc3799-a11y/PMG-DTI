# 旧版口袋生成脚本放置说明

这套脚本来自你之前的 `scripts.zip`，建议作为“旧版口袋流水线”单独保留在：

`preprocess/pocket_pipeline_legacy/`

原因：
- 这套脚本是完整链路（UniProt映射 + AFDB下载 + P2Rank + 汇总），和你当前 `preprocess/scripts/00~07` 是并行方案，不应混写。
- 脚本里有 `.sh`（Linux/bash），与当前 Windows/PowerShell 环境分开更安全。
- 便于后续做对照实验：`legacy` vs `new`.

## 与当前 Pha-DTI 目录的推荐对接

1. 输入 pairs csv：
- 用 `data/interim/human/pairs_with_pid/full_with_pid.csv`
- 该文件至少要有 `Protein` 序列列。

2. 结构下载输出目录：
- 推荐写到 `data/raw/human/alphafold_structures/`

3. P2Rank 输出目录：
- 推荐写到 `data/raw/human/p2rank_raw/p2rank_out/`

4. pockets 汇总文件：
- 推荐写到 `data/raw/human/p2rank_raw/protein_pockets.csv`

5. 供训练读取的 pdb 索引：
- 建议额外生成/维护 `data/raw/human/p2rank_raw/pdb_index.csv`

## 脚本清单（原样保留）

- `01_make_unique_proteins.py`
- `02_download_uniprot_human_reviewed_fasta.sh`
- `03_map_exact.py`
- `04_map_similarity_mmseqs.sh`
- `05_merge_mapping.py`
- `06_download_afdb_structures.py`
- `07_run_p2rank_batch.sh`
- `08_collect_pockets.py`
- `09_join_pairs_with_pockets.py`

## 注意

- `02/04/07` 是 bash 脚本，Windows 直接运行需要 WSL 或 Git Bash。
- `09_join_pairs_with_pockets.py` 内部按 `Protein -> md5 -> Protein_ID` 重新生成 ID；你当前新流程用的是 `sha1`。两种 ID 不兼容，混用会导致对不上。建议二选一并保持一致。

