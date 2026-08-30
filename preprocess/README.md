# Preprocessing workflow

Run all commands from the PMG-DTI repository root. Raw third-party datasets,
protein structures, P2Rank outputs, ESM-2 weights and large feature caches are
not included. Their official sources, verified table hashes, licences and exact
acquisition commands are documented in [`data/README.md`](../data/README.md).

## Execution order

| Stage | Script or external tool | Main output |
| --- | --- | --- |
| Interaction table retrieval | `preprocess/scripts/00_fetch_moltrans.py` | `data/raw/<dataset>/.../full.csv` |
| Row alignment and protein IDs | `preprocess/scripts/01_align_pid_and_clean.py` | `full_with_pid.csv`, `protein_table.csv` |
| Pharmacophore features | `preprocess/scripts/02_build_pharm_features.py` | `pharm_features.pt` |
| UniProt sequence mapping | `preprocess/scripts/02b_map_uniprot_by_sequence.py` | `protein_uniprot_map.csv` |
| Drug graph cache | `preprocess/scripts/02c_build_drug_graph_cache.py` | `drug_graph_cache.pt` |
| ESM-2 residue embeddings | `preprocess/scripts/03_extract_esm2_tokens.py` | `esm2_tokens_t6_fp16.pt` |
| AlphaFold DB structures | `preprocess/scripts/03b_fetch_alphafold_structures.py` | PDB files, `pdb_index.csv` |
| Pocket prediction | external P2Rank 2.5.1 | prediction and residue CSV files |
| Pocket-output index | `preprocess/scripts/04b_build_protein_pockets_index.py` | `protein_pockets.csv` |
| Pocket parsing/cache | `preprocess/scripts/04_parse_p2rank_pockets.py`, `preprocess/scripts/05_build_pocket_graph_cache.py` | pocket JSONL and graph cache |
| Coverage checks | `preprocess/scripts/07_coverage_report.py`, `preprocess/scripts/07_sanity_check_preproc.py` | coverage and validation reports |

The script defaults target Human. For BioSNAP, pass the corresponding
`data/raw/biosnap`, `data/interim/biosnap` and `data/processed/biosnap` paths as
shown in `data/README.md`. Do not change the row order of either
`full_with_pid.csv`; the deterministic split generator addresses rows by their
stored zero-based indices.

## Split generation

After all inputs are prepared, regenerate the manuscript five-fold split files:

```bash
python scripts/make_fivefold_splits.py --dataset all --seed 42
```

See [`REPRODUCIBILITY.md`](../REPRODUCIBILITY.md) for source-table hashes,
software versions, split semantics and the distinction between split and
training seeds.

## RCSB interpretation assets

`03d_query_rcsb_holo_structures.py` through
`03g_generate_chimerax_scripts.py` support the experimentally determined
protein-ligand complexes used for interpretation and visualization. These RCSB
structures do not replace the AlphaFold DB structures used to construct the
training pocket graphs.
