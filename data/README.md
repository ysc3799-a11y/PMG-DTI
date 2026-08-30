# Data sources, acquisition and local layout

PMG-DTI uses third-party interaction tables, predicted protein structures and
pretrained-model resources. Raw third-party files, AlphaFold structures,
P2Rank outputs, ESM-2 weights and large derived caches are **not** redistributed
in this repository. Download them from their official sources and review the
upstream terms before use. The source links and file identities below were
verified on 2026-08-30.

## 1. Human and BioSNAP interaction tables

The exact Human and BioSNAP tables used here are the copies distributed in the
official [DrugBAN repository](https://github.com/peizhenbai/DrugBAN). The local
files were compared byte for byte with the corresponding files in that
repository.

| Dataset | Official source path | Local raw path | Rows | Required columns | SHA-256 |
| --- | --- | --- | ---: | --- | --- |
| Human | [`datasets/human/full.csv`](https://github.com/peizhenbai/DrugBAN/blob/main/datasets/human/full.csv) | `data/raw/human/moltrans/full.csv` | 6,728 | `SMILES, Protein, Y` | `ccda331c2697dbd12c96e5c4e855e98f162b9766086d945913bd0b90154aede3` |
| BioSNAP | [`datasets/biosnap/full.csv`](https://github.com/peizhenbai/DrugBAN/blob/main/datasets/biosnap/full.csv) | `data/raw/biosnap/full.csv` | 27,464 | `SMILES, Protein, Y` (`drug_cluster` and `target_cluster` are also present upstream) | `b0155eb273f0ceea2b883c2eb66108365e679c1d66c96cef413133257f991585` |

Retrieve the files from the repository root:

```bash
python preprocess/scripts/00_fetch_moltrans.py --dataset human
python preprocess/scripts/00_fetch_moltrans.py --dataset biosnap
```

The script also accepts `--src_csv /path/to/full.csv` for an already downloaded
copy. Check the identity before preprocessing:

```bash
sha256sum data/raw/human/moltrans/full.csv
sha256sum data/raw/biosnap/full.csv
```

On PowerShell, use `Get-FileHash <path> -Algorithm SHA256`. A mismatch means the
upstream table or its byte-level serialization differs from the version used by
PMG-DTI. For BioSNAP, cite Huang *et al.*,
[MolTrans](https://doi.org/10.1093/bioinformatics/btaa880); for Human, cite Chen
*et al.*, [TransformerCPI](https://doi.org/10.1093/bioinformatics/btaa524). Also
cite the [DrugBAN paper](https://doi.org/10.1038/s42256-022-00605-1) for the
distribution from which these exact files were obtained. Dataset reuse remains
subject to the upstream terms; this repository does not grant additional rights
over those tables.

Create the row-aligned tables used by the model:

```bash
python preprocess/scripts/01_align_pid_and_clean.py

python preprocess/scripts/01_align_pid_and_clean.py \
  --in_csv data/raw/biosnap/full.csv \
  --out_csv data/interim/biosnap/pairs_with_pid/full_with_pid.csv \
  --out_protein_table data/interim/biosnap/pairs_with_pid/protein_table.csv \
  --out_summary_json data/interim/biosnap/pairs_with_pid/summary.json
```

The resulting experiment tables have the following fixed identities. Do not
sort, deduplicate or otherwise rewrite them before generating split indices.

| Dataset | Processed rows | Columns | SHA-256 |
| --- | ---: | --- | --- |
| Human | 6,728 | `SMILES, Protein, Y, Protein_ID, Protein_Len` | `f52517071e32bc28437df974e913521835394d0be75015df477f57f166d3b908` |
| BioSNAP | 27,464 | `SMILES, Protein, Y, Protein_ID, Protein_Len` | `b6056790a65910e22bc2dcddd77fb6d200cff38f5b58474675357005b7eac90b` |

## 2. UniProt mapping prerequisite

AlphaFold DB retrieval is keyed by UniProt accession. Export the appropriate
reviewed protein table, including accession and sequence columns, from
[UniProtKB](https://www.uniprot.org/uniprotkb), place it at the path supplied to
`--uniprot_tsv`, and map target sequences by exact sequence identity:

```bash
python preprocess/scripts/02b_map_uniprot_by_sequence.py \
  --protein_table data/interim/human/pairs_with_pid/protein_table.csv \
  --uniprot_tsv data/raw/human/moltrans/uniprot_human_reviewed.tsv \
  --out_map_csv data/interim/human/pairs_with_pid/protein_uniprot_map.csv
```

Use dataset-specific paths for BioSNAP. Only mappings with the accepted status
are passed to the structure downloader. For a final archival release, preserve
the UniProt export query, retrieval date and SHA-256 beside the mapping table.

## 3. AlphaFold Protein Structure Database

Protein structures used for pocket prediction are downloaded by UniProt
accession from the [AlphaFold Protein Structure Database](https://alphafold.ebi.ac.uk/).
`03b_fetch_alphafold_structures.py` first queries the official
[prediction API](https://alphafold.ebi.ac.uk/api-docs) and uses the returned PDB
URL. For compatibility with older records, it then tries the `model_v4`,
`model_v3` and `model_v2` file URLs if the API query fails. The generated
`download_report.json` and `pdb_index.csv` record download status and source
paths.

```bash
python preprocess/scripts/03b_fetch_alphafold_structures.py

python preprocess/scripts/03b_fetch_alphafold_structures.py \
  --map_csv data/interim/biosnap/pairs_with_pid/protein_uniprot_map.csv \
  --out_dir data/raw/biosnap/alphafold_structures \
  --out_pdb_index_csv data/raw/biosnap/p2rank_raw/pdb_index.csv \
  --out_report_json data/raw/biosnap/alphafold_structures/download_report.json
```

AlphaFold DB structures are provided under the database's
[CC BY 4.0 licence and disclaimer](https://alphafold.ebi.ac.uk/assets/License-Disclaimer.pdf)
and remain subject to the [EMBL-EBI terms of use](https://www.ebi.ac.uk/about/terms-of-use/).
The files are predictions, not experimentally determined structures; consult
the confidence values before biological interpretation.

## 4. P2Rank pocket predictions

Binding pockets were predicted locally with
[P2Rank](https://github.com/rdk/p2rank), version **2.5.1**, downloaded from the
official [release archive](https://github.com/rdk/p2rank/releases). The retained
run logs show that the experiments used P2Rank's `models/default` profile. This
is recorded here as the actual experimental setting; changing to another
profile creates a different derived dataset.

After installing Java and unpacking P2Rank 2.5.1, run its `prank` launcher on
each PDB file listed in `pdb_index.csv`, for example:

```bash
prank predict -f path/to/protein.pdb -o data/raw/human/p2rank_raw/p2rank_out/protein_id
```

Do not add `-c alphafold` when reproducing the released PMG-DTI pocket cache,
because the audited runs used `models/default`. Preserve each generated
`run.log`, `*_predictions.csv` and `*_residues.csv`. Then build the index used by
PMG-DTI:

```bash
python preprocess/scripts/04b_build_protein_pockets_index.py \
  --p2rank_out_dir data/raw/human/p2rank_raw/p2rank_out \
  --out_csv data/raw/human/p2rank_raw/protein_pockets.csv
```

Repeat with BioSNAP paths. P2Rank is external software and is not bundled here;
follow its upstream licence and cite the P2Rank publication specified by its
official repository.

## 5. ESM-2 residue embeddings

Residue-level protein representations are generated with Meta FAIR's
[ESM](https://github.com/facebookresearch/esm) package (`fair-esm==2.0.0`) and
the pretrained **ESM-2 `esm2_t6_8M_UR50D`** checkpoint. This model has 6 layers,
approximately 8 million parameters and a 320-dimensional representation; the
official model table identifies its training data as UR50/D 2021_04. The PMG-DTI
script stores final-layer per-residue embeddings in FP16, truncates sequences at
1,022 residues and does not include beginning/end special tokens.

```bash
python preprocess/scripts/03_extract_esm2_tokens.py \
  --in_csv data/interim/human/pairs_with_pid/full_with_pid.csv \
  --model_name esm2_t6_8M_UR50D \
  --pool last \
  --dtype fp16 \
  --max_len 1022 \
  --out_pt data/interim/human/esm2_tokens/esm2_tokens_t6_fp16.pt

python preprocess/scripts/03_extract_esm2_tokens.py \
  --in_csv data/interim/biosnap/pairs_with_pid/full_with_pid.csv \
  --model_name esm2_t6_8M_UR50D \
  --pool last \
  --dtype fp16 \
  --max_len 1022 \
  --out_pt data/interim/biosnap/esm2_tokens/esm2_tokens_t6_fp16.pt
```

`fair-esm` downloads the upstream pretrained checkpoint on first use. Neither
the upstream weights nor the derived embedding caches are committed to this
repository. Follow the [ESM licence](https://github.com/facebookresearch/esm/blob/main/LICENSE)
and cite the ESM-2 publication listed by the official project.

## 6. RCSB Protein Data Bank

Experimentally determined holo structures from
[RCSB PDB](https://www.rcsb.org/) are queried only for the real-complex mapping
and interpretation/visualization cases. They do **not** replace the AlphaFold DB
structures used to construct the training pocket graphs and do not define the
DTI labels or data splits.

The workflow uses the official [Search and Data APIs](https://www.rcsb.org/docs/programmatic-access/web-apis-overview)
through `preprocess/scripts/03d_query_rcsb_holo_structures.py`. Selected mmCIF,
legacy PDB and ligand SDF assets are obtained with
`preprocess/scripts/03f_download_rcsb_visualization_assets.py` from the official
[file-download service](https://www.rcsb.org/docs/programmatic-access/file-download-services).
Generated candidate tables and `metadata.json` files retain the PDB entry ID,
ligand component ID, source URLs and download status.

PDB archive files and RCSB programmatic API data are available under the
[CC0 1.0 usage policy](https://www.rcsb.org/pages/usage-policy). Cite RCSB PDB
and the primary structure depositors for the final selected complexes wherever
possible.

## 7. Local directory layout

```text
data/
|-- raw/
|   |-- human/
|   `-- biosnap/
|       |-- alphafold_structures/
|       `-- p2rank_raw/
|-- interim/
|   |-- human/
|   `-- biosnap/
|       |-- pairs_with_pid/
|       |-- pharm_features/
|       `-- esm2_tokens/
`-- processed/
    |-- human/
    `-- biosnap/
        |-- drug_graph_cache/
        `-- pocket_graph_cache_top3_prob01_expand0/
```

The exact runtime paths are defined in `configs/human.yaml` and
`configs/biosnap.yaml`. Split indices are regenerated deterministically with
`scripts/make_fivefold_splits.py`; source-table hashes, row-order rules,
software versions and commands are documented in `REPRODUCIBILITY.md`.

## 8. Redistribution and archival checklist

- Do not commit raw Human/BioSNAP files, AlphaFold PDB files, P2Rank output
  directories, ESM-2 weights or large feature caches to Git history.
- Preserve checksums and machine-readable manifests for every released derived
  artifact.
- Preserve the exact UniProt export query/date/hash and the AlphaFold/RCSB
  source URL or accession for every structure used in a published case.
- Put large derived artifacts and pretrained PMG-DTI checkpoints in a versioned
  release or durable research-data repository, with an archival identifier.
- Include the required original publications in the manuscript reference list;
  URLs in this file are acquisition metadata, not substitutes for citations.
