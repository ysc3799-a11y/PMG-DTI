# Reproducibility

## Input table identity

The split indices refer to the stored row order of each `full_with_pid.csv` file. The split generator loads the table with `pandas.read_csv()` and immediately applies `reset_index(drop=True)`. It does not sort, shuffle, or deduplicate the table before assigning row indices.

Do not reorder or deduplicate either CSV before regenerating the splits.

| Dataset | Rows | Columns | SHA-256 |
| --- | ---: | --- | --- |
| Human | 6,728 | `SMILES, Protein, Y, Protein_ID, Protein_Len` | `f52517071e32bc28437df974e913521835394d0be75015df477f57f166d3b908` |
| BioSNAP | 27,464 | `SMILES, Protein, Y, Protein_ID, Protein_Len` | `b6056790a65910e22bc2dcddd77fb6d200cff38f5b58474675357005b7eac90b` |

The generator verifies these hashes before writing any split. A hash mismatch means that the dataset version or byte-level CSV representation differs from the one used to define the released experiment.

## Software versions

- Python 3.9.23
- pandas 2.3.1
- scikit-learn 1.6.1

The complete environment is pinned in `requirements.txt` and `environment.yml`.

## Deterministic five-fold split generation

The outer stratified five-fold assignment uses split seed 42. Within each fold, 10% of the four-fold development partition is selected as validation data using seed `42 + fold_index`. Thus each fold contains approximately 72% training, 8% validation and 20% test data.

Generate both datasets from the repository root:

```bash
python scripts/make_fivefold_splits.py --dataset all --seed 42
```

Or generate them separately:

```bash
python scripts/make_fivefold_splits.py --dataset human --seed 42
python scripts/make_fivefold_splits.py --dataset biosnap --seed 42
```

The command creates `splits/<dataset>/fivefold_seed42/`, including five JSON index files, a statistics CSV and a machine-readable generation manifest. The generated `splits/` directory is intentionally excluded from Git because it can be reproduced deterministically.

## Split seed versus training seed

The split seed controls only fold membership. The training seed controls model initialization, mini-batch order, dropout and other stochastic training operations. A fixed `fivefold_seed42` split can therefore be reused with multiple training seeds.

Run one training seed over all five folds with:

```bash
python scripts/run_fivefold.py --dataset human --seed <TRAINING_SEED>
python scripts/run_fivefold.py --dataset biosnap --seed <TRAINING_SEED>
```

Repeat these commands for every training seed reported in the manuscript. Do not substitute an unverified seed list. In the retained PMG-DTI five-fold records available when this preview was assembled, only Human training seed 42 could be independently verified; additional seeds and a complete BioSNAP five-fold record must be verified against the final experiment logs before a multiple-seed claim is made.
