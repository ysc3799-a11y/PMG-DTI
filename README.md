# PMG-DTI

Paper-aligned implementation of PMG-DTI for drug--target interaction prediction using molecular graph, pharmacophore, protein-sequence and binding-pocket evidence.

## Main settings

The release configurations use stratified five-fold cross-validation and the manuscript loss weights:

```text
lambda_local = 0.05
lambda_cal   = 0.02
lambda_align = 0.01
lambda_rank  = 0.20
batch size   = 16
grad accum   = 1
```

Pocket nodes include ESM-2 embeddings, residue-type and physicochemical descriptors, pLDDT and spatial features (`POCKET_USE_STRUCT_FEAT=1`, `POCKET_STRUCT_DIM=29`).

## Environment

```bash
conda env create -f environment.yml
conda activate pmg-dti
```

CUDA-specific PyTorch and DGL wheels may need to be installed from their official indexes for the local CUDA version.

## Data and split preparation

Raw third-party datasets and large derived caches are not redistributed. See `data/README.md`, `preprocess/README.md` and `REPRODUCIBILITY.md`.

After preparing the input CSV files, deterministically regenerate the five-fold indices:

```bash
python scripts/make_fivefold_splits.py --dataset all --seed 42
```

The generator verifies the source CSV SHA-256 values and writes the derived files under `splits/`. Generated split indices are not committed to Git.

## Five-fold training

Human:

```bash
python scripts/run_fivefold.py --dataset human --device cuda:0 --seed <TRAINING_SEED>
```

BioSNAP:

```bash
python scripts/run_fivefold.py --dataset biosnap --device cuda:0 --seed <TRAINING_SEED>
```

Repeat the command for each training seed actually reported in the manuscript. Split seed and training seed are distinct; see `REPRODUCIBILITY.md`.

## Evaluation

The training entry point supports checkpoint-only evaluation with `--eval_only_ckpt`. The classification threshold is selected on the validation set and then applied unchanged to the test set.

## Availability

Large pretrained checkpoints should be distributed through a versioned GitHub Release or a durable archival repository rather than committed to Git history. Add the final licence and citation metadata before making this repository public.
