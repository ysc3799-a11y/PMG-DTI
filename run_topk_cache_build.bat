@echo off
setlocal

cd /d "%~dp0"

echo [INFO] Building missing Top-K pocket caches from:
echo %CD%
echo.

call :build_cache human 1 "data/interim/human/pairs_with_pid/full_with_pid.csv" "data/interim/human/esm2_tokens/esm2_tokens_t6_fp16.pt" "data/raw/human/p2rank_raw/protein_pockets.csv" "data/raw/human/p2rank_raw/pdb_index.csv"
if errorlevel 1 exit /b 1
call :build_cache human 2 "data/interim/human/pairs_with_pid/full_with_pid.csv" "data/interim/human/esm2_tokens/esm2_tokens_t6_fp16.pt" "data/raw/human/p2rank_raw/protein_pockets.csv" "data/raw/human/p2rank_raw/pdb_index.csv"
if errorlevel 1 exit /b 1
call :build_cache human 3 "data/interim/human/pairs_with_pid/full_with_pid.csv" "data/interim/human/esm2_tokens/esm2_tokens_t6_fp16.pt" "data/raw/human/p2rank_raw/protein_pockets.csv" "data/raw/human/p2rank_raw/pdb_index.csv"
if errorlevel 1 exit /b 1
call :build_cache human 5 "data/interim/human/pairs_with_pid/full_with_pid.csv" "data/interim/human/esm2_tokens/esm2_tokens_t6_fp16.pt" "data/raw/human/p2rank_raw/protein_pockets.csv" "data/raw/human/p2rank_raw/pdb_index.csv"
if errorlevel 1 exit /b 1

call :build_cache biosnap 1 "data/interim/biosnap/pairs_with_pid/full_with_pid.csv" "data/interim/biosnap/esm2_tokens/esm2_tokens_t6_fp16.pt" "data/raw/biosnap/p2rank_raw/protein_pockets.csv" "data/raw/biosnap/p2rank_raw/pdb_index.csv"
if errorlevel 1 exit /b 1
call :build_cache biosnap 2 "data/interim/biosnap/pairs_with_pid/full_with_pid.csv" "data/interim/biosnap/esm2_tokens/esm2_tokens_t6_fp16.pt" "data/raw/biosnap/p2rank_raw/protein_pockets.csv" "data/raw/biosnap/p2rank_raw/pdb_index.csv"
if errorlevel 1 exit /b 1
call :build_cache biosnap 3 "data/interim/biosnap/pairs_with_pid/full_with_pid.csv" "data/interim/biosnap/esm2_tokens/esm2_tokens_t6_fp16.pt" "data/raw/biosnap/p2rank_raw/protein_pockets.csv" "data/raw/biosnap/p2rank_raw/pdb_index.csv"
if errorlevel 1 exit /b 1
call :build_cache biosnap 5 "data/interim/biosnap/pairs_with_pid/full_with_pid.csv" "data/interim/biosnap/esm2_tokens/esm2_tokens_t6_fp16.pt" "data/raw/biosnap/p2rank_raw/protein_pockets.csv" "data/raw/biosnap/p2rank_raw/pdb_index.csv"
if errorlevel 1 exit /b 1

echo.
echo [INFO] Top-K pocket cache build finished.
pause
exit /b 0

:build_cache
set "DATASET=%~1"
set "TOPK=%~2"
set "IN_CSV=%~3"
set "ESM2_PT=%~4"
set "PROTEIN_POCKETS_CSV=%~5"
set "PDB_INDEX_CSV=%~6"
set "OUT_DIR=data/processed/%DATASET%/pocket_graph_cache_top%TOPK%_prob01_expand0"

if exist "%OUT_DIR%\*.pt" (
  echo [SKIP] %DATASET% K=%TOPK% cache already exists at %OUT_DIR%
  goto :eof
)

echo [RUN ] %DATASET% K=%TOPK% -^> %OUT_DIR%
python preprocess/scripts/05_build_pocket_graph_cache.py ^
  --in_csv "%IN_CSV%" ^
  --pid_col Protein_ID ^
  --seq_col Protein ^
  --esm2_pt "%ESM2_PT%" ^
  --protein_pockets_csv "%PROTEIN_POCKETS_CSV%" ^
  --pdb_index_csv "%PDB_INDEX_CSV%" ^
  --out_dir "%OUT_DIR%" ^
  --topk %TOPK% ^
  --max_nodes 96 ^
  --knn_k 16 ^
  --expand_min_nodes 0 ^
  --prob_thres 0.1 ^
  --min_residues 8 ^
  --seqdist_norm log1p ^
  --seqdist_norm_denom 128.0 ^
  --raw_only 1 ^
  --edge_mode radius ^
  --radius_cutoff 10.0 ^
  --radius_max_neighbors 32 ^
  --edge_rbf_bins 16 ^
  --edge_use_seqdist 1 ^
  --use_struct_feat 0
goto :eof
