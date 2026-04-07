@echo off
setlocal

cd /d "%~dp0"

echo [INFO] Running Top-K pocket ablation experiments from:
echo %CD%
echo [INFO] Run run_topk_cache_build.bat first if K=1/2/5 caches are missing.
echo [INFO] This batch is currently narrowed to the unfinished Table 9 runs only.
echo.

rem Human: table already has K=1 and K=3 filled.
call :run_cfg_if_missing "human_topk_pocket_k2_3view_soft_pocket_bias_emaes" "configs/topk_ablation/human_topk_pocket_k2_3view_soft_pocket_bias_emaes.yaml"
if errorlevel 1 exit /b 1
call :run_cfg_if_missing "human_topk_pocket_k5_3view_soft_pocket_bias_emaes" "configs/topk_ablation/human_topk_pocket_k5_3view_soft_pocket_bias_emaes.yaml"
if errorlevel 1 exit /b 1

rem BioSNAP: table already has K=3 filled.
call :run_cfg_if_missing "biosnap_topk_pocket_k1_3view_soft_pocket_bias_emaes" "configs/topk_ablation/biosnap_topk_pocket_k1_3view_soft_pocket_bias_emaes.yaml"
if errorlevel 1 exit /b 1
call :run_cfg_if_missing "biosnap_topk_pocket_k2_3view_soft_pocket_bias_emaes" "configs/topk_ablation/biosnap_topk_pocket_k2_3view_soft_pocket_bias_emaes.yaml"
if errorlevel 1 exit /b 1
call :run_cfg_if_missing "biosnap_topk_pocket_k5_3view_soft_pocket_bias_emaes" "configs/topk_ablation/biosnap_topk_pocket_k5_3view_soft_pocket_bias_emaes.yaml"
if errorlevel 1 exit /b 1

echo.
echo [INFO] Top-K pocket ablation finished.
pause
exit /b 0

:run_cfg_if_missing
if exist "runs\topk_ablation\%~1_*" (
echo [SKIP] %~1 already has an existing run under runs\topk_ablation
goto :eof
)
call :run_cfg "%~2"
goto :eof

:run_cfg
echo [RUN ] %~1
python train/train_gated_canon_real_pocket_seqdist_3view_soft_pocket_bias_emaes.py --cfg "%~1"
goto :eof
