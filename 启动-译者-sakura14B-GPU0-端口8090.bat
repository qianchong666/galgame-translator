@echo off
chcp 936 >nul
title sakura-14B translator GPU0 port 8090
cd /d "%~dp0"
echo ============================================
echo   sakura-galtransl-14B   GPU0   :8090
echo ============================================
curl -s -m 2 -o nul "http://127.0.0.1:8090/v1/models"
if not errorlevel 1 (
  echo 已在运行，无需重复启动。
  if /i not "%~1"=="nopause" pause
  exit /b 0
)
echo 正在启动... 首次加载约 1-3 分钟
echo 本窗口不要关闭，关掉就等于停服务
echo.
wsl.exe -d ubuntu-2404 -- bash -c "CUDA_VISIBLE_DEVICES=0 VLLM_USE_FLASHINFER_SAMPLER=0 /root/venvs/sakura/bin/vllm serve /mnt/g/aimodels/Sakura-GalTransl-14B-v3.8-GPTQ-Int4 --served-model-name sakura-galtransl-14b --max-model-len 16384 --gpu-memory-utilization 0.9 --host 0.0.0.0 --port 8090"
echo.
echo 服务已退出。
if /i not "%~1"=="nopause" pause
