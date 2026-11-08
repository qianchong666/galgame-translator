@echo off
chcp 936 >nul
title Qwen3.5-9B-AWQ reviewer GPU1 port 9092
cd /d "%~dp0"
echo ============================================
echo   Qwen3.5-9B-AWQ   GPU1   :9092
echo ============================================
curl -s -m 2 -o nul "http://127.0.0.1:9092/v1/models"
if not errorlevel 1 (
  echo 已在运行，无需重复启动。
  if /i not "%~1"=="nopause" pause
  exit /b 0
)
echo 正在启动... 首次加载约 1-2 分钟
echo 本窗口不要关闭，关掉就等于停服务
echo.
wsl.exe -d ubuntu-2404 -- bash -c "CUDA_VISIBLE_DEVICES=1 VLLM_USE_FLASHINFER_SAMPLER=0 /root/venvs/sakura/bin/vllm serve /mnt/g/aimodels/Qwen35-9B-DefiantFable-W4A16-AWQ --served-model-name qwen35-9b-awq --max-model-len 16384 --gpu-memory-utilization 0.9 --enable-auto-tool-choice --tool-call-parser hermes --host 0.0.0.0 --port 9092"
echo.
echo 服务已退出。
if /i not "%~1"=="nopause" pause
