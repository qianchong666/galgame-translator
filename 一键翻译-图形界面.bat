@echo off
chcp 65001 >nul
title Yinuo Translator - sakura14B GPU0:8090 + Qwen3.5-9B GPU1:9092
cd /d "%~dp0"
if exist ".venv\Scripts\pythonw.exe" (
  start "" ".venv\Scripts\pythonw.exe" "tools\one_click_gui.py"
  exit /b 0
)
if exist ".venv\Scripts\python.exe" (
  start "" ".venv\Scripts\python.exe" "tools\one_click_gui.py"
  exit /b 0
)
echo [ERROR] .venv not found. Expected folder: %~dp0.venv
pause
exit /b 2
