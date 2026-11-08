@echo off
chcp 65001 >nul
title Yinuo Translator CLI - workers 3 / chunk 400
cd /d "%~dp0"
set "SRC=%~1"
if "%SRC%"=="" set /p "SRC=Drop a Japanese txt here and press Enter: "
set "SRC=%SRC:"=%"
if not exist "%SRC%" (
  echo [ERROR] file not found: %SRC%
  pause
  exit /b 2
)
".venv\Scripts\python.exe" tools\run_all.py --src "%SRC%" --workers 3 --chunk 400
if /i not "%~2"=="nopause" pause
exit /b 0
