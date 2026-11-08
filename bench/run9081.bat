@echo off
cd /d G:\llama-cpp\b11874-cu13.4
llama-server.exe -m G:\aimodels\GalTransl-v4-4B-2601\Galtransl-v4-4B-2601.gguf ^
 --ctx-size 16384 --n-gpu-layers 999 --flash-attn on --parallel 4 --jinja ^
 -ctk q8_0 -ctv q8_0 --port 9081 --no-webui > F:\aiwork\galgame小说翻译\bench\srv9081.log 2>&1
