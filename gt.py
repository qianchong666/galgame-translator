#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""gtlib 入口。

用法示例
--------
  # 1) 只验证清洗
  python gt.py clean "G:\\book\\src.txt" -o out

  # 2) 看专名提取质量
  python gt.py terms "G:\\book\\src.txt" --show 60

  # 3) 完整翻译（自己拉起 llama-server）
  python gt.py translate "G:\\book\\src.txt" --server-cmd "G:\\llama-cpp\\...\\llama-server.exe -m G:\\aimodels\\xxx.gguf --ctx-size 8192 -ngl 999 -fa on" --gpus 0

  # 4) 连接已经跑着的服务
  python gt.py translate src.txt --base-url http://127.0.0.1:9080

  # 5) 中断后重跑同一条命令 = 断点续传
  # 6) 自检
  python gt.py selftest
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gtlib.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
