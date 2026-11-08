# -*- coding: utf-8 -*-
"""测生成速度（长输出），顺便看 GPU 时钟是否真的在boost。"""
import subprocess
import sys
import time

import requests

port = int(sys.argv[1]) if len(sys.argv) > 1 else 9080
SYS = "你是翻译。把用户给的日文翻成中文，保持行数。"
user = "\n".join(["「これはテストです。ちゃんと長く訳してください。」"] * 60)

for trial in range(3):
    t0 = time.time()
    r = requests.post(f"http://127.0.0.1:{port}/v1/chat/completions",
                      json={"messages": [{"role": "system", "content": SYS},
                                         {"role": "user", "content": user}],
                            "max_tokens": 900, "temperature": 0.3}, timeout=600)
    el = time.time() - t0
    d = r.json()
    u = d.get("usage", {})
    clk = subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm,clocks.mem,power.draw,utilization.gpu",
                          "--format=csv,noheader"], capture_output=True, text=True,
                         encoding="utf-8", errors="replace").stdout.strip().replace("\n", " | ")
    print(f"trial{trial} el={el:.1f}s prompt={u.get('prompt_tokens')} out={u.get('completion_tokens')} "
          f"gen={(u.get('completion_tokens') or 0)/el:.1f} tok/s   [{clk}]")
