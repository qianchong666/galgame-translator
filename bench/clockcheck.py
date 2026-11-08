# -*- coding: utf-8 -*-
import json
import threading
import time

import requests

res = {}


def go():
    t0 = time.time()
    r = requests.post("http://127.0.0.1:9080/v1/chat/completions",
                      json={"messages": [{"role": "user",
                                          "content": "次の文章を中国語に翻訳して、できるだけ長く書いてください：\n" + "あ" * 20}],
                            "max_tokens": 600, "temperature": 0.3}, timeout=300)
    d = r.json()
    u = d.get("usage", {})
    res["el"] = time.time() - t0
    res["tok"] = u.get("completion_tokens")
    res["txt"] = d["choices"][0]["message"]["content"][:80]


t = threading.Thread(target=go, daemon=True)
t.start()
time.sleep(2)
import subprocess
for _ in range(6):
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,clocks.sm,utilization.gpu,power.draw,temperature.gpu",
                          "--format=csv,noheader"], capture_output=True, text=True,
                         encoding="utf-8", errors="replace").stdout.strip()
    print(out.replace("\n", " | "))
    time.sleep(1)
t.join()
print("el=%.1fs tok=%s  %.1f tok/s" % (res["el"], res["tok"], (res["tok"] or 0) / res["el"]))
