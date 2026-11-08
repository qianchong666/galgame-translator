# -*- coding: utf-8 -*-
"""27B 并发能力 + 批量质检 JSON 可靠性。"""
import json
import threading
import time
from pathlib import Path

import requests

URL = "http://127.0.0.1:19092/v1/chat/completions"
SYS = ("你是日译中质检员。用户给出若干条「编号. 日文原文 ⟹ 中文译文」。逐条判断译文：\n"
       "- ok：意思正确；\n- order：意思对但中文语序/表达别扭；\n"
       "- wrong：有漏译、错译、人名错等实质错误。\n"
       "只输出 JSON 数组，每项 {\"i\":编号,\"v\":\"ok|order|wrong\",\"fix\":可选}。不要其他文字。")
BASE = Path(r"F:\aiwork\galgame小说翻译\v2\OUT_book10_final\sample_ja_full")


def ask(user, max_tokens=1200):
    t0 = time.time()
    r = requests.post(URL, json={"model": "qwen3.8-27b",
                                 "messages": [{"role": "system", "content": SYS},
                                              {"role": "user", "content": user}],
                                 "max_tokens": max_tokens, "temperature": 0.1}, timeout=900)
    r.raise_for_status()
    d = r.json()
    return d["choices"][0]["message"].get("content") or "", time.time() - t0, d.get("usage", {})


def pairs(ids=(3, 4, 5, 6)):
    out = []
    for i in ids:
        t = json.loads((BASE / "trans" / f"{i:06d}.json").read_text(encoding="utf-8"))
        src = (BASE / "chunks" / f"{i:06d}.txt").read_text(encoding="utf-8").split("\n")
        for s, d in zip(src, t["text"].split("\n")):
            if s.strip():
                out.append((s.strip(), d.strip()))
    return out


def run(workers, ps, per=8):
    groups = [ps[i:i + per] for i in range(0, len(ps), per)]
    t0 = time.time()
    lock = threading.Lock()
    okjson = [0]
    out = []

    def w():
        while True:
            with lock:
                if not groups:
                    return
                g = groups.pop(0)
            body = "\n".join(f"{k}. {s} ⟹ {d}" for k, (s, d) in enumerate(g, 1))
            txt, el, u = ask(body)
            try:
                arr = json.loads(txt[txt.find("["): txt.rfind("]") + 1])
                with lock:
                    okjson[0] += 1
            except Exception:
                arr = None
            out.append((len(g), el, arr is not None, u.get("completion_tokens")))

    ts = [threading.Thread(target=w) for _ in range(workers)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    wall = time.time() - t0
    print(f"workers={workers} 批次数={len(out)} 单元={len(ps)} wall={wall:.1f}s "
          f"→ {len(ps)/wall:.2f} 单元/s  JSON合格={okjson[0]}/{len(out)}  "
          f"平均单批 {sum(o[1] for o in out)/len(out):.1f}s")
    return wall


if __name__ == "__main__":
    ps = pairs()
    print(f"共 {len(ps)} 条单元；每条约 {sum(len(s) for s, _ in ps)/len(ps):.0f} 字\n")
    run(1, ps)
    run(4, ps)
