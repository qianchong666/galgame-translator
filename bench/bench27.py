# -*- coding: utf-8 -*-
"""测 27B（19092）逐条检查 vs 批量检查的开销与判定质量。"""
import json
import sys
import time
from pathlib import Path

import requests

BASE = Path(r"F:\aiwork\galgame小说翻译\v2\OUT_book10_final\sample_ja_full")
URL = "http://127.0.0.1:19092/v1/chat/completions"

SYS = (
    "你是日译中质检员。用户给出若干条「编号. 日文原文 ⟹ 中文译文」。逐条判断译文：\n"
    "- ok：意思正确、无漏译错译；\n"
    "- order：意思对但中文语序/表达别扭，需要调语序；\n"
    "- wrong：有漏译、错译、人名错、语气错误等实质错误，需要重译。\n"
    "只输出 JSON 数组，每项形如 {\"i\": 编号, \"v\": \"ok|order|wrong\", \"fix\": 可选修正后的中文}。"
    "不要输出任何其他文字。"
)


def ask(user, max_tokens=900):
    t0 = time.time()
    r = requests.post(URL, json={"model": "qwen3.8-27b",
                                 "messages": [{"role": "system", "content": SYS},
                                              {"role": "user", "content": user}],
                                 "max_tokens": max_tokens, "temperature": 0.1,
                                 "stream": False}, timeout=600)
    r.raise_for_status()
    d = r.json()
    el = time.time() - t0
    msg = d["choices"][0]["message"]
    txt = msg.get("content") or ""
    if not txt and msg.get("reasoning_content"):
        txt = msg["reasoning_content"]
    return txt, el, d.get("usage", {})


def pairs():
    trans = json.loads((BASE / "trans" / "000003.json").read_text(encoding="utf-8"))
    src = (BASE / "chunks" / "000003.txt").read_text(encoding="utf-8").split("\n")
    dst = trans["text"].split("\n")
    out = [(s.strip(), d.strip()) for s, d in zip(src, dst) if s.strip()]
    # 人为注入错误：第 2 条漏一半、第 6 条换人名、第 8 条语序打乱
    if len(out) > 8:
        out[1] = (out[1][0], out[1][1][: len(out[1][1]) // 3])
        out[5] = (out[5][0], out[5][1].replace("优", "武"))
        out[7] = (out[7][0], "在胯间动着纤细的手腕每次时，美丽的脸扭曲了" + out[7][1][-6:])
    return out


def main():
    ps = pairs()
    print(f"待检 {len(ps)} 条\n")
    # ---- 单条
    t0 = time.time()
    for k, (s, d) in enumerate(ps, 1):
        user = f"1. {s} ⟹ {d}"
        txt, el, u = ask(user)
        print(f"[单条{k}] {el:.1f}s out={u.get('completion_tokens')} → {txt.strip()[:90]}")
    print(f"单条合计 {time.time()-t0:.1f}s / {len(ps)} 条\n")
    # ---- 批量 6 条
    t0 = time.time()
    for gi in range(0, len(ps), 6):
        grp = ps[gi:gi + 6]
        body = "\n".join(f"{k}. {s} ⟹ {d}" for k, (s, d) in enumerate(grp, 1))
        txt, el, u = ask(body, max_tokens=1200)
        print(f"[批{gi//6+1} n={len(grp)}] {el:.1f}s out={u.get('completion_tokens')} → {txt.strip()[:400]}")
    print(f"批量合计 {time.time()-t0:.1f}s / {len(ps)} 条")


if __name__ == "__main__":
    main()
