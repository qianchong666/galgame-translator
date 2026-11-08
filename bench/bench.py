# -*- coding: utf-8 -*-
"""测多 slot / 短句粒度 / 27B 抽查 的真实耗时。只用标准库 + requests。"""
import json
import queue
import sys
import threading
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "v2"))
from gtlib.job import PROMPT_TMPL              # noqa: E402
from gtlib.placeholder import mask_code       # noqa: E402
from gtlib.segment import split_sentences     # noqa: E402

SYS = PROMPT_TMPL.format(ph=mask_code("bracket", 1))
BASE = Path(r"F:\aiwork\galgame小说翻译\v2\OUT_book10_final\sample_ja_full")


def chat(port, user, system=SYS, max_tokens=1024, temperature=0.3, timeout=300):
    t0 = time.time()
    r = requests.post(f"http://127.0.0.1:{port}/v1/chat/completions",
                      json={"messages": [{"role": "system", "content": system},
                                         {"role": "user", "content": user}],
                            "max_tokens": max_tokens, "temperature": temperature,
                            "stream": False},
                      timeout=timeout)
    r.raise_for_status()
    d = r.json()
    txt = d["choices"][0]["message"]["content"]
    el = time.time() - t0
    u = d.get("usage") or {}
    return txt, el, u


def units_from_chunks(n_chunks=6, max_chars=40):
    """把前面若干块拆成句子级短单元（≈20 token 量级）。"""
    units = []
    for i in range(1, n_chunks + 1):
        p = BASE / "chunks" / f"{i:06d}.txt"
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").split("\n"):
            line = line.strip()
            if not line:
                continue
            for s in split_sentences(line):
                s = s.strip()
                if not s:
                    continue
                if len(s) <= max_chars:
                    units.append((i, s))
                else:                     # 过长句再按逗号粗切
                    buf = ""
                    for part in s.replace("、", "、\x00").replace("，", "，\x00").split("\x00"):
                        if buf and len(buf) + len(part) > max_chars:
                            units.append((i, buf))
                            buf = part
                        else:
                            buf += part
                    if buf:
                        units.append((i, buf))
    return units


def run_many(port, items, nworkers, fn, tag):
    wq = queue.Queue()
    for it in items:
        wq.put(it)
    out = []
    lock = threading.Lock()
    t0 = time.time()

    def worker():
        while True:
            try:
                it = wq.get_nowait()
            except queue.Empty:
                return
            try:
                r = fn(port, it)
            except Exception as e:                      # noqa: BLE001
                r = (None, 0.0, {"error": str(e)[:120]})
            with lock:
                out.append((it, r))

    ts = [threading.Thread(target=worker, daemon=True) for _ in range(nworkers)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    wall = time.time() - t0
    lat = [r[1] for _, r in out if r[0]]
    ptok = sum((r[2] or {}).get("prompt_tokens", 0) for _, r in out)
    ctok = sum((r[2] or {}).get("completion_tokens", 0) for _, r in out)
    print(f"[{tag}] 请求 {len(out)}  wall {wall:.1f}s  平均 {lat and sum(lat)/len(lat):.2f}s/请求  "
          f"prompt {ptok} tok  completion {ctok} tok  "
          f"({ctok/max(wall,1e-9):.1f} tok/s 端到端)")
    return out, wall


def one_unit(port, it):
    return chat(port, it[1])


BATCH_TMPL = ("下面有若干条互不相关的日文短句，编号从 1 开始。逐条翻译成简体中文，"
              "严格按「编号|译文」的格式输出同样数量的行，不要合并、不要解释。\n\n{body}")


def one_batch(port, group):
    body = "\n".join(f"{k+1}| {s}" for k, (_, s) in enumerate(group))
    return chat(port, BATCH_TMPL.format(body=body))


def main():
    scen = sys.argv[1] if len(sys.argv) > 1 else "A"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 9080
    nw = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    n_chunks = int(sys.argv[4]) if len(sys.argv) > 4 else 6
    units = units_from_chunks(n_chunks=n_chunks, max_chars=40)
    print(f"短单元 {len(units)} 条，平均 {sum(len(u[1]) for u in units)/len(units):.1f} 字")
    sample = units[:48]
    tag = f"{scen} port={port} workers={nw}"

    if scen in ("A", "B"):
        out, w = run_many(port, sample, nw, one_unit, tag + " 逐句")
    elif scen == "C":
        groups = [sample[i:i + 6] for i in range(0, len(sample), 6)]
        out, w = run_many(port, groups, nw, one_batch, tag + " 6句/请求")
    else:  # D 块级
        chunks = []
        for i in range(1, 17):
            p = BASE / "chunks" / f"{i:06d}.txt"
            if p.exists():
                chunks.append((i, p.read_text(encoding="utf-8")))
        out, w = run_many(port, chunks, nw, one_unit, tag + " 320字块")
        print(f"  共译 {sum(len(c[1]) for c in chunks)} 字源文 → {sum(len(c[1]) for c in chunks)/w:.1f} 字/s")
        return dump(tag.replace(' ', '_'), out, w)
    src_chars = sum(len(it[1]) for it in sample) if scen in ("A", "B") else \
        sum(len(s) for g in groups for _, s in g)
    print(f"  共译 {src_chars} 字源文 → {src_chars/w:.1f} 字/s")
    dump(tag.replace(' ', '_'), out, w)


def dump(tag, out, wall):
    p = Path(__file__).with_name(f"out_{tag}.json")
    p.write_text(json.dumps([{"k": it, "text": r[0], "el": r[1], "u": r[2]}
                             for it, r in out], ensure_ascii=False, indent=1),
                 encoding="utf-8")
    print(f"  → {p.name}  wall={wall:.1f}s")


if __name__ == "__main__":
    main()
