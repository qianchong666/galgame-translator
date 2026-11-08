# -*- coding: utf-8 -*-
"""校对台架：给候选「校对模型」注入已知错误，量它能不能修对、会不会改坏。

为什么需要：质检门只说明"没坏"，无法比较两个校对模型（27B / 9B / 在线大模型）。
这里用黄金集本身造错——从参考译文里删占位符、删行、插入复读指令、打乱行序、
把占位符换成真名——再看候选模型交回的补丁能不能把错误修回去，且不碰没出错的行。

用法（在 v2 目录下）：
  python bench/prbench.py --golden bench/golden --limit 8 \
      --cand flash@http://127.0.0.1:19095/v1@qwen3.8-flash-next-gptq-nvfp4 \
      --cand sakura@http://127.0.0.1:8090/v1@sakura-galtransl-14b \
      --out bench/prbench.json
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gtlib.core import Config, Logger, write_json_atomic            # noqa: E402
from gtlib.director import Director                                # noqa: E402
from gtlib.job import Translator                                   # noqa: E402
from gtlib.score import _lines, load_golden                        # noqa: E402

import difflib                                                     # noqa: E402

STRICT_SYS = ("你是日译中校对员，只做最低限度修正。给你逐行编号的日文原文与机器译文。" "规则：1) 只有原文与译文意思明显不符（漏译、错译、占位符丢失、假名残留）的行才提交补丁；" "2) 风格、措辞、语序的小差别一律不动；3) 最多提交 3 条补丁；" "4) old 必须与给出的该行中文完全一致；5) new 只能是一行，占位符数量须与原文该行一致；" "6) 没有该改的行就提交空 patches。必须调用 submit_patches。")

PH = re.compile(r"【#\s*\d+\s*#】|⟦\s*\d+\s*⟧")
ECHO = ("硬性要求：\n1. 只输出译文本身，不要解释。\n2. 严格保持原文的行数与断行位置。\n"
        "3. 保留「」引号结构。\n4. 占位符必须原样保留。\n5. 不要输出任何说明性文字。")


def sim(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


# ---------------------------------------------------------------- 造错
PROBLEM = {
    "drop_ph": "占位符出现次数与原文不符：有占位符被删掉了",
    "drop_line": "译文行数比原文少 1 行：有一行被删掉了",
    "echo": "译文开头混入了提示词内容（复读了指令），需删掉这些说明文字",
    "swap": "相邻两行的顺序被交换了，与原文行序不一致",
    "bad_name": "占位符被替换成了真实译名，占位符丢失",
}


def corrupt(kind: str, text: str, rng: random.Random):
    """返回 (坏文本, 判定修好函数, 出错的行号集合)。判定一律按行号精确比对。"""
    lines = _lines(text)
    ref = list(lines)

    if kind == "drop_ph":                      # 删掉一个占位符
        for i, ln in enumerate(lines):
            m = PH.search(ln)
            if m:
                lines[i] = ln[:m.start()] + ln[m.end():]
                hit = {i}
                break
        else:
            return None, None, set()
        want = len(PH.findall(ref[i]))
        def ok(t, i=i, want=want):
            L = _lines(t)
            return 0 <= i < len(L) and len(PH.findall(L[i])) == want
        return "\n".join(lines), ok, hit

    if kind == "drop_line":                    # 删掉一整行
        cand = [i for i, ln in enumerate(lines) if ln.strip()]
        if len(cand) < 4:
            return None, None, set()
        d = rng.choice(cand[1:-1])
        bad = lines[:d] + lines[d + 1:]
        def ok(t, d=d):
            L = _lines(t)
            return len(L) == len(ref) and L[d] == ref[d]
        return "\n".join(bad), ok, {d}

    if kind == "echo":                         # 前面粘一段模型复读的指令
        bad = ECHO + "\n\n" + text
        def ok(t):
            return ("硬性要求" not in t) and ("只输出译文本身" not in t)                    and len(_lines(t)) == len(ref)
        return bad, ok, set()

    if kind == "swap":                         # 相邻两行打乱
        cand = [i for i in range(len(lines) - 1) if lines[i].strip() and lines[i + 1].strip()]
        if not cand:
            return None, None, set()
        i = rng.choice(cand)
        bad = lines[:i] + [lines[i + 1], lines[i]] + lines[i + 2:]
        def ok(t, i=i):
            L = _lines(t)
            if len(L) != len(ref) or i + 1 >= len(L):
                return False
            return (sim(L[i], ref[i]) > sim(L[i], ref[i + 1])
                    and sim(L[i + 1], ref[i + 1]) > sim(L[i + 1], ref[i]))
        return "\n".join(bad), ok, {i, i + 1}

    if kind == "bad_name":                     # 占位符被模型"译成"了真名
        m = PH.search(text)
        if not m:
            return None, None, set()
        bad = text[:m.start()] + "礼子" + text[m.end():]
        want = len(PH.findall(text))
        def ok(t, want=want):
            return len(PH.findall(t)) >= want
        return bad, ok, {-1}

    raise ValueError(kind)


# ---------------------------------------------------------------- 主流程
def bench_candidate(label: str, base: str, model: str, items, kinds, limit: int,
                    timeout: float, no_think: bool = False) -> dict:
    cfg = Config()
    cfg.director_no_think = no_think
    cfg.director_base = base
    cfg.director_model = model
    cfg.director_timeout = timeout
    cfg.director_review = True
    cfg.director_terms = False
    cfg.request_timeout = timeout
    cfg.stall_timeout = timeout
    log = Logger(None, echo=False)
    d = Director(cfg, log)
    if not d.probe():
        return {"label": label, "error": f"端点不可达: {base}"}

    rng = random.Random(20261107)
    rows = []
    for it in items[:limit]:
        ref = it["ref"]
        for kind in kinds:
            bad, fixed_fn, hit_lines = corrupt(kind, ref, rng)
            if bad is None:
                continue
            t0 = time.time()
            try:
                got = d.proofread_patches(it["src"], bad, [PROBLEM.get(kind, "有错误")])
            except Exception as e:
                rows.append({"chunk": it["id"], "kind": kind, "error": str(e)[:80]})
                continue
            applied = dropped = 0
            out = bad
            if got:
                out, applied, dropped = Translator._apply_patches(bad, got.get("patches"))
            fixed = bool(fixed_fn(out))
            rows.append({
                "chunk": it["id"], "kind": kind,
                "fixed": fixed, "applied": applied, "dropped": dropped,
                "sim_before": round(sim(bad, ref), 4), "sim_after": round(sim(out, ref), 4),
                "delta": round(sim(out, ref) - sim(bad, ref), 4),
                "secs": round(time.time() - t0, 1),
                "changed_lines": sum(1 for a, b in zip(_lines(out), _lines(ref)) if a.strip() != b.strip()),
                "collateral": sum(1 for j, (a, b) in enumerate(zip(_lines(out), _lines(ref))) if a.strip() != b.strip() and j not in hit_lines),
            })
    return {"label": label, "rows": rows}


def summarize(res: dict) -> str:
    if res.get("error"):
        return f"### {res['label']}  ✗ {res['error']}"
    rows = res["rows"]
    errs = [r for r in rows if r.get("error")]
    ok = [r for r in rows if not r.get("error")]
    out = [f"### {res['label']}"]
    if not ok:
        return out[0] + "  ✗ 无有效样本" + (f"（{errs[0]['error']}）" if errs else "")
    fix = sum(1 for r in ok if r["fixed"]) / len(ok)
    delta = sum(r["delta"] for r in ok) / len(ok)
    secs = sum(r["secs"] for r in ok) / len(ok)
    out.append(f"- 修好率 **{fix:.0%}**（{sum(1 for r in ok if r['fixed'])}/{len(ok)}）"
               f"｜相似度增益 {delta:+.4f}"
               f"｜误伤行 {sum(r['collateral'] for r in ok) / len(ok):.1f}｜平均 {secs:.1f}s/次"
               f"｜补丁 生效 {sum(r['applied'] for r in ok)} 丢弃 {sum(r['dropped'] for r in ok)}")
    by = {}
    for r in ok:
        by.setdefault(r["kind"], []).append(r)
    for k, rs in by.items():
        fr = sum(1 for r in rs if r["fixed"]) / len(rs)
        dl = sum(r["delta"] for r in rs) / len(rs)
        out.append(f"  - {k:10s} 修好 {fr:.0%}（{len(rs)} 例）｜增益 {dl:+.4f}")
    if errs:
        out.append(f"  - 请求异常 {len(errs)} 次：{errs[0]['error']}")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description="校对台架：注入已知错误，量候选校对模型能不能修对")
    ap.add_argument("--golden", default="bench/golden")
    ap.add_argument("--cand", action="append", required=True,
                    help="候选：标签@端点@模型名，可给多个")
    ap.add_argument("--kinds", default="drop_ph,drop_line,echo,swap,bad_name")
    ap.add_argument("--limit", type=int, default=8, help="用黄金集前 N 段（默认 8）")
    ap.add_argument("--timeout", type=float, default=240.0)
    ap.add_argument("--no-think", action="store_true", help="中控带 enable_thinking=false")
    ap.add_argument("--strict", action="store_true", help="用严厉的补丁提示词（限制最少改动）")
    ap.add_argument("--out", help="结果写成 json")
    a = ap.parse_args()

    if a.strict:
        from gtlib import director as _dmod
        _dmod.PATCH_SYS = STRICT_SYS
    _meta, items = load_golden(Path(a.golden))
    kinds = [k.strip() for k in a.kinds.split(",") if k.strip()]
    results = []
    for spec in a.cand:
        parts = spec.split("@")
        if len(parts) != 3:
            print(f"候选格式错误：{spec}（应为 标签@端点@模型名）")
            return 2
        label, base, model = parts
        res = bench_candidate(label, base, model, items, kinds, a.limit, a.timeout, a.no_think)
        results.append(res)
        print(summarize(res))
        print()
    if a.out:
        write_json_atomic(Path(a.out), results)
        print(f"明细已写入 {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
