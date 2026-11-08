# -*- coding: utf-8 -*-
"""全书一致性体检（长文专用，纯确定性，不调用模型）。

为什么需要
----------
块级质检门只看「这一块对不对」，看不到「前后一致不一致」：
· 同一个人这块叫「小姐」、下块叫「同学」；
· 名字一会儿「礼子」一会儿「九条礼子」（有时是正常的，需人工判断）；
· 敬语级别在中途从「您/吗」漂成「你/吧」；
· 人称代词密度突变（叙述视点漂移）。

这里做全书扫描，只输出「可疑清单 + 定位到块」，不改文本；
要修就按清单定点重译（删掉对应 trans/*.json 再续跑）。
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List

#: 常见中文敬称后缀
HON = ("小姐", "女士", "同学", "先生", "君", "大人", "夫人", "阿姨", "酱", "桑")

_PRON = re.compile(r"[他她它]")
_FORMAL = re.compile(r"您")
_CASUAL = re.compile(r"[你妳]")


def _suffix_usage(text: str, name: str) -> Dict[str, int]:
    """统计某个人名后面跟了哪些敬称。"""
    out: Dict[str, int] = {}
    for h in HON:
        n = len(re.findall(re.escape(name) + r"\s*" + h, text))
        if n:
            out[h] = n
    return out


def check_honorifics(text: str, names: List[str]) -> List[dict]:
    issues: List[dict] = []
    for name in names:
        if not name or len(name) < 2:
            continue
        used = _suffix_usage(text, name)
        if len(used) >= 2:
            issues.append({"type": "honorific", "name": name, "usage": used,
                           "hint": "同一人被用了多种称呼，建议统一"})
    return issues


def check_name_variants(text: str, names: List[str]) -> List[dict]:
    """同长度、仅差一字的两个名字都高频出现 → 可能是译名不统一。"""
    issues: List[dict] = []
    pool = [n for n in dict.fromkeys(names) if 2 <= len(n) <= 5]
    for i, a in enumerate(pool):
        for b in pool[i + 1:]:
            if a == b or a in b or b in a:
                continue
            if len(a) != len(b):
                continue
            diff = sum(1 for x, y in zip(a, b) if x != y)
            if diff != 1:
                continue
            ca, cb = text.count(a), text.count(b)
            if ca >= 2 and cb >= 2:
                issues.append({"type": "name_variant", "a": a, "b": b,
                               "count": [ca, cb], "hint": "两字形近且都在用，确认是否同一人"})
    return issues


def _segments(text: str, n: int = 10) -> List[str]:
    """优先按行切；行太少时按字数均分，保证总能切出 n 段。"""
    lines = text.splitlines()
    if len(lines) >= n:
        size = max(1, len(lines) // n)
        out = [chr(10).join(lines[i:i + size]) for i in range(0, len(lines), size)]
        return out[:n]
    size = max(1, len(text) // n)
    return [text[i:i + size] for i in range(0, len(text), size)][:n]


def check_style_drift(text: str, n: int = 10) -> List[dict]:
    """按段统计「敬体/口语」比例与代词密度，找出明显偏离全书基线的段落。"""
    segs = _segments(text, n)
    if len(segs) < 3:
        return []
    stats = []
    for i, s in enumerate(segs):
        L = max(1, len(s))
        f = len(_FORMAL.findall(s))
        c = len(_CASUAL.findall(s))
        stats.append({"seg": i + 1, "f": f, "c": c,
                      "formality": f / (f + c) if (f + c) else None,
                      "pron": len(_PRON.findall(s)) / L * 1000})
    issues: List[dict] = []
    vals = [x["formality"] for x in stats if x["formality"] is not None]
    if vals:
        base = sum(vals) / len(vals)
        if 0.08 < base < 0.92:                       # 全书本身就是混用，才谈得上漂移
            for x in stats:
                v = x["formality"]
                if v is None:
                    continue
                if abs(v - base) > 0.35:
                    issues.append({"type": "style_drift", "metric": "敬语比例",
                                   "seg": x["seg"], "value": round(v, 2),
                                   "base": round(base, 2),
                                   "hint": f"第 {x['seg']} 段敬语比例 {v:.2f}，全书基线 {base:.2f}"})
    prons = sorted(x["pron"] for x in stats)
    med = prons[len(prons) // 2]
    if med > 0.1:
        for x in stats:
            if x["pron"] > med * 2.5:
                issues.append({"type": "style_drift", "metric": "代词密度",
                               "seg": x["seg"], "value": round(x["pron"], 2),
                               "base": round(med, 2),
                               "hint": f"第 {x['seg']} 段代词密度 {x['pron']:.2f}，中位 {med:.2f}"})
    return issues


def run(out_dir: Path) -> dict:
    """扫描一个输出目录，返回体检报告（不写文件、不改文本）。"""
    out_dir = Path(out_dir)
    merged = sorted(out_dir.glob("合并_*_成品.txt")) or sorted(out_dir.glob("合并_*.txt"))
    if not merged:
        inner = sorted(out_dir.glob("*/合并_*_成品.txt")) or sorted(out_dir.glob("*/合并_*.txt"))
        merged = inner
    text = merged[0].read_text(encoding="utf-8", errors="replace") if merged else ""

    names: List[str] = []
    for tp in list(out_dir.glob("*/terms.json")) + list(out_dir.glob("terms.json")):
        try:
            data = json.loads(tp.read_text(encoding="utf-8"))
        except Exception:                                  # noqa: BLE001
            continue
        for e in data.get("terms", []):
            if str(e.get("type", "")).startswith("人名"):
                tgt = e.get("target") or e.get("term")
                if tgt:
                    names.append(tgt)
        break

    issues: List[dict] = []
    if text:
        issues += check_honorifics(text, names)
        issues += check_name_variants(text, names)
        issues += check_style_drift(text)
    return {"file": str(merged[0]) if merged else "",
            "chars": len(text), "names": len(dict.fromkeys(names)),
            "issues": issues}


def summary(rep: dict) -> str:
    lines = [f"一致性体检：{rep.get('chars', 0)} 字 · 人名 {rep.get('names', 0)} 个 · "
             f"可疑 {len(rep.get('issues', []))} 项"]
    for it in rep.get("issues", [])[:20]:
        if it["type"] == "honorific":
            lines.append(f"  · 称呼混用：{it['name']} → {it['usage']}")
        elif it["type"] == "name_variant":
            lines.append(f"  · 形近译名：{it['a']} / {it['b']}（{it['count'][0]}/{it['count'][1]} 次）")
        elif it["type"] == "style_drift":
            lines.append(f"  · 语体漂移：{it['hint']}")
    if len(rep.get("issues", [])) > 20:
        lines.append(f"  … 还有 {len(rep['issues']) - 20} 项，详见 JSON")
    return "\n".join(lines)
