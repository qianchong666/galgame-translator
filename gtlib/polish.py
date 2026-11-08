# -*- coding: utf-8 -*-
"""译文收尾：称呼统一 + 特殊字符清理（纯 Python，必要时用 9B 判性别）。

为什么要做：
1. 日语的「さん」是中性敬称，模型会按语境随机译成 小姐/女士/同学/君/先生，
   同一个人前后不一致。这里让 9B 只判一次"这个人该用哪个称呼"（女性→小姐、
   男性→君），之后全部用确定性替换，不再让模型每次自由发挥。
2. 译文里常残留零宽字符、控制符、全角空格、孤立的【】⟦⟧ 残片、重复标点。

两条都是纯文本处理，不重写内容。
"""
from __future__ import annotations

import json
import re
from typing import Dict, List, Optional

#: 常见的「名字 + 敬称」后缀（模型可能用其中任意一个）
HONORIFICS = ("小姐", "女士", "同学", "先生", "君", "酱", "桑", "姐", "大人", "阿姨", "夫人")

#: 零宽 / 控制字符
_ZERO = re.compile(r"[\u200b-\u200f\u202a-\u202e\ufeff\u2060]")
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

GENDER_SYS = (
    "你在给日译中小说整理人物称呼表。用户给出人名列表（中文译名）。\n"
    "请判断每个人物的性别，并给出这部作品里「さん」该统一译成什么：\n"
    "- 女性 → 小姐；男性 → 君；无法判断或非人类 → 无（不加敬称）。\n"
    "拿不准就按上下文最常见的性别填，宁可填无也不要瞎猜。\n"
    "必须调用 submit_persona 工具。"
)

GENDER_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_persona",
        "parameters": {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "gender": {"type": "string", "enum": ["男", "女", "无"]},
                            "honorific": {"type": "string", "enum": ["君", "小姐", "无"]},
                        },
                        "required": ["name", "gender", "honorific"],
                    },
                }
            },
            "required": ["items"],
        },
    },
}


def clean_special(text: str) -> tuple[str, dict]:
    """去掉零宽/控制符、规整空格与重复标点、清理占位符残片。返回 (文本, 统计)。"""
    stats = {"zero": 0, "ctrl": 0, "punct": 0, "orphan": 0, "blank": 0}
    if not text:
        return text, stats

    stats["zero"] = len(_ZERO.findall(text))
    text = _ZERO.sub("", text)
    stats["ctrl"] = len(_CTRL.findall(text))
    text = _CTRL.sub("", text)

    # 占位符残片（正常情况还原阶段已处理，这里是最后一道兜底）
    before = text
    text = re.sub(r"【\s*#*\s*\d{0,5}\s*#*\s*】", "", text)
    text = re.sub(r"[⟦⟧]", "", text)
    stats["orphan"] = before.count("【") - text.count("【")

    # 重复标点：「。。。」→「……」，「？？」→「？」，但保留「！？」「……」
    def _collapse(m: re.Match) -> str:
        stats["punct"] += 1
        return m.group(1) + m.group(2)[-1]

    text = re.sub(r"([。，、！？])([。，、！？]{2,})", _collapse, text)
    text = re.sub(r"……{2,}", "……", text)

    # 行内规整：全角空格压成普通空格、去行尾空白、空行不留空格
    out_lines: List[str] = []
    for ln in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        s = ln.replace("\u3000", " ").rstrip()
        if s != ln:
            stats["blank"] += 1
        out_lines.append(s)
    text = "\n".join(out_lines)
    return text, stats


def gender_map(names: List[str], cfg, log) -> Dict[str, str]:
    """让 9B 判一次人物性别 → {人名: 统一敬称}。中控不可用时返回 {}。"""
    from .director import make_director

    names = [n for n in dict.fromkeys(names) if n]
    if not names:
        return {}
    d = make_director(cfg, log)
    if d is None:
        return {}
    out: Dict[str, str] = {}
    batch = 40
    for i in range(0, len(names), batch):
        part = names[i:i + batch]
        user = "人名列表：\n" + "\n".join(f"- {n}" for n in part)
        try:
            data = d._call(GENDER_SYS, user, GENDER_TOOL, max_tokens=200 + 40 * len(part))
        except Exception as e:
            log.warn(f"  称呼判定失败（本批 {len(part)} 人）：{str(e)[:80]}")
            continue
        for row in (data.get("items") or []):
            name = str(row.get("name") or "").strip()
            hon = str(row.get("honorific") or "无").strip()
            if name and hon in ("君", "小姐"):
                out[name] = hon
    return out


def unify_honorifics(text: str, pmap: Dict[str, str]) -> tuple[str, int]:
    """把「人名+任意敬称」统一成「人名+指定敬称」。返回 (文本, 替换次数)。"""
    if not pmap:
        return text, 0
    n = 0
    for name, target in pmap.items():
        if not name:
            continue
        alts = [h for h in HONORIFICS if h != target]
        if not alts:
            continue
        pat = re.compile(re.escape(name) + r"\s*(?:" + "|".join(alts) + r")")
        text, k = pat.subn(name + target, text)
        n += k
    return text, n
