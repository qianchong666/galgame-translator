# -*- coding: utf-8 -*-
"""让大模型（9B）辅助清洗：规则判不了的行，交模型判「删/留」。

规则清洗（clean.py）是确定性的、快的，但有两类它做不好：
  1. **误删**：`作者的话`、`第X话`、单独一行的数字、短句……被当广告/装饰删掉，
     而它们有时是正文的一部分（尤其是盗版站把正文拆得七零八落时）；
  2. **漏删**：URL、公众号、群号、乱码残片这类没进规则的垃圾留在正文里，
     切进块里就是给译文灌噪声。

做法：先用规则跑一遍，把「被判删除的行」和「看起来可疑的保留行」凑成候选，
按批送模型判定（每批 40 行，一次调用），再按模型结论重跑清洗。
模型只回答"这行保留还是删掉"，不碰正文内容——零改写风险。
"""
from __future__ import annotations

import json
import re
from typing import List, Optional, Sequence

from . import clean as cleanmod
from .core import Config, Logger

#: 明显是垃圾的行（URL、群号、公众号、纯数字页码）
_JUNK = re.compile(
    r"https?://|www\.|\.com|\.net|\.cc|公众号|微信|QQ群|qq群|书友群|群号|"
    r"最新章节|全文阅读|手机阅读|无弹窗|请记住本站|加入书签|投推荐票|"
    r"^\s*\d{1,4}\s*$|^\s*[-—]{2,}\s*$", re.I)

CLEAN_SYS = (
    "你在做小说文本清洗的最后一道复核。给你若干「行号. 原文行」，每行后面标了规则引擎的初判。\n"
    "你的任务只有一个：判断这一行**该保留还是删掉**。\n"
    "规则：\n"
    "1. 属于正文（叙述、对白、场景描写、章节标题、拟声词、单独成行的短句）→ 保留；\n"
    "2. 属于无关内容（盗版站广告、网址、公众号、群号、水印、页码、乱码残片、"
    "与剧情无关的作者求票/求收藏）→ 删掉；\n"
    "3. 拿不准就保留（宁可留一句废话，也不要删掉正文）；\n"
    "4. 不要改写内容，只给 drop 字段。\n"
    "必须调用 submit_clean 工具。"
)

CLEAN_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_clean",
        "description": "提交每行该删还是该留的判定",
        "parameters": {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "idx": {"type": "integer", "description": "行号（原样回填）"},
                            "drop": {"type": "boolean", "description": "true=删掉，false=保留"},
                            "why": {"type": "string", "description": "一句话原因（可选）"},
                        },
                        "required": ["idx", "drop"],
                    },
                }
            },
            "required": ["items"],
        },
    },
}


def _candidates(raw: str, lang: str, cleaned_lines: set) -> List[dict]:
    """挑出需要模型复核的行：规则想删的 + 规则留下但明显可疑的。"""
    out: List[dict] = []
    for i, line in enumerate(raw.replace("\r\n", "\n").replace("\r", "\n").split("\n")):
        s = line.strip()
        if not s:
            continue
        wanted_drop = cleanmod.is_boilerplate(s, lang) or cleanmod.is_decorative(s)
        suspicious = bool(_JUNK.search(s)) or (len(s) <= 3 and not re.search(r"[。！？…」]", s))
        if wanted_drop:
            out.append({"idx": i, "text": s, "hint": "规则判：删"})
        elif suspicious and s not in cleaned_lines:
            out.append({"idx": i, "text": s, "hint": "规则判：留（但可疑）"})
    return out


def review_clean(raw: str, cfg: Config, log: Logger, batch: int = 40,
                 lang: str = "auto") -> str:
    """规则清洗 + 模型复核，返回最终清洗文本。中控不可用时退回纯规则。"""
    from .director import make_director

    if lang == "auto":
        lang = cleanmod.detect_lang(raw)
    base = cleanmod.clean(raw, lang=lang)
    d = make_director(cfg, log)
    if d is None:
        log.warn("  未配置中控，跳过 AI 清洗复核（纯规则）")
        return base.text

    cands = _candidates(raw, lang, set(base.text.split("\n")))
    if not cands:
        log.info("  清洗复核：没有可疑行")
        return base.text
    log.info(f"  清洗复核：{len(cands)} 行可疑，交模型判定（每批 {batch} 行）")

    drop_idx: set = set()
    keep_idx: set = set()
    for i in range(0, len(cands), batch):
        part = cands[i:i + batch]
        lines = [f"{c['idx']}. {c['text'][:120]}   ←（{c['hint']}）" for c in part]
        user = "请逐行判断该保留还是删掉：\n" + "\n".join(lines)
        try:
            data = d._call(CLEAN_SYS, user, CLEAN_TOOL, max_tokens=200 + 30 * len(part))
        except Exception as e:
            log.warn(f"  清洗复核失败（本批 {len(part)} 行，跳过）：{str(e)[:80]}")
            continue
        got = 0
        for row in (data.get("items") or []):
            try:
                idx = int(row.get("idx"))
            except Exception:
                continue
            want_drop = bool(row.get("drop"))
            # 规则已删的行：模型说留 → 白名单；规则留下的可疑行：模型说删 → 黑名单
            txt = next((c["text"] for c in part if c["idx"] == idx), "")
            is_junk_only = not re.search(r"[ぁ-んァ-ヺ]", txt)   # 整行没有日文假名才算垃圾
            if any(c["idx"] == idx and c["hint"].startswith("规则判：删") for c in part):
                (drop_idx if want_drop else keep_idx).add(idx)
            elif want_drop:
                if is_junk_only:
                    drop_idx.add(idx)
                else:                       # 广告和正文被并到一行了：宁可不删，别丢正文
                    log.info(f"    跳过删除（行内含正文）：{txt[:40]}")
                    keep_idx.add(idx)
            got += 1
        log.info(f"    批 {i // batch + 1}: 判定 {got}/{len(part)} 行")

    log.info(f"  清洗复核结果：救回 {len(keep_idx)} 行、追加删除 {len(drop_idx)} 行")
    out = cleanmod.clean(raw, lang=lang, keep_lines=keep_idx, drop_lines=drop_idx)
    return out.text
