# -*- coding: utf-8 -*-
"""长文记忆（实验特性）：文档简报 + 场景台账。

解决的问题
----------
现在的翻译是「块级独立」：400 字一块，块与块之间只有一句上文。
于是长文常出现三类漂移：
  · 指代断：这块用「他」，读者已不知道是谁；
  · 语体漂：第一章「です・ます」，第八章变「だ・である」；
  · 称呼乱：同一个角色一会「佐藤小姐」一会「佐藤同学」。

做法（不照搬任何模型的长上下文方案，纯本地增量）
----------
1) 文档简报 doc brief：翻译前用中控读一遍开头 + 人名表，产出
   {语体, 视点, 角色表(含固定称呼), 禁译项}，之后每块请求都带上（短、固定）。
2) 场景台账 ledger：把块按场景分组（默认 8 块一组），预先按**原文顺序**串行
   跑一遍中控，产出每组的 {人物, 称呼状态, 未解决指代, 语体提醒, 剧情一句}。
   注意：这是在**翻译前**对原文做的，所以不受翻译并发乱序影响。
3) 落盘 memory.jsonl（每块一行），翻译时按块号查表注入提示词。
   关掉开关就完全退回原行为，不影响既有产物。

设计取舍
--------
· 用「预扫一遍原文」而不是「翻完一块更新一次台账」：后者在 3 并发下必然乱序，
  且会把 9B 调用塞进翻译热路径（每块 +3s）。预扫是一次性成本（约 2 分钟/本）。
· 台账只描述「原文此时此刻的状态」，不参与翻译判定，出错也不会污染译文——
  最坏情况是提示词里多了一句没用的话。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

MEMORY_FILE = "memory.jsonl"

BRIEF_SYS = (
    "你是长篇日文小说的翻译统筹。用户给你作品的开头片段与专名表。\n"
    "请产出一份**全书通用的翻译简报**，用于让译者保持一致。\n"
    "只根据材料判断，拿不准就留空，不要编造剧情；taboos 只在材料里明确出现时才填，"
    "没有就返回空数组。必须调用 submit_brief。"
)

BRIEF_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_brief",
        "description": "提交全书翻译简报",
        "parameters": {
            "type": "object",
            "properties": {
                "work_type": {"type": "string", "description": "体裁，如 galgame剧本/轻小说/官能小说"},
                "narration": {"type": "string", "description": "叙述视点，如 一人称(男主)/三人称"},
                "style": {"type": "string", "description": "原文语体，如 です・ます / だ・である / 混用"},
                "tone": {"type": "string", "description": "整体基调，20 字内"},
                "characters": {
                    "type": "array",
                    "description": "主要人物（最多 12 个）",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "中文译名，与专名表一致"},
                            "role": {"type": "string", "description": "身份/关系"},
                            "address": {"type": "string", "description": "他人对他的固定称呼，如 小姐/君/先生/直呼其名"},
                            "self": {"type": "string", "description": "自称，如 我/人家/老夫"},
                        },
                        "required": ["name"],
                    },
                },
                "taboos": {
                    "type": "array",
                    "description": "全书禁译/保留项，如 拟声词保留假名、称呼不本地化、不加解释性增译",
                    "items": {"type": "string"},
                },
            },
            "required": ["narration", "style"],
        },
    },
}

LEDGER_SYS = (
    "你在给长篇小说的翻译做「分场记录」，供后续段落翻译时参考。\n"
    "用户给出上一场的记录与这一段的原文。请只描述**原文里确凿的信息**：\n"
    "- 这一场出现了谁（用中文译名）、彼此怎么称呼；\n"
    "- 有没有代词/省略主语指向上一场的人物，指谁；\n"
    "- 语体是否与前面一致（敬体/常体）；\n"
    "- 一句话说清这一场发生了什么（供下一场理解省略）。\n"
    "不要评价译文，不要翻译原文。必须调用 submit_ledger。"
)

LEDGER_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_ledger",
        "parameters": {
            "type": "object",
            "properties": {
                "characters": {"type": "array", "items": {"type": "string"},
                               "description": "本场出场人物（中文译名）"},
                "address": {"type": "string", "description": "本场称呼关系，如 佐藤小姐叫铃木君「铃木君」"},
                "refs": {"type": "string", "description": "指向前面人物的代词/省略主语，没有就留空"},
                "style_ok": {"type": "boolean", "description": "语体与前文是否一致"},
                "style_note": {"type": "string", "description": "如 仍为敬体；或 此处转常体"},
                "events": {"type": "string", "description": "一句话概括本场事件"},
            },
            "required": ["characters", "events"],
        },
    },
}


def _head_text(chunks: List[str], limit: int = 6000) -> str:
    buf: List[str] = []
    n = 0
    for c in chunks:
        if n >= limit:
            break
        buf.append(c)
        n += len(c)
    return "\n".join(buf)[:limit]


def _names_line(terms: List[dict]) -> str:
    names = [f"{t.get('term')}→{t.get('target')}" for t in terms
             if str(t.get("type", "")).startswith("人名") and t.get("target")]
    return "、".join(dict.fromkeys(names))[:800]


def build_brief(cfg, log, chunks: List[str], terms: List[dict]) -> dict:
    """读开头 + 人名表 → 全书简报。中控不可用则返回空 dict。"""
    from .director import make_director

    d = make_director(cfg, log)
    if d is None:
        return {}
    user = ("【专名表（原词→译名）】\n" + (_names_line(terms) or "（无）")
            + "\n\n【作品开头】\n" + _head_text(chunks))
    try:
        data = d._call(BRIEF_SYS, user, BRIEF_TOOL, max_tokens=1200)
    except Exception as e:                        # noqa: BLE001
        log.warn(f"  文档简报生成失败：{str(e)[:80]}")
        return {}
    if not isinstance(data, dict):
        return {}
    log.info(f"  文档简报：{data.get('work_type', '?')} · {data.get('style', '?')}"
             f" · 角色 {len(data.get('characters') or [])} 个")
    return data


def _demask(text: str, names: Dict[str, str]) -> str:
    """把 【#N#】 还原成中文译名，台账里不该出现系统标记。"""
    if not text:
        return text
    import re as _re

    def rep(m):
        return names.get(m.group(1), m.group(0))

    return _re.sub(r"【#\s*(\d+)\s*#】", rep, str(text))


def build_ledgers(cfg, log, chunks: List[str], brief: dict,
                  scene: int = 8, names: Optional[Dict[str, str]] = None,
                  presence: Optional[List[List[str]]] = None) -> List[dict]:
    """按原文顺序扫一遍，每 scene 块产出一条台账。返回 [ {cid, ...}, ... ]"""
    from .director import make_director

    d = make_director(cfg, log)
    if d is None:
        return []
    brief_line = _brief_line(brief)
    prev = ""
    rows: List[dict] = []
    total = len(chunks)
    step = max(1, int(scene))
    for start in range(0, total, step):
        group = chunks[start:start + step]
        body = "\n".join(f"[{start + i + 1}] {c}" for i, c in enumerate(group))
        user = (brief_line + "\n\n【上一场记录】\n" + (prev or "（这是第一场）")
                + "\n\n【本场原文】\n" + body[:4000])
        try:
            data = d._call(LEDGER_SYS, user, LEDGER_TOOL, max_tokens=700)
        except Exception as e:                    # noqa: BLE001
            log.warn(f"  台账 {start}-{start + len(group)} 失败：{str(e)[:60]}")
            data = {}
        if not isinstance(data, dict):
            data = {}
        if names:
            data = {k: (_demask(v, names) if isinstance(v, str)
                        else [_demask(x, names) for x in v] if isinstance(v, list) else v)
                    for k, v in data.items()}
        prev_chars = (presence[start] if (presence and start < len(presence)) else []) or []
        prev = (f"人物：{'、'.join(prev_chars) or '、'.join(data.get('characters') or []) or '?'}；"
                f"称呼：{data.get('address') or '-'}；"
                f"事件：{data.get('events') or '-'}")
        for i in range(start, min(start + step, total)):
            row = dict(data)
            if presence and i < len(presence):
                row["characters"] = presence[i]      # 确定性名单，覆盖模型猜的
            rows.append({"cid": i + 1, "scene_from": start + 1,
                         "scene_to": min(start + step, total), **row})
        log.info(f"  台账 {start + 1}-{min(start + step, total)}/{total}"
                 f"：{(data.get('events') or '')[:30]}")
    return rows


def _brief_line(brief: dict) -> str:
    if not brief:
        return ""
    parts = []
    if brief.get("work_type"):
        parts.append(f"体裁：{brief['work_type']}")
    if brief.get("narration"):
        parts.append(f"视点：{brief['narration']}")
    if brief.get("style"):
        parts.append(f"语体：{brief['style']}")
    if brief.get("tone"):
        parts.append(f"基调：{brief['tone']}")
    chars = brief.get("characters") or []
    if chars:
        c = "、".join(
            f"{x.get('name')}"
            + (f"（{x.get('role')}）" if x.get("role") else "")
            + (f"[{x.get('address')}]" if x.get("address") else "")
            for x in chars[:10])
        parts.append("人物：" + c)
    if brief.get("taboos"):
        parts.append("禁译：" + "、".join(brief["taboos"][:4]))
    return "\n".join(parts)


def render_block(brief: dict, row: Optional[dict]) -> str:
    """压成 2~3 行的中性「背景资料」。

    注意：译者模型（sakura/GalTransl 系）对**指令式**文字极敏感，多一句
    「不要翻译」就可能整段复读。所以这里只给名词性信息，不带祈使句，
    也不加【】标题。
    """
    parts: List[str] = []
    head: List[str] = []
    if brief.get("style"):
        head.append(f"语体 {brief['style']}")
    if brief.get("narration"):
        head.append(f"视点 {brief['narration']}")
    if brief.get("tone"):
        head.append(f"基调 {brief['tone']}")
    if head:
        parts.append("背景：" + "｜".join(head))

    chars = brief.get("characters") or []
    if chars:
        seen, items = set(), []
        for x in chars:
            nm = str(x.get("name") or "")
            if not nm or "#" in nm or nm in seen:
                continue
            seen.add(nm)
            items.append(nm + (f"（{x.get('address')}）" if x.get("address") else ""))
            if len(items) >= 8:
                break
        if items:
            parts.append("人物：" + "、".join(items))

    if row:
        seg: List[str] = []
        if row.get("characters"):
            seg.append("本场 " + "、".join(row["characters"][:6]))
        if row.get("address"):
            a = row["address"]
            seg.append(("、".join(a) if isinstance(a, list) else str(a))[:50])
        if row.get("refs"):
            r = row["refs"]
            seg.append("指代 " + (("、".join(r) if isinstance(r, list) else str(r))[:40]))
        if row.get("style_note"):
            seg.append(str(row["style_note"])[:20])
        if seg:
            parts.append("；".join(seg))
    return chr(10).join(parts)


def save_memory(out_dir: Path, brief: dict, rows: List[dict]) -> Path:
    p = Path(out_dir) / MEMORY_FILE
    with p.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"type": "brief", **brief}, ensure_ascii=False) + "\n")
        for r in rows:
            f.write(json.dumps({"type": "chunk", **r}, ensure_ascii=False) + "\n")
    return p


def load_memory(out_dir: Path) -> tuple[dict, Dict[int, dict]]:
    """返回 (brief, {cid: row})。文件不存在则空。"""
    p = Path(out_dir) / MEMORY_FILE
    brief: dict = {}
    rows: Dict[int, dict] = {}
    if not p.exists():
        return brief, rows
    try:
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            if d.get("type") == "brief":
                brief = d
            elif d.get("type") == "chunk":
                rows[int(d.get("cid") or 0)] = d
    except Exception:                              # noqa: BLE001
        return {}, {}
    return brief, rows

def strip_echo(text: str, block: str) -> str:
    """译者把注入的记忆块抄进译文时，把它剥掉。

    sakura/GalTransl 系模型有个稳定行为：只要提示词里出现「像正文的段落」，
    它就可能原样抄进输出。这里只处理**开头连续命中**的情形（记忆块永远在开头），
    命中行数 ≥2 或首行完全一致才剥，避免误伤正常译文。
    """
    if not text or not block:
        return text
    lines = text.splitlines()
    blk = [b.strip() for b in block.splitlines() if b.strip()]
    if not blk:
        return text
    hit = 0
    for i, b in enumerate(blk):
        if i >= len(lines):
            break
        got = lines[i].strip()
        if got.startswith(b[:12]) or b.startswith(got[:12]) or (got and b and got[0] == b[0] and len(got) > 6
                                                                and abs(len(got) - len(b)) < 12):
            hit += 1
        else:
            break
    if hit == 0:
        return text
    if hit < len(blk) and hit < 2 and not lines[0].startswith(blk[0][:12]):
        return text
    return chr(10).join(lines[hit:]).lstrip()

def presence_from_masks(chunk_texts: List[str], style: str,
                        names: Dict[str, str]) -> List[List[str]]:
    """按遮蔽标记确定每块出场人物（零模型、零误差）。

    masked.txt 里出现 【#10#】 就说明 10 号角色在这一块出场；
    这比让模型读日文猜可靠得多（实测模型会把婆媳、母女搞混）。
    """
    from .placeholder import indices_in

    allowed = set()
    for k in names:
        try:
            allowed.add(int(k))
        except (TypeError, ValueError):
            continue
    out: List[List[str]] = []
    for t in chunk_texts:
        got = indices_in(t, style, allowed) or {}
        ordered = []
        for idx in sorted(got, key=lambda i: got[i], reverse=True):
            nm = names.get(str(idx))
            if nm and nm not in ordered:
                ordered.append(nm)
        out.append(ordered)
    return out

def narrator_of(brief: dict) -> Optional[str]:
    """从简报里找出叙述者（一人称作品里他每块都在场，但不是每次都被点名）。"""
    narration = str(brief.get("narration") or "")
    if "一人称" not in narration and "第一人称" not in narration:
        return None
    for x in brief.get("characters") or []:
        role = str(x.get("role") or "")
        if any(k in role for k in ("男主", "主角", "主人公", "叙述者")):
            return str(x.get("name") or "") or None
    import re as _re
    m = _re.search(r"[（(]([^）)]{1,6})[）)]", narration)
    return m.group(1) if m else None


def merge_narrator(presence: List[List[str]], brief: dict) -> List[List[str]]:
    """一人称作品：把叙述者补进每一块（排在最前，去重）。"""
    nm = narrator_of(brief)
    if not nm:
        return presence
    out: List[List[str]] = []
    for row in presence:
        cur = [nm] + [x for x in row if x != nm]
        out.append(cur)
    return out
