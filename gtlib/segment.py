# -*- coding: utf-8 -*-
"""切块：把清洗后的文本切成"适合一次请求翻译"的块。

相对旧 step3_split 的改动
--------------------------------
1. **段落优先**：旧版把每句拼成一个 chunk，靠 `_merge_into_chunks` 用 30% 阈值兜底，
   容易出现"一句话一块"（请求数爆炸）与"跨场景拼一块"（人称错乱）。这里以段为原子，
   在 `---` 场景边界处强制断开，其余按容量贪心合并。
2. **引号安全**：`「…」` 内部不拆（除非超长），避免把对白切成两半让模型各译一半。
3. **占位符安全**：硬切时不允许切在 `⟦12⟧` 中间。
4. **上文锚点**：每块可带上文尾部若干字符（context），用于人称/语气延续。
5. **纯函数**：同一输入永远产出同样的块序列（含 line_start/line_end），断点续传才可信。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

# 占位符（与 placeholder.py 的样式表保持一致，这里只用于"不要切在中间"）
_PROTECT = re.compile(r"⟦\d+⟧|\[\[\d+\]\]|【#\d+#】|【#[\w\u4e00-\u9fff]+_\d+#】|\{#\d+#\}|｛\d+｝")

_OPEN = set("「『（(【〈《〔［")
_CLOSE = set("」』）)】〉》〕］")
_SENT_END = set("。！？!?…‥")
_SENT_TAIL = _SENT_END | _CLOSE
_HARD_END = "。！？…」』｣!?"
_SOFT_BREAK = set("、，,；;：:")


@dataclass
class Chunk:
    id: int
    text: str
    chars: int = 0
    is_dialogue: bool = False
    line_start: int = 0
    line_end: int = 0
    context: str = ""
    scene_before: bool = False      # 本块前有一个场景分隔线 `---`

    def __post_init__(self) -> None:
        if not self.chars:
            self.chars = len(self.text)
        if not self.is_dialogue:
            first = (self.text.split("\n") or [""])[0].lstrip()
            self.is_dialogue = first[:1] in _OPEN

    def to_meta(self) -> dict:
        return {"id": self.id, "chars": self.chars, "is_dialogue": self.is_dialogue,
                "line_start": self.line_start, "line_end": self.line_end,
                "context": self.context, "scene_before": self.scene_before}


# ------------------------------------------------------------------ 句子切分
def split_sentences(line: str) -> List[str]:
    """按句末标点切分，但不切在引号/括号内部。"""
    out: List[str] = []
    buf: List[str] = []
    depth = 0
    for ch in line:
        buf.append(ch)
        if ch in _OPEN:
            depth += 1
            continue
        if ch in _CLOSE:
            depth = max(0, depth - 1)
            if depth == 0:
                out.append("".join(buf))
                buf = []
            continue
        if ch in _SENT_END and depth == 0:
            out.append("".join(buf))
            buf = []
    if buf:
        tail = "".join(buf)
        if tail.strip():
            out.append(tail)
    return [s for s in (x.strip() for x in out) if s]


def _protected_spans(s: str) -> List[tuple]:
    return [(m.start(), m.end()) for m in _PROTECT.finditer(s)]


def _cut_ok(s: str, i: int, spans: List[tuple]) -> bool:
    for a, b in spans:
        if a < i < b:
            return False
    return True


def split_long_text(text: str, max_chars: int) -> List[str]:
    """把超长文本切成 ≤max_chars 的片段，优先在句末/逗号处下刀。

    关键：**保留原文换行**。旧实现把句子片段用字符串相加合并，把换行吞掉了，
    导致多行段落变成一行，行数守恒校验全部误报。
    """
    if len(text) <= max_chars:
        return [text]

    units: List[str] = []
    for line in text.split("\n"):
        if not line.strip():
            continue
        if len(line) <= max_chars:
            units.append(line)
        else:
            units.extend(_split_line(line, max_chars))

    pieces: List[str] = []
    buf: List[str] = []
    buf_len = 0
    for u in units:
        add = len(u) + (1 if buf else 0)
        if buf and buf_len + add > max_chars:
            pieces.append("\n".join(buf))
            buf, buf_len = [u], len(u)
        else:
            buf.append(u)
            buf_len += add
    if buf:
        pieces.append("\n".join(buf))
    return pieces or [text]


def _split_line(sent: str, max_chars: int) -> List[str]:
    """单行过长时切开（句末 → 逗号 → 硬切），不切断占位符。"""
    pieces: List[str] = []
    for part in split_sentences(sent):
        if len(part) <= max_chars:
            pieces.append(part)
            continue
        spans = _protected_spans(part)
        start = 0
        n = len(part)
        while start < n:
            end = min(start + max_chars, n)
            if end < n:
                cut = -1
                for i in range(end, start + max(1, max_chars // 3) - 1, -1):
                    if i < n and part[i - 1] in _SOFT_BREAK and _cut_ok(part, i, spans):
                        cut = i
                        break
                if cut <= start:
                    # 绝不切在句子中间：向后找句末；实在没有就整段当一行
                    nb = -1
                    for i in range(end, min(n, end + max_chars)):
                        if part[i - 1] in _HARD_END and _cut_ok(part, i, spans):
                            nb = i
                            break
                    if nb <= start:
                        # 单句比上限还长：最多放大到 3 倍，切在逗号处；再不行就整段
                        far = min(n, start + max_chars * 3)
                        for i in range(far, end, -1):
                            if _cut_ok(part, i, spans) and part[i - 1] in _SOFT_BREAK:
                                nb = i
                                break
                    # 连逗号都没有的病态长句：只能硬切，但放大到 3 倍上限
                    cut = nb if nb > start else min(start + max_chars * 3, n)
                end = cut if cut > start else n
            pieces.append(part[start:end])
            start = end

    merged: List[str] = []
    buf = ""
    for p in pieces:
        if buf and len(buf) + len(p) > max_chars:
            merged.append(buf)
            buf = p
        else:
            buf += p
    if buf:
        merged.append(buf)
    return merged


# ------------------------------------------------------------------ 主入口
def segment(text: str,
            chunk_chars: int = 320,
            hard_max: int = 520,
            context_chars: int = 0,
            keep_scene_break: bool = True) -> List[Chunk]:
    """切块主入口。返回带 id 的 Chunk 列表。"""
    if not text.strip():
        return []
    hard_max = max(chunk_chars, hard_max)

    lines = text.split("\n")

    # ---- 1. 分组：一行=一个原子（clean 已把软换行合并），场景行单列
    groups: List[tuple] = []   # (kind, 行文本, line_start)
    for idx, line in enumerate(lines):
        s = line.strip()
        if not s:
            continue
        if keep_scene_break and s == "---":
            groups.append(("scene", "---", idx))
            continue
        groups.append(("text", line.rstrip(), idx))

    # ---- 2. 段落 → 块（贪心合并，场景边界强制断开）
    chunks: List[Chunk] = []
    cur_text: List[str] = []
    cur_chars = 0
    cur_start = 0
    cur_end = 0
    scene_pending = False

    def flush_chunk() -> None:
        nonlocal cur_text, cur_chars, cur_start, cur_end, scene_pending
        if cur_text:
            t = "\n".join(cur_text).strip("\n")
            if t.strip():
                chunks.append(Chunk(id=0, text=t, line_start=cur_start,
                                    line_end=cur_end, scene_before=scene_pending))
                scene_pending = False
        cur_text = []
        cur_chars = 0

    for kind, group_line, line_start in groups:
        if kind == "scene":
            flush_chunk()
            scene_pending = True
            continue

        para = group_line
        para_len = len(para)
        if para_len == 0:
            continue

        # 超长段落：先切成若干 ≤chunk_chars 的片段，各自成块
        if para_len > hard_max:
            flush_chunk()
            pieces = split_long_text(para, chunk_chars)
            for k, piece in enumerate(pieces):
                chunks.append(Chunk(id=0, text=piece.strip("\n"),
                                    line_start=line_start, line_end=line_start,
                                    scene_before=(scene_pending if k == 0 else False)))
            if pieces:
                scene_pending = False
            continue

        if cur_text and cur_chars + para_len + 1 > chunk_chars and cur_chars >= chunk_chars * 0.45:
            flush_chunk()

        if not cur_text:
            cur_start = line_start
        cur_text.append(para)
        cur_end = line_start
        cur_chars += para_len + (1 if len(cur_text) > 1 else 0)

    flush_chunk()

    # ---- 3. 编号 + 上文锚点
    for i, ch in enumerate(chunks, 1):
        ch.id = i

    if context_chars > 0:
        for i in range(1, len(chunks)):
            prev = chunks[i - 1].text
            if len(prev) <= context_chars:
                tail = prev
            else:
                tail = prev[-context_chars:]
                # 不要在占位符中间截断
                cut = tail.find("⟧") + 1
                if 0 < cut < len(tail) and tail[:1] == "⟦":
                    tail = tail[cut:]
            chunks[i].context = tail.strip("\n")

    return chunks


def merge_back(chunks: List[Chunk], translations: dict, scene_lines: bool = True) -> str:
    """把译文按块顺序拼回全文（不含/含场景行由调用方决定是否穿插 '---'）。"""
    parts = [translations.get(c.id, "") for c in chunks]
    return "\n".join(p for p in parts if p is not None)
