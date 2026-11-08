# -*- coding: utf-8 -*-
"""翻译单元：**完整句 + 不超过 N token**，可打包发送、可按编号装回。

语义对齐 `F:\\aiwork\\猫仔v4\\猫仔AI分章器.py` 的分块器（v7_split_sentences /
v6_group_chunks）：

* 只在句末标点后断句，句末标点连同紧跟的收尾引号/括号一起带走
  （`「……あぁ」` 整句成一条，`これは、礼子（れいこ）さんだ。` 不会在 `）` 处被切开）；
* 绝不从句中间切；单句自身就超过上限时整句独占一条（宁可超一点，也不切碎）；
* 贪心打包：累计 token 尽量接近上限但不超；
* 纯标点/纯符号行（如孤立的「？」）判为 contentless，原样透传不送模型
  （实测：把孤立的「？」单独发给 4B，它会开始复述系统提示词）。

装回时按「源行号 + 行内序号」拼：同一行的单元直接相连，不同行之间保留换行，
因此行数、空行、`---` 场景线都与清洗后的原文严格一致。
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

#: 句末标点（与 猫仔v4 的 V7_STOP 一致）
STOP = "。！？!?…‥"
#: 句末标点后允许一起带走的收尾符
CLOSE = "」』）)】〉》〕］\"'" + "’”"

#: 占位符样式（与 placeholder.py 保持一致）
PH_RE = re.compile(r"⟦\d+⟧|\[\[\d+\]\]|【#\d+#】|【#[\w\u4e00-\u9fff]+_\d+#】|\{#\d+#\}|｛\d+｝")
#: 「有实义」的字符：字母/数字/假名/汉字
MEAT_RE = re.compile(r"[0-9A-Za-z\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff]")
KANA_RE = re.compile(r"[\u3040-\u309f\u30a0-\u30ff]")
#: 5 个以上连续重复字符 = 复读（4 个容易误伤「哈哈哈哈」「字字字字」这类正常行文）
REPEAT_RE = re.compile(r"(.)\1{4,}")

DEFAULT_CPT = 1.6          # 1 token ≈ 1.6 字（日文经验值，与 猫仔v4 默认一致）


# ------------------------------------------------------------------ token 估算
def count_tokens(text: str, chars_per_token: float = DEFAULT_CPT) -> int:
    """估 token 数。装了 tiktoken 就用真 token，否则按字数折算（同 猫仔v4）。"""
    t = text or ""
    if not t:
        return 0
    try:                                    # 可选依赖，没有就走估算
        import tiktoken                      # type: ignore
        enc = tiktoken.get_encoding("cl100k_base")
        return max(1, len(enc.encode(t)))
    except Exception:
        pass
    return max(1, int(len(t) / max(0.5, float(chars_per_token or DEFAULT_CPT))))


def token_method() -> str:
    try:
        import tiktoken                      # type: ignore  # noqa: F401
        return "tiktoken(cl100k_base)"
    except Exception:
        return f"按字数估算（1 token≈{DEFAULT_CPT} 字）"


# ------------------------------------------------------------------ 断句
def split_sentences(line: str) -> List[str]:
    """把一行切成完整句；句末标点后跟着的收尾引号/括号归同一句。

    两条规则（都是为了「一条就是一个能独立读的完整句」）：
    * `「`/`」` 算对话边界：括号闭合且里面已经有实义内容时断句
      （`「あぁ」と言った。` → `「あぁ」` + `と言った。`）；
    * `（）【】` 等普通括号**不**断句（`これは、礼子（れいこ）さんだ。` 不能被切开）；
    * 句首的省略号/标点不单独成句（`「……` 不往外吐），免得出现「？」「……」这种无实义单元。
    """
    out: List[str] = []
    buf: List[str] = []
    depth = 0                                   # 「」『』的嵌套层数
    i, n = 0, len(line)
    while i < n:
        ch = line[i]
        buf.append(ch)
        i += 1
        if ch in "「『":
            depth += 1
            continue
        if ch in "」』":
            depth = max(0, depth - 1)
            if depth == 0 and not is_contentless("".join(buf)):
                while i < n and (line[i] in CLOSE or line[i] in STOP):
                    buf.append(line[i])
                    i += 1
                out.append("".join(buf))
                buf = []
            continue
        if ch in STOP:
            if is_contentless("".join(buf)):     # 句首的「……」「？」先攒着
                continue
            while i < n and (line[i] in CLOSE or line[i] in STOP):
                buf.append(line[i])
                i += 1
            out.append("".join(buf))
            buf = []
    if buf:
        out.append("".join(buf))
    return [s for s in out if s]


def is_contentless(s: str) -> bool:
    """去掉占位符后没有任何实义字符（例：「？」「……」「!?」）。"""
    return MEAT_RE.search(PH_RE.sub("", s or "")) is None


# ------------------------------------------------------------------ 单元
@dataclass
class Unit:
    id: int
    line: int                       # 源行号（1 基）
    order: int                      # 行内序号（1 基）
    src: str
    tokens: int = 0
    kind: str = "text"              # text / punct / scene

    def to_dict(self) -> dict:
        return {"id": self.id, "line": self.line, "order": self.order,
                "src": self.src, "tokens": self.tokens, "kind": self.kind}

    @staticmethod
    def from_dict(d: dict) -> "Unit":
        return Unit(id=int(d["id"]), line=int(d.get("line", 0)),
                    order=int(d.get("order", 1)), src=str(d.get("src", "")),
                    tokens=int(d.get("tokens", 0)), kind=str(d.get("kind", "text")))

    @property
    def translatable(self) -> bool:
        return self.kind == "text"


def build_units(text: str, max_tokens: int = 20,
                chars_per_token: float = DEFAULT_CPT,
                scene_mark: str = "---") -> Tuple[List[Unit], List[str]]:
    """切翻译单元。返回 (units, 原行列表)。

    每行内部按句子贪心打包到 ≤max_tokens；单句超限就整句占一条。
    """
    lines = text.split("\n")
    units: List[Unit] = []
    uid = 0
    for ln, raw in enumerate(lines, 1):
        if not raw.strip():
            continue
        if scene_mark and raw.strip() == scene_mark:
            uid += 1
            units.append(Unit(id=uid, line=ln, order=1, src=raw,
                              tokens=0, kind="scene"))
            continue
        buf: List[str] = []
        bt = 0
        order = 0
        for sent in split_sentences(raw):
            tk = count_tokens(sent, chars_per_token)
            if buf and bt + tk > max_tokens:
                uid += 1
                order += 1
                body = "".join(buf)
                units.append(Unit(id=uid, line=ln, order=order, src=body, tokens=bt,
                                  kind="punct" if is_contentless(body) else "text"))
                buf, bt = [], 0
            buf.append(sent)
            bt += tk
        if buf:
            uid += 1
            order += 1
            body = "".join(buf)
            units.append(Unit(id=uid, line=ln, order=order, src=body, tokens=bt,
                              kind="punct" if is_contentless(body) else "text"))
    return units, lines


def merge_back(lines: Sequence[str], units: Sequence[Unit],
               texts: Dict[int, str]) -> str:
    """按行号装回全文。缺译文的单元用原文顶上（不静默丢内容）。"""
    per: Dict[int, List[Unit]] = {}
    for u in units:
        per.setdefault(u.line, []).append(u)
    out: List[str] = []
    for i, raw in enumerate(lines, 1):
        us = per.get(i)
        if not us:
            out.append(raw)                       # 空行原样
            continue
        us = sorted(us, key=lambda u: u.order)
        out.append("".join(texts.get(u.id) or u.src for u in us))
    return "\n".join(out)


# ------------------------------------------------------------------ 打包发送
def pack_units(units: Sequence[Unit], size: int = 8,
               max_chars: int = 700) -> List[List[Unit]]:
    """把可译单元按顺序打包成请求批次（不跨场景线，不超条数/字数）。"""
    size = max(1, int(size))
    packs: List[List[Unit]] = []
    cur: List[Unit] = []
    chars = 0
    for u in units:
        if not u.translatable:
            continue
        if cur and (len(cur) >= size or chars + len(u.src) > max_chars):
            packs.append(cur)
            cur, chars = [], 0
        cur.append(u)
        chars += len(u.src)
    if cur:
        packs.append(cur)
    return packs


BATCH_RULES = (
    "\n\n【本批规则】\n"
    "1. 下面是一批互相独立的短句，每条以「编号|」开头，只翻译竖线后面的日文；\n"
    "2. 逐条译成简体中文，输出的行数与条数完全一致，每行写成「编号|译文」，编号顺序不变；\n"
    "3. 不要把两条合并成一行，也不要把一条拆成多行；一条原文再短也要给出完整的译文；\n"
    "4. 不要输出除「编号|译文」以外的任何内容（不要解释、不要列大纲、不要复述规则）；\n"
    "5. 省略主语时按上文补全，人称代词要跟同一角色的既定叫法一致。"
)


def build_batch_user(pack: Sequence[Unit]) -> str:
    from .job import BODY_HEADER
    body = "\n".join(f"{i}| {u.src.rstrip()}" for i, u in enumerate(pack, 1))
    return f"{BODY_HEADER}\n{body}"


def build_batch_system(base_prompt: str, context: str = "") -> str:
    sys_p = (base_prompt or "") + BATCH_RULES
    if context:
        sys_p += f"\n\n【故事上文·仅供理解用，不要翻译、不要复述】\n{context}"
    return sys_p


_REPLY_RE = re.compile(r"^\s*[\[\(]?\s*(\d{1,3})\s*[\]\)]?\s*[|｜:：.、,]\s*(.*)$")


def parse_batch_reply(text: str, n: int) -> Dict[int, str]:
    """解析「编号|译文」回复。返回 {1..n 中成功解析的: 译文}。

    容错顺序：严格编号 → 编号带各种分隔符 → 没写编号但行数刚好（按顺序对上）。
    """
    got: Dict[int, str] = {}
    lines = [l for l in (text or "").split("\n")]
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        m = _REPLY_RE.match(line)
        if not m:
            continue
        k = int(m.group(1))
        body = m.group(2).strip()
        if 1 <= k <= n and body and k not in got:
            got[k] = body
    if len(got) == n:
        return got
    plain = [l.strip() for l in lines if l.strip() and not _REPLY_RE.match(l.strip())]
    if not got and plain and len(plain) == n:
        return {i + 1: t for i, t in enumerate(plain)}
    return got


# ------------------------------------------------------------------ 单元质检
def _max_run(s: str) -> Tuple[str, int]:
    """最长连续重复字符（用于区分「忠实于原文的啊啊啊啊」和真的复读）。"""
    best, cur, prev = "", 0, ""
    run, who = 0, ""
    for ch in s:
        if ch == prev:
            run += 1
        else:
            run = 1
            who = ch
        prev = ch
        if run > cur:
            cur, best = run, who
    return best, cur


def check_unit(src: str, dst: str, min_ratio: float = 0.30,
               max_ratio: float = 3.0) -> Tuple[str, List[str]]:
    """单元级快速质检：占位符 / 假名残留 / 长度 / 复读 / 空。"""
    probs: List[str] = []
    t = (dst or "").strip()
    if not t:
        return "failed", ["空译文"]
    want = Counter(PH_RE.findall(src or ""))
    got = Counter(PH_RE.findall(t))
    if want - got:
        probs.append("缺占位符 " + "/".join(sorted((want - got).elements()))[:60])
    if got - want:
        probs.append("多占位符 " + "/".join(sorted((got - want).elements()))[:60])
    status = "failed" if probs else "ok"
    if KANA_RE.search(PH_RE.sub("", t)):
        probs.append(f"假名残留 {len(KANA_RE.findall(PH_RE.sub('', t)))}")
        status = "failed" if status == "failed" else "degraded"
    r = len(t) / max(1, len(src or ""))
    if r < min_ratio or r > max_ratio:
        probs.append(f"长度比 {r:.2f}")
        status = "failed" if status == "failed" else "degraded"
    if REPEAT_RE.search(t):
        # 原文自己就是「ああああ」「……」这种长串 → 译文跟着长串是对的，不算复读
        ch, run = _max_run(t)
        same, srun = _max_run(PH_RE.sub("", src or ""))
        faithful = (same == ch and srun >= 3) or (srun >= 3 and run <= srun + 2)
        if not faithful:
            probs.append("复读")
            status = "failed" if status == "failed" else "degraded"
    return status, probs


def fix_passable(src: str, fix: str) -> bool:
    """中控给的修正稿能不能直接用：不能空、占位符要齐、不能留日文假名。"""
    t = (fix or "").strip()
    if not t:
        return False
    want = Counter(PH_RE.findall(src or ""))
    got = Counter(PH_RE.findall(t))
    if want != got:
        return False
    if KANA_RE.search(PH_RE.sub("", t)):
        return False
    r = len(t) / max(1, len(src or ""))
    return 0.25 <= r <= 3.2
