# -*- coding: utf-8 -*-
"""占位符：专名 / 术语的遮蔽与还原。

为什么重写
----------
旧版用 `【#人名_1#】` 这种长标签：
* token 贵（一个标签 ≈ 8~10 token），一个块里出现十几个就吃掉 1/3 上下文；
* 模型很容易改坏——实测 1181 个占位符丢了 99 个（8.4%），53 个括号被改成 `【…#]`；
* 还原是"逐条 str.replace"，20,000 块要 12.68 秒（平方级），且丢一个就整块漏还原。

新版做法：
1. **短标记**：默认 `⟦12⟧`（全局唯一序号，不带类别），两侧都是罕见括号，token 极少。
2. **单次正则还原**：一次 `re.sub` 扫全文（实测 0.013s / 538k 字），复杂度线性。
3. **容错**：`⟦12⟧` 被模型写成 `[12]`、`【12】`、`⟦ 12 ⟧`、全角数字、甚至丢了右括号，都能还原；
   只有序号在映射表里才替换，避免把正文里本来就有的 `[1]` 注释误改。
4. **完整性校验**：还原前先比对"源块期望的序号 multiset"与"译文实际出现的序号"，
   给出 missing / extra / duplicate / residual 四类报告，交给 validate.py 判定三态。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

# ------------------------------------------------------------------ 样式
STYLES: Dict[str, str] = {
    "bracket": "⟦{n}⟧",
    "double": "[[{n}]]",
    "hash": "【#{n}#】",
    "brace": "{#{n}#}",
}

#: 通用容错：任意"成对"的罕见括号包住纯数字/带 # 的数字
_GENERIC_LEFT = "⟦〖【\\[［{(｛〔"
_GENERIC_RIGHT = "⟧〗】\\]］})｝〕"

TIER1: Dict[str, str] = {
    "bracket": r"⟦\s*#?\s*([0-9０-９]{1,5})\s*#?\s*⟧",
    "double": r"\[\[\s*#?\s*([0-9０-９]{1,5})\s*#?\s*\]\]",
    "hash": r"【\s*#\s*([0-9０-９]{1,5})\s*#\s*】",
    "brace": r"\{#?\s*([0-9０-９]{1,5})\s*#?\}",
}

TIER2 = r"[%s]\s*#*\s*([0-9０-９]{1,5})\s*#*\s*[%s]" % (_GENERIC_LEFT, _GENERIC_RIGHT)

#: 只有左括号（模型漏了右括号）
TIER3 = r"⟦\s*#?\s*([0-9０-９]{1,5})\s*#?"

_FW_DIGITS = str.maketrans("０１２３４５６７８９", "0123456789")


def _to_int(s: str) -> Optional[int]:
    try:
        return int(s.translate(_FW_DIGITS))
    except Exception:
        return None


def mask_code(style: str, idx: int) -> str:
    return STYLES.get(style, STYLES["bracket"]).format(n=idx)


# ------------------------------------------------------------------ 数据结构
@dataclass
class TermEntry:
    idx: int
    term: str
    type: str = "专名"
    target: str = ""
    conf: str = "high"           # high / medium / low
    count: int = 0
    note: str = ""

    def to_dict(self) -> dict:
        return {"idx": self.idx, "term": self.term, "type": self.type,
                "target": self.target, "conf": self.conf,
                "count": self.count, "note": self.note}


@dataclass
class TermMap:
    """序号 → 词条。序号全局唯一（不按类别分段），还原时无需类别信息。"""

    style: str = "bracket"
    entries: Dict[int, TermEntry] = field(default_factory=dict)

    # -- 构建 --------------------------------------------------
    @classmethod
    def build(cls, terms: Iterable[tuple], style: str = "bracket") -> "TermMap":
        """terms: 可迭代的 (term, type, conf, count) 或 TermEntry。"""
        tm = cls(style=style)
        idx = 0
        seen = set()
        for item in terms:
            if isinstance(item, TermEntry):
                term, typ, conf, cnt = item.term, item.type, item.conf, item.count
            else:
                term, typ = item[0], item[1]
                conf = item[2] if len(item) > 2 else "high"
                cnt = item[3] if len(item) > 3 else 0
            if not term or term in seen:
                continue
            seen.add(term)
            idx += 1
            tm.entries[idx] = TermEntry(idx=idx, term=term, type=typ,
                                        conf=conf, count=cnt)
        return tm

    # -- 序列化 ------------------------------------------------
    def to_dict(self) -> dict:
        return {"version": "4.0", "style": self.style,
                "terms": [e.to_dict() for e in self.entries.values()]}

    @classmethod
    def from_dict(cls, d: dict) -> "TermMap":
        tm = cls(style=(d or {}).get("style", "bracket"))
        for e in (d or {}).get("terms", []):
            entry = TermEntry(idx=int(e.get("idx", 0)), term=e.get("term", ""),
                              type=e.get("type", "专名"), target=e.get("target", ""),
                              conf=e.get("conf", "high"), count=int(e.get("count", 0)),
                              note=e.get("note", ""))
            if entry.idx:
                tm.entries[entry.idx] = entry
        return tm

    # -- 查询 --------------------------------------------------
    def by_term(self, term: str) -> Optional[TermEntry]:
        for e in self.entries.values():
            if e.term == term:
                return e
        return None

    def unresolved(self) -> List[TermEntry]:
        return [e for e in self.entries.values() if not e.target.strip()]

    def targets(self) -> Dict[int, str]:
        return {i: (e.target.strip() or e.term) for i, e in self.entries.items()}

    def __len__(self) -> int:
        return len(self.entries)


# ------------------------------------------------------------------ 遮蔽
def mask_text(text: str, tm: TermMap) -> Tuple[str, Dict[str, int]]:
    """单次正则完成全部遮蔽（长词优先），返回 (遮蔽文本, {序号: 命中次数})。

    左边界保护：前面紧贴着假名的不遮。
    例子：专名表里有 `クリ`，原文 `びっクリした`——直接替换会造出 `ビッ⟦8⟧` 这种
    模型无法处理的碎词（实测里模型会直接把它删掉，于是整块占位符校验报错）。
    """
    if not tm.entries:
        return text, {}
    pairs = sorted(((e.term, e.idx) for e in tm.entries.values()),
                   key=lambda x: len(x[0]), reverse=True)
    mapping = {t: mask_code(tm.style, i) for t, i in pairs}
    term_idx = {t: i for t, i in pairs}
    body = "|".join(re.escape(t) for t, _ in pairs)
    pat = re.compile(r"(?<![ぁ-ゖァ-ヺ])(?:" + body + r")")
    hits: Dict[int, int] = {}

    def _rep(m: re.Match) -> str:
        t = m.group(0)
        idx = term_idx.get(t)
        if idx is None:
            return t
        hits[idx] = hits.get(idx, 0) + 1
        return mapping[t]

    out = pat.sub(_rep, text)
    return out, hits


def mask_counts(text: str, tm: TermMap) -> Dict[int, int]:
    """只统计（不替换）各词条在文本中的出现次数。"""
    _, hits = mask_text(text, tm)
    return hits


# ------------------------------------------------------------------ 还原
@dataclass
class RestoreReport:
    replaced: int = 0
    missing: List[int] = field(default_factory=list)      # 源有、译无
    extra: List[int] = field(default_factory=list)        # 译有、源无
    duplicate: List[int] = field(default_factory=list)    # 译文里多出的重复
    residual: List[str] = field(default_factory=list)     # 还原后仍残留的残渣

    @property
    def clean(self) -> bool:
        return not (self.missing or self.extra or self.residual)

    def to_dict(self) -> dict:
        return {"replaced": self.replaced, "missing": self.missing,
                "extra": self.extra, "duplicate": self.duplicate,
                "residual": self.residual[:20]}


def _count_indices(text: str, style: str, allowed: Optional[set] = None) -> Dict[int, int]:
    """统计文本里出现的（可能已损坏的）占位符序号及次数。"""
    counts: Dict[int, int] = {}
    for pat in (TIER1.get(style, TIER1["bracket"]), TIER2, TIER3):
        for m in re.finditer(pat, text):
            idx = _to_int(m.group(1))
            if idx is None:
                continue
            if allowed is not None and idx not in allowed:
                continue
            counts[idx] = counts.get(idx, 0) + 1
        if pat is TIER3:
            break
    return counts


def restore_text(text: str, tm: TermMap) -> Tuple[str, RestoreReport]:
    """把译文里的占位符还原成中文译名（容错 + 报告）。"""
    rep = RestoreReport()
    if not text:
        return text, rep
    targets = tm.targets()
    if not targets:
        return text, rep

    def _sub(pat: str, s: str) -> str:
        def _rep(m: re.Match) -> str:
            idx = _to_int(m.group(1))
            if idx is None or idx not in targets:
                return m.group(0)
            rep.replaced += 1
            return targets[idx]
        return re.sub(pat, _rep, s)

    out = _sub(TIER1.get(tm.style, TIER1["bracket"]), text)   # 1) 精确样式
    out = _sub(TIER2, out)                                    # 2) 括号被换掉
    out = _sub(TIER3, out)                                    # 3) 丢了右括号

    # 3.5) 模型把 【#10#】 写成 【%102;】 这种"实体化"形态 → 用已知序号做前缀匹配
    _ENT = re.compile(r"【%\s*(\d{1,6})\s*;?】?")

    def _rep_ent(m: re.Match) -> str:
        num = m.group(1)
        for k in range(len(num), 0, -1):
            idx = _to_int(num[:k])
            if idx is not None and idx in targets:
                rep.replaced += 1
                return targets[idx]
        return m.group(0)

    out = _ENT.sub(_rep_ent, out)

    # 3) 残渣：还原后还剩下的尖括号/【# 片段
    for m in re.finditer(r"⟦[^⟧\n]{0,12}⟧?|【\s*#[^】\n]{0,12}】?|\[\[\s*\d{1,5}\s*\]?", out):
        rep.residual.append(m.group(0))
        if len(rep.residual) >= 50:
            break
    # 4) 彻底无法还原的碎片（如模型只吐了半个 ⟦）→ 直接去掉，决不把系统符号留给读者
    if rep.residual:
        out = re.sub(r"⟦|⟧", "", out)
        out = re.sub(r"【\s*#+.{0,12}?】?", "", out)     # 【##10】这类残片（. 不跨行）
        out = re.sub(r"[\[［]\s*#+\s*\d{1,5}\s*#*\s*[\]］]?", "", out)
    return out, rep


def check_placeholders(src: str, dst: str, tm: TermMap) -> RestoreReport:
    """对比源块与译块的占位符完整性（还原前调用，用于质检）。"""
    style = tm.style
    allowed = set(tm.entries.keys())
    exp = _count_indices(src, style, allowed)
    got = _count_indices(dst, style, allowed)
    rep = RestoreReport()
    for idx, n in exp.items():
        m = got.get(idx, 0)
        if m == 0:
            rep.missing.append(idx)
        elif m < n:
            rep.duplicate.append(idx)      # 次数不足，按重复/缺失记录
    for idx, n in got.items():
        e = exp.get(idx, 0)
        if e == 0:
            rep.extra.append(idx)
        elif n > e:
            rep.duplicate.append(idx)
    rep.missing.sort()
    rep.extra.sort()
    rep.duplicate = sorted(set(rep.duplicate))
    return rep


def indices_in(text: str, style: str = "bracket",
               allowed: Optional[set] = None) -> Dict[int, int]:
    """公开接口：统计文本里出现的占位符序号及次数（容错）。"""
    return _count_indices(text, style, allowed)


def placeholder_ratio(text: str, style: str = "bracket") -> float:
    """占位符字符占比（用于判断块是否"全是名字"、长度比阈值要不要放宽）。"""
    if not text:
        return 0.0
    n = sum(len(m.group(0)) for m in re.finditer(TIER1.get(style, TIER1["bracket"]), text))
    return n / len(text)

def audit_masks(text: str, style: str = "hash", allowed=None) -> tuple[str, dict]:
    """翻译前的体检：把畸形占位符（【##10】/【%102;】/【#10】）规整回标准样式。

    为什么要在翻译前做：模型看到畸形标记会跟着学坏（实测它把 【#10#】 写成 【%102;】），
    一旦写坏，译文既翻不出内容、也还原不回来。这里在送译者之前先统一成标准标记。
    """
    import re as _re
    stats = {"total": 0, "fixed": 0, "samples": []}
    std = STYLES.get(style, STYLES["hash"])

    def _canon(m: _re.Match) -> str:
        stats["total"] += 1
        raw = m.group(0)
        num = None
        for g in m.groups():
            if g is not None:
                num = _to_int(g)
                break
        if num is None:
            stats["fixed"] += 1
            if len(stats["samples"]) < 5:
                stats["samples"].append(raw)
            return ""
        if allowed is not None and num not in allowed:
            # 【%102;】里的 102 是 【#10#】被写坏的产物 → 试前缀匹配已知序号
            hit = None
            txt_num = str(num)
            for k in range(len(txt_num) - 1, 0, -1):
                cand = _to_int(txt_num[:k])
                if cand is not None and cand in allowed:
                    hit = cand
                    break
            if hit is None:
                return raw
            num = hit
        good = std.format(n=num)
        if raw != good:
            stats["fixed"] += 1
            if len(stats["samples"]) < 5:
                stats["samples"].append(raw)
        return good

    # 各种括号 + 任意数量的 # / % / 数字
    pat = _re.compile(r"【\s*[#%]*\s*(\d{1,5})\s*[#%;]*\s*】|⟦\s*[#%]*\s*(\d{1,5})\s*[#%;]*\s*[⟧]?")
    out = pat.sub(_canon, text)
    return out, stats
