# -*- coding: utf-8 -*-
"""译文质检门：把"看不出来但已经坏了"的块拦在成品之外。

旧版没有校验，真实产物里出现了三类硬伤：
  * 7/67 个块整块返回**系统提示词复读**（模型把 prompt 念了一遍）；
  * 99/1181 个占位符丢失，43/60 块受影响，还原后正文出现 `【#人名_16#】` 残渣；
  * 53 个标签括号被改坏（`【…#]` / `[…]`），旧还原逻辑完全无法识别。

这里给出 9 类检查，结论三态：
  ok        —— 全部通过
  degraded  —— 有 warn（可接受但记录，默认计入成品）
  failed    —— 有 fail（默认剔除，strict 模式下整文件不出成品）

所有阈值集中在 core.DEFAULT_THRESHOLDS，可按语种/模型调整。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from .core import Config
from .placeholder import TermMap, check_placeholders

_KANA = re.compile(r"[ぁ-ゖァ-ヺー]")
_PLACEHOLDER = re.compile(r"⟦\d+⟧|\[\[\d+\]\]|【#\d+#】|\{#\d+#\}")

#: 模型把提示词念出来的强特征串（命中直接 failed）
_ECHO_SIGNS = (
    "系统保护占位符", "只输出中文翻译", "你是一个专业的日文翻译", "翻译成中文",
    "保持原文的格式", "不要输出任何其他内容", "请你翻译", "以下是翻译",
    # v2 自带提示词（sakura/GalTransl 走 system 角色时会整段复读）
    "硬性要求", "只输出译文本身", "不要输出任何说明性文字", "严格保持原文的行数",
)

#: 占位符被模型写坏的痕迹（【#10#】→【%102;】、【10】这类）
_PH_MALFORMED = re.compile(r"【[^】]{0,12}[%&][^】]{0,12}】|【\s*\d{1,5}\s*;?\s*】|⟦[^⟧]{0,8}%")

#: 弱特征（只在行首位置命中时才算，降低误伤对白的概率）
_ECHO_PREFIX = ("译文：", "翻译：", "输出：", "Output:")

#: 常见退化输出
_DEGENERATE = (
    "抱歉", "我无法", "无法翻译", "as an ai", "作为AI", "对不起，我不能",
)


@dataclass
class Check:
    name: str
    level: str          # ok / warn / fail
    detail: str = ""

    def to_dict(self) -> dict:
        return {"name": self.name, "level": self.level, "detail": self.detail}


@dataclass
class Validation:
    status: str = "ok"
    checks: List[Check] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)

    @property
    def failed(self) -> bool:
        return self.status == "failed"

    @property
    def degraded(self) -> bool:
        return self.status == "degraded"

    def add(self, name: str, level: str, detail: str = "") -> None:
        self.checks.append(Check(name, level, detail))

    def finalize(self) -> "Validation":
        levels = {c.level for c in self.checks}
        self.status = "failed" if "fail" in levels else ("degraded" if "warn" in levels else "ok")
        return self

    def to_dict(self) -> dict:
        return {"status": self.status,
                "checks": [c.to_dict() for c in self.checks if c.level != "ok"],
                "metrics": {k: round(v, 4) for k, v in self.metrics.items()}}

    def summary(self) -> str:
        bad = [f"{c.name}:{c.level}({c.detail})" for c in self.checks if c.level != "ok"]
        return f"{self.status}" + ((" | " + "; ".join(bad)) if bad else "")


# ------------------------------------------------------------------ 工具
def strip_placeholders(s: str) -> str:
    return _PLACEHOLDER.sub("", s or "")


def kana_ratio(s: str) -> float:
    plain = strip_placeholders(s)
    if not plain:
        return 0.0
    return len(_KANA.findall(plain)) / len(plain)


def _lines(s: str) -> List[str]:
    return [x for x in (s or "").split("\n") if x.strip()]


def _quote_balance(s: str) -> int:
    return s.count("「") - s.count("」")


def strip_context_echo(dst: str, context: str) -> str:
    """模型有时把上文锚点也一起翻了，这里把开头重复的部分切掉。"""
    if not context or not dst:
        return dst
    ctx = context.strip()
    if not ctx:
        return dst
    head = dst.lstrip()
    for n in (len(ctx), len(ctx) - 3, len(ctx) // 2):
        if n >= 6 and head[:n] == ctx[:n]:
            out = head[n:].lstrip("　 \n")
            return out or dst
    # 上文被"翻译"后的形态：以 ctx 最后 8 字为锚点
    tail = ctx[-8:]
    pos = head.find(tail)
    if 0 <= pos <= 40:
        return head[pos + len(tail):].lstrip("　 \n") or dst
    return dst


# ------------------------------------------------------------------ 主入口
def validate_chunk(src: str, dst: str,
                   tm: Optional[TermMap] = None,
                   cfg: Optional[Config] = None,
                   prompt: str = "",
                   context: str = "",
                   allowed_extra: Optional[Set[int]] = None) -> Validation:
    cfg = cfg or Config()
    th = cfg.thresholds
    v = Validation()

    src = src or ""
    dst = (dst or "").strip()

    # 1) 空输出
    if not dst:
        v.add("nonempty", "fail", "译文为空")
        return v.finalize()
    v.add("nonempty", "ok")

    # 2) 提示词复读 / 拒译
    low = dst.lower()
    hit = [s for s in _ECHO_SIGNS if s.lower() in low]
    if not hit:
        first = dst.lstrip().split("\n", 1)[0]
        hit = [s for s in _ECHO_PREFIX if first.startswith(s)]
    if hit:
        v.add("prompt_echo", "fail", f"命中特征: {','.join(hit[:3])}")
    else:
        v.add("prompt_echo", "ok")
    deg = [s for s in _DEGENERATE if s.lower() in low]
    if deg and len(dst) < 120:
        v.add("refusal", "fail", f"疑似拒译/道歉: {deg[0]}")
    else:
        v.add("refusal", "ok")

    plain_src = strip_placeholders(src)
    plain_dst = strip_placeholders(dst)

    # 3) 行数守恒
    n_src, n_dst = len(_lines(src)), len(_lines(dst))
    v.metrics["lines_src"] = n_src
    v.metrics["lines_dst"] = n_dst
    if n_src:
        diff = abs(n_dst - n_src) / n_src
        v.metrics["line_diff"] = diff
        if n_dst != n_src:
            # 行数是硬不变量：块内一行对应原文一行，丢/增一行都是真错误
            # （大偏差仍是 fail，小偏差以前只 warn，实测漏掉了"每块少一行"的内容丢失）
            v.add("line_count", "fail", f"{n_src}→{n_dst}")
        else:
            v.add("line_count", "ok")
    else:
        v.add("line_count", "ok")

    # 3.9) 占位符是否被写坏（畸形标记既翻不出来也还原不了）
    bad_ph = _PH_MALFORMED.findall(dst)
    if bad_ph:
        v.add("ph_malformed", "fail", "占位符畸形: " + " ".join(bad_ph[:3]))
    else:
        v.add("ph_malformed", "ok")

    # 4) 占位符完整性
    if tm is not None and len(tm):
        rep = check_placeholders(src, dst, tm)
        extra = [i for i in rep.extra if not (allowed_extra and i in allowed_extra)]
        v.metrics["ph_expected"] = float(len(re.findall(r"⟦\d+⟧", src)))
        v.metrics["ph_missing"] = float(len(rep.missing))
        if rep.missing:
            v.add("placeholder", "fail", f"缺失{len(rep.missing)}: {rep.missing[:8]}")
        elif extra:
            v.add("placeholder", "fail", f"凭空出现: {extra[:8]}")
        elif rep.duplicate:
            v.add("placeholder", "warn", f"次数不符: {rep.duplicate[:8]}")
        elif rep.residual:
            v.add("placeholder", "warn", f"残渣: {rep.residual[:3]}")
        else:
            v.add("placeholder", "ok")

    # 5) 引号锚点（对白结构）
    qs, qd = src.count("「"), dst.count("「")
    v.metrics["quote_src"] = qs
    v.metrics["quote_dst"] = qd
    if qs and qd == 0:
        v.add("quote_anchor", "fail", "原文有对白，译文全丢")
    elif qs and abs(qs - qd) > max(1, int(qs * 0.5)):
        v.add("quote_anchor", "warn", f"{qs}→{qd}")
    elif _quote_balance(dst) > 0 and _quote_balance(src) == 0:
        v.add("quote_anchor", "warn", "引号未闭合")
    else:
        v.add("quote_anchor", "ok")

    # 6) 假名残留（没翻的情况）
    kr = kana_ratio(dst)
    v.metrics["kana_ratio"] = kr
    if kr > th["kana_residue_max"]:
        v.add("kana_residue", "fail", f"{kr:.1%}")
    elif kr > th["kana_residue_warn"]:
        v.add("kana_residue", "warn", f"{kr:.1%}")
    else:
        v.add("kana_residue", "ok")

    # 7) 长度比
    if plain_src:
        ratio = len(plain_dst) / len(plain_src)
        v.metrics["length_ratio"] = ratio
        lo, hi = th["length_ratio_min"], th["length_ratio_max"]
        wlo, whi = th["length_ratio_warn"]
        if ratio < lo or ratio > hi:
            v.add("length_ratio", "fail", f"{ratio:.2f}")
        elif ratio < wlo or ratio > whi:
            v.add("length_ratio", "warn", f"{ratio:.2f}")
        else:
            v.add("length_ratio", "ok")
    else:
        v.add("length_ratio", "ok")

    # 8) 逐字照抄（没翻 / 原样返回）
    if plain_src and plain_dst:
        common = sum(1 for a, b in zip(plain_src, plain_dst) if a == b)
        same = common / max(1, len(plain_src))
        v.metrics["similarity"] = same
        if same > th["same_as_src_ratio"] and len(plain_src) > 20:
            v.add("untranslated", "fail", f"与原文相似 {same:.0%}")
        else:
            v.add("untranslated", "ok")

    # 9) 退化重复
    lines = _lines(dst)
    run = 1
    max_run = 1
    for i in range(1, len(lines)):
        if lines[i] == lines[i - 1]:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 1
    v.metrics["max_repeat"] = max_run
    if max_run > th["repeat_line_max"]:
        v.add("repetition", "warn", f"连续重复{max_run}行")
    elif len(plain_dst) > 200:
        uniq = len(set(plain_dst)) / len(plain_dst)
        v.metrics["unique_char"] = uniq
        if uniq < th["unique_char_min"]:
            v.add("repetition", "fail", f"字符重复率异常 {uniq:.2f}")
        else:
            v.add("repetition", "ok")
    else:
        v.add("repetition", "ok")

    return v.finalize()
def repair_line_count(src: str, dst: str) -> tuple[str, bool]:
    """模型把相邻两句并成一行（11→10）时，按句末确定性拆回来，不重译。"""
    s_lines = src.splitlines()
    d_lines = dst.splitlines()
    if len(d_lines) == len(s_lines) or len(s_lines) - len(d_lines) != 1:
        return dst, False
    end_ch = "。！？…?!.」』"
    for i, ln in enumerate(d_lines):
        for j in range(6, len(ln) - 6):
            if ln[j - 1] not in end_ch:
                continue
            cand = d_lines[:i] + [ln[:j], ln[j:]] + d_lines[i + 1:]
            if len(cand) == len(s_lines):
                return chr(10).join(cand), True
    return dst, False