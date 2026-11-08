# -*- coding: utf-8 -*-
"""清洗：把抓来的小说文本整理成"可切块、可校验"的规范文本。

和旧 step1（分段 + 清洗两趟、各写一套正则）相比的关键改动
--------------------------------------------------------------------------------
1. **只跑一趟**：旧流程 step1_segment 把「。」后的句子强行断行，破坏段落与引号结构；
   这里保留原始行/段结构，句子级切分交给 segment.py（它需要按需切，且保护占位符）。
2. **装饰行归一而不是删除**：`─────`、`※※※` 这类场景分隔是叙事信号，直接删会让
   上下文串味；这里统一替换成 `---`，切块时作为"硬边界"。
3. **全角 ASCII 不再压平**：旧 _clean_line 把 U+FF01..U+FF5E 转半角，会把 `！？，` 变成
   `!?,`，中文/日文标点全丢，还破坏了模型对停顿的判断。改为默认不转（可选开启）。
4. **软换行合并**：日文小说常见 40 字硬折行，旧版不合并 → 切块时把一句话切成两半。
5. **去盗版站广告行**（中文源常见）。
6. 所有判断都是纯函数，同样的输入永远得到同样的输出——断点续传才能对齐块 ID。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# 装饰性字符（整行都是这些字符 → 视为分隔线）
_DECO = set("…‥・･—―─━═⎯▬▭▮▯▰▱﹏〜~～-−_=*＊+＋#＃◆◇◈●○◎■□▲△▼▽★☆※§¶†‡")


@dataclass
class CleanResult:
    text: str
    stats: Dict[str, int] = field(default_factory=dict)


# ------------------------------------------------------------------ 语种
def detect_lang(text: str) -> str:
    """粗略判断 ja / zh。以假名出现率为主。"""
    sample = text[:20000]
    kana = sum(1 for c in sample if "\u3041" <= c <= "\u30ff")
    han = sum(1 for c in sample if "\u4e00" <= c <= "\u9fff")
    if kana == 0:
        return "zh"
    # 日文里假名一般占汉字量的一半以上；纯汉字书名/术语多的文本仍按 ja 处理
    return "ja" if kana * 4 >= max(1, han) else "zh"


# ------------------------------------------------------------------ 判定
def is_decorative(line: str) -> bool:
    """整行是不是纯装饰/分隔。"""
    s = line.strip("　 \t")
    if len(s) < 2:
        return False
    if any(q in s for q in "「」『』"):     # 台词行（哪怕全是省略号）不是装饰
        return False
    nondeco = [c for c in s if c not in _DECO]
    if not nondeco:
        return True
    # 重复同一字符 ≥5 次
    if len(s) >= 5 and len(set(s)) == 1:
        return True
    # HTML 标签行 <br> <p> 等
    if re.fullmatch(r"(?:<[^<>\n]{1,12}>)+", s):
        return True
    # 「─── 第三章 ───」这类：装饰占多数、实义字 ≤4
    deco_ratio = sum(1 for c in s if c in _DECO) / len(s)
    if deco_ratio > 0.6 and len(nondeco) <= 4:
        return True
    # *** / ~~~ 分隔
    if re.fullmatch(r"[-=_*~#·．.\s]{3,}", s):
        return True
    return False


_ZH_BOILER = (
    re.compile(r"^(?:本书|本书由|本书来自|更多(?:精彩|小说|章节|好书)|请记住|一秒记住|"
               r"手机用户请|手机版阅读|最新网址|本章未完|全文阅读|百度搜索|搜[索狗]|"
               r"章节错误|内容来源|本站|无弹窗|更新最快|加入书签|投推荐票|"
               r"\(未完待续\)|\(本章完\)|（未完待续）|（本章完）)"),
    re.compile(r"^(?:https?://|www\.|m\.|wap\.)", re.I),
    re.compile(r"^[\w.-]+\.(?:com|net|org|cc|xyz|cn|la|info)(?:/\S*)?$", re.I),
)

_JA_BOILER = (
    re.compile(r"^(?:作者|著者|イラスト|挿絵|底本|校正|入力)[：:]"),
    re.compile(r"^(?:※|＊)\s*(?:この|本作品|無断)"),
)


def is_boilerplate(line: str, lang: str) -> bool:
    """广告/站点行。中日的特征都查（日文抓取源里也常混中文站广告）。"""
    s = line.strip()
    if not s:
        return False
    if any(q in s for q in "「」『』"):     # 对白行一律不当广告删
        return False
    for p in (tuple(_ZH_BOILER) + tuple(_JA_BOILER)):
        if p.match(s):
            return True
    return False


_SENT_END = set("。！？!?…‥")
_CLOSE = set("」』）)】〉》〕］\"'")


def _ends_sentence(line: str) -> bool:
    s = line.rstrip("　 \t")
    if not s:
        return True
    return s[-1] in _SENT_END or s[-1] in _CLOSE or s[-1] in "：:、，,"


def _starts_new_block(line: str) -> bool:
    s = line.lstrip("　 \t")
    if not s:
        return True
    return s[0] in "「『（(【〈《〔［" or is_decorative(s)


_ZERO_WIDTH = re.compile(r"[\u200b-\u200f\u2028\u2029\u202a-\u202e\u2060-\u2064\ufeff\ufffe]")
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_FULLWIDTH_ASCII = re.compile(r"[\uff01-\uff5e]")


def _normalize_line(line: str, to_half_ascii: bool) -> str:
    line = line.replace("\r", "")
    line = _ZERO_WIDTH.sub("", line)
    line = _CTRL.sub("", line)
    line = line.rstrip()                      # 去行尾空白
    line = line.lstrip("　 ")                 # 去首行缩进（全/半角）
    if to_half_ascii:
        line = _FULLWIDTH_ASCII.sub(lambda m: chr(ord(m.group()) - 0xFEE0), line)
    line = re.sub(r"[ \t]{2,}", " ", line)
    return line.strip()


def clean(raw: str,
          lang: str = "auto",
          merge_lines: Optional[bool] = None,
          drop_boilerplate: bool = True,
          keep_scene_break: bool = True,
          to_half_ascii: bool = False,
          keep_lines: Optional[set] = None,
          drop_lines: Optional[set] = None) -> CleanResult:
    """主清洗入口。

    Parameters
    ----------
    lang             : auto / ja / zh
    merge_lines      : None 表示按语种自动（ja 默认合并软换行，zh 不合并）
    keep_scene_break : 装饰行是否保留为 `---`（False 则直接删）
    """
    stats = {"lines_in": 0, "lines_out": 0, "deco": 0, "boiler": 0,
             "merged": 0, "blanks": 0, "dropped": 0}
    if not raw:
        return CleanResult("", stats)

    if lang == "auto":
        lang = detect_lang(raw)
    if merge_lines is None:
        merge_lines = (lang == "ja")

    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    lines = raw.split("\n")
    stats["lines_in"] = len(lines)

    # ---- 1. 逐行规范化 + 分类
    kind: List[str] = []      # 'text' | 'blank' | 'scene'
    body: List[str] = []
    for _idx, line in enumerate(lines):
        s = _normalize_line(line, to_half_ascii)
        if drop_lines and _idx in drop_lines:      # 模型判定要删的可疑行
            stats["dropped"] += 1
            continue
        if not s:
            kind.append("blank")
            body.append("")
            continue
        if keep_lines and _idx in keep_lines:      # 模型/白名单要求保留 → 原样进正文
            kind.append("text")
            body.append(s)
            continue
        if drop_boilerplate and is_boilerplate(s, lang):
            stats["boiler"] += 1
            continue
        if is_decorative(s):
            stats["deco"] += 1
            if keep_scene_break:
                kind.append("scene")
                body.append("---")
            continue
        kind.append("text")
        body.append(s)

    # ---- 2. 合并软换行（句未结束且下一行不像新块开头）
    if merge_lines:
        out_k, out_b = [], []
        i = 0
        n = len(body)
        while i < n:
            k, s = kind[i], body[i]
            if k == "text":
                while i + 1 < n and kind[i + 1] == "text":
                    cur = s
                    nxt = body[i + 1]
                    if (_ends_sentence(cur) or len(cur) < 12
                            or _starts_new_block(nxt)):
                        break
                    s = cur + nxt
                    stats["merged"] += 1
                    i += 1
                out_k.append("text")
                out_b.append(s)
            else:
                out_k.append(k)
                out_b.append(s)
            i += 1
        kind, body = out_k, out_b

    # ---- 3. 规范空行：连续空行压成一个；首尾去掉；scene 前后不加空行
    final: List[str] = []
    prev = None
    for k, s in zip(kind, body):
        if k == "blank":
            if prev is None or prev == "blank":
                continue
            final.append("")
            prev = "blank"
            stats["blanks"] += 1
            continue
        if k == "scene":
            # 场景分隔前后各去掉多余空行
            if final and final[-1] == "":
                final.pop()
            if final and final[-1] != "---":
                final.append("---")
            elif not final:
                final.append("---")
            prev = "scene"
            continue
        final.append(s)
        prev = "text"

    while final and final[-1] in ("", "---"):
        final.pop()

    text = "\n".join(final)
    stats["lines_out"] = len(final)
    stats["dropped"] = stats["lines_in"] - stats["lines_out"]
    return CleanResult(text, stats)
