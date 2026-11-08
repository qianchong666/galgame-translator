# -*- coding: utf-8 -*-
"""内置自检：不依赖 GPU / 网络，覆盖清洗、切块、占位符、专名、质检、续传、组装。"""
from __future__ import annotations

import json
import re
import shutil
import tempfile
import threading
from pathlib import Path
from typing import List

from . import clean as cleanmod
from . import segment as segmod
from . import terms as termmod
from .core import Config, Logger, read_json, read_text_auto, write_json_atomic, write_text_atomic
from .job import Translator, default_prompt
from .llama import ChatResult
from .placeholder import (TermEntry, TermMap, check_placeholders, indices_in,
                          mask_text, restore_text)
from .validate import validate_chunk

APP_DIR = Path(__file__).resolve().parents[1]

_FAILS: List[str] = []
_PASS = 0


def ck(cond: bool, msg: str, extra: str = "") -> None:
    global _PASS
    if cond:
        _PASS += 1
        return
    _FAILS.append(f"{msg}" + (f"  [{extra}]" if extra else ""))


# ------------------------------------------------------------------ 假客户端
_KANA = re.compile(r"[ぁ-ゖァ-ヺー]")


def fake_zh(src: str) -> str:
    """把假名换成汉字，保留占位符/标点/汉字 → 结构合法但明显不等于原文。"""
    return "".join(("字" if _KANA.match(ch) else ch) for ch in src)


class ScriptedClient:
    """按需返回伪译文的客户端，用于验证流水线而不启模型。"""

    def __init__(self, bad_chunk: int = -1, log=None):
        self.calls: List[str] = []
        self.bad_chunk = bad_chunk
        self.log = log
        self._lock = threading.Lock()

    def chat(self, user, system=None, max_tokens=None, temperature=None, retries=None):
        with self._lock:
            self.calls.append(user)
        if "现在翻译（只输出对应行）" in user:
            return ChatResult(text="\n".join(
                m.group(0) for m in re.finditer(r"^\d+\. .+$", user, re.M)), finish="stop")
        if user.startswith("把下面的日文专名"):
            return ChatResult(text=user.splitlines()[-1].strip(), finish="stop")
        if self.bad_chunk >= 0 and self._hits() == self.bad_chunk:
            return ChatResult(text="你是一个专业的日文翻译。请将下面的日文文本翻译成中文。"
                                   "文中形如 ⟦1⟧ 的标签为系统保护占位符。", finish="stop")
        return ChatResult(text=fake_zh(user), finish="stop")

    def _hits(self) -> int:
        return sum(1 for c in self.calls
                   if "现在翻译（只输出对应行）" not in c
                   and not c.startswith("把下面的日文专名")) - 1

    @property
    def translate_calls(self) -> int:
        return sum(1 for c in self.calls
                   if "现在翻译（只输出对应行）" not in c
                   and not c.startswith("把下面的日文专名"))


# ------------------------------------------------------------------ 测试项
def t_encoding(tmp: Path) -> None:
    ja = "「おはよう」と彼女は言った。\r\n窓の外は雨だった。"
    for enc in ("utf-8", "cp932", "utf-16"):
        p = tmp / f"enc_{enc}.txt"
        p.write_bytes(ja.encode(enc))
        got, detected = read_text_auto(p)
        ck(got.replace("\r\n", "\n") == ja.replace("\r\n", "\n"),
           f"编码探测 {enc}", f"拿到 {detected}: {got[:20]!r}")


def t_clean() -> None:
    raw = ("序章\n"
           "──────────\n"
           "彼は窓の外を見ていた。雨が\n"
           "静かに降っていた。\n"
           "「なあ、友香」\n"
           "「なに？」\n"
           "更多精彩小说请访问 www.example.com\n"
           "！？\n")
    res = cleanmod.clean(raw, lang="ja")
    t = res.text
    ok = True
    ck("---" in t, "装饰行归一为 ---", t)
    ck("静かに降っていた" in t and "雨が静かに" in t, "软换行已合并", t)
    ck("！？" in t, "全角标点不被压平（旧版会变成 !?）", t)
    ck("example.com" not in t, "去盗版站广告行", t)
    ck(res.stats["deco"] >= 1 and res.stats["merged"] >= 1, "清洗统计正确", str(res.stats))
    # 装饰行数量守恒：不应把正文删掉
    ck("序章" in t and "なに？" in t, "正文未丢失")


def t_segment() -> None:
    text = ("一行目の文です。\n二行目の文です。\n"
            "---\n"
            "「これは長い対話で、中の句読点では切らないでほしいのです。"
            "それでも続きます。」\n"
            + "あ" * 900 + "\n")
    ch = segmod.segment(text, chunk_chars=120, hard_max=200, context_chars=40)
    ck(len(ch) >= 3, "切块数量合理", str(len(ch)))
    ck([c.id for c in ch] == list(range(1, len(ch) + 1)), "块 ID 连续")
    ck(all(c.chars <= 600 for c in ch), "硬上限生效(≤3倍)",
       str(max(c.chars for c in ch)))
    # 场景分隔：第一个 scene_before 出现在 --- 之后的块
    ck(any(c.scene_before for c in ch), "场景边界被记录")
    # 对白不被拆成"只有左引号"的块
    bad = [c.text for c in ch if c.text.count("「") != c.text.count("」")
           and "「" in c.text and len(c.text) < 100]
    ck(not bad, "引号未被拆坏", str(bad)[:120])
    # 确定性
    ch2 = segmod.segment(text, chunk_chars=120, hard_max=200, context_chars=40)
    ck([c.text for c in ch] == [c.text for c in ch2], "切块是纯函数（可续传）")
    # 占位符不被切断
    t3 = "⟦12⟧" * 60
    ch3 = segmod.segment(t3, chunk_chars=20, hard_max=30)
    ck(all(c.text.count("⟦") == c.text.count("⟧") for c in ch3),
       "占位符不被切断", str([c.text[:20] for c in ch3]))


def t_placeholder() -> None:
    tm = TermMap.build([("友香", "人名"), ("アンジェリカ", "人名"), ("聖マリア学園", "地名")])
    src = "「おはよう、友香」アンジェリカが聖マリア学園の門をくぐる。"
    masked, hits = mask_text(src, tm)
    ck("友香" not in masked and "⟦1⟧" in masked, "遮蔽生效", masked)
    ck(hits.get(1) == 1 and hits.get(2) == 1, "遮蔽命中统计正确", str(hits))

    # 正常还原
    out, rep = restore_text(masked, tm)
    ck(rep.clean and "友香" in out, "正常还原", out)

    # 各种损坏形态
    variants = [
        ("⟦1⟧", "[1]"),
        ("⟦1⟧", "【1】"),
        ("⟦1⟧", "⟦ 1 ⟧"),
        ("⟦1⟧", "⟦１⟧"),
        ("⟦1⟧", "⟦1"),
        ("⟦1⟧", "(1)"),
    ]
    for orig, broken in variants:
        t = f"「おはよう、{orig}」"
        t2, r2 = restore_text(t.replace(orig, broken), tm)
        ck("友香" in t2, f"容错还原 {broken}", t2)

    # 不是占位符的数字不变
    t3 = "第[9]章について"
    t4, _ = restore_text(t3, tm)
    ck(t4 == t3, "正文中的 [9] 不被误改", t4)

    # 左边界保护：不能把 `びっクリ` 切成 `びっ⟦n⟧`
    tm2 = TermMap.build([("クリ", "专名")])
    m2, h2 = mask_text("びっクリしたでしょ？", tm2)
    ck("クリ" in m2 and not h2, "词中假名不被拆开遮蔽", m2)
    m3, h3 = mask_text("「クリを触る」", tm2)
    ck(h3.get(1) == 1, "独立的片假名词仍能遮蔽", m3)

    # 缺失检测
    rep2 = check_placeholders("⟦1⟧と⟦2⟧", "だけ", tm)
    ck(sorted(rep2.missing) == [1, 2], "缺失检测", str(rep2.missing))
    rep3 = check_placeholders("⟦1⟧", "⟦1⟧⟦3⟧", tm)
    ck(rep3.extra == [3], "多余占位符检测", str(rep3.extra))


def t_terms() -> None:
    text = ("絶対友香さんが来た。全部友香さんも来た。一度友香さんに聞いた。\n"
            "「アンジェリカ、待って」と友香さんが言った。\n"
            "ベッドの上でドアを開けた。キスをした。\n"
            "聖マリア学園の前で待ち合わせた。\n")
    found = termmod.extract_terms(text, min_freq=1)
    names = {t.term for t in found}
    ck("友香" in names, "敬称人名提取 + 前缀修剪（絶対友香→友香）", str(sorted(names)))
    ck(not {"絶対友香", "全部友香", "一度友香"} & names, "错误拼接的人名已被消除",
       str(sorted(names)))
    ck("アンジェリカ" in names, "片假名专名保留")
    ck("ベッド" not in names and "ドア" not in names and "キス" not in names,
       "常用外来词不被当专名", str(sorted(names)))
    ck(any(t == "聖マリア学園" for t in names), "地名后缀识别", str(sorted(names)))


def t_validate() -> None:
    tm = TermMap.build([("友香", "人名")])
    src = "「おはよう、⟦1⟧」と彼女は言った。今日はいい天気だ。"
    good = "「早上好，⟦1⟧」她说道。今天天气真好。"
    v = validate_chunk(src, good, tm)
    ck(v.status == "ok", "正常译文通过质检", v.summary())

    v = validate_chunk(src, "你是一个专业的日文翻译，请翻译。", tm)
    ck(v.status == "failed" and any(c.name == "prompt_echo" for c in v.checks),
       "提示词复读被抓到", v.summary())

    v = validate_chunk(src, "「早上好」她说道。今天天气真好。", tm)
    ck(v.status == "failed" and any(c.name == "placeholder" for c in v.checks),
       "占位符丢失被抓到", v.summary())

    v = validate_chunk(src, "「おはようございます」と彼女は言った。今日はいい天気だ。", tm)
    ck(v.status == "failed", "假名残留/未翻译被抓到", v.summary())

    v = validate_chunk(src, "好。", tm)
    ck(v.status in ("degraded", "failed"), "长度比异常被标记", v.summary())


def t_pipeline(tmp: Path) -> None:
    src_text = (
        "序章\n────────\n"
        "「おはよう、友香さん」と彼は言った。\n窓の外は雨だった。\n\n"
        "「おはよう、アンジェリカ」と友香さんは答えた。\n"
        "そして二人は聖マリア学園へ向かった。\n\n"
        "---\n\n"
        "次の日の朝も、雨が降っていた。\n"
        "「今日は行かないの？」と彼女が聞いた。\n"
    ) * 3
    src = tmp / "sample.txt"
    write_text_atomic(src, src_text)

    cfg = Config(do_clean=True, chunk_chars=90, chunk_hard_max=140,
                 context_chars=0, strict=True, translate_terms=True,
                 retries=2, mask_terms=True, lang="ja")
    log = Logger(tmp / "test.log", echo=False, level="warn")
    client = ScriptedClient()
    out = tmp / "OUT"
    tr = Translator(cfg, log, out, clients=[client])
    rep = tr.process(src)
    base = out / "sample"
    ck(rep.status == "ok", "流水线跑通（全部块通过）", rep.to_dict().__str__()[:200])
    ck((base / "sample_仅翻译.txt").exists(), "产出 仅翻译.txt")
    ck((base / "sample_对照.txt").exists(), "产出 对照.txt")
    src_out = (base / "sample_原文.txt").read_text(encoding="utf-8")
    ck("友香" in src_out and "⟦" not in src_out, "原文版已还原占位符", src_out[:60])
    tr_txt = (base / "sample_仅翻译.txt").read_text(encoding="utf-8")
    ck("⟦" not in tr_txt, "成品里没有占位符残渣", tr_txt[:80])
    ck("友香" in tr_txt, "占位符已还原为中文名")
    ck(rep.ok == rep.chunks, "块数统计一致", f"{rep.ok}/{rep.chunks}")

    # --- 续传：第二次调用不应再请求模型
    before = client.translate_calls
    rep2 = Translator(cfg, log, out, clients=[client]).process(src)
    ck(client.translate_calls == before, "续传：没有重复请求",
       f"{client.translate_calls - before} 次")
    ck(rep2.status == "ok", "续传结果仍为 ok")

    # --- 只删一个块 → 只重跑那一块
    trans = sorted((base / "trans").glob("*.json"))
    ck(len(trans) >= 2, "块结果已落盘", str(len(trans)))
    trans[0].unlink()
    before2 = client.translate_calls
    rep3 = Translator(cfg, log, out, clients=[client]).process(src)
    ck(client.translate_calls == before2 + 1, "只重译被删的那一块",
       f"{client.translate_calls - before2} 次")

    # --- strict 模式下的失败语义
    out2 = tmp / "OUT2"
    bad = ScriptedClient(bad_chunk=1)
    cfg2 = Config(do_clean=True, chunk_chars=90, chunk_hard_max=140, strict=True,
                  translate_terms=False, retries=1, lang="ja")
    rep4 = Translator(cfg2, log, out2, clients=[bad]).process(src)
    b2 = out2 / "sample"
    ck(rep4.status == "failed", "strict：有失败块 → 文件失败", rep4.status)
    ck(not (b2 / "sample_仅翻译.txt").exists(), "strict：失败文件不产出成品")
    ck(list(b2.glob("*_FAILED.json")), "strict：写出失败报告")

    # --- 非 strict 模式允许部分成品
    cfg3 = Config(do_clean=True, chunk_chars=90, chunk_hard_max=140, strict=False,
                  translate_terms=False, retries=1, lang="ja")
    out3 = tmp / "OUT3"
    rep5 = Translator(cfg3, log, out3, clients=[ScriptedClient(bad_chunk=1)]).process(src)
    b3 = out3 / "sample"
    ck((b3 / "sample_仅翻译.txt").exists(), "非 strict：仍产出成品")
    ck(rep5.status == "partial", "非 strict：状态为 partial", rep5.status)

    # --- 改了遮蔽规则（术语表变化）→ 旧译文必须归档而不是错位复用
    cfg_g = Config(do_clean=True, chunk_chars=90, chunk_hard_max=140, context_chars=0,
                   strict=True, translate_terms=True, retries=2, mask_terms=True,
                   mask_katakana=True, lang="ja")
    gl = tmp / "glossary.json"
    write_json_atomic(gl, {"友香": {"type": "人名"}, "雨": {"type": "术语"}})
    old_count = len(list((base / "trans").glob("*.json")))
    Translator(cfg_g, log, out, clients=[client], glossary=gl).process(src)
    ck(bool(list(base.glob("trans_stale_*"))),
       "遮蔽结果变化时旧译文被归档而不是错位复用", f"旧 {old_count} 块")


def t_echo_guard(tmp: Path) -> None:
    """模型回吐指令 / 半个占位符：不能混进成品。"""
    from .job import Translator as _T
    dirty = ("「早上好」他说道。\n【中控复核意见】所；有；\"；⟦")
    cleaned = _T._post_clean(dirty)
    ck("中控复核意见" not in cleaned, "回吐防护: 指令标记被截掉", cleaned[:60])
    ck("早上好" in cleaned, "回吐防护: 正文保留")

    # 模型会把标题改写（漏字/换词），字面匹配漏过一次 → 正则兜底
    para = ("「早上好」他说道。\n「今天也是好天气」\n"
            "【上次译文的问题】占位符 ⟦3⟧ 的出现次数不对，请与原文一一对应。")
    cleaned = _T._post_clean(para)
    ck("译文的问题" not in cleaned and "⟦3⟧ 的出现次数" not in cleaned,
       "回吐防护: 改写过的指令标题也截掉", cleaned[-40:])
    ck("今天也是好天气" in cleaned, "回吐防护: 改写标记前的正文保留")

    front = "【中控复核意见】请把名字改回来\n「早上好」他说道。"
    cleaned = _T._post_clean(front)
    ck("中控复核意见" not in cleaned and "早上好" in cleaned,
       "回吐防护: 标记在开头时只删那一行，不误删正文", cleaned[:40])

    cfg = Config(lang="ja", chunk_chars=200, chunk_hard_max=300)
    tm = TermMap.build([("友香", "人名", "high", 3)], style="bracket")
    src = "⟦1⟧「おはよう」\n⟦1⟧は笑った。"
    broken = "⟦1⟧「早上好」\n⟦1は笑った。"      # 右边括号丢了
    out, rep = restore_text(broken, tm)
    ck("⟦" not in out and "⟧" not in out, "回吐防护: 半个占位符也被清掉", out[:60])
    ck("友香" in out, "回吐防护: 能救的占位符仍然还原成译名", out[:60])

    # 通过整条流水线：模型回吐指令的块必须降级而不是把指令写进成品
    srcf = tmp / "echo.txt"
    write_text_atomic(srcf, "「おはよう」と彼は言った。\n窓の外は雨だった。\n" * 3)

    class EchoClient(ScriptedClient):
        def chat(self, user, system=None, max_tokens=None, temperature=None,
                 retries=None):
            base = super().chat(user, system, max_tokens, temperature, retries)
            return ChatResult(text=base.text + "\n【中控复核意见】把这句改成…\n⟦",
                              finish="stop")

    cfg2 = Config(lang="ja", chunk_chars=90, chunk_hard_max=140, translate_terms=False,
                  retries=1, strict=False)
    out2 = tmp / "ECHO_OUT"
    rep2 = Translator(cfg2, Logger(None, echo=False), out2,
                      clients=[EchoClient()]).process(srcf)
    txt = (out2 / "echo" / "echo_仅翻译.txt").read_text(encoding="utf-8")
    ck("中控复核意见" not in txt, "回吐防护: 成品里没有指令残句", txt[-40:])
    ck("⟦" not in txt and "⟧" not in txt, "回吐防护: 成品里没有半个占位符")
    ck(rep2.failed == 0, "回吐防护: 清理后不算失败块", str(rep2.status))


def t_dryrun(tmp: Path) -> None:
    src = tmp / "d.txt"
    write_text_atomic(src, "「おはよう」\n今日は晴れです。\n")
    cfg = Config(do_clean=True, chunk_chars=80, lang="ja")
    log = Logger(None, echo=False, level="error")
    rep = Translator(cfg, log, tmp / "DRY", clients=[], dry_run=True).process(src)
    ck(rep.status == "ok", "dry-run 可跑通", rep.status)
    ck((tmp / "DRY" / "d" / "d_仅翻译.txt").exists(), "dry-run 产出文件")


def t_prompt_echo(tmp: Path) -> None:
    """整段提示词被抄进译文：确定性剥掉，且不误删正文。"""
    from gtlib.job import strip_prompt_echo, default_prompt
    NL = chr(10)
    pr = default_prompt("hash")
    body_txt = "【#3#】「早上好」" + NL + "他这么说着，走出了房间。"
    dirty = pr + NL + NL + body_txt
    clean = strip_prompt_echo(dirty, pr)
    ck(clean.strip() == body_txt.strip(), "复读剥离: 开头整段提示词被删", repr(clean[:40]))
    ck("硬性要求" not in clean, "复读剥离: 提示词特征消失")
    dirty2 = body_txt + NL + NL + pr
    ck(strip_prompt_echo(dirty2, pr).strip() == body_txt.strip(), "复读剥离: 结尾复读也能删")
    safe = "他说：「只输出译文本身。」然后笑了。" + NL + "她点点头。"
    ck(strip_prompt_echo(safe, pr).strip() == safe.strip(), "复读剥离: 正文相似短句不误删")


def t_patches(tmp: Path) -> None:
    """行级补丁套用：定位、行数守恒、越界/对不上就丢弃。"""
    dst = "行一" + chr(10) + "行二" + chr(10) + "行三"
    patches = [
        {"line": 2, "old": "行二", "new": "第二行改好"},
        {"line": 9, "old": "根本不存在", "new": "x"},
        {"line": 1, "old": "错误的old", "new": "不该生效"},
        {"line": 3, "old": "行三", "new": "多行" + chr(10) + "内容"},
    ]
    new, applied, dropped = Translator._apply_patches(dst, patches)
    ck(applied == 2, "补丁: 生效行数", f"applied={applied}")
    ck(dropped == 2, "补丁: 越界/对不上被丢弃", f"dropped={dropped}")
    ck(new.count(chr(10)) == 2, "补丁: 行数守恒", repr(new))
    ck("行二" not in new and "第二行改好" in new, "补丁: 目标行被替换", repr(new))
    ck("多行 内容" in new, "补丁: new 里的换行压平", repr(new))
    new2, applied2, _ = Translator._apply_patches(dst, [{"line": 1, "old": "行三", "new": "三号"}])
    ck(applied2 == 1 and new2.split(chr(10))[2] == "三号",
       "补丁: old 唯一定位可纠行号", repr(new2))


def t_patch_ops(tmp: Path) -> None:
    """补丁的 delete / insert 动作：删行、插行、行号漂移。"""
    NL = chr(10)
    dst = "第一行" + NL + "第二行" + NL + "第三行"
    out, applied, dropped = Translator._apply_patches(dst, [{"line": 2, "op": "delete"}])
    ck(applied == 1 and len(_lines_local(out)) == 2 and "第二行" not in out,
       "补丁ops: delete 删掉一行", repr(out))
    out, applied, _ = Translator._apply_patches(dst, [{"line": 1, "op": "delete", "count": 2}])
    ck(applied == 1 and len(_lines_local(out)) == 1 and out.strip() == "第三行",
       "补丁ops: delete count=2 删两行", repr(out))
    out, applied, _ = Translator._apply_patches(dst, [{"line": 1, "op": "insert", "new": "插入行"}])
    L = _lines_local(out)
    ck(applied == 1 and len(L) == 4 and L[1] == "插入行", "补丁ops: insert 插在指定行后", repr(out))
    # 同时删+改：逆序应用，行号不漂移
    out, applied, _ = Translator._apply_patches(
        dst, [{"line": 1, "op": "delete"}, {"line": 3, "op": "replace", "new": "第三行改好"}])
    L = _lines_local(out)
    ck(applied == 2 and "第三行改好" in L and "第一行" not in L,
       "补丁ops: 删除+替换混合不漂移", repr(out))
    # new 里的换行必须压平，不能凭空多行
    out, _, _ = Translator._apply_patches(dst, [{"line": 2, "op": "replace",
                                                 "new": "多行" + NL + "内容"}])
    ck(len(_lines_local(out)) == 3 and "多行 内容" in out, "补丁ops: new 换行压平", repr(out))


def _lines_local(t: str):
    return t.replace(chr(13), "").split(chr(10))


def t_api_mode(tmp: Path) -> None:
    """用本地假 OpenAI 端点验证 API 模式：鉴权头、SSE 解析、429 退避、远程 payload、全流程。"""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    state: dict = {"auth": [], "payloads": [], "n429": 0, "chat": 0}

    def mock_answer(user: str) -> str:
        user = user.split("\n\n【上一次译文的问题】")[0]
        if "现在翻译（只输出对应行）" in user:
            return "\n".join(m.group(0) for m in re.finditer(r"^\d+\. .+$", user, re.M))
        if user.startswith("把下面的日文专名"):
            lines = [l.strip() for l in user.splitlines() if l.strip()]
            return lines[-1] if lines else ""
        return fake_zh(user)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):        # 静音
            return

        def do_GET(self):
            auth = self.headers.get("Authorization")
            state["auth"].append(auth)
            if self.path.endswith("/models") and auth == "Bearer sk-test":
                body = json.dumps({"data": [{"id": "mock-model"}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            code = 401 if self.path.endswith("/models") else 404
            self.send_response(code)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(n) if n else b"{}"
            try:
                payload = json.loads(raw.decode("utf-8"))
            except Exception:
                payload = {}
            state["payloads"].append(payload)
            state["auth"].append(self.headers.get("Authorization"))
            if state["n429"] < 1:      # 先限流一次，验证退避重试
                state["n429"] += 1
                self.send_response(429)
                self.send_header("Retry-After", "0")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            state["chat"] += 1
            user = ""
            for m in reversed(payload.get("messages") or []):
                if m.get("role") == "user":
                    user = m.get("content") or ""
                    break
            text = mock_answer(user)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            for i in range(0, len(text), 24):
                chunk = json.dumps({"choices": [{"delta": {"content": text[i:i + 24]},
                                                 "finish_reason": None}]}, ensure_ascii=False)
                self.wfile.write(f"data: {chunk}\n\n".encode("utf-8"))
            tail = json.dumps({"choices": [{"delta": {}, "finish_reason": "stop"}],
                               "usage": {"prompt_tokens": 10, "completion_tokens": 10}})
            self.wfile.write(f"data: {tail}\n\ndata: [DONE]\n\n".encode("utf-8"))
            self.wfile.flush()
            self.close_connection = True

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    log = Logger(tmp / "api.log", echo=False, level="warn")
    try:
        from .llama import LlamaClient, LlamaServer, list_models, normalize_api_base

        ck(normalize_api_base("https://api.deepseek.com") == "https://api.deepseek.com/v1",
           "API: 自动补 /v1", normalize_api_base("https://api.deepseek.com"))
        ck(normalize_api_base("https://api.deepseek.com/v1/") == "https://api.deepseek.com/v1",
           "API: 去尾斜杠")
        ck(normalize_api_base("http://127.0.0.1:9080") == "http://127.0.0.1:9080",
           "API: 本地地址不被改")

        server = LlamaServer.remote(f"http://127.0.0.1:{port}/v1", log, api_key="sk-test")
        ck(list_models(server) == ["mock-model"], "API: /models 能拿到模型名")
        ck(server.health(5.0) is True, "API: 远程探活不走 /health")

        cfg = Config(model_name="mock-model", retries=3, translate_terms=False,
                     chunk_chars=90, chunk_hard_max=140, lang="ja")
        client = LlamaClient(server, log, cfg)
        res = client.chat("こんにちは\n今日は晴れです。")
        ck("字" in res.text, "API: 流式 SSE 解析成功", res.text[:40])
        ck(state["auth"][-1] == "Bearer sk-test", "API: 请求带 Authorization",
           str(state["auth"][-1]))
        ck(state["n429"] == 1 and state["chat"] >= 1,
           "API: 429 限流后自动退避重试", f"429={state['n429']} ok={state['chat']}")
        last = state["payloads"][-1] or {}
        ck(not ({"repeat_penalty", "cache_prompt"} & set(last)),
           "API: 远程模式不发 llama.cpp 专属字段", str(sorted(last)))
        ck(last.get("model") == "mock-model", "API: payload 带上模型名")

        src = tmp / "api_src.txt"
        write_text_atomic(src, "第一章\n「おはよう」と彼女は言った。\n窓の外は雨だった。\n\n「行こうか」\n")
        out = tmp / "API_OUT"
        rep = Translator(cfg, log, out, clients=[client]).process(src)
        ck(rep.status == "ok", "API: 完整流水线跑通", str(rep.to_dict())[:160])
        f = out / "api_src" / "api_src_仅翻译.txt"
        ck(f.exists(), "API: 产出成品")
        if f.exists():
            txt = f.read_text(encoding="utf-8")
            ck("⟦" not in txt, "API: 成品无占位符残渣", txt[:60])
    finally:
        httpd.shutdown()
        httpd.server_close()


def t_director(tmp: Path) -> None:
    """验证中控（大模型）模式：工具调用定名 / 复核 / 端点不可达时的降级。"""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from .director import Director, DirectorError, make_director
    from .llama import LlamaServer

    state = {"glossary_calls": 0, "review_calls": 0, "proof_calls": 0,
             "proof_mode": "fix"}

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            return

        def do_GET(self):
            body = json.dumps({"data": [{"id": "mock-director"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            payload = json.loads(self.rfile.read(n) or b"{}")
            user = ""
            for m in reversed(payload.get("messages") or []):
                if m.get("role") == "user":
                    user = m.get("content") or ""
                    break
            tools = payload.get("tools") or []
            name = ((tools[0] or {}).get("function") or {}).get("name") if tools else ""
            if name == "submit_glossary":
                state["glossary_calls"] += 1
                entries = []
                for ln in user.splitlines():
                    ln = ln.strip()
                    if not ln.startswith("- "):
                        continue
                    term = re.split(r"[（(]", ln[2:], 1)[0].strip()
                    if not term:
                        continue
                    if term == "雨":
                        entries.append({"term": term, "zh": "", "type": "其他",
                                        "keep": False})
                    else:
                        entries.append({"term": term, "zh": "中" + term[:2],
                                        "type": "人名", "keep": True})
                msg = {"role": "assistant", "content": None, "tool_calls": [
                    {"id": "c1", "type": "function",
                     "function": {"name": "submit_glossary",
                                  "arguments": json.dumps({"entries": entries},
                                                          ensure_ascii=False)}}]}
            elif name == "submit_review":
                state["review_calls"] += 1
                args = {"verdict": "ok", "reason": "mock 放行", "notes": []}
                msg = {"role": "assistant", "content": None, "tool_calls": [
                    {"id": "c2", "type": "function",
                     "function": {"name": "submit_review",
                                  "arguments": json.dumps(args, ensure_ascii=False)}}]}
            elif name == "submit_proofread":
                state["proof_calls"] += 1
                if state["proof_mode"] == "fix":
                    src = ""
                    if "【日文原文】" in user:
                        part = user.split("【日文原文】", 1)[1]
                        src = part.split("【机器译文】", 1)[0].strip()
                    args = {"text": fake_zh(src), "changed": True,
                            "notes": ["mock 校对修正了占位符"]}
                else:
                    args = {"text": "嗯", "changed": True}
                msg = {"role": "assistant", "content": None, "tool_calls": [
                    {"id": "c3", "type": "function",
                     "function": {"name": "submit_proofread",
                                  "arguments": json.dumps(args, ensure_ascii=False)}}]}
            else:
                msg = {"role": "assistant", "content": "{\"verdict\": \"ok\"}"}
            body = json.dumps({"choices": [{"index": 0, "message": msg,
                                            "finish_reason": "stop"}]}, ensure_ascii=False).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    log = Logger(tmp / "dir.log", echo=False, level="warn")
    try:
        cfg = Config(director_base=f"http://127.0.0.1:{port}/v1",
                     director_model="", lang="ja")
        d = Director(cfg, log)
        ck(d.probe(), "中控: 探活 + 自动取模型名", d.model)
        ck(d.model == "mock-director", "中控: 模型名来自 /models", d.model)

        cands = [TermMap.from_dict({}).entries, ]      # 占位，保持导入被使用
        entries = [TermEntry(idx=1, term="友香", type="人名", count=3),
                   TermEntry(idx=2, term="アンジェリカ", type="人名", count=2),
                   TermEntry(idx=3, term="雨", type="术语", count=9)]
        got = d.decide_terms(entries, {"友香": "「おはよう、友香さん」"})
        ck(state["glossary_calls"] == 1, "中控: 术语定名走了一次工具调用")
        ck(got.get(1, {}).get("zh"), "中控: 产出中文定名", str(got.get(1)))
        ck(got.get(3, {}).get("keep") is False, "中控: 能判定非专名(keep=false)")

        rev = d.review("「おはよう」", "「早上好」", ["length_ratio:warn(0.42)"])
        ck(rev and rev["verdict"] == "ok", "中控: 复核返回结构化结论", str(rev))
        ck(state["review_calls"] == 1, "中控: 复核也走工具调用")

        pr = d.proofread("「おはよう」と彼は言った。", "「早上好」他说。", ["kana_residue"])
        ck(pr and "字" in pr["text"], "中控: 校对直接交出改稿", str(pr)[:80])
        ck(state["proof_calls"] == 1, "中控: 校走 submit_proofread 工具")

        # --- 接到流水线上：术语定名真的用了中控给的名字
        src = tmp / "dir_src.txt"
        write_text_atomic(src, "「おはよう、友香さん」と彼は言った。\n"
                               "「おはよう、アンジェリカ」と友香さんは答えた。\n" * 3)
        cfg2 = Config(director_base=f"http://127.0.0.1:{port}/v1",
                      director_model="mock-director", lang="ja",
                      chunk_chars=90, chunk_hard_max=140, translate_terms=True,
                      retries=2, strict=True)
        log2 = Logger(tmp / "dir2.log", echo=False, level="warn")
        judge = make_director(cfg2, log2)
        ck(judge is not None, "中控: make_director 能用")
        out = tmp / "DIR_OUT"
        rep = Translator(cfg2, log2, out, clients=[ScriptedClient()],
                         director=judge).process(src)
        ck(rep.status == "ok", "中控+译者: 流水线跑通", rep.status)
        tm = read_json(out / "dir_src" / "terms.json", {}) or {}
        targets = [e.get("target", "") for e in (tm.get("terms") or [])]
        ck(any(t.startswith("中") for t in targets),
           "中控+译者: terms.json 用了中控定名", str(targets)[:120])
        ck((out / "dir_src" / "terms_director.json").exists(),
           "中控: 定名决定已缓存（续跑不漂移）")
        txt = (out / "dir_src" / "dir_src_仅翻译.txt").read_text(encoding="utf-8")
        ck("中友" in txt, "中控+译者: 成品里出现中控定的译名", txt[:60])

        # --- 校对（改稿）：失败块被 27B 改好了，要采纳
        out2 = tmp / "DIR_PROOF"
        cfg3 = Config(director_base=f"http://127.0.0.1:{port}/v1",
                      director_model="mock-director", lang="ja",
                      chunk_chars=90, chunk_hard_max=140, translate_terms=True,
                      retries=1, strict=True, director_proofread=True,
                      director_review=False)
        rep2 = Translator(cfg3, log2, out2, clients=[ScriptedClient(bad_chunk=1)],
                          director=make_director(cfg3, log2)).process(src)
        ck(rep2.status == "ok", "中控校对: 失败块被改稿后整体 ok",
           f"{rep2.status} ok={rep2.ok} deg={rep2.degraded} fail={rep2.failed}")
        ck(rep2.failed == 0, "中控校对: 没有残留失败块", str(rep2.failed))
        proved = [json.loads(p.read_text(encoding="utf-8"))
                  for p in sorted((out2 / "dir_src" / "trans").glob("*.json"))]
        ck(any((r.get("detail") or "").startswith("中控校对") for r in proved),
           "中控校对: 采纳的块标了来源",
           str([r.get("detail") for r in proved]))
        ck(state["proof_calls"] >= 1, "中控校对: 真的调用了校对工具",
           str(state["proof_calls"]))

        # --- 校对（改稿）：改得更差 → 必须丢掉，不能混进成品，且改稿次数有上限
        state["proof_mode"] = "junk"
        calls_before = state["proof_calls"]
        out3 = tmp / "DIR_PROOF_BAD"
        cfg4 = Config(director_base=f"http://127.0.0.1:{port}/v1",
                      director_model="mock-director", lang="ja",
                      chunk_chars=90, chunk_hard_max=140, translate_terms=False,
                      retries=1, strict=False, director_proofread=True,
                      director_review=False)
        rep3 = Translator(cfg4, log2, out3, clients=[ScriptedClient(bad_chunk=1)],
                          director=make_director(cfg4, log2)).process(src)
        ck(rep3.failed >= 1, "中控校对: 改差的稿子被拒，仍计失败块", str(rep3.failed))
        junk = [json.loads(p.read_text(encoding="utf-8"))
                for p in sorted((out3 / "dir_src" / "trans").glob("*.json"))
                if json.loads(p.read_text(encoding="utf-8")).get("status") == "failed"]
        ck(all("嗯" not in (r.get("text") or "") for r in junk),
           "中控校对: 差稿没有写进块结果")
        ck(state["proof_calls"] - calls_before <= cfg4.director_proofread_tries,
           "中控校对: 单块改稿次数有上限（不烧 token）",
           f"{state['proof_calls'] - calls_before} 次 / 上限 {cfg4.director_proofread_tries}")
        state["proof_mode"] = "fix"

        # --- 端点不可达 → 自动退化，不抛到上层
        dead = Config(director_base="http://127.0.0.1:9/v1", lang="ja")
        ck(make_director(dead, log2) is None, "中控: 端点不可达时自动退化")
        ddead = Director(dead, log2)
        ck(ddead.decide_terms(entries, {}) == {}, "中控: 不可达时定名返回空表（不炸）")
        ck(ddead.review("あ", "啊", ["x"]) is None, "中控: 不可达时复核返回 None")
    finally:
        httpd.shutdown()
        httpd.server_close()


def t_unit(tmp: Path) -> None:
    """单元模式：≤N token 的完整句 + 打包翻译 + 27B 三档定性 + 装回。"""
    from . import unit as um

    # --- 1. 断句：括号内不能切开，收尾引号/标点要跟着走
    a = um.split_sentences("これは、礼子（れいこ）さんだ。")
    ck(a == ["これは、礼子（れいこ）さんだ。"], "断句：括号内不切", str(a))
    b = um.split_sentences("「……あぁ」と言った。")
    ck(b == ["「……あぁ」", "と言った。"], "断句：引号整句带走", str(b))
    c = um.split_sentences("そうか！？そうか。")
    ck(c == ["そうか！？", "そうか。"], "断句：连写标点算一句", str(c))

    # --- 2. 单元：不超过上限、单句超限就独占一条、绝不切句
    text = ("「おはよう、友香さん」と彼は言った。\n"
            "窓の外は雨だった。\n\n"
            "これは、礼子（れいこ）さんだ。友達の貴史のお母さん。\n"
            "？\n"
            "---\n"
            "ああああああああああああああああああああああああああああああああああああああああああ。\n")
    units, lines = um.build_units(text, max_tokens=20, chars_per_token=1.6)
    ck(all(u.tokens <= max(20, 1) or "。" in u.src for u in units),
       "单元：不超上限（超限的只能是一整句）",
       str([(u.id, u.tokens) for u in units]))
    long_u = [u for u in units if u.tokens > 20]
    ck(len(long_u) == 1 and long_u[0].src.count("。") == 1,
       "单元：超长句整句独占一条，不从中间切", str([u.src[:20] for u in long_u]))
    ck(any(u.kind == "punct" for u in units), "单元：纯标点判为 punct（不送模型）")
    ck(any(u.kind == "scene" for u in units), "单元：--- 场景线单独成条")
    ck(all(len(um.split_sentences(u.src)) >= 1 for u in units), "单元：单元内部可回收")

    # --- 3. 装回：行数/空行/原文逐字可逆
    back = um.merge_back(lines, units, {u.id: u.src for u in units})
    ck(back == text, "装回：单元按行号+行内序号拼接 == 原文", repr(back[:60]))

    # --- 4. 打包与解析
    texty = [u for u in units if u.translatable]
    packs = um.pack_units(texty, size=2, max_chars=700)
    ck(all(len(p) <= 2 for p in packs), "打包：不超条数", str([len(p) for p in packs]))
    ck([u.id for p in packs for u in p] == [u.id for u in texty],
       "打包：顺序与覆盖面不变")
    ck(um.parse_batch_reply("1| 甲\n2| 乙", 2) == {1: "甲", 2: "乙"}, "解析：编号|译文")
    ck(um.parse_batch_reply("1. 甲\n2、乙", 2) == {1: "甲", 2: "乙"}, "解析：容忍别种分隔符")
    ck(um.parse_batch_reply("甲\n乙", 2) == {1: "甲", 2: "乙"}, "解析：没编号但行数对齐")
    ck(um.parse_batch_reply("1| 甲", 2) == {1: "甲"}, "解析：缺条只回已有的")

    # --- 5. 单元质检
    st, _p = um.check_unit("⟦1⟧「あ」", "「啊」")
    ck(st == "failed", "质检：缺占位符 → failed", st)
    st, _p = um.check_unit("「そうか」", "「そうか」")
    ck(st == "degraded", "质检：假名残留 → degraded", st)
    ck(um.fix_passable("⟦1⟧「あ」", "⟦1⟧「啊」") and
       not um.fix_passable("⟦1⟧「あ」", "「啊」"), "质检：中控修正稿的门禁")

    # --- 6. 整条流水线（假客户端 + 假中控）
    class UnitClient:
        """把「N| 原文」批次回成「N| 伪译文」，并记下调用。"""

        def __init__(self):
            self.calls: List[str] = []
            self.lock = threading.Lock()

        def chat(self, user, system=None, max_tokens=None, temperature=None, retries=None):
            with self.lock:
                self.calls.append(user)
            out = []
            for line in user.split("\n"):
                m = re.match(r"^(\d+)\|\s?(.*)$", line.strip())
                if m:
                    out.append(f"{m.group(1)}| {fake_zh(m.group(2))}")
            return ChatResult(text="\n".join(out), finish="stop")

    class UnitDirector:
        """按内容定档：含「おはよう」→ order+可用的修正稿；「窓」首轮 wrong、次轮 ok。"""

        def __init__(self):
            self.calls = 0
            self.seen: List[str] = []

        def available(self):
            return True

        def check_units(self, pairs, hint="", max_tokens=1400):
            self.calls += 1
            self.seen += [s for _i, s, _d in pairs]
            out = {}
            for i, s, _d in pairs:
                if "おはよう" in s:
                    out[i] = {"v": "order", "fix": "调整后的" + fake_zh(s), "why": "语序"}
                elif "窓" in s and self.calls <= 2:
                    out[i] = {"v": "wrong", "fix": "", "why": "漏译"}
                else:
                    out[i] = {"v": "ok", "fix": "", "why": ""}
            return out

    src = tmp / "unit_sample.txt"
    write_text_atomic(src, ("「おはよう、友香さん」と彼は言った。\n"
                            "窓の外は雨だった。\n\n"
                            "これは、礼子さんだ。\n") * 2)
    cfg = Config(do_clean=True, lang="ja", strict=True, retries=2,
                 mode="unit", unit_tokens=20, unit_pack=4, unit_context=40,
                 unit_check=True, unit_check_pack=8, unit_check_rounds=2,
                 final_check="all", translate_terms=True, mask_terms=True,
                 director_terms=False, director_veto=False)
    log = Logger(tmp / "unit.log", echo=False, level="warn")
    cli = UnitClient()
    direc = UnitDirector()
    out = tmp / "OUTU"
    rep = Translator(cfg, log, out, clients=[cli], director=direc).process(src)
    base = out / "unit_sample"
    ck(rep.status in ("ok", "partial"), "单元模式：跑通", f"{rep.status} {rep.notes}")
    ck((base / "unit_sample_仅翻译.txt").exists(), "单元模式：产出成品")
    ck((base / "units.jsonl").exists(), "单元模式：落盘 units.jsonl")
    if (base / "unit_sample_仅翻译.txt").exists():
        got = (base / "unit_sample_仅翻译.txt").read_text(encoding="utf-8")
        want_lines = (base / "cleaned.txt").read_text(encoding="utf-8").split("\n")
        ck(len(got.split("\n")) == len(want_lines), "单元模式：行数守恒",
           f"{len(got.split(chr(10)))} vs {len(want_lines)}")
        ck("⟦" not in got, "单元模式：成品无占位符残渣", got[:60])
        ck("调整后的" in got, "单元模式：order 判定用中控修正稿替换", got[:80])
    ck(direc.calls >= 2, "单元模式：中控检查跑了多轮（初检+复检/终检）", str(direc.calls))
    n_units = len([l for l in (base / "units.jsonl").read_text(encoding="utf-8").split("\n")
                   if l.strip()]) if (base / "units.jsonl").exists() else 0
    ck(n_units > 0, "单元模式：单元表非空", str(n_units))
    # --- 续传：第二次不应再发翻译请求
    before = len(cli.calls)
    Translator(cfg, log, out, clients=[cli], director=direc).process(src)
    ck(len(cli.calls) == before, "单元模式：续传不重复翻译",
       f"{len(cli.calls) - before} 次")


# ------------------------------------------------------------------ 入口
def run_selftest(verbose: bool = True) -> bool:
    global _FAILS, _PASS
    _FAILS, _PASS = [], 0
    tmp = Path(tempfile.mkdtemp(prefix="gtlib_selftest_"))
    try:
        t_encoding(tmp)
        t_clean()
        t_segment()
        t_placeholder()
        t_terms()
        t_validate()
        t_dryrun(tmp)
        t_pipeline(tmp)
        t_echo_guard(tmp)
        t_patches(tmp)
        t_patch_ops(tmp)
        t_prompt_echo(tmp)
        t_api_mode(tmp)
        t_director(tmp)
        t_unit(tmp)
    except Exception:
        import traceback
        _FAILS.append("异常: " + traceback.format_exc())
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    if verbose:
        print(f"\n通过 {_PASS} 项，失败 {len(_FAILS)} 项")
        for f in _FAILS:
            print("  ✗ " + f)
    return not _FAILS
