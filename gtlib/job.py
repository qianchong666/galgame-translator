# -*- coding: utf-8 -*-
"""作业编排：准备 → 术语翻译 → 逐块翻译（可续跑） → 组装 → 报告。

旧版 step3 的核心缺陷在这里被逐条修掉
--------------------------------------
1. **清单永远 pending**：GalTrans 模式产出 `_对照/_仅翻译/_原文.txt`，但清单/跳过/合并只看
   `_processed.txt`，于是"跳过已处理"永远失效、进度永远不对。现在**状态由文件推导**：
   每块落一个 `trans/000123.json`，有它就是完成，无需清单位。
2. **无块级续传**：中断就从头来。现在块级 resume，且校验不通过的块会被重译。
3. **失败也出成品**：旧版缺块时装 `【未翻译 #id】` 照样拼成成品文件，下游当成正常译文。
   现在三态：strict 下有任何 failed 块 → 该文件不出成品，只写 `_FAILED.json`。
4. **一块一请求、没有语义重试**：现在 failed 块按 `正常 → 加温 → 拆半` 三档重试，
   拆半后两半都通过才采用，避免"整块糊成一坨"。
5. **截断当成功**：`finish_reason=length` 现在会放大 max_tokens 重取，最多两次。
6. **日志在 worker 里直接 root.update()**：现在只通过 Logger 回调，GUI 自己 queue 消费。
"""
from __future__ import annotations

import json
import queue
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence

from . import clean as cleanmod
from . import segment as segmod
from . import terms as termmod
from .core import (Config, JsonlWriter, Logger, human_dur, now_iso, read_json,
                   read_jsonl, read_text_auto, safe_stem, sha1_text, stamp,
                   write_json_atomic, write_text_atomic)
from .llama import ChatResult, LlamaClient, LlamaError
from .placeholder import (STYLES, TermEntry, TermMap, audit_masks, check_placeholders,
                          indices_in, mask_code, mask_text, restore_text)
from .validate import Validation, strip_context_echo, validate_chunk

#: 默认系统提示词（占位符说明按 style 动态生成）
PROMPT_TMPL = (
    "你是一名专业的日文小说/美少女游戏文本翻译。把用户给出的日文原文翻译成简体中文。\n"
    "硬性要求：\n"
    "1. 只输出译文本身，不要解释、不要加“译文：”之类的前缀、不要保留日文原文。\n"
    "2. 严格保持原文的行数与断行位置，不要合并行、不要拆分行、不要增删空行。\n"
    "3. 保留「」『』等引号与标点结构，对白要口语自然，叙述要符合中文书面语。\n"
    "4. 形如 {ph} 的标记是系统占位符，代表已定名的人名/地名/术语，必须原样保留，"
    "不得翻译、改写、增删或移动位置。\n"
    "5. 不要输出任何说明性文字。"
)

CONTEXT_HEADER = "【上文参考·不要翻译】"
BODY_HEADER = "【需要翻译的正文】"

#: 我们自己往 prompt 里塞的标记：一旦出现在“译文”里，说明模型回吐了指令
ECHO_MARKS = ("【中控复核意见】", "【上一次译文的问题】", "【上文参考", "【故事上文",
              BODY_HEADER)

#: 模型回吐时会把标记改写（漏字/换词），比如「【上一次译文的问题】」→「【上次译文的问题】」，
#: 字面匹配会漏，所以再加一条正则兜底（实测漏过一次，直接混进成品）。
ECHO_RE = re.compile(r"【[^】\n]{0,16}(?:译文的问题|中控复核意见|上文参考|故事上文|"
                     r"需要翻译的正文)[^】\n]{0,16}】")


def strip_prompt_echo(text: str, prompt: str) -> str:
    """模型把提示词抄进译文时，按行剥掉。

    实测：模型复读时常常漏掉首句、把两条规则并成一条，所以不能"从第一行对齐"，
    而是在提示词里滑动找**最长连续命中段**（>=3 行，字符前缀比对），首尾各剥一次。
    要求连续 >=3 行，避免正文里碰巧出现同样的短句被误删。
    """
    if not text or not prompt:
        return text
    plines = [ln.strip() for ln in prompt.replace(chr(13), "").split(chr(10)) if ln.strip()]
    if len(plines) < 3:
        return text
    lines = text.replace(chr(13), "").split(chr(10))

    def _same(a: str, b: str) -> bool:
        a, b = a.strip(), b.strip()
        if not a or not b:
            return False
        if a == b:
            return True
        k = min(len(a), len(b), 10)
        return k >= 6 and a[:k] == b[:k]

    def _best_at_head(seq, plines):
        """返回 (起始提示词行号, 连续命中行数)，取最长。"""
        best = (0, 0)
        for k in range(len(plines)):
            n = 0
            for j in range(k, len(plines)):
                if n < len(seq) and _same(seq[n], plines[j]):
                    n += 1
                else:
                    break
            if n > best[1]:
                best = (k, n)
        return best

    k, n = _best_at_head(lines, plines)
    if n >= 3:
        lines = lines[n:]
    rk, rn = _best_at_head(list(reversed(lines)), list(reversed(plines)))
    if rn >= 3:
        lines = lines[:len(lines) - rn]
    return chr(10).join(lines).strip()


def strip_echo_marks(t: str) -> str:
    """清掉回吐的指令标题（含被模型改写过的）：
    出现在后半段→从标记处截断；出现在开头/中间→只删那一行，不误删正文。
    """
    for _ in range(6):
        idx = -1
        for mark in ECHO_MARKS:
            i = t.find(mark)
            if i >= 0 and (idx < 0 or i < idx):
                idx = i
        m = ECHO_RE.search(t)
        if m is not None and (idx < 0 or m.start() < idx):
            idx = m.start()
        if idx < 0:
            return t
        if idx >= len(t) * 0.5:          # 追加在译文后面 → 从标记处截断
            return t[:idx].rstrip()
        ls = t.rfind("\n", 0, idx) + 1   # 开头/中间 → 丢掉整行
        le = t.find("\n", idx)
        t = (t[:ls] + (t[le:] if le >= 0 else "")).strip("\n")
    return t


def default_prompt(style: str = "bracket") -> str:
    from .placeholder import mask_code
    return PROMPT_TMPL.format(ph=mask_code(style, 1))


def build_user_prompt(src: str, context: str = "") -> str:
    """上下文放到 system 里（见 _translate_one），这里只作备用。"""
    if not context:
        return src
    return f"{BODY_HEADER}\n{src}"


def build_system_prompt(base: str, context: str = "") -> str:
    if not context:
        return base
    return (f"{base}\n\n【故事上文·仅供理解用，不要翻译、不要复述】\n{context}")


# ------------------------------------------------------------------ 报告
@dataclass
class FileReport:
    source: str = ""
    out_dir: str = ""
    status: str = "pending"          # ok / partial / failed
    chunks: int = 0
    ok: int = 0
    degraded: int = 0
    failed: int = 0
    chars: int = 0
    duration: float = 0.0
    failed_ids: List[int] = field(default_factory=list)
    bad_checks: Dict[str, int] = field(default_factory=dict)
    outputs: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"source": self.source, "out_dir": self.out_dir, "status": self.status,
                "chunks": self.chunks, "ok": self.ok, "degraded": self.degraded,
                "failed": self.failed, "chars": self.chars,
                "duration": round(self.duration, 2),
                "failed_ids": self.failed_ids[:200],
                "bad_checks": self.bad_checks, "outputs": self.outputs,
                "notes": self.notes}


# ------------------------------------------------------------------ 主体
_RANK = {"failed": 0, "degraded": 1, "ok": 2}


class _Cand:
    """给中控看的候选（Term 与 TermEntry 的统一视图）。"""
    __slots__ = ("idx", "term", "type", "count")

    def __init__(self, idx: int, term: str, type_: str = "专名", count: int = 1):
        self.idx = idx
        self.term = term
        self.type = type_
        self.count = count


class Translator:
    def __init__(self,
                 cfg: Config,
                 log: Logger,
                 out_root: Path,
                 clients: Optional[Sequence[LlamaClient]] = None,
                 cancel: Optional[threading.Event] = None,
                 progress: Optional[Callable[[int, int], None]] = None,
                 dry_run: bool = False,
                 glossary: Optional[Path] = None,
                 director=None):
        self.cfg = cfg
        self.log = log
        self.out_root = Path(out_root)
        self.clients = list(clients or [])
        self.cancel = cancel or threading.Event()
        self.mem_brief: dict = {}       # 长文记忆：文档简报
        self.mem_rows: dict = {}        # 长文记忆：按块号的台账
        self.term_judge = director          # 中控：专名定名 + 块复核（可缺）
        self.progress = progress
        self.dry_run = dry_run
        self.glossary = Path(glossary) if glossary else None
        self.style = cfg.placeholder_style
        self.prompt = cfg.prompt or default_prompt(self.style)
        self._done = 0
        self._total = 0
        self._plock = threading.Lock()

    # ---------------------------------------------------------------- 路径
    def base_dir(self, src: Path) -> Path:
        # 扁平模式：out 目录本身就是这本书的输出目录（out/时间_书名/）
        if getattr(self.cfg, "out_flat", False):
            return self.out_root
        return self.out_root / safe_stem(src.name)

    # ---------------------------------------------------------------- 入口
    def process(self, src: Path) -> FileReport:
        t0 = time.time()
        src = Path(src)
        rep = FileReport(source=str(src))
        base = self.base_dir(src)
        rep.out_dir = str(base)
        base.mkdir(parents=True, exist_ok=True)

        meta = self.prepare(src, base)
        if meta.get("_failed"):
            rep.status = "failed"
            rep.notes.append(meta.get("_failed"))
            return rep

        tm = TermMap.from_dict(read_json(base / "terms.json", {}) or {})
        chunks = read_jsonl(base / "chunks.jsonl")
        rep.chunks = len(chunks)
        rep.chars = int(meta.get("chars", 0))

        self.log.info(f"📖 {src.name}  →  {base.name}  "
                      f"({len(chunks)} 块 / {rep.chars} 字 / 术语 {len(tm)})")

        # --- 术语译文
        if self.cfg.translate_terms and len(tm) and not self.dry_run:
            tm = self.translate_terms(base, tm)

        # --- 单元模式：完整句拆条 → 打包翻译 → 27B 定性/重译 → 装回 → 全文终检
        if self.cfg.mode == "unit":
            from .unitjob import run_unit_pipeline
            rep = run_unit_pipeline(self, src, base, tm, rep)
            rep.duration = time.time() - t0
            write_json_atomic(base / "_report.json", rep.to_dict())
            return rep

        # --- 长文记忆（实验）：翻译前按原文顺序预扫，产出文档简报 + 分场台账
        self.mem_brief, self.mem_rows = {}, {}
        if getattr(self.cfg, "doc_memory", False) and not self.dry_run:
            from . import memory as memmod
            if (base / memmod.MEMORY_FILE).exists():
                self.mem_brief, self.mem_rows = memmod.load_memory(base)
                self.log.info(f"  ↺ 复用长文记忆（{len(self.mem_rows)} 条台账）")
            else:
                t_mem = time.time()
                texts = [self._chunk_text(base, int(c["id"])) for c in chunks]
                terms_all = (read_json(base / "terms.json", {}) or {}).get("terms", [])
                brief = memmod.build_brief(self.cfg, self.log, texts, terms_all)
                nmap = {str(e.get("idx")): (e.get("target") or e.get("term"))
                        for e in terms_all if e.get("idx") is not None}
                # 出场人物：按遮蔽标记确定性统计，不问模型
                presence = memmod.presence_from_masks(texts, self.style,
                                                      {k: v for k, v in nmap.items() if v})
                presence = memmod.merge_narrator(presence, brief)
                hit = sum(1 for x in presence if x)
                self.log.info(f"  出场人物：{hit}/{len(texts)} 块按遮蔽标记确定"
                              f"（零模型调用）")
                rows = memmod.build_ledgers(self.cfg, self.log, texts, brief,
                                            scene=int(getattr(self.cfg, "memory_scene", 8)),
                                            names=nmap, presence=presence)
                if brief or rows:
                    memmod.save_memory(base, brief, rows)
                    self.mem_brief = brief
                    self.mem_rows = {int(r["cid"]): r for r in rows}
                    self.log.info(f"  📝 长文记忆就绪：{len(rows)} 条台账"
                                  f"（{time.time() - t_mem:.0f}s）")

        # --- 逐块翻译
        results = self.translate_chunks(base, chunks, tm)

        # --- 组装
        self.assemble(base, chunks, tm, results, rep)
        rep.duration = time.time() - t0
        write_json_atomic(base / "_report.json", rep.to_dict())
        self.log.info(f"✅ {src.name}: {rep.status}  ok={rep.ok} degraded={rep.degraded} "
                      f"failed={rep.failed}  {human_dur(rep.duration)}")
        return rep

    # ---------------------------------------------------------------- 准备
    def prepare(self, src: Path, base: Path) -> dict:
        """读入 → 清洗 → 提词 → 遮蔽 → 切块。可重复调用（幂等）。

        续传判定不只靠参数指纹，还看**遮蔽后正文的 sha1**：改了遮蔽规则（如改了术语表、
        改了占位符样式）但参数没变时，旧块与旧译文必须作废，否则会错位复用。
        """
        fp = self.cfg.segment_fingerprint()
        meta_path = base / "job.json"
        meta_old = read_json(meta_path, {}) or {}
        chunks_path = base / "chunks.jsonl"

        raw, enc = read_text_auto(src)
        if not raw.strip():
            return {"_failed": "源文件为空"}
        lang = self.cfg.lang
        if lang == "auto":
            lang = cleanmod.detect_lang(raw)

        text = raw
        cstats: Dict[str, int] = {}
        if self.cfg.do_clean:
            if getattr(self.cfg, "clean_ai", False) and self._can_judge():
                from .clean_ai import review_clean
                text = review_clean(raw, self.cfg, self.log, lang=lang)
                cstats = {"ai": 1}
            else:
                cres = cleanmod.clean(
                    raw, lang=lang,
                    merge_lines=self.cfg.merge_wrapped_lines,
                    drop_boilerplate=self.cfg.drop_boilerplate,
                    keep_scene_break=self.cfg.keep_scene_break,
                )
                text = cres.text
                cstats = cres.stats

        # --- 专名提取 + 遮蔽（先把新正文算出来，才能做续传判定）
        extra: List[dict] = []
        if self.glossary and self.glossary.exists():
            g = read_json(self.glossary, {}) or {}
            if isinstance(g, dict):
                for k, v in g.items():
                    extra.append({"term": k, "type": (v or {}).get("type", "专名")
                                  if isinstance(v, dict) else "专名"})
            elif isinstance(g, list):
                extra.extend(g)
        found = termmod.extract_terms(text, min_freq=self.cfg.term_min_freq,
                                      extra_terms=extra,
                                      mask_katakana=self.cfg.mask_katakana)
        # cleaned.txt 先落盘：中控定名要拿例句
        write_text_atomic(base / "cleaned.txt", text)
        # --- 中控可否决“假专名”（如 教師）：遮蔽前就把非专名剔出表
        if (self.term_judge is not None and self.cfg.director_terms
                and self.cfg.director_veto and found):
            cache = self._director_cache(base, found)
            vetoed = [t for t in found
                      if isinstance(cache.get(t.term), dict)
                      and cache[t.term].get("keep") is False]
            if vetoed:
                self.log.info("  中控否决遮蔽: "
                              + "、".join(t.term for t in vetoed[:8])
                              + (" …" if len(vetoed) > 8 else ""))
                names = {t.term for t in vetoed}
                found = [t for t in found if t.term not in names]
        tm = (termmod.build_term_map(found, style=self.style)
              if self.cfg.mask_terms else TermMap(style=self.style))
        masked, hits = mask_text(text, tm)
        # 翻译前体检：把畸形占位符（【##10】/【%102;】）规整回标准样式，
        # 否则模型看到坏标记会跟着学坏，译文既翻不出内容也还原不回来。
        masked, mask_stats = audit_masks(masked, self.style, allowed=set(tm.entries.keys()))
        if mask_stats.get("fixed"):
            self.log.warn(f"  占位符体检: 修正 {mask_stats['fixed']}/{mask_stats['total']} 处畸形标记"
                          f"（例：{', '.join(mask_stats['samples'][:3])}）")
        tm.entries = {i: e for i, e in tm.entries.items() if hits.get(i, 0) > 0}
        for i, e in tm.entries.items():
            e.count = hits.get(i, e.count)
        masked_sha = sha1_text(masked)

        # --- 续传判定
        if (meta_old.get("fingerprint") == fp
                and meta_old.get("masked_sha1") == masked_sha
                and chunks_path.exists() and (base / "masked.txt").exists()):
            self.log.info("  ↺ 复用已有切块（指纹一致）")
            return meta_old

        if meta_old and (meta_old.get("fingerprint") != fp
                         or meta_old.get("masked_sha1") not in (None, masked_sha)):
            self.log.warn("  ⚠ 切块参数或遮蔽结果变了 → 旧块与旧译文作废")
            if (base / "trans").exists():
                stale = base / f"trans_stale_{stamp()}"
                n = 1
                while stale.exists():
                    n += 1
                    stale = base / f"trans_stale_{stamp()}_{n}"
                (base / "trans").rename(stale)
                self.log.warn(f"    旧译文已移到 {stale.name}")

        if cstats:
            if cstats.get("ai"):
                self.log.info("  清洗: 规则 + 中控复核（AI 判定可疑行删/留）")
            else:
                self.log.info(f"  清洗: {cstats.get('lines_in')}→{cstats.get('lines_out')} 行, "
                              f"合并 {cstats.get('merged', 0)}, 装饰 {cstats.get('deco', 0)}, "
                              f"广告 {cstats.get('boiler', 0)}")
        write_text_atomic(base / "cleaned.txt", text)
        write_json_atomic(base / "terms.json", tm.to_dict())
        # 全部候选（含未遮蔽的 low 档）另存一份，供人工审阅/做术语表
        write_json_atomic(base / "terms_candidates.json", [
            {"term": t.term, "type": t.type, "conf": t.conf,
             "count": t.count, "origin": t.origin,
             "masked": tm.by_term(t.term) is not None}
            for t in found])
        write_text_atomic(base / "masked.txt", masked)
        self.log.info(f"  专名: 候选 {len(found)} → 遮蔽 {len(tm)} 条 "
                      f"({sum(hits.values())} 处)")

        # --- 切块
        chunks = segmod.segment(masked,
                                chunk_chars=self.cfg.chunk_chars,
                                hard_max=self.cfg.chunk_hard_max,
                                context_chars=self.cfg.context_chars,
                                keep_scene_break=self.cfg.keep_scene_break)
        cdir = base / "chunks"
        cdir.mkdir(parents=True, exist_ok=True)
        for old in cdir.glob("*.txt"):
            old.unlink()
        with open(chunks_path, "w", encoding="utf-8", newline="\n") as f:
            for ch in chunks:
                write_text_atomic(cdir / f"{ch.id:06d}.txt", ch.text)
                f.write(json.dumps(ch.to_meta() if hasattr(ch, "to_meta") else
                                   {"id": ch.id, "chars": ch.chars}, ensure_ascii=False) + "\n")

        meta = {
            "version": "4.0",
            "source": str(src),
            "source_name": src.name,
            "encoding": enc,
            "lang": lang,
            "chars": len(text),
            "masked_chars": len(masked),
            "masked_sha1": masked_sha,
            "fingerprint": fp,
            "clean_stats": cstats,
            "terms": len(tm),
            "chunks": len(chunks),
            "prompt": self.prompt,
            "created_at": now_iso(),
        }
        write_json_atomic(meta_path, meta)
        self.log.info(f"  切块: {len(chunks)} 块 "
                      f"(均值 {len(text) // max(1, len(chunks))} 字/块)")
        return meta

    # ---------------------------------------------------------------- 术语
    def translate_terms(self, base: Path, tm: TermMap) -> TermMap:
        todo = tm.unresolved()
        if not todo:
            return tm
        client = self.clients[0] if self.clients else None
        if client is None:
            for e in todo:
                e.target = e.term
                e.note = "no-llm"
            write_json_atomic(base / "terms.json", tm.to_dict())
            return tm

        # --- 第一优先：中控（大模型）定名，失败的条目再回落小模型
        done = 0
        if self.term_judge is not None and self.cfg.director_terms:
            self._director_terms(base, tm, todo)
            done = sum(1 for e in tm.entries.values() if e.target.strip())

        todo = tm.unresolved()
        if todo:
            self.log.info(f"  术语翻译: {len(todo)} 条（批 {self.cfg.terms_batch}，交译者小模型）")
        for batch in termmod.chunk_batches(todo, self.cfg.terms_batch):
            if self.cancel.is_set():
                break
            sp, up = termmod.build_terms_prompt(batch)
            got: Dict[int, str] = {}
            try:
                res = client.chat(up, system=sp,
                                  max_tokens=min(512, 24 * len(batch) + 64),
                                  temperature=0.1, retries=2)
                got = termmod.parse_terms_reply(res.text, batch)
            except LlamaError as e:
                self.log.warn(f"    批次失败: {e}")
            for e in batch:
                val = got.get(e.idx)
                if not val:
                    val = self._translate_one_term(client, e)
                if val:
                    e.target = val
                    done += 1
                else:
                    e.target = e.term
                    e.conf = "low"
                    e.note = "未译出"
        write_json_atomic(base / "terms.json", tm.to_dict())
        self.log.info(f"  术语翻译完成: {done}/{len(tm)}")
        return tm

    def _director_terms(self, base: Path, tm: TermMap, todo: List) -> None:
        """让中控批量定名；决定缓存到 terms_director.json，续跑不漂移。"""
        cache = self._director_cache(base, todo)
        # 非专名（keep=false）的条目：不自己做主，交回小模型翻译
        for e in todo:
            dec = cache.get(e.term)
            if not isinstance(dec, dict):
                continue
            if dec.get("keep") is False:
                e.note = "中控:非专名"
                continue
            if dec.get("zh"):
                e.target = dec["zh"]
                e.type = dec.get("type") or e.type
                e.note = "中控定名" if dec["zh"] != e.term else "中控定名(同形)"
        changed = sum(1 for e in tm.entries.values()
                      if e.note.startswith("中控定名") and e.target != e.term)
        self.log.info(f"  中控定名生效: {changed} 条改动"
                      f"（共 {sum(1 for e in tm.entries.values() if e.note)} 条判定）")

    def _director_cache(self, base: Path, items: List) -> Dict[str, dict]:
        """查/写中控定名缓存：{term: {zh, type, keep}}。无中控时返回空表。"""
        cache_path = base / "terms_director.json"
        cache = read_json(cache_path, {}) or {}
        if not isinstance(cache, dict):
            cache = {}
        if self.term_judge is None or not self.cfg.director_terms:
            return cache
        pending = [e for e in items if e.term not in cache]
        if not pending:
            return cache
        # 候选既可能是 Term（无 idx）也可能是 TermEntry（有 idx）→ 统一包一层
        cands = [_Cand(i + 1, e.term, getattr(e, "type", "专名"),
                       getattr(e, "count", 1))
                 for i, e in enumerate(pending)]
        samples = self._term_samples(base, cands)
        self.log.info(f"  中控定名: {len(cands)} 条（{self.cfg.director_model or 'auto'}）")
        try:
            got = self.term_judge.decide_terms(cands, samples)
        except Exception as e:
            self.log.warn(f"  中控定名异常，回落小模型: {e}")
            return cache
        for idx, dec in (got or {}).items():
            if not isinstance(dec, dict):
                continue
            pos = int(idx) - 1
            if pos < 0 or pos >= len(cands):
                continue
            c = cands[pos]
            if dec.get("keep") is False:
                cache[c.term] = {"zh": "", "type": c.type, "keep": False}
            elif dec.get("zh"):
                cache[c.term] = {"zh": dec["zh"],
                                 "type": dec.get("type") or c.type,
                                 "keep": True}
        write_json_atomic(cache_path, cache)
        return cache

    @staticmethod
    def _term_samples(base: Path, todo: List) -> Dict[str, str]:
        """给每个专名找一句原文例句，帮大模型判断语境。"""
        samples: Dict[str, str] = {}
        p = base / "cleaned.txt"
        if not p.exists():
            return samples
        try:
            lines = p.read_text(encoding="utf-8").splitlines()
        except Exception:
            return samples
        for e in todo:
            for ln in lines:
                if e.term in ln:
                    samples[e.term] = ln.strip()[:90]
                    break
        return samples

    def _translate_one_term(self, client: LlamaClient, e: TermEntry) -> str:
        try:
            res = client.chat(
                f"把下面的日文专名翻译成简体中文，只输出译名本身，不要标点、不要解释：\n{e.term}",
                system=termmod.TERM_SYS, max_tokens=24, temperature=0.0, retries=2)
        except LlamaError:
            return ""
        line = ""
        for raw in (res.text or "").splitlines():
            raw = raw.strip().strip("「」『』\"'。、，,.：: ")
            if raw:
                line = raw
                break
        line = re.sub(r"^\d+\s*[.、．)）]\s*", "", line)
        line = line.strip("「」『』\"'。、，,.：: ")
        if not line or len(line) > 24:
            return ""
        kana = len(re.findall(r"[ぁ-ゖァ-ヺ]", line))
        if kana > len(line) * 0.4:
            return ""
        return line

    # ---------------------------------------------------------------- 翻译
    def translate_chunks(self, base: Path, chunks: List[dict],
                         tm: TermMap) -> Dict[int, dict]:
        trans_dir = base / "trans"
        trans_dir.mkdir(parents=True, exist_ok=True)
        state = JsonlWriter(base / "state.jsonl")

        results: Dict[int, dict] = {}
        pending: List[dict] = []
        for ch in chunks:
            rec = read_json(trans_dir / f"{int(ch['id']):06d}.json", None)
            reusable = (rec and rec.get("status") in ("ok", "degraded") and rec.get("text")
                        and not rec.get("review_note"))     # 被复核打回的块必须重译
            if reusable:
                results[int(ch["id"])] = rec
            else:
                pending.append(ch)

        self._total = len(chunks)
        self._done = len(results)
        if self.progress:
            self.progress(self._done, self._total)
        if results:
            self.log.info(f"  ↺ 续跑: 已有 {len(results)}/{len(chunks)} 块")
        if not pending:
            return results

        nworkers = max(1, min(len(self.clients) or 1, len(pending)))
        wq: "queue.Queue[dict]" = queue.Queue()
        for ch in pending:
            wq.put(ch)
        lock = threading.Lock()
        errors: List[str] = []

        def worker(idx: int) -> None:
            client = self.clients[idx % len(self.clients)] if self.clients else None
            while not self.cancel.is_set():
                try:
                    ch = wq.get_nowait()
                except queue.Empty:
                    return
                cid = int(ch["id"])
                src_text = self._chunk_text(base, cid)
                rec = self._translate_one(client, base, cid, src_text, ch, tm)
                with lock:
                    results[cid] = rec
                    self._done += 1
                    state.append({"ts": now_iso(), "chunk": cid, "status": rec["status"],
                                  "attempts": rec.get("attempts", 1),
                                  "elapsed": round(rec.get("elapsed", 0), 2),
                                  "chars": len(src_text)})
                    if self.progress:
                        self.progress(self._done, self._total)
                if rec["status"] == "failed":
                    with lock:
                        errors.append(f"#{cid:06d}")
                    self.log.warn(f"  ✗ 块 {cid:06d} 失败: {rec.get('detail', '')}")
                else:
                    self.log.info(f"  · 块 {cid:06d} {rec['status']} "
                                  f"{rec.get('elapsed', 0):.1f}s "
                                  f"{rec.get('attempts', 1)}次")

        threads = [threading.Thread(target=worker, args=(i,), daemon=True)
                   for i in range(nworkers)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        if errors:
            self.log.warn(f"  失败块 {len(errors)} 个: {', '.join(errors[:12])}"
                          + (" …" if len(errors) > 12 else ""))
        return results

    def _chunk_text(self, base: Path, cid: int) -> str:
        p = base / "chunks" / f"{cid:06d}.txt"
        return p.read_text(encoding="utf-8") if p.exists() else ""

    def _translate_one(self, client: Optional[LlamaClient], base: Path, cid: int,
                       src: str, meta: dict, tm: TermMap) -> dict:
        if not src.strip():
            return {"id": cid, "status": "ok", "text": "", "attempts": 0,
                    "elapsed": 0.0, "detail": "空块"}
        context = meta.get("context") or ""
        allowed = set(indices_in(context, self.style).keys()) if context else set()

        if self.dry_run or client is None:
            return {"id": cid, "status": "ok", "text": f"（dry-run）{src[:40]}",
                    "attempts": 0, "elapsed": 0.0,
                    "checks": [], "detail": "dry-run"}

        attempts_log: List[dict] = []
        t0 = time.time()
        last_val: Optional[Validation] = None
        last_text = ""
        proof_n = 0                      # 本块已让中控改稿的次数
        retry_note = ""
        # 上一轮「批量复核」打回时留下的意见，直接作为第一次重译的提示
        try:
            prev_p = base / "trans" / f"{cid:06d}.json"
            if prev_p.exists():
                prev = json.loads(prev_p.read_text(encoding="utf-8"))
                if prev.get("review_note"):
                    retry_note = "【中控复核意见】" + str(prev["review_note"])[:200]
                    self.log.info(f"    块{cid:06d} 带复核意见重译：{retry_note[:50]}")
        except Exception:
            pass
        bare = False                     # 上轮把指令复读了 → 下一轮只发正文（sakura 系模型）

        for rnd in range(1, max(1, self.cfg.retries) + 1):
            if self.cancel.is_set():
                break
            user = src + (("\n\n" + retry_note) if retry_note else "")
            system = "" if bare else build_system_prompt(self.prompt, context)
            mem_txt = ""
            if not bare and (self.mem_brief or self.mem_rows):
                from .memory import render_block
                mem_txt = render_block(self.mem_brief, self.mem_rows.get(cid))
                if mem_txt:
                    # 放最前面：紧贴正文时几乎必被抄（sakura 系模型）
                    system = mem_txt + "\n\n" + system
            mtokens = int(min(self.cfg.max_tokens_cap,
                              max(96, len(src) * self.cfg.max_tokens_per_char)))
            try:
                res = client.chat(user, system=system, max_tokens=mtokens,
                                  temperature=self.cfg.temperature + 0.05 * (rnd - 1))
            except LlamaError as e:
                attempts_log.append({"round": rnd, "error": str(e)[:200]})
                self.log.warn(f"    块{cid:06d} 第{rnd}轮请求失败: {e}")
                continue

            text = (res.text or "").strip()
            # 截断 → 放大上限重取一次
            if res.truncated and mtokens < self.cfg.max_tokens_cap:
                try:
                    res2 = client.chat(user, system=system,
                                       max_tokens=min(self.cfg.max_tokens_cap, mtokens * 2),
                                       temperature=self.cfg.temperature)
                    if res2.text and len(res2.text) > len(text):
                        res, text = res2, res2.text.strip()
                except LlamaError:
                    pass

            text = strip_context_echo(text, context)
            if mem_txt:
                from .memory import strip_echo as _strip_mem
                t2 = _strip_mem(text, mem_txt)
                if t2 != text:
                    self.log.info(f"    块{cid:06d} 记忆块被抄进译文 → 已剥离")
                    text = t2
            text = strip_prompt_echo(text, self.prompt)   # 整段复读提示词 → 确定性剥掉
            text = self._post_clean(text)
            val = validate_chunk(src, text, tm, self.cfg, prompt=self.prompt,
                                 context=context, allowed_extra=allowed)
            attempts_log.append({"round": rnd, "status": val.status,
                                 "checks": [c.to_dict() for c in val.checks if c.level != "ok"],
                                 "chars": len(text)})
            last_val, last_text = val, text
            # 复读指令 → 下一轮只发正文（不给任何说明文字，sakura/GalTransl 系最稳）
            bare = any(c.name == "prompt_echo" and c.level == "fail" for c in val.checks)
            if bare:
                self.log.info(f"    块{cid:06d} 命中提示词复读 → 下一轮改发纯正文")

            # --- 中控校对（默认只可疑块）：直接让大模型改稿，优于小模型自己猜
            if (self.cfg.director_proofread and self._can_judge()
                    and proof_n < max(1, int(self.cfg.director_proofread_tries))
                    and (val.status != "ok" or self.cfg.director_proofread_all
                         or self._audit_hit(cid))):
                proof_n += 1
                got = self._proofread_chunk(cid, src, text, val, tm,
                                            context=context, allowed=allowed)
                if got is not None:
                    fixed, fval = got
                    attempts_log.append({"round": "proofread", "status": fval.status,
                                         "checks": [c.to_dict() for c in fval.checks
                                                    if c.level != "ok"]})
                    if _RANK.get(fval.status, 0) > _RANK.get(val.status, 0):
                        self.log.info(f"    ✎ 块{cid:06d} 中控校对: {val.status} → {fval.status}")
                        rec = {"id": cid, "status": fval.status, "text": fixed,
                               "attempts": len(attempts_log),
                               "elapsed": time.time() - t0,
                               "checks": [c.to_dict() for c in fval.checks if c.level != "ok"],
                               "metrics": fval.metrics, "attempts_log": attempts_log,
                               "detail": "中控校对", "finish": res.finish}
                        self._save_chunk(base, cid, rec)
                        return rec
                    if fval.status != "failed":
                        # 校对没变好但也不差 → 保留校对版（通常可读性更好）
                        text, val = fixed, fval

            # 失败里最常见的"少一行"：先确定性地按句末拆回来，不重译
            if val.status == "failed":
                try:
                    from .validate import repair_line_count
                    fixed, did = repair_line_count(src, text)
                    if did:
                        v2 = validate_chunk(src, fixed, tm, self.cfg, prompt=self.prompt,
                                            context=context, allowed_extra=allowed)
                        attempts_log.append({"round": "line_repair", "status": v2.status,
                                             "checks": [c.to_dict() for c in v2.checks
                                                        if c.level != "ok"]})
                        if v2.status != "failed":
                            self.log.info(f"    ⚒ 块{cid:06d} 行数确定性修复成功（{v2.status}）")
                            text, val = fixed, v2
                except Exception as e:
                    self.log.warn(f"    行数修复异常: {str(e)[:60]}")

            if val.status != "failed":
                rec = {"id": cid, "status": val.status, "text": text,
                       "attempts": rnd, "elapsed": time.time() - t0,
                       "checks": [c.to_dict() for c in val.checks if c.level != "ok"],
                       "metrics": val.metrics, "attempts_log": attempts_log,
                       "finish": res.finish}
                # --- degraded 块交中控复核：也许只是自动质检误报
                if val.status == "degraded" and self._can_review():
                    judge = self._review_chunk(cid, src, text, val)
                    if judge is not None:
                        rec["director"] = judge
                        if judge["verdict"] == "ok":
                            rec["status"] = "ok"
                            rec["detail"] = "中控放行: " + (judge.get("reason") or "")
                            self._save_chunk(base, cid, rec)
                            self.log.info(f"    ✓ 块{cid:06d} 降级 → 中控放行"
                                          f"（{judge.get('reason','')[:40]}）")
                            return rec
                        note = "【中控复核意见】" + "；".join(judge["notes"])
                        if judge["notes"] and rnd < max(1, self.cfg.retries):
                            retry_note = note
                            self.log.info(f"    ↻ 块{cid:06d} 中控要求重译："
                                          f"{judge['notes'][0][:50]}")
                            continue
                self._save_chunk(base, cid, rec)
                return rec

            # 失败：把具体问题反馈给下一轮（比如漏了哪些占位符）
            retry_note = self._feedback(src, text, tm, val)
            # 中控再补一刀：指出具体哪句/哪个符号错了（只在前两轮问，别炸 token）
            if self._can_review() and rnd <= 2:
                judge = self._review_chunk(cid, src, text, val)
                if judge is not None and judge["verdict"] == "retry" and judge["notes"]:
                    extra = "【中控复核意见】" + "；".join(judge["notes"])
                    retry_note = (retry_note + "\n" + extra) if retry_note else extra

        # --- 定向修复：只重译漏掉占位符的那几行，再拼回去
        if not self.cancel.is_set():
            repaired = self._repair_lines(client, src, last_text, tm)
            if repaired is not None:
                val = validate_chunk(src, repaired, tm, self.cfg, prompt=self.prompt,
                                     context=context, allowed_extra=allowed)
                attempts_log.append({"round": "repair", "status": val.status,
                                     "checks": [c.to_dict() for c in val.checks if c.level != "ok"]})
                if val.status != "failed":
                    rec = {"id": cid, "status": val.status, "text": repaired,
                           "attempts": len(attempts_log), "elapsed": time.time() - t0,
                           "checks": [c.to_dict() for c in val.checks if c.level != "ok"],
                           "metrics": val.metrics, "attempts_log": attempts_log,
                           "detail": "line-repair"}
                    self._save_chunk(base, cid, rec)
                    return rec
                last_val, last_text = val, repaired

        # --- 最后一招：拆半重译
        if self.cfg.retry_split and len(src) > 120 and not self.cancel.is_set():
            halves = segmod.split_long_text(src, max(40, len(src) // 2))
            if len(halves) >= 2:
                mid = len(halves) // 2
                left, right = "".join(halves[:mid]), "".join(halves[mid:])
                outs: List[str] = []
                ok_all = True
                for part in (left, right):
                    try:
                        res = client.chat(part, system=build_system_prompt(self.prompt, context),
                                          max_tokens=max(96, int(len(part) * self.cfg.max_tokens_per_char)),
                                          temperature=self.cfg.temperature)
                    except LlamaError:
                        ok_all = False
                        break
                    pt = self._post_clean(strip_context_echo((res.text or "").strip(), context))
                    pv = validate_chunk(part, pt, tm, self.cfg, prompt=self.prompt)
                    if pv.status == "failed":
                        ok_all = False
                        break
                    outs.append(pt)
                if ok_all and outs:
                    text = "\n".join(outs)
                    val = validate_chunk(src, text, tm, self.cfg, prompt=self.prompt,
                                         context=context, allowed_extra=allowed)
                    attempts_log.append({"round": "split", "status": val.status,
                                         "checks": [c.to_dict() for c in val.checks if c.level != "ok"]})
                    if val.status != "failed":
                        rec = {"id": cid, "status": val.status, "text": text,
                               "attempts": len(attempts_log),
                               "elapsed": time.time() - t0,
                               "checks": [c.to_dict() for c in val.checks if c.level != "ok"],
                               "metrics": val.metrics, "attempts_log": attempts_log,
                               "detail": "split-retry"}
                        self._save_chunk(base, cid, rec)
                        return rec

        # --- 最后一道：行数对不上就逐行重译（保证 1:1，最慢但确定性最强）
        if client is not None and last_val is not None:
            bad = {c.name for c in last_val.checks if c.level == "fail"}
            if "line_count" in bad and getattr(self.cfg, "line_mode_fallback", False):
                text, val = self._line_mode_translate(client, src, tm, context, allowed)
                attempts_log.append({"round": "line", "status": val.status,
                                     "checks": [c.to_dict() for c in val.checks
                                                if c.level != "ok"]})
                if val.status != "failed":
                    rec = {"id": cid, "status": val.status, "text": text,
                           "attempts": len(attempts_log),
                           "elapsed": time.time() - t0,
                           "checks": [c.to_dict() for c in val.checks if c.level != "ok"],
                           "metrics": val.metrics, "attempts_log": attempts_log,
                           "detail": "line-mode-retry"}
                    self._save_chunk(base, cid, rec)
                    return rec

        rec = {"id": cid, "status": "failed", "text": last_text,
               "attempts": len(attempts_log), "elapsed": time.time() - t0,
               "checks": [c.to_dict() for c in (last_val.checks if last_val else [])
                          if c.level != "ok"],
               "metrics": (last_val.metrics if last_val else {}),
               "attempts_log": attempts_log,
               "detail": (last_val.summary() if last_val else "全部请求失败")}
        self._save_chunk(base, cid, rec)
        return rec

    def _line_mode_translate(self, client, src: str, tm, context: str, allowed):
        """逐行重译：一行一请求，天然保证行数 1:1。行数类失败的最后一招。"""
        out: List[str] = []
        for ln in src.split("\n"):
            if not ln.strip():
                out.append("")
                continue
            if not re.search(r"[ぁ-んァ-ヺ]", ln):      # 已经是中文/纯符号
                out.append(ln)
                continue
            try:
                res = client.chat(ln, system="",
                                  max_tokens=int(min(self.cfg.max_tokens_cap,
                                                     max(96, len(ln) * self.cfg.max_tokens_per_char))),
                                  temperature=0.1)
                t = self._post_clean(strip_prompt_echo((res.text or "").strip(), self.prompt))
                t = t.split("\n")[0].strip() if t.strip() else ""
                out.append(t or ln)
            except Exception:
                out.append(ln)
        text = "\n".join(out)
        val = validate_chunk(src, text, tm, self.cfg, prompt=self.prompt,
                             context=context, allowed_extra=allowed)
        return text, val

    def _can_judge(self) -> bool:
        return (self.term_judge is not None
                and getattr(self.term_judge, "available", lambda: False)())

    def _can_review(self) -> bool:
        return self.cfg.director_review and self._can_judge()

    @staticmethod
    def _apply_patches(dst: str, patches) -> Tuple[str, int, int]:
        """把中控的行级补丁确定性套用：返回 (新译文, 生效行数, 丢弃数)。

        定位规则：优先用 line 指的行，且要求 old 对得上；对不上就全局找 old 精确匹配；
        都找不到就丢弃这个补丁（绝不瞎改）。new 里的换行压成空格，保证行数不变。
        """
        lines = dst.split("\n")
        applied = dropped = 0
        ops = []                     # (行号, 动作, 行数, 新内容)
        for p in patches or []:
            if not isinstance(p, dict):
                dropped += 1
                continue
            try:
                n = int(p.get("line"))
            except Exception:
                dropped += 1
                continue
            op = str(p.get("op") or "replace").strip().lower()
            new = str(p.get("new") or "").replace(chr(13), " ").replace(chr(10), " ").strip()
            old = str(p.get("old") or "").strip()
            try:
                cnt = max(1, int(p.get("count") or 1))
            except Exception:
                cnt = 1
            if op not in ("replace", "delete", "insert"):
                op = "replace"
            if op != "delete" and not new:
                dropped += 1
                continue
            idx = n - 1 if 0 <= n - 1 < len(lines) else -1
            if idx >= 0 and old and old not in lines[idx]:
                alt = next((i for i, ln in enumerate(lines) if ln.strip() == old), -1)
                if alt >= 0:
                    idx = alt
                else:
                    idx = -1
            if idx < 0:
                alt = next((i for i, ln in enumerate(lines) if old and ln.strip() == old), -1)
                idx = alt if alt >= 0 else -1
            if idx < 0:
                dropped += 1
                continue
            if op == "replace" and lines[idx].strip() == new:
                continue
            ops.append((idx, op, cnt, new))
        # 逆序应用，避免行号漂移（delete/insert 会改变行数）
        for idx, op, cnt, new in sorted(ops, key=lambda x: -x[0]):
            if op == "delete":
                del lines[idx:idx + cnt]
            elif op == "insert":
                lines.insert(idx + 1, new)
            else:
                lines[idx] = new
            applied += 1
        return "\n".join(lines), applied, dropped

    def _audit_hit(self, cid: int) -> bool:
        """抽检：按块号做确定性抽样（同一配置重跑抽到同一批，便于 A/B）。"""
        r = float(getattr(self.cfg, "audit_ratio", 0.0) or 0.0)
        if r <= 0:
            return False
        seed = int(getattr(self.cfg, "chunk_chars", 320) or 320)
        return ((hash((seed, cid)) & 0x7FFFFFFF) % 10000) / 10000.0 < r

    def _proofread_chunk(self, cid: int, src: str, dst: str, val: Validation, tm,
                         context: str = "", allowed=None):
        """中控改稿：默认只交行级补丁（省 token、风格零漂移），拿不到有效补丁才退回整稿重写。"""
        if self.term_judge is None or not getattr(self.term_judge, "available",
                                                 lambda: False)():
            return None
        problems = [c.detail or c.name for c in (val.checks or []) if c.level != "ok"]
        fixed = None
        notes: List[str] = []

        if getattr(self.cfg, "proofread_patch", True):
            try:
                got = self.term_judge.proofread_patches(src, dst, problems)
            except Exception as e:
                self.log.warn(f"    中控补丁异常（忽略）: {e}")
                got = None
            if got:
                notes = got.get("notes") or []
                patched, applied, dropped = self._apply_patches(dst, got.get("patches"))
                if applied:
                    self.log.debug(f"    块{cid:06d} 中控补丁 {applied} 行（丢弃 {dropped}）")
                    fixed = patched
                elif dropped:
                    self.log.debug(f"    块{cid:06d} 中控补丁全部丢弃（{dropped} 条定位失败）→ 退回整稿")

        if fixed is None:                       # 补丁没拿到 / 全丢弃 → 整稿重写
            try:
                got = self.term_judge.proofread(src, dst, problems)
            except Exception as e:
                self.log.warn(f"    中控校对异常（忽略）: {e}")
                return None
            if not got or not got.get("text"):
                return None
            notes = got.get("notes") or []
            fixed = self._post_clean(strip_context_echo(got["text"].strip(), context))

        fixed = self._post_clean(strip_context_echo(fixed.strip(), context))
        if not fixed.strip():
            return None
        try:
            fval = validate_chunk(src, fixed, tm, self.cfg, prompt=self.prompt,
                                  context=context, allowed_extra=allowed)
        except Exception as e:
            self.log.warn(f"    校对稿质检异常（忽略）: {e}")
            return None
        if got.get("notes"):
            self.log.debug(f"    块{cid:06d} 校对说明: " + "；".join(got["notes"][:2]))
        return fixed, fval

    def _review_chunk(self, cid: int, src: str, dst: str, val: Validation):
        """请中控复核一块可疑译文；失败一律返回 None（不影响主流程）。"""
        problems = [c.detail or c.name for c in (val.checks or []) if c.level != "ok"]
        if not problems:
            problems = [val.summary()] if hasattr(val, "summary") else []
        probe = ""
        try:
            if getattr(self, "style", "") and "⟦" in (src or ""):
                probe = "（注：⟦数字⟧ 是已定名专名的占位符，必须原样保留。）"
            return self.term_judge.review(src, dst, problems, style_probe=probe)
        except Exception as e:
            self.log.warn(f"    中控复核异常（忽略）: {e}")
            return None

    def _save_chunk(self, base: Optional[Path], cid: int, rec: dict) -> None:
        if base is None:
            return
        write_json_atomic(base / "trans" / f"{cid:06d}.json", rec)

    def _feedback(self, src: str, dst: str, tm: TermMap, val=None) -> str:
        """把上一轮的具体毛病写成一句话喂回模型（比单纯高温重试有效得多）。"""
        if not len(tm):
            return ""
        rep = check_placeholders(src, dst, tm)
        notes: List[str] = []
        bad = [c for c in ((val.checks if val is not None else []) or [])
               if getattr(c, "name", "") == "ph_malformed" and c.level != "ok"]
        if bad:
            notes.append("占位符被写坏了（出现 【%102;】 这类畸形标记），"
                         "请把标记原样输出为形如 【#10#】 的样式，不要改写、不要加 % 或 ;")
        if rep.missing:
            marks = " ".join(mask_code(self.style, i) for i in rep.missing[:3])
            quotes: List[str] = []
            lines = src.split("\n")
            for i in rep.missing[:2]:
                mk = mask_code(self.style, i)
                for ln in lines:
                    if mk in ln:
                        quotes.append(ln.strip()[:70])
                        break
            msg = f"漏掉了占位符 {marks}，请让它们原样出现在译文对应位置"
            if quotes:
                msg += "（对应原文行：" + " / ".join(quotes) + "）"
            notes.append(msg)
        if rep.duplicate:
            marks = " ".join(mask_code(self.style, i) for i in rep.duplicate[:8])
            notes.append(f"占位符 {marks} 的出现次数不对，请与原文一一对应")
        if rep.extra:
            notes.append("出现了原文没有的占位符，请删除")
        if not notes:
            return ""
        return "【上一次译文的问题】" + "；".join(notes) + "。"

    def _repair_lines(self, client: Optional[LlamaClient], src: str, dst: str,
                      tm: TermMap) -> Optional[str]:
        """定位漏掉占位符的那几行，单独重译并回填。行数不等时不处理。"""
        if client is None or not dst or not len(tm):
            return None
        miss = check_placeholders(src, dst, tm).missing
        if not miss:
            return None
        src_lines = src.split("\n")
        dst_lines = dst.split("\n")
        if len(src_lines) != len(dst_lines):
            return None
        targets: List[int] = []
        for idx in miss[:3]:
            mk = mask_code(self.style, idx)
            for n, ln in enumerate(src_lines):
                if mk in ln and n not in targets:
                    targets.append(n)
                    break
        if not targets:
            return None
        out = list(dst_lines)
        changed = False
        for n in targets:
            need = [i for i in miss if mask_code(self.style, i) in src_lines[n]]
            if not need:
                continue
            marks = ", ".join(mask_code(self.style, i) for i in need)
            sys_p = (f"{self.prompt}\n请注意：这一行里的 {marks} 是必须保留的占位符，"
                     "必须原样出现在译文对应位置，不得省略、不得改写。")
            try:
                res = client.chat(src_lines[n], system=sys_p,
                                  max_tokens=max(64, int(len(src_lines[n]) * 2)),
                                  temperature=0.0, retries=2)
            except LlamaError:
                continue
            t = self._post_clean((res.text or "").strip())
            got = indices_in(t, self.style, set(tm.entries.keys()))
            if t and all(got.get(i, 0) >= 1 for i in need):
                out[n] = t
                changed = True
        return "\n".join(out) if changed else None

    @staticmethod
    def _post_clean(text: str, keep_wrap: bool = False) -> str:
        """去掉模型爱加的包装（整体引号 / 「译文：」/ 回吐提示词 / 多余空行）。

        keep_wrap=True 时**不**剥整体引号：单元模式下一条对白就是 `「……」`，
        剥了就把原文的引号弄丢了。
        """
        if not text:
            return ""
        t = text.strip()
        # 模型可能把正文标题也复述一遍 → 只取标题后的内容
        if BODY_HEADER in t:
            t = t.split(BODY_HEADER)[-1].strip()
        if CONTEXT_HEADER in t:
            t = t.split(CONTEXT_HEADER)[0].strip()
        # 模型把我们的重试/复核指令也吐出来了 → 从标记处截断/删行
        # （实测漏过一次：模型把「【上一次译文的问题】」写成「【上次译文的问题】」，
        #   字面匹配没命中，指令直接混进了成品，所以这里交给 strip_echo_marks 兼管）
        t = strip_echo_marks(t)
        t = strip_echo_marks(t)
        t = re.sub(r"^(?:译文|翻譯|翻译|Chinese|中文)\s*[：:]\s*", "", t)
        lines = [ln.rstrip() for ln in t.split("\n")]
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()
        t = "\n".join(lines)
        # 整体被一对引号包住（且内部没有换行）→ 去掉（单元模式不能去：那是对白本身的引号）
        if (not keep_wrap) and len(t) > 2 and t[0] in "「『\"" and t[-1] in "」』\"" \
                and "\n" not in t:
            t = t[1:-1].strip()
        return t

    # ---------------------------------------------------------------- 组装
    def assemble(self, base: Path, chunks: List[dict], tm: TermMap,
                 results: Dict[int, dict], rep: FileReport) -> None:
        for cid, rec in results.items():
            st = rec.get("status", "failed")
            if st == "ok":
                rep.ok += 1
            elif st == "degraded":
                rep.degraded += 1
                for c in rec.get("checks", []):
                    rep.bad_checks[c["name"]] = rep.bad_checks.get(c["name"], 0) + 1
            else:
                rep.failed += 1
                rep.failed_ids.append(cid)

        missing = [int(c["id"]) for c in chunks if int(c["id"]) not in results]
        if missing:
            rep.failed += len(missing)
            rep.failed_ids.extend(missing)
            rep.notes.append(f"未完成块 {len(missing)} 个: {missing[:10]}")
        rep.failed_ids.sort()

        usable = [c for c in results.values()
                  if c.get("status") == "ok" or
                  (c.get("status") == "degraded" and self.cfg.allow_degraded)]

        if self.cfg.strict and rep.failed:
            rep.status = "failed"
            detail = {
                "source": rep.source,
                "reason": f"{rep.failed} 个块未通过质检（strict 模式不产出成品）",
                "failed_ids": rep.failed_ids,
                "failed_detail": [
                    {"id": cid,
                     "detail": results[cid].get("detail", ""),
                     "checks": results[cid].get("checks", [])}
                    for cid in rep.failed_ids[:50] if cid in results
                ],
                "hint": "可用 --allow-partial 生成部分译文，或调大 chunk 尺寸/换模型重跑",
            }
            write_json_atomic(base / f"{safe_stem(rep.source)}_FAILED.json", detail)
            self.log.error(f"  ✗ {Path(rep.source).name}: 有 {rep.failed} 块失败，"
                           f"按 strict 规则不产出成品（详见 *_FAILED.json）")
            return

        tr_parts: List[str] = []
        pair_parts: List[str] = []
        origin = TermMap(style=tm.style)
        for i, e in tm.entries.items():
            origin.entries[i] = TermEntry(idx=e.idx, term=e.term, type=e.type,
                                          target=e.term, conf=e.conf, count=e.count)
        for ch in chunks:
            cid = int(ch["id"])
            rec = results.get(cid, {})
            raw_src = self._chunk_text(base, cid)
            src_show, _ = restore_text(raw_src, origin) if len(origin) else (raw_src, None)
            if ch.get("scene_before"):
                tr_parts.append("---")
                pair_parts.append("---")
            if rec.get("status") in ("ok", "degraded") and rec.get("text"):
                # 还原前体检：把译者写坏的标记（【%102;】等）先规整回标准样式
                txt_in, mstats = audit_masks(rec["text"], self.style,
                                             allowed=set(tm.entries.keys()))
                if mstats.get("fixed"):
                    rep.notes.append(f"块{cid:06d} 还原前修正畸形标记 {mstats['fixed']} 处")
                restored, rrep = restore_text(txt_in, tm)
                if rrep.missing or rrep.residual:
                    rep.notes.append(f"块{cid:06d} 还原残留: missing={rrep.missing[:5]} "
                                     f"residual={rrep.residual[:3]}")
                tr_parts.append(restored)
                pair_parts.append(src_show)
                pair_parts.append(restored)
                pair_parts.append("")
            else:
                marker = f"【未翻译 #{cid:06d}】"
                tr_parts.append(marker)
                pair_parts.append(src_show)
                pair_parts.append(marker)
                pair_parts.append("")

        stem = safe_stem(rep.source)
        out_tr = base / f"{stem}_仅翻译.txt"
        out_pair = base / f"{stem}_对照.txt"
        out_src = base / f"{stem}_原文.txt"
        write_text_atomic(out_tr, "\n".join(p for p in tr_parts if p is not None))
        write_text_atomic(out_pair, "\n".join(pair_parts).replace("\n\n\n", "\n\n"))
        cleaned = base / "cleaned.txt"
        if cleaned.exists():
            write_text_atomic(out_src, cleaned.read_text(encoding="utf-8"))
        rep.outputs = [str(out_tr), str(out_pair)]
        rep.status = "partial" if rep.failed else "ok"

    # ---------------------------------------------------------------- 合并
    @staticmethod
    def merge(out_root: Path, log: Logger, only_ok: bool = True) -> Optional[Path]:
        parts: List[str] = []
        used = 0
        reps_glob = sorted(out_root.glob("*/_report.json"))
        if not reps_glob and (out_root / "_report.json").exists():
            reps_glob = [out_root / "_report.json"]      # 扁平布局：out 目录本身即该书目录
        for rep_path in reps_glob:
            rep = read_json(rep_path, {}) or {}
            if only_ok and rep.get("status") == "failed":
                log.warn(f"  跳过失败文件: {Path(rep.get('source', '')).name}")
                continue
            stem = safe_stem(rep.get("source", rep_path.parent.name))
            f = rep_path.parent / f"{stem}_仅翻译.txt"
            if not f.exists():
                continue
            parts.append(f.read_text(encoding="utf-8"))
            used += 1
        if not parts:
            log.warn("  没有可合并的译文")
            return None
        out = out_root / f"合并_{out_root.name}.txt"
        write_text_atomic(out, "\n\n".join(parts))
        log.info(f"  合并 {used} 个文件 → {out}")
        return out
