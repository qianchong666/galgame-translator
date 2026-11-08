# -*- coding: utf-8 -*-
"""单元模式流水线：按「≤N token 的完整句」翻译 → 27B 逐批定性 → 定点重译 → 装回 → 全文终检。

与块模式（job.Translator）的区别只在翻译与校验粒度：

    clean/mask/术语  ← 复用 Translator.prepare / translate_terms（完全不变）
    切分            ← unit.build_units（完整句、贪心打包到 ≤unit_tokens）
    翻译            ← 每请求打包 unit_pack 条（多 slot 并发），带上一小段上文
    检查            ← 27B 每 unit_check_pack 条一批，判 ok / order / wrong
                      ok     → 通过
                      order  → 用 27B 的 fix 直接替换（先过占位符/假名门）
                      wrong  → 带「问题」重译该条，再复检（最多 unit_check_rounds 轮）
    装回            ← unit.merge_back（行号+行内序号，行数/空行/场景线与原文一致）
    全文终检        ← 按 final_window 条一组再过一遍 27B，只看连贯性与遗留错误

所有中间态落在 base/ 下：units.jsonl（单元表）、utrans/NNNNNN.json（单条状态），
所以中断后原命令重跑就是续传。
"""
from __future__ import annotations

import json
import queue
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from . import unit as unitmod
from .core import (Config, Logger, human_dur, now_iso, read_json, write_json_atomic,
                   write_text_atomic, sha1_text, safe_stem)
from .llama import LlamaClient, LlamaError
from .placeholder import restore_text
from .validate import validate_chunk


class UnitRun:
    """一次单元模式作业。构造后调 go()。"""

    def __init__(self, tr, src: Path, base: Path, tm, rep) -> None:
        self.tr = tr
        self.cfg: Config = tr.cfg
        self.log: Logger = tr.log
        self.clients: Sequence[LlamaClient] = tr.clients
        self.director = tr.term_judge
        self.cancel = tr.cancel
        self.base = Path(base)
        self.tm = tm
        self.rep = rep
        self.udir = self.base / "utrans"
        self.units: List[unitmod.Unit] = []
        self.lines: List[str] = []
        self.recs: Dict[int, dict] = {}
        self._done = 0
        self._total = 0
        self._lock = threading.Lock()

    # ---------------------------------------------------------------- 入口
    def go(self):
        t0 = time.time()
        masked_p = self.base / "masked.txt"
        if not masked_p.exists():
            self.rep.status = "failed"
            self.rep.notes.append("缺少 masked.txt（prepare 未完成）")
            return self.rep
        masked = masked_p.read_text(encoding="utf-8")
        self.units, self.lines = unitmod.build_units(
            masked, max_tokens=self.cfg.unit_tokens,
            chars_per_token=self.cfg.unit_chars_per_token,
            scene_mark="---" if self.cfg.keep_scene_break else "")
        if not self.units:
            self.rep.status = "failed"
            self.rep.notes.append("没有可翻译单元")
            return self.rep
        texty = sum(1 for u in self.units if u.translatable)
        self.rep.chunks = len(self.units)
        self.log.info(f"🧩 {texty} 条可译单元 / {len(self.units)} 条（含标点+场景）/ "
                      f"{len(self.lines)} 行 · 上限 {self.cfg.unit_tokens} token "
                      f"（{unitmod.token_method()}）· 打包 {self.cfg.unit_pack} 条/请求")

        self._sync_units()
        self._load()
        self._translate_pending()
        if not self.cancel.is_set():
            self._check_round(stage="初检", only_ids=None)
        if self.cfg.unit_check_rounds > 1 and not self.cancel.is_set():
            # 第二轮：只复检「重译过、还没定性」的（27B 改过稿的不再重查，避免来回抖）
            dirty = [u.id for u in self.units
                     if u.id in self.recs and not self.recs[u.id].get("check_done")
                     and self.recs[u.id].get("touched")]
            if dirty:
                self.log.info(f"  ↻ 复检 {len(dirty)} 条改过/重译过的单元")
                self._check_round(stage="复检", only_ids=set(dirty))
        self._assemble_and_write(final=False)
        if self.cfg.final_check != "off" and not self.cancel.is_set():
            self._final_check()
            self._assemble_and_write(final=True)
        self.rep.duration = time.time() - t0
        self.rep.status = "partial" if self.rep.failed else "ok"
        write_json_atomic(self.base / "_report.json", self.rep.to_dict())
        self.log.info(f"✅ {Path(self.rep.source).name}: {self.rep.status}  "
                      f"ok={self.rep.ok} degraded={self.rep.degraded} failed={self.rep.failed}  "
                      f"{human_dur(self.rep.duration)}")
        return self.rep

    # ---------------------------------------------------------------- 单元表
    def _fingerprint(self) -> str:
        raw = "|".join(f"{u.line}:{u.order}:{u.kind}:{u.src}" for u in self.units)
        return sha1_text(raw + f"#{self.cfg.unit_tokens}#{self.cfg.unit_chars_per_token}")

    def _sync_units(self) -> None:
        """单元表落盘；指纹变了就把旧译文挪走（否则会错位复用）。"""
        self.udir.mkdir(parents=True, exist_ok=True)
        up = self.base / "units.jsonl"
        fp = self._fingerprint()
        old = read_json(self.base / "units_meta.json", {}) or {}
        if up.exists() and old.get("fingerprint") not in (None, fp):
            stale = self.base / f"utrans_stale_{int(time.time())}"
            if self.udir.exists() and any(self.udir.iterdir()):
                self.udir.rename(stale)
                self.udir.mkdir(parents=True, exist_ok=True)
            self.log.warn(f"  ⚠ 单元切分变了 → 旧单元译文作废（{stale.name}）")
        with open(up, "w", encoding="utf-8", newline="\n") as f:
            for u in self.units:
                f.write(json.dumps(u.to_dict(), ensure_ascii=False) + "\n")
        write_json_atomic(self.base / "units_meta.json", {
            "fingerprint": fp, "units": len(self.units),
            "tokens": self.cfg.unit_tokens, "chars_per_token": self.cfg.unit_chars_per_token,
            "pack": self.cfg.unit_pack, "created_at": now_iso(),
        })

    def _load(self) -> None:
        for u in self.units:
            rec = read_json(self.udir / f"{u.id:06d}.json", None)
            if rec and rec.get("status") in ("ok", "degraded") and "text" in rec:
                self.recs[u.id] = rec
        self._total = sum(1 for u in self.units if u.translatable)
        self._done = sum(1 for i in self.recs if self.recs[i].get("text") is not None)
        if self.recs:
            self.log.info(f"  ↺ 续跑: 已有 {self._done}/{self._total} 条单元译文")
        self._progress()

    def _progress(self) -> None:
        if self.tr.progress:
            self.tr.progress(min(self._done, self._total), self._total)

    def _save(self, uid: int, rec: dict) -> None:
        self.recs[uid] = rec
        write_json_atomic(self.udir / f"{uid:06d}.json", rec)

    # ---------------------------------------------------------------- 翻译
    def _translate_pending(self) -> None:
        todo = [u for u in self.units if u.translatable and u.id not in self.recs]
        if not todo:
            return
        packs = unitmod.pack_units(todo, size=self.cfg.unit_pack,
                                  max_chars=self.cfg.unit_max_chars)
        self.log.info(f"  ⚙ 待译 {len(todo)} 条 → {len(packs)} 个批次 "
                      f"· {max(1, len(self.clients))} worker")
        wq: "queue.Queue[List[unitmod.Unit]]" = queue.Queue()
        for p in packs:
            wq.put(p)
        errors: List[int] = []

        def worker(idx: int) -> None:
            client = self.clients[idx % len(self.clients)] if self.clients else None
            while not self.cancel.is_set():
                try:
                    pack = wq.get_nowait()
                except queue.Empty:
                    return
                got = self._do_pack(client, pack)
                with self._lock:
                    for u in pack:
                        rec = got.get(u.id)
                        if rec is None:
                            continue
                        self._save(u.id, rec)
                        self._done += 1
                        if rec.get("status") == "failed":
                            errors.append(u.id)
                self._progress()

        n = max(1, min(len(self.clients) or 1, len(packs)))
        threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        if errors:
            self.log.warn(f"  翻译阶段失败 {len(errors)} 条: {errors[:10]}")

    def _context_for(self, first: unitmod.Unit) -> str:
        """给批次开头找一段上文（同文件前面 1~2 条，限长）。"""
        n = int(self.cfg.unit_context)
        if n <= 0:
            return ""
        prev = [u for u in self.units
                if u.translatable and (u.line, u.order) < (first.line, first.order)]
        tail = "".join(u.src for u in prev[-2:])
        return tail[-n:].strip()

    def _do_pack(self, client: Optional[LlamaClient],
                 pack: Sequence[unitmod.Unit],
                 hints: Optional[Dict[int, str]] = None) -> Dict[int, dict]:
        """译一个批次；编号对不上的条目再单独补译。"""
        out: Dict[int, dict] = {}
        t0 = time.time()
        if self.tr.dry_run or client is None:
            return {u.id: {"id": u.id, "status": "ok", "text": f"（dry-run）{u.src[:24]}",
                           "src": u.src, "attempts": 0, "elapsed": 0.0,
                           "detail": "dry-run"} for u in pack}
        system = unitmod.build_batch_system(self.tr.prompt, self._context_for(pack[0]))
        user = unitmod.build_batch_user(pack)
        if hints:
            rows = [f"{i}: {hints[u.id][:80]}" for i, u in enumerate(pack, 1)
                    if hints.get(u.id)]
            if rows:
                user += ("\n\n【上一次译文的问题·这几乎要求改正】\n" + "\n".join(rows))
        chars = sum(len(u.src) for u in pack)
        mtokens = int(min(self.cfg.max_tokens_cap,
                          max(128, chars * self.cfg.max_tokens_per_char + 64)))
        try:
            res = client.chat(user, system=system, max_tokens=mtokens,
                              temperature=self.cfg.temperature)
            raw = res.text or ""
        except LlamaError as e:
            self.log.warn(f"  批次请求失败（{len(pack)} 条）: {e}")
            raw = ""
        got = unitmod.parse_batch_reply(raw, len(pack))
        for i, u in enumerate(pack, 1):
            text = self.tr._post_clean(got.get(i, ""), keep_wrap=True)
            if not text:
                text = self._translate_single(client, u, t0)      # 编号丢了 → 单条补译
                if text is None:
                    out[u.id] = {"id": u.id, "status": "failed", "text": u.src, "src": u.src,
                                 "attempts": 1, "elapsed": time.time() - t0,
                                 "detail": "批次未返回该条，单条补译也失败"}
                    self.log.warn(f"  ✗ 单元 {u.id:06d} 未译出: {u.src[:30]}")
                    continue
            status, probs = unitmod.check_unit(u.src, text)
            out[u.id] = {"id": u.id, "status": status, "text": text, "src": u.src,
                         "attempts": 1, "elapsed": time.time() - t0,
                         "checks": probs, "detail": "初译"}
            if status != "ok":
                self.log.info(f"  · 单元 {u.id:06d} {status}: {'; '.join(probs)[:60]}")
        return out

    def _translate_single(self, client: LlamaClient, u: unitmod.Unit,
                          t0: float) -> Optional[str]:
        ctx = self._context_for(u)
        system = unitmod.build_batch_system(self.tr.prompt, ctx)
        user = unitmod.build_batch_user([u])
        for rnd in range(1, max(1, self.cfg.retries) + 1):
            if self.cancel.is_set():
                return None
            try:
                res = client.chat(user, system=system,
                                  max_tokens=max(96, len(u.src) * 3),
                                  temperature=self.cfg.temperature + 0.05 * (rnd - 1))
            except LlamaError:
                continue
            got = unitmod.parse_batch_reply(res.text or "", 1)
            text = self.tr._post_clean(got.get(1, ""), keep_wrap=True)
            if text:
                return text
        return None

    # ---------------------------------------------------------------- 27B 判定
    STAGE_RANK = {"初检": 1, "复检": 2, "全文终检": 3}

    def _check_round(self, stage: str, only_ids: Optional[set] = None) -> None:
        if not self.cfg.unit_check or self.director is None:
            return
        if not getattr(self.director, "available", lambda: False)():
            self.log.warn("  中控不可用 → 跳过单元质检（只保留本地规则门）")
            return
        rank = self.STAGE_RANK.get(stage, 0)
        cand = [u for u in self.units
                if u.translatable and u.id in self.recs
                and (only_ids is None or u.id in only_ids)
                and self.STAGE_RANK.get(self.recs[u.id].get("check_done") or "", 0) < rank]
        if not cand:
            return
        packs = unitmod.pack_units(cand, size=self.cfg.unit_check_pack, max_chars=1200)
        t0 = time.time()
        fixed = retry = kept = 0
        retry_ids: List[int] = []
        for pi, pack in enumerate(packs, 1):
            if self.cancel.is_set():
                break
            pairs = [(i, u.src, self.recs[u.id].get("text", ""))
                     for i, u in enumerate(pack, 1)]
            hint = ("【这一批是连续正文】判定时可以看上下文，但每条只判它自己那一条。"
                    if stage == "全文终检" else "")
            verdicts = self.director.check_units(pairs, hint=hint)
            if not verdicts:
                self.log.warn(f"  {stage}: 第 {pi}/{len(packs)} 批没有拿到判定，跳过")
                continue
            for i, u in enumerate(pack, 1):
                v = verdicts.get(i)
                rec = self.recs[u.id]
                if not v:
                    continue
                rec["verdict"] = v.get("v")
                rec["why"] = v.get("why", "")
                rec["check_done"] = stage
                fix = v.get("fix") or ""
                # 本地规则门判 failed 的（占位符多/少）：绝不能因为中控说 ok 就放行，
                # 只能「中控给了能过门的修正稿」或「重译」。
                if rec.get("status") == "failed":
                    if fix and unitmod.fix_passable(u.src, fix):
                        rec["text"] = self.tr._post_clean(fix, keep_wrap=True) or fix
                        rec["status"] = "ok"
                        rec["touched"] = True
                        rec["fix_count"] = int(rec.get("fix_count", 0)) + 1
                        rec["detail"] = f"{stage}:中控改正(failed)"
                        fixed += 1
                        self._save(u.id, rec)
                    else:
                        rec["check_done"] = ""
                        retry_ids.append(u.id)
                        self._save(u.id, rec)
                    continue
                if v["v"] == "ok":
                    if rec.get("status") == "degraded":
                        rec["status"] = "ok"
                        rec["detail"] = f"{stage}:中控放行"
                        kept += 1
                    self._save(u.id, rec)
                    continue
                if fix and unitmod.fix_passable(u.src, fix) and not rec.get("fix_count"):
                    # 每条最多被中控改一次：实测 27B 每看一次都能“找点东西改”，
                    # 不封顶就会反复抖（同一批 394 条第二轮又改了 107 条）——那是纯消耗。
                    rec["text"] = self.tr._post_clean(fix, keep_wrap=True) or fix
                    rec["status"] = "ok"
                    rec["touched"] = True
                    rec["fix_count"] = int(rec.get("fix_count", 0)) + 1
                    rec["detail"] = f"{stage}:{'语序' if v['v'] == 'order' else '中控改正'}"
                    fixed += 1
                    self._save(u.id, rec)
                    continue
                if rec.get("fix_count"):
                    self._save(u.id, rec)          # 已经改过一次 → 不再折腾也不算重译理由
                    continue
                # order 但修正稿过不了门（占位符丢了/带日文）→ 当 wrong 重译，不能默默放过
                retry_ids.append(u.id)
                rec["check_done"] = ""
                self._save(u.id, rec)
            if pi % 5 == 0 or pi == len(packs):
                self.log.info(f"  {stage} {pi}/{len(packs)} 批 · 改稿 {fixed} · "
                              f"待重译 {len(retry_ids)} · {time.time() - t0:.0f}s")
        # --- 重译 wrong 的条目（带 27B 的意见）
        redone = 0
        if retry_ids and not self.cancel.is_set():
            todo = [u for u in self.units if u.id in set(retry_ids)]
            packs2 = unitmod.pack_units(todo, size=self.cfg.unit_pack, max_chars=700)
            for pack in packs2:
                if self.cancel.is_set():
                    break
                hint = "；".join(f"{u.src[:12]}:{self.recs[u.id].get('why') or '重译'}"
                                for u in pack if self.recs[u.id].get("why"))
                problems = {u.id: (self.recs[u.id].get("why") or
                                   "；".join(self.recs[u.id].get("checks") or []) or
                                   "译文不可用")
                            for u in pack}
                got = self._do_pack(self.clients[0] if self.clients else None, pack,
                                    hints=problems)
                for u in pack:
                    rec = got.get(u.id)
                    if not rec:
                        continue
                    rec["touched"] = True
                    rec["check_done"] = ""          # 改过了 → 还得再过一次判定
                    rec["attempts"] = int(self.recs[u.id].get("attempts", 1)) + 1
                    rec["detail"] = f"{stage}:重译"
                    rec["hint"] = hint[:160]
                    if rec.get("status") != "failed":
                        redone += 1
                    self._save(u.id, rec)
        if fixed or redone or kept:
            self.log.info(f"  {stage}小计: 中控改稿 {fixed} · 放行 {kept} · 重译 {redone}")

    # ---------------------------------------------------------------- 全文终检
    def _final_check(self) -> None:
        if self.cfg.final_check == "fixed":
            dirty = {u.id for u in self.units
                     if self.recs.get(u.id, {}).get("touched")}
            if not dirty:
                self.log.info("  全文终检: 没有改过的条目 → 跳过")
                return
            ids = set()
            for u in self.units:
                if u.id in dirty:
                    ids.update(range(max(1, u.id - 3), u.id + 4))
            self.log.info(f"  全文终检: 只查改过条目及其邻域（{len(ids)} 条）")
        else:
            ids = None
            self.log.info("  全文终检: 全量再过一遍")
        self._check_round(stage="全文终检", only_ids=ids)

    # ---------------------------------------------------------------- 装回
    def _assemble_and_write(self, final: bool) -> None:
        texts = {uid: rec.get("text", "") for uid, rec in self.recs.items()
                 if rec.get("status") in ("ok", "degraded") and rec.get("text")}
        merged_masked = unitmod.merge_back(self.lines, self.units, texts)
        restored, rrep = restore_text(merged_masked, self.tm)
        if rrep.missing or rrep.residual:
            self.rep.notes.append(f"还原残留: missing={rrep.missing[:5]} "
                                 f"residual={rrep.residual[:3]}")
        # 行数守恒：装回来必须和清洗后的原文一样多行
        src_lines = (self.base / "cleaned.txt").read_text(encoding="utf-8")
        if len(restored.split("\n")) != len(src_lines.split("\n")):
            self.rep.notes.append("⚠ 装回后行数与原文不一致")
            self.log.warn("  ⚠ 装回后行数与原文不一致")

        # 单元级统计
        self.rep.ok = self.rep.degraded = self.rep.failed = 0
        self.rep.failed_ids = []
        self.rep.bad_checks = {}
        for u in self.units:
            rec = self.recs.get(u.id)
            if rec is None:
                if u.translatable:
                    self.rep.failed += 1
                    self.rep.failed_ids.append(u.id)
                continue
            st = rec.get("status", "failed")
            if st == "ok":
                self.rep.ok += 1
            elif st == "degraded":
                self.rep.degraded += 1
                for p in (rec.get("checks") or []):
                    key = str(p).split()[0][:8] if p else "?"
                    self.rep.bad_checks[key] = self.rep.bad_checks.get(key, 0) + 1
            else:
                self.rep.failed += 1
                self.rep.failed_ids.append(u.id)
        self.rep.failed_ids.sort()

        stem = safe_stem(self.rep.source)
        out_tr = self.base / f"{stem}_仅翻译.txt"
        out_pair = self.base / f"{stem}_对照.txt"
        out_src = self.base / f"{stem}_原文.txt"
        write_text_atomic(out_tr, restored)
        # 对照：原文行 / 译文行 交替
        pair: List[str] = []
        for a, b in zip(src_lines.split("\n"), restored.split("\n")):
            pair.append(a)
            pair.append(b)
            pair.append("")
        write_text_atomic(out_pair, "\n".join(pair).replace("\n\n\n", "\n\n"))
        write_text_atomic(out_src, src_lines)
        self.rep.outputs = [str(out_tr), str(out_pair)]
        tag = "终检后" if final else "装回"
        self.log.info(f"  📦 {tag}: {len(self.lines)} 行 · ok={self.rep.ok} "
                      f"degraded={self.rep.degraded} failed={self.rep.failed}")


def run_unit_pipeline(tr, src: Path, base: Path, tm, rep):
    return UnitRun(tr, src, base, tm, rep).go()
