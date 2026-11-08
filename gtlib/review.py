# -*- coding: utf-8 -*-
"""独立的「批量复核」阶段。

把流水线拆成两段，避免翻译与中控互相堵：
  ① 批量并发翻译（gt.py translate --no-director-review --workers N）
  ② 批量并发复核（gt.py review OUT目录 --workers M）→ 判定 retry 的块把状态改回 degraded
  ③ 再跑一次 translate --resume（只重译非 ok 块），必要时重复 ②③
  ④ 最后 merge

复核不改稿：判定不通过就把这块打回给译者重译（用户口径：复核只负责挑错，
重写交给翻译模型）。判定 ok 的块不动，只写一份 review 记录便于追溯。
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .core import Config, Logger, read_text_auto, write_json_atomic
from .director import make_director


def _read(p: Path) -> str:
    return read_text_auto(p)[0]


def _file_dirs(root: Path) -> List[Path]:
    root = Path(root)
    if (root / "chunks.jsonl").exists():
        return [root]
    out = []
    for d in sorted(root.iterdir()):
        if d.is_dir() and (d / "chunks.jsonl").exists():
            out.append(d)
    return out


def review_dir(root: Path, cfg: Config, log: Logger, workers: int = 3,
               audit_ratio: float = 0.0, rounds_tag: str = "") -> dict:
    """对一个输出目录里的块做批量复核；retry 的块状态改回 degraded。"""
    d = make_director(cfg, log)
    if d is None:
        log.warn("  未配置/不可达中控，跳过复核")
        return {"reviewed": 0}

    total_reviewed = total_retry = total_ok = total_err = 0
    for fd in _file_dirs(Path(root)):
        trans = sorted((fd / "trans").glob("*.json"))
        if not trans:
            continue
        cand: List[Tuple[int, str, str, dict, Path]] = []
        for p in trans:
            try:
                rec = json.loads(_read(p))
            except Exception:
                continue
            st = rec.get("status")
            text = (rec.get("text") or "").strip()
            if not text:
                continue
            # 默认只复核"没通过质检"的块；ok 块按 audit_ratio 抽样
            if st == "ok":
                if audit_ratio <= 0:
                    continue
                seed = int(hash((rounds_tag or "r", int(rec["id"]))) & 0x7FFFFFFF)
                if (seed % 10000) / 10000.0 >= audit_ratio:
                    continue
            cid = int(rec["id"])
            src_p = fd / "chunks" / f"{cid:06d}.txt"
            if not src_p.exists():
                continue
            cand.append((cid, _read(src_p), text, rec, p))

        if not cand:
            continue
        log.info(f"  {fd.name}: 待复核 {len(cand)} 块（并发 {workers}）")
        lock = threading.Lock()

        def work(item):
            cid, src, text, rec, path = item
            # 确定性预检：行数 1:1 是硬不变量，对不上直接打回，不浪费一次模型调用
            n_src = len(src.replace(chr(13), "").split(chr(10)))
            n_dst = len(text.replace(chr(13), "").split(chr(10)))
            if n_src != n_dst:
                return cid, {"verdict": "retry",
                             "notes": [f"行数不符：原文 {n_src} 行，译文 {n_dst} 行"],
                             "reason": "line_count"}, "", 0.0
            t0 = time.time()
            try:
                hint = ("判定要点：只有在出现实质错误（漏译、错译、占位符丢失、行数不符、"
                        "假名残留、人称/称谓确实译错）时才 retry；"
                        "措辞更优美、语气词位置、同义表达不同一律判 ok。")
                got = d.review(src, text, [hint], style_probe="")
            except Exception as e:
                return cid, None, str(e)[:120], 0.0
            return cid, got, "", time.time() - t0

        with cf.ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            for cid, got, err, secs in ex.map(work, cand):
                with lock:
                    total_reviewed += 1
                    if err or not got:
                        total_err += 1
                        log.warn(f"    块{cid:06d} 复核失败（忽略）: {err or '空响应'}")
                        continue
                    verdict = str(got.get("verdict") or "").lower()
                    notes = [str(n).strip() for n in (got.get("notes") or []) if str(n).strip()]
                    path_obj = next(p for c, _, _, r, p in cand if c == cid)
                    try:
                        rec = json.loads(_read(path_obj))
                    except Exception:
                        continue
                    rec["review"] = {"verdict": verdict, "notes": notes[:4],
                                     "reason": str(got.get("reason") or "")[:160],
                                     "at": time.strftime("%Y-%m-%d %H:%M:%S"),
                                     "secs": round(secs, 1)}
                    if verdict == "retry":
                        rec["status"] = "degraded"     # 打回重译
                        rec["review_note"] = "；".join(notes[:3]) or str(got.get("reason") or "")[:120]
                        total_retry += 1
                        log.info(f"    ↺ 块{cid:06d} 复核不通过 → 打回重译：{rec['review_note'][:60]}")
                    else:
                        total_ok += 1
                    write_json_atomic(path_obj, rec)

    log.info(f"  复核完成：{total_reviewed} 块｜放行 {total_ok}｜打回 {total_retry}｜异常 {total_err}")
    return {"reviewed": total_reviewed, "ok": total_ok,
            "retry": total_retry, "errors": total_err}
