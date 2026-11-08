# -*- coding: utf-8 -*-
"""黄金集回归评分。

为什么要有它：9 道质检门只能证明"没坏"（行数够、占位符没丢、没复读），
证明不了"更好"。换模型/换提示词/改校对策略时，只有固定黄金集上的分数能横向比较。

黄金集按 **cleaned.txt 的行号区间** 定义（清洗是确定性的，行号跨运行不变；
占位符样式/切块参数变了也不影响），参考译文取自一次高质量输出。
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .core import read_text_auto as _rta, write_json_atomic, write_text_atomic


def _read(path) -> str:
    """read_text_auto 返回 (文本, 编码)，这里只要文本。"""
    return _rta(path)[0]

KANA = re.compile(r"[ぁ-んァ-ヺ]")
PH = re.compile(r"⟦\s*\d+\s*⟧|【#\s*\d+\s*#】|【#?人名_?\d+#?】|\[\d+\]")


# ---------------------------------------------------------------- 工具
def _lines(text: str) -> List[str]:
    return text.replace("\r\n", "\n").replace("\r", "\n").split("\n")


def _find_file_dir(root: Path) -> Path:
    """输出根目录 → 单个源文件的产物目录（含 chunks.jsonl 的那层）。"""
    root = Path(root)
    if (root / "chunks.jsonl").exists():
        return root
    for d in sorted(root.iterdir()):
        if d.is_dir() and (d / "chunks.jsonl").exists():
            return d
    raise FileNotFoundError(f"{root} 下找不到 chunks.jsonl")


def _find_output(file_dir: Path) -> Path:
    """产物目录 → 仅翻译成品。"""
    cands = sorted(file_dir.glob("*_仅翻译.txt"))
    if not cands:
        raise FileNotFoundError(f"{file_dir} 下找不到 *_仅翻译.txt")
    return cands[0]


# ---------------------------------------------------------------- 黄金集
def build_golden(run_root: Path, out_dir: Path, samples: int = 24,
                 only_ok: bool = True) -> dict:
    """从一次高质量运行里抽黄金集（均匀取样，按块边界切）。"""
    file_dir = _find_file_dir(Path(run_root))
    cleaned = _lines(_read(file_dir / "cleaned.txt"))
    out_lines = _lines(_read(_find_output(file_dir)))
    chunks = [json.loads(x) for x in _read(file_dir / "chunks.jsonl").splitlines()
              if x.strip()]

    ok_ids = set()
    if only_ok:
        for p in sorted((file_dir / "trans").glob("*.json")):
            try:
                rec = json.loads(_read(p))
                if rec.get("status") in ("ok", "degraded"):
                    ok_ids.add(int(rec["id"]))
            except Exception:
                continue

    pool = [c for c in chunks if not ok_ids or c["id"] in ok_ids]
    if not pool:
        pool = chunks
    step = max(1, len(pool) // max(1, samples))
    picked = pool[::step][:samples]

    out_dir = Path(out_dir)
    (out_dir / "chunks").mkdir(parents=True, exist_ok=True)
    (out_dir / "ref").mkdir(parents=True, exist_ok=True)
    items = []
    for i, c in enumerate(picked, 1):
        s, e = int(c["line_start"]), int(c["line_end"])
        src = "\n".join(cleaned[s:e + 1]) if e < len(cleaned) else "\n".join(cleaned[s:])
        ref = "\n".join(out_lines[s:e + 1]) if e < len(out_lines) else "\n".join(out_lines[s:])
        if not src.strip() or not ref.strip():
            continue
        name = f"{i:04d}"
        write_text_atomic(out_dir / "chunks" / f"{name}.txt", src)
        write_text_atomic(out_dir / "ref" / f"{name}.txt", ref)
        items.append({"id": name, "chunk_id": c["id"], "line_start": s,
                      "line_end": e, "chars": len(src)})

    meta = {
        "version": 1,
        "source_file": str(file_dir),
        "source_lines": len(cleaned),
        "output_lines": len(out_lines),
        "samples": len(items),
        "items": items,
    }
    write_json_atomic(out_dir / "meta.json", meta)
    return meta


def load_golden(golden_dir: Path) -> Tuple[dict, List[dict]]:
    golden_dir = Path(golden_dir)
    meta = json.loads(_read(golden_dir / "meta.json"))
    items = []
    for it in meta["items"]:
        items.append({
            "id": it["id"],
            "src": _read(golden_dir / "chunks" / f"{it['id']}.txt"),
            "ref": _read(golden_dir / "ref" / f"{it['id']}.txt"),
            "line_start": it["line_start"], "line_end": it["line_end"],
        })
    return meta, items


# ---------------------------------------------------------------- 确定性指标
def metrics(src: str, ref: str, cand: str) -> dict:
    import difflib

    sl, rl, cl = _lines(src), _lines(ref), _lines(cand)
    sim = difflib.SequenceMatcher(None, ref, cand).ratio()
    return {
        "line_src": len(sl), "line_ref": len(rl), "line_cand": len(cl),
        "line_delta": len(cl) - len(rl),
        "kana": len(KANA.findall(cand)),
        "ph_left": len(PH.findall(cand)),
        "chars": len(cand),
        "len_ratio": round(len(cand) / max(1, len(ref)), 3),
        "sim_ref": round(sim, 4),
    }


# ---------------------------------------------------------------- 裁判
JUDGE_SYS = (
    "你是日译中译文的评分员。用户给出「日文原文」「参考译文」「待评译文」。\n"
    "请独立评价**待评译文**（参考译文只当锚点，不要直接抄它的分）：\n"
    "- accuracy：1-5，意思是否准确，有无漏译/错译/加戏；\n"
    "- fluency：1-5，中文是否通顺自然，不像翻译腔；\n"
    "- style：1-5，语气/人称/称谓是否贴合原文人物与场景；\n"
    "- format：1-5，行数是否与原文一致、引号「」与标点结构是否保留、有无残留假名或占位符；\n"
    "- vs_ref：待评译文与参考译文相比，整体是 better / equal / worse。\n"
    "只输出一个 JSON 对象：{\"accuracy\":n,\"fluency\":n,\"style\":n,\"format\":n,"
    "\"vs_ref\":\"better|equal|worse\",\"note\":\"一句话指出最要紧的问题\"}"
)


def _parse_json(text: str) -> Optional[dict]:
    text = text.strip()
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def judge(base: str, model: str, key: str, src: str, ref: str, cand: str,
          timeout: float = 120.0) -> Optional[dict]:
    import requests

    user = (f"【日文原文】\n{src}\n\n【参考译文】\n{ref}\n\n【待评译文】\n{cand}")
    payload = {
        "model": model or "judge",
        "messages": [{"role": "system", "content": JUDGE_SYS},
                     {"role": "user", "content": user}],
        "temperature": 0.0, "max_tokens": 1600,   # 思考型模型会吃光预算，留足
    }
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    url = base.rstrip("/")
    if not url.endswith("/v1"):
        url += "/v1"
    r = requests.post(url + "/chat/completions", json=payload, headers=headers,
                      timeout=timeout)
    r.raise_for_status()
    ch = r.json()["choices"][0]
    text = ch["message"].get("content") or ""
    if not text and ch.get("finish_reason") == "length":     # 思考吃光预算 → 放大重试
        payload["max_tokens"] = 4000
        r = requests.post(url + "/chat/completions", json=payload, headers=headers,
                          timeout=timeout)
        r.raise_for_status()
        text = r.json()["choices"][0]["message"].get("content") or ""
    got = _parse_json(text)
    if not got:
        return None
    out = {}
    for k in ("accuracy", "fluency", "style", "format"):
        try:
            out[k] = float(got.get(k))
        except Exception:
            out[k] = None
    out["vs_ref"] = str(got.get("vs_ref") or "").lower()
    out["note"] = (got.get("note") or "")[:120]
    vals = [v for k, v in out.items() if k in ("accuracy", "fluency", "style", "format")
            and v is not None]
    out["overall"] = round(sum(vals) / len(vals), 2) if vals else None
    return out


# ---------------------------------------------------------------- 主流程
def score_run(target: Path, golden_dir: Path, judge_base: str = "",
              judge_model: str = "", judge_key: str = "",
              limit: int = 0, workers: int = 4) -> dict:
    """给一个成品（目录或 txt）打分。"""
    meta, items = load_golden(golden_dir)
    target = Path(target)
    if target.is_dir():
        out_path = _find_output(_find_file_dir(target))
    else:
        out_path = target
    lines = _lines(_read(out_path))
    drift = len(lines) - int(meta.get("source_lines", len(lines)))

    rows = []
    for it in items[: limit or None]:
        s, e = it["line_start"], it["line_end"]
        cand = "\n".join(lines[s:e + 1]) if e < len(lines) else "\n".join(lines[s:])
        row = {"id": it["id"], **metrics(it["src"], it["ref"], cand)}
        rows.append((row, cand, it))

    if judge_base:
        import concurrent.futures as cf

        def work(pair):
            row, cand, it = pair
            try:
                row["judge"] = judge(judge_base, judge_model, judge_key,
                                     it["src"], it["ref"], cand)
            except Exception as e:
                row["judge"] = None
                row["judge_error"] = str(e)[:120]
            return row, cand, it

        with cf.ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            rows = list(ex.map(work, rows))

    scored = [r for r, _, _ in rows]
    js = [r["judge"] for r in scored if r.get("judge")]
    def _avg(key: str) -> Optional[float]:
        vals = [j[key] for j in js if j.get(key) is not None]
        return round(sum(vals) / len(vals), 2) if vals else None

    vs = {}
    for j in js:
        vs[j["vs_ref"]] = vs.get(j["vs_ref"], 0) + 1

    agg = {
        "target": str(out_path),
        "line_drift": drift,
        "n": len(scored),
        "sim_ref": round(sum(r["sim_ref"] for r in scored) / max(1, len(scored)), 4),
        "line_ok": sum(1 for r in scored if r["line_delta"] == 0),
        "kana_total": sum(r["kana"] for r in scored),
        "ph_left_total": sum(r["ph_left"] for r in scored),
        "judge_n": len(js),
        "accuracy": _avg("accuracy"), "fluency": _avg("fluency"),
        "style": _avg("style"), "format": _avg("format"),
        "overall": _avg("overall"),
        "vs_ref": vs,
    }
    return {"agg": agg, "rows": scored}


def fmt_report(res: dict, label: str = "") -> str:
    a = res["agg"]
    head = f"### {label or a['target']}"
    if a["line_drift"]:
        head += f"  ⚠ 行数与黄金集不一致({a['line_drift']:+d})"
    parts = [
        head,
        f"- 样本 {a['n']} 段｜与参考相似度 {a['sim_ref']}｜行数守恒 {a['line_ok']}/{a['n']}"
        f"｜假名残留 {a['kana_total']}｜占位符残渣 {a['ph_left_total']}",
    ]
    if a["judge_n"]:
        parts.append(f"- 裁判(1-5)：准确 {a['accuracy']}｜通顺 {a['fluency']}"
                     f"｜语气 {a['style']}｜格式 {a['format']}｜**综合 {a['overall']}**"
                     f"｜对比参考 {a['vs_ref']}")
    return "\n".join(parts)
