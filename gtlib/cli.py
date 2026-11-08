# -*- coding: utf-8 -*-
"""命令行入口：gt.py <子命令> ..."""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse

from . import __version__
from . import clean as cleanmod
from . import segment as segmod
from . import terms as termmod
from .core import (APP_DIR, Config, Logger, human_dur, read_json, read_text_auto,
                   safe_stem, stamp, write_json_atomic, write_text_atomic)
from .job import FileReport, Translator, default_prompt
from .llama import LlamaClient, LlamaServer, ServerPool, make_servers, port_free
from .director import make_director
from .placeholder import TermMap, mask_text
from .validate import validate_chunk


# ------------------------------------------------------------------ 工具
def add_cfg_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("服务")
    g.add_argument("--server-cmd", help="llama-server 启动命令模板（含模型路径）")
    g.add_argument("--base-url", help="直接连已运行的 llama-server，如 http://127.0.0.1:9080")
    g.add_argument("--api-base", help="OpenAI 兼容 API 地址，如 https://api.deepseek.com/v1")
    g.add_argument("--api-key", help="API Key（也可用环境变量 GT_API_KEY）")
    g.add_argument("--model", help="模型名（远程必填，否则取 /models 第一个）")
    g.add_argument("--director-base",
                   help="中控（大模型）端点，如 http://127.0.0.1:19092/v1；不填则单模型跑")
    g.add_argument("--director-model", help="中控模型名，如 qwen3.8-27b（留空取第一个）")
    g.add_argument("--director-key", help="中控 API Key（本地引擎不需要）")
    g.add_argument("--director-no-think", action="store_true",
                   help="中控请求带 chat_template_kwargs.enable_thinking=false（Qwen3 系 vLLM）")
    g.add_argument("--no-director", action="store_true", help="禁用中控（纯单模型）")
    g.add_argument("--no-director-review", action="store_true",
                   help="中控只定名，不复核译文")
    g.add_argument("--no-director-veto", action="store_true",
                   help="不允许中控否决遮蔽（默认允许）")
    g.add_argument("--proofread", choices=["off", "flagged", "all"],
                   help="中控改稿：off=不改稿（默认，只做复核）/ flagged=只改可疑块 / all=全量改稿")
    g.add_argument("--proofread-tries", type=int, help="单块最多改几次稿（默认 2）")
    g.add_argument("--no-proofread", action="store_true", help="等同 --proofread off")
    g.add_argument("--audit-ratio", type=float,
                   help="抽检比例 0~1：对质检通过的块也随机送中控校对（0.1 = 十分之一）")
    g.add_argument("--proofread-mode", choices=["patch", "rewrite"],
                   help="校对产出：patch=只交改动的行（默认）／rewrite=整稿重写")
    g.add_argument("--port-base", type=int, default=9080)
    g.add_argument("--gpus", default="0", help="逗号分隔的 GPU 号，如 0,1")
    g.add_argument("--workers", type=int, default=1,
                   help="并发 worker 数（建议 2~3；实测 6 路并发会让 sakura 复读率飙升、中控排队超时）")
    g.add_argument("--slots", type=int, default=1,
                   help="给自启的 llama-server 开多少 slot（--parallel N，N 个并发请求共用一卡）")
    g.add_argument("--startup-timeout", type=float, default=240.0)

    m = p.add_argument_group("翻译")
    m.add_argument("-c", "--config", help="从 job.json / config.json 载入全部参数")
    m.add_argument("--mode", choices=["chunk", "unit"],
                   help="chunk=按块翻（默认）/ unit=完整句单元（揃 unit-tokens）")
    m.add_argument("--unit-tokens", type=int, help="单条单元 token 上限（默认 20）")
    m.add_argument("--unit-pack", type=int, help="一个请求装多少条单元（默认 8）")
    m.add_argument("--unit-chars", type=float, dest="unit_chars_per_token",
                   help="无 tiktoken 时的 token 估算比例（默认 1.6 字/token）")
    m.add_argument("--unit-context", type=int, help="单元模式上文字数（默认 80）")
    m.add_argument("--unit-check-pack", type=int, help="中控一次判多少条（默认 8）")
    m.add_argument("--no-unit-check", action="store_true", help="单元模式不交中控定性")
    m.add_argument("--final-check", choices=["off", "fixed", "all"],
                   help="全文终检：off / fixed=只查改过条目 / all=全量（默认）")
    m.add_argument("--prompt", help="系统提示词（默认内置）")
    m.add_argument("--system-in-user", action="store_true",
                   help="把指令并入 user 消息、不发 system 角色（sakura 系模型防复读）")
    m.add_argument("--think", action="store_true",
                   help="译者保留思考模式（默认关闭：请求带 enable_thinking=false）")
    m.add_argument("--clean-ai", action="store_true",
                   help="清洗阶段让中控复核可疑行（广告/乱码/误删的正文）")
    m.add_argument("--line-mode", action="store_true",
                   help="行数对不上时逐行重译兜底（保证 1:1，最慢）")
    m.add_argument("--chunk", type=int, help="每块目标字符数")
    m.add_argument("--chunk-max", type=int, help="单块硬上限字符数")
    m.add_argument("--context", type=int, help="上文锚点字符数（0=不给）")
    m.add_argument("--temp", type=float)
    m.add_argument("--retries", type=int)
    m.add_argument("--timeout", type=float, help="单块总超时秒")
    m.add_argument("--stall", type=float, help="无数据流超时秒（看门狗）")
    m.add_argument("--style", choices=["bracket", "double", "hash", "brace"])
    m.add_argument("--glossary", help="外部术语表 json（{\"词\": {\"type\": \"人名\"}}）")
    m.add_argument("--dry-run", action="store_true", help="不调模型，只验证清洗/切块/组装")
    m.add_argument("--save-config", help="把本次参数存成 json 以便复跑")

    q = p.add_argument_group("质量")
    q.add_argument("--no-clean", action="store_true")
    q.add_argument("--no-mask", action="store_true", help="不遮蔽专名（只翻译）")
    q.add_argument("--mask-katakana", action="store_true",
                   help="连纯片假名专名一起遮蔽（默认只遮人名/地名/术语表）")
    q.add_argument("--no-term-translate", action="store_true", help="跳过术语翻译")
    q.add_argument("--allow-partial", action="store_true",
                   help="有块失败也产出成品（默认 strict：不出成品）")
    q.add_argument("--out-flat", action="store_true",
                   help="输出目录本身就是该书目录（不额外套一层文件名子目录）")
    q.add_argument("--lang", choices=["auto", "ja", "zh"], default=None)
    q.add_argument("--log-level", choices=["debug", "info", "warn", "error"], default=None)


def apply_args(cfg: Config, args: argparse.Namespace) -> Config:
    if getattr(args, "api_base", None): cfg.api_base = args.api_base
    if getattr(args, "api_key", None): cfg.api_key = args.api_key
    if getattr(args, "model", None): cfg.model_name = args.model
    if getattr(args, "director_base", None): cfg.director_base = args.director_base
    if getattr(args, "director_model", None): cfg.director_model = args.director_model
    if getattr(args, "director_key", None): cfg.director_key = args.director_key
    if getattr(args, "director_no_think", False): cfg.director_no_think = True
    if getattr(args, "no_director", False): cfg.director_base = ""
    if getattr(args, "no_director_review", False): cfg.director_review = False
    if getattr(args, "no_director_veto", False): cfg.director_veto = False
    if getattr(args, "think", False): cfg.translator_no_think = False
    if getattr(args, "clean_ai", False): cfg.clean_ai = True
    if getattr(args, "line_mode", False): cfg.line_mode_fallback = True
    if getattr(args, "audit_ratio", None) is not None:
        cfg.audit_ratio = max(0.0, min(1.0, float(args.audit_ratio)))
    pm = getattr(args, "proofread_mode", None)
    if pm:
        cfg.proofread_patch = (pm == "patch")
    pf = getattr(args, "proofread", None)
    if pf == "off" or getattr(args, "no_proofread", False):
        cfg.director_proofread = False
        cfg.director_proofread_all = False
    elif pf == "flagged":
        cfg.director_proofread = True
        cfg.director_proofread_all = False
    elif pf == "all":
        cfg.director_proofread = True
        cfg.director_proofread_all = True
    if getattr(args, "proofread_tries", None) is not None:
        cfg.director_proofread_tries = max(1, int(args.proofread_tries))
    if getattr(args, "server_cmd", None): cfg.server_cmd = args.server_cmd
    if getattr(args, "port_base", None): cfg.port_base = args.port_base
    if getattr(args, "gpus", None):
        cfg.gpus = tuple(int(x) for x in str(args.gpus).split(",") if x.strip() != "")
    if getattr(args, "workers", None): cfg.workers = max(1, args.workers)
    if getattr(args, "slots", None): cfg.slots = max(1, args.slots)
    if getattr(args, "mode", None): cfg.mode = args.mode
    if getattr(args, "unit_tokens", None): cfg.unit_tokens = max(4, args.unit_tokens)
    if getattr(args, "unit_pack", None): cfg.unit_pack = max(1, args.unit_pack)
    if getattr(args, "unit_chars_per_token", None):
        cfg.unit_chars_per_token = max(0.5, float(args.unit_chars_per_token))
    if getattr(args, "unit_context", None) is not None:
        cfg.unit_context = max(0, args.unit_context)
    if getattr(args, "unit_check_pack", None):
        cfg.unit_check_pack = max(1, args.unit_check_pack)
    if getattr(args, "no_unit_check", False): cfg.unit_check = False
    if getattr(args, "final_check", None): cfg.final_check = args.final_check
    if getattr(args, "startup_timeout", None): cfg.server_start_timeout = args.startup_timeout
    if getattr(args, "prompt", None): cfg.prompt = args.prompt
    if getattr(args, "system_in_user", False): cfg.system_in_user = True
    if getattr(args, "chunk", None): cfg.chunk_chars = args.chunk
    if getattr(args, "chunk_max", None): cfg.chunk_hard_max = args.chunk_max
    if getattr(args, "context", None) is not None and args.context is not None:
        cfg.context_chars = args.context
    if getattr(args, "temp", None) is not None: cfg.temperature = args.temp
    if getattr(args, "retries", None): cfg.retries = args.retries
    if getattr(args, "timeout", None): cfg.request_timeout = args.timeout
    if getattr(args, "stall", None): cfg.stall_timeout = args.stall
    if getattr(args, "style", None): cfg.placeholder_style = args.style
    if getattr(args, "no_clean", False): cfg.do_clean = False
    if getattr(args, "no_mask", False): cfg.mask_terms = False
    if getattr(args, "mask_katakana", False): cfg.mask_katakana = True
    if getattr(args, "no_term_translate", False): cfg.translate_terms = False
    if getattr(args, "allow_partial", False): cfg.strict = False
    cfg.out_flat = bool(getattr(args, "out_flat", False))
    if getattr(args, "lang", None): cfg.lang = args.lang
    if getattr(args, "log_level", None): cfg.log_level = args.log_level
    if not cfg.prompt:
        cfg.prompt = default_prompt(cfg.placeholder_style)
    return cfg


def collect_inputs(paths: List[str], pattern: str = "*.txt") -> List[Path]:
    out: List[Path] = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            out.extend(sorted(path.glob(pattern)))
        elif path.is_file():
            out.append(path)
        else:
            print(f"!! 找不到: {p}")
    seen, uniq = set(), []
    for f in out:
        r = f.resolve()
        if r not in seen:
            seen.add(r)
            uniq.append(f)
    return uniq


def _attach_server(base_url: str, log: Logger) -> LlamaServer:
    port = urlparse(base_url).port or 9080
    srv = LlamaServer(cmd=[], port=port, log=log, name="attach")
    srv.owned = False
    if not srv.health(5.0):
        raise SystemExit(f"!! 无法连接 {base_url}/health")
    log.info(f"🔗 连接到已有服务 {base_url}")
    return srv


# ------------------------------------------------------------------ 子命令
def cmd_clean(args: argparse.Namespace) -> int:
    files = collect_inputs(args.inputs, args.pattern)
    if not files:
        return 2
    merge = {"auto": None, "on": True, "off": False}[args.merge]
    ai = getattr(args, "ai_review", False)
    cfg = None
    log = None
    if ai:
        from .clean_ai import review_clean
        cfg = Config()
        cfg.director_base = getattr(args, "director_base", "") or ""
        cfg.director_model = getattr(args, "director_model", "") or ""
        cfg.director_no_think = bool(getattr(args, "director_no_think", False))
        cfg.director_timeout = 240.0
        log = Logger(None, echo=True)
    for f in files:
        raw, enc = read_text_auto(f)
        if ai:
            text = review_clean(raw, cfg, log, lang=args.lang)
            stats = {"ai": True}
            res = cleanmod.CleanResult(text, stats)
        else:
            res = cleanmod.clean(raw, lang=args.lang, merge_lines=merge,
                                 drop_boilerplate=not args.keep_ads)
        out = Path(args.out) / f"{safe_stem(f.name)}_clean.txt" if args.out else \
            f.with_name(f"{safe_stem(f.name)}_clean.txt")
        write_text_atomic(out, res.text)
        print(f"{f.name} [{enc}] {res.stats} → {out}")
    return 0


def cmd_terms(args: argparse.Namespace) -> int:
    files = collect_inputs(args.inputs, args.pattern)
    if not files:
        return 2
    for f in files:
        raw, enc = read_text_auto(f)
        res = cleanmod.clean(raw, lang=args.lang) if not args.raw else cleanmod.CleanResult(raw)
        found = termmod.extract_terms(res.text, min_freq=args.min_freq)
        tm = termmod.build_term_map(found, style=args.style)
        masked, hits = mask_text(res.text, tm)
        tm.entries = {i: e for i, e in tm.entries.items() if hits.get(i, 0) > 0}
        out = f.with_name(f"{safe_stem(f.name)}_terms.json")
        write_json_atomic(out, tm.to_dict())
        print(f"\n=== {f.name} ===  候选 {len(found)} → 遮蔽 {len(tm)} 条")
        for t in found[:args.show]:
            e = tm.by_term(t.term)
            print(f"  [{t.type}/{t.conf}] {t.term}  x{t.count}"
                  + (f"  → 占位符#{e.idx}" if e else "  (未遮蔽)"))
        print(f"→ {out}")
    return 0


def cmd_translate(args: argparse.Namespace) -> int:
    cfg = Config.load(args.config) if args.config else Config()
    cfg = apply_args(cfg, args)
    if getattr(args, "save_config", None):
        cfg.save(args.save_config)
        print(f"参数已保存: {args.save_config}")

    files = collect_inputs(args.inputs, args.pattern)
    if not files:
        print("!! 没有输入文件")
        return 2

    out_root = Path(args.out) if args.out else (APP_DIR / "OUT" / f"{stamp()}_{safe_stem(files[0].name)}")
    out_root.mkdir(parents=True, exist_ok=True)
    log = Logger(out_root / "run.log", echo=not args.quiet, level=cfg.log_level)

    log.info(f"=== gtlib v{__version__} ===")
    log.info(f"输入 {len(files)} 个文件 → {out_root}")
    log.info(f"块 {cfg.chunk_chars}±{cfg.chunk_hard_max} 字 | 重试 {cfg.retries} | "
             f"strict={cfg.strict} | 遮蔽={cfg.mask_terms} | 占位符={cfg.placeholder_style}")

    if args.fresh:
        for f in files:
            stale = out_root / safe_stem(f.name)
            if stale.exists():
                import shutil
                shutil.rmtree(stale, ignore_errors=True)
                log.warn(f"  --fresh: 已删除 {stale.name}")

    clients: List[LlamaClient] = []
    stop_all = None
    judge = None
    cancel = __import__("threading").Event()
    reports: List[FileReport] = []

    try:
        if args.dry_run:
            log.warn("  ⚠ dry-run：不调用模型，只验证清洗/切块/组装")
        else:
            # 中控先连（只做判断，和译者服务互不影响）
            if not args.no_director:
                try:
                    judge = make_director(cfg, log, cancel)
                except Exception as e:
                    log.warn(f"  中控初始化失败（按单模型跑）: {e}")
            if args.base_url:
                server = _attach_server(args.base_url, log)
                n = max(1, cfg.workers) * max(1, cfg.slots)
                clients = [LlamaClient(server, log, cfg, cancel=cancel)
                           for _ in range(n)]
            else:
                if not cfg.server_cmd and not cfg.use_api:
                    log.error("缺少 --server-cmd / --api-base（或 --base-url 连接已有服务）")
                    return 3
                servers, stop_fn = make_servers(cfg, log, out_root / "log")
                stop_all = stop_fn
                clients = []
                for s in servers:
                    clients += [LlamaClient(s, log, cfg, cancel=cancel)
                                for _ in range(max(1, int(cfg.slots or 1)))]
            log.info(f"  {len(clients)} 个 worker 就绪")

        done_total = 0
        for i, src in enumerate(files, 1):
            log.info(f"\n--- [{i}/{len(files)}] {src.name} ---")
            tr = Translator(cfg, log, out_root, clients=clients,
                            cancel=cancel, dry_run=args.dry_run,
                            glossary=Path(args.glossary) if args.glossary else None,
                            director=judge)
            rep = tr.process(src)
            reports.append(rep)

        ok = sum(1 for r in reports if r.status == "ok")
        part = sum(1 for r in reports if r.status == "partial")
        bad = sum(1 for r in reports if r.status == "failed")
        log.info(f"\n=== 完成: ok={ok} partial={part} failed={bad} ===")
        write_json_atomic(out_root / "_summary.json", {
            "version": __version__, "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "ok": ok, "partial": part, "failed": bad,
            "reports": [r.to_dict() for r in reports],
        })
        if args.merge:
            Translator.merge(out_root, log, only_ok=not args.allow_partial)
        return 0 if bad == 0 else 1
    except KeyboardInterrupt:
        cancel.set()
        log.warn("已中断（已完成的块已落盘，可原命令重跑续传）")
        return 130
    finally:
        if stop_all is not None:
            stop_all()


def cmd_merge(args: argparse.Namespace) -> int:
    log = Logger(None, echo=True)
    out = Translator.merge(Path(args.out_dir), log, only_ok=not args.allow_partial)
    return 0 if out else 1


def cmd_status(args: argparse.Namespace) -> int:
    root = Path(args.out_dir)
    rows = []
    reps = sorted(root.glob("*/_report.json"))
    if not reps and (root / "_report.json").exists():
        reps = [root / "_report.json"]          # 扁平布局：目录本身即该书目录
    for rep in reps:
        d = read_json(rep, {}) or {}
        rows.append((Path(d.get("source", "")).name, d.get("status", "?"),
                     d.get("chunks", 0), d.get("ok", 0), d.get("degraded", 0),
                     d.get("failed", 0), d.get("duration", 0)))
    if not rows:
        print("没有找到 _report.json")
        return 1
    print(f"{'文件':40s} {'状态':8s} {'块':>5s} {'ok':>5s} {'deg':>5s} {'fail':>5s} {'耗时':>8s}")
    for r in rows:
        print(f"{r[0][:38]:40s} {r[1]:8s} {r[2]:5d} {r[3]:5d} {r[4]:5d} {r[5]:5d} {human_dur(r[6]):>8s}")
    return 0


def cmd_review(args: argparse.Namespace) -> int:
    """批量复核：把没通过质检的块交中控判 ok/retry；retry 的块打回重译。"""
    from .review import review_dir

    cfg = Config.load(args.config) if getattr(args, "config", None) else Config()
    cfg = apply_args(cfg, args)
    out_root = Path(args.out_dir)
    if not out_root.exists():
        print(f"!! 目录不存在: {out_root}")
        return 2
    log = Logger(out_root / "review.log", echo=True, level=cfg.log_level)
    log.info(f"=== 批量复核 {out_root} ===")
    res = review_dir(out_root, cfg, log, workers=max(1, args.workers),
                     audit_ratio=max(0.0, min(1.0, args.audit_ratio or 0.0)),
                     rounds_tag=args.tag or "")
    if res.get("retry"):
        print(f"\n有 {res['retry']} 块被打回重译：同参数再跑一次 gt.py translate（续传只补这些块）")
    return 0


def cmd_selftest(args: argparse.Namespace) -> int:
    """内置冒烟测试：不依赖 GPU / 网络。"""
    from .selftest import run_selftest
    return 0 if run_selftest(verbose=not args.quiet) else 1


def cmd_polish(args: argparse.Namespace) -> int:
    """收尾：称呼统一（9B 判性别一次）+ 特殊字符清理（纯 Python）。"""
    from .polish import clean_special, gender_map, unify_honorifics

    target = Path(args.target)
    terms_path = None
    if target.is_dir():
        merged = sorted(target.glob("合并_*.txt"))
        if not merged:
            print(f"!! {target} 下没有 合并_*.txt")
            return 2
        target_txt = merged[0]
        cand = list(target.glob("*/terms.json"))
        terms_path = cand[0] if cand else None
    else:
        target_txt = target
        terms_path = None
        if getattr(args, "terms", None):
            terms_path = Path(args.terms)

    text = target_txt.read_text(encoding="utf-8", errors="replace")
    before_len = len(text)

    # 1) 称呼统一（需要中控 + 术语表里的人名）
    replaced = 0
    if getattr(args, "director_base", "") and terms_path and terms_path.exists():
        cfg = Config()
        cfg.director_base = args.director_base
        cfg.director_model = getattr(args, "director_model", "") or ""
        cfg.director_no_think = bool(getattr(args, "director_no_think", False))
        log = Logger(None, echo=True)
        try:
            import json as _json
            terms = _json.loads(terms_path.read_text(encoding="utf-8"))
            names = [e.get("target") or e.get("term") for e in terms.get("terms", [])
                     if (e.get("type") or "").startswith("人名")]
            names = [n for n in names if n]
            print(f"  待统一称呼的人名：{len(names)} 个")
            pmap = gender_map(names, cfg, log)
            print(f"  判定出敬称：{pmap}")
            text, replaced = unify_honorifics(text, pmap)
        except Exception as e:
            print(f"  !! 称呼统一失败（跳过）：{str(e)[:100]}")

    # 2) 特殊字符清理
    text, stats = clean_special(text)

    out_path = Path(args.out) if getattr(args, "out", None) else         target_txt.with_name(target_txt.stem + "_成品.txt")
    write_text_atomic(out_path, text)
    print(f"  称呼统一：替换 {replaced} 处｜字符清理：{stats}")
    print(f"  {before_len} → {len(text)} 字 → {out_path}")
    return 0


def cmd_golden(args: argparse.Namespace) -> int:
    from .score import build_golden
    meta = build_golden(Path(args.from_run), Path(args.out), samples=args.samples)
    print(f"黄金集已建：{args.out}｜{meta['samples']} 段｜源行数 {meta['source_lines']}")
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    from .score import fmt_report, score_run

    results = []
    for t in args.targets:
        res = score_run(Path(t), Path(args.golden), judge_base=args.judge_base,
                        judge_model=args.judge_model, judge_key=args.judge_key,
                        limit=args.limit, workers=args.workers)
        results.append(res)
        print(fmt_report(res, label=Path(t).name))
        print()

    if len(results) == 2:                      # A/B：逐段比较综合分
        a, b = results
        wins = losses = ties = 0
        by_id = {r["id"]: r for r in a["rows"]}
        for rb in b["rows"]:
            ra = by_id.get(rb["id"])
            ja = (ra or {}).get("judge") or {}
            jb = rb.get("judge") or {}
            if ja.get("overall") is None or jb.get("overall") is None:
                continue
            if jb["overall"] > ja["overall"] + 1e-6:
                wins += 1
            elif jb["overall"] < ja["overall"] - 1e-6:
                losses += 1
            else:
                ties += 1
        print(f"### A/B（{Path(args.targets[1]).name} 相对 {Path(args.targets[0]).name}）"
              f"：胜 {wins}｜平 {ties}｜负 {losses}")

    if args.out:
        write_json_atomic(Path(args.out),
                          [{"agg": r["agg"], "rows": r["rows"]} for r in results])
        print(f"明细已写入 {args.out}")
    return 0


# ------------------------------------------------------------------ 入口
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="gt.py", description=f"gtlib v{__version__} — 日文小说/Galgame 翻译流水线")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("translate", help="完整流水线：清洗→提词→遮蔽→翻译→还原→组装")
    p.add_argument("inputs", nargs="+", help="文件或目录")
    p.add_argument("-o", "--out", help="输出根目录（默认 OUT/<时间戳>_<文件名>）")
    p.add_argument("--pattern", default="*.txt")
    p.add_argument("--fresh", action="store_true", help="丢弃已有切块与译文重跑")
    p.add_argument("--merge", action="store_true", help="结束后合并所有译文")
    p.add_argument("--quiet", action="store_true")
    add_cfg_args(p)
    p.set_defaults(func=cmd_translate)

    p = sub.add_parser("clean", help="只做清洗")
    p.add_argument("inputs", nargs="+")
    p.add_argument("-o", "--out")
    p.add_argument("--pattern", default="*.txt")
    p.add_argument("--lang", default="auto", choices=["auto", "ja", "zh"])
    p.add_argument("--merge", default="auto", choices=["auto", "on", "off"],
                   help="是否合并软换行（auto=日文合并）")
    p.add_argument("--keep-ads", action="store_true")
    p.add_argument("--ai-review", action="store_true",
                   help="让中控（9B）复核清洗：判定可疑行该删还是该留")
    p.add_argument("--director-base", help="中控端点，如 http://127.0.0.1:9092/v1")
    p.add_argument("--director-model", help="中控模型名，如 qwen35-9b-awq")
    p.add_argument("--director-no-think", action="store_true",
                   help="带 chat_template_kwargs.enable_thinking=false")
    p.set_defaults(func=cmd_clean)

    p = sub.add_parser("terms", help="只看专名提取与遮蔽结果")
    p.add_argument("inputs", nargs="+")
    p.add_argument("--pattern", default="*.txt")
    p.add_argument("--lang", default="auto")
    p.add_argument("--min-freq", type=int, default=2)
    p.add_argument("--style", default="bracket",
                   choices=["bracket", "double", "hash", "brace"])
    p.add_argument("--show", type=int, default=40)
    p.add_argument("--raw", action="store_true", help="不清洗直接提取")
    p.set_defaults(func=cmd_terms)

    p = sub.add_parser("merge", help="合并一个输出目录里的所有译文")
    p.add_argument("out_dir")
    p.add_argument("--allow-partial", action="store_true")
    p.set_defaults(func=cmd_merge)

    p = sub.add_parser("status", help="查看输出目录里各文件的进度/质量")
    p.add_argument("out_dir")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("selftest", help="内置自检（不需要 GPU）")
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=cmd_selftest)

    p = sub.add_parser("review", help="批量复核：没通过质检的块交中控判定，retry 的打回重译")
    p.add_argument("out_dir", help="translate 的输出目录")
    p.add_argument("--tag", default="", help="抽样用的轮次标记（同标记重复跑抽到同一批）")
    add_cfg_args(p)
    p.set_defaults(func=cmd_review, workers=3, audit_ratio=0.0)

    p = sub.add_parser("polish", help="收尾：称呼统一 + 特殊字符清理")
    p.add_argument("target", help="输出目录（含 合并_*.txt 与 terms.json）或单个 txt")
    p.add_argument("-o", "--out", help="输出文件（默认 原名_成品.txt）")
    p.add_argument("--terms", help="txt 模式下手工指定 terms.json")
    p.add_argument("--director-base", help="中控端点（判人物性别用）")
    p.add_argument("--director-model", help="中控模型名")
    p.add_argument("--director-no-think", action="store_true")
    p.set_defaults(func=cmd_polish)

    p = sub.add_parser("golden", help="从一次高质量运行里抽黄金集（回归评分用）")
    p.add_argument("--from-run", required=True, help="输出根目录或其中一个源文件的产物目录")
    p.add_argument("--out", required=True, help="黄金集目录，如 bench/golden")
    p.add_argument("--samples", type=int, default=24, help="抽多少段（默认 24）")
    p.set_defaults(func=cmd_golden)

    p = sub.add_parser("score", help="按黄金集给成品打分（可两两 A/B）")
    p.add_argument("targets", nargs="+", help="成品 txt 或输出目录；给两个则做 A/B")
    p.add_argument("--golden", default="bench/golden", help="黄金集目录")
    p.add_argument("--judge-base", default="", help="裁判模型端点（OpenAI 兼容），留空只算确定性指标")
    p.add_argument("--judge-model", default="", help="裁判模型名（留空取 /models 第一个）")
    p.add_argument("--judge-key", default="", help="裁判 API Key")
    p.add_argument("--limit", type=int, default=0, help="只评前 N 段（0=全部）")
    p.add_argument("--workers", type=int, default=4, help="裁判并发（默认 4）")
    p.add_argument("--out", help="把逐段明细写成 json")
    p.set_defaults(func=cmd_score)

    return ap


def main(argv: Optional[List[str]] = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    return args.func(args)
