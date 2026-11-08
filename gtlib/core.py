# -*- coding: utf-8 -*-
"""基础设施：日志、编码探测、原子写、JSONL、配置。

设计要点
--------
* 日志单消费者：所有 worker 线程只往 Logger 里写，Logger 内部加锁后同时落盘 + 回调
  （GUI 侧把回调塞进 queue 再由 Tk 主线程消费，绝不在工作线程里 root.update()）。
* 编码探测打分：utf-8 严格优先，其次对 cp932/gb18030/big5/shift_jis 解码后打分。
* 所有落盘都是"临时文件 + os.replace"，断电/中断不会留下半个文件。
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import threading
import time
import unicodedata
from dataclasses import dataclass, asdict, field, fields
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

APP_DIR = Path(__file__).resolve().parents[1]

# ============================================================ 日志

_LEVELS = {"debug": 10, "info": 20, "warn": 30, "error": 40}


class Logger:
    """线程安全日志器：落盘 + 控制台 + 任意回调（GUI 用）。"""

    def __init__(self, path: Optional[os.PathLike | str] = None,
                 echo: bool = True, level: str = "info"):
        self.path = Path(path) if path else None
        self.echo = echo
        self.level = _LEVELS.get(level, 20)
        self._lock = threading.RLock()
        self._sinks: List[Callable[[str], None]] = []
        self._file = None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._file = open(self.path, "a", encoding="utf-8", buffering=1)

    # -- 回调 --------------------------------------------------
    def add_sink(self, fn: Callable[[str], None]) -> None:
        with self._lock:
            self._sinks.append(fn)

    def remove_sink(self, fn: Callable[[str], None]) -> None:
        with self._lock:
            if fn in self._sinks:
                self._sinks.remove(fn)

    def close(self) -> None:
        with self._lock:
            if self._file:
                try:
                    self._file.close()
                finally:
                    self._file = None

    # -- 内部 --------------------------------------------------
    def _emit(self, lvl: str, msg: str, **fields: Any) -> None:
        if _LEVELS[lvl] < self.level:
            return
        ts = datetime.now().strftime("%H:%M:%S")
        extra = ""
        if fields:
            extra = " " + " ".join(f"{k}={v}" for k, v in fields.items())
        line = f"[{ts}][{lvl.upper():<5}]{msg}{extra}"
        with self._lock:
            if self._file:
                try:
                    self._file.write(line + "\n")
                except Exception:
                    pass
            if self.echo:
                try:
                    print(line, flush=True)
                except Exception:
                    pass
            for s in list(self._sinks):
                try:
                    s(line)
                except Exception:
                    pass

    def debug(self, msg: str, **f: Any) -> None: self._emit("debug", msg, **f)
    def info(self, msg: str, **f: Any) -> None: self._emit("info", msg, **f)
    def warn(self, msg: str, **f: Any) -> None: self._emit("warn", msg, **f)
    def error(self, msg: str, **f: Any) -> None: self._emit("error", msg, **f)


class NullLogger(Logger):
    def __init__(self) -> None:
        super().__init__(echo=False, level="error")


# ============================================================ 编码

_ENCODINGS = ("utf-8-sig", "utf-8", "cp932", "shift_jis", "gb18030", "big5", "euc-jp", "utf-16")

_HIRA = range(0x3041, 0x30A0 + 1)
_KATA = range(0x30A1, 0x30FF + 1)


def _score_text(s: str) -> float:
    """给"解码结果像不像正常日文/中文"打分，越大越好。"""
    if not s:
        return -1e9
    score = 0.0
    for ch in s:
        o = ord(ch)
        if o in _HIRA:
            score += 3
        elif o in _KATA:
            score += 3
        elif 0x4E00 <= o <= 0x9FFF:
            score += 1
        elif ch in "\n\r\t 　、。！？「」『』（）…—・":
            score += 0.2
        elif o < 0x20:
            score -= 8
        elif o == 0xFFFD:
            score -= 60
        elif 0x80 <= o <= 0x9F:
            score -= 30
        elif 0xC0 <= o <= 0xFF:
            # Latin-1 区在日文/中文文本里几乎只可能是乱码
            score -= 12
    return score / max(1, len(s))


def decode_bytes(data: bytes, prefer: Sequence[str] = _ENCODINGS) -> tuple[str, str, bool]:
    """返回 (文本, 编码名, 是否干净解码)。"""
    if data.startswith(b"\xef\xbb\xbf"):
        return data.decode("utf-8-sig"), "utf-8-sig", True
    if data[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return data.decode("utf-16"), "utf-16", True

    try:
        return data.decode("utf-8"), "utf-8", True
    except UnicodeDecodeError:
        pass

    best: tuple[float, str, str] = (-1e18, "utf-8", "")
    for enc in prefer:
        try:
            s = data.decode(enc)
            clean = True
        except UnicodeDecodeError:
            s = data.decode(enc, errors="replace")
            clean = False
        sc = _score_text(s) - (0 if clean else 5)
        if sc > best[0]:
            best = (sc, enc, s)
    return best[2], best[1], True


def read_text_auto(path: os.PathLike | str) -> tuple[str, str]:
    """读取文本文件，自动探测编码。返回 (文本, 编码)。"""
    data = Path(path).read_bytes()
    text, enc, _ = decode_bytes(data)
    return text, enc


# ============================================================ 落盘

def write_text_atomic(path: os.PathLike | str, text: str, encoding: str = "utf-8") -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    with open(tmp, "w", encoding=encoding, newline="\n") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)
    return p


def write_json_atomic(path: os.PathLike | str, obj: Any) -> Path:
    return write_text_atomic(path, json.dumps(obj, ensure_ascii=False, indent=2))


def read_json(path: os.PathLike | str, default: Any = None) -> Any:
    p = Path(path)
    if not p.exists():
        return default
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return default


class JsonlWriter:
    """追加式 JSONL，自带锁与 flush。用 state.jsonl 做块级断点记录。"""

    def __init__(self, path: os.PathLike | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def append(self, obj: Dict[str, Any]) -> None:
        line = json.dumps(obj, ensure_ascii=False)
        with self._lock:
            with open(self.path, "a", encoding="utf-8", newline="\n") as f:
                f.write(line + "\n")


def read_jsonl(path: os.PathLike | str) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    out: List[Dict[str, Any]] = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


# ============================================================ 小工具

def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def human_dur(sec: float) -> str:
    if sec < 60:
        return f"{sec:.1f}s"
    m, s = divmod(int(sec), 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


def safe_stem(name: str) -> str:
    """把文件名压成安全的目录名（去掉扩展名与非法字符）。"""
    stem = Path(name).stem
    stem = re.sub(r'[\\/:*?"<>|\r\n\t]+', "_", stem).strip(" .")
    return stem[:120] or "untitled"


def normalize_ws(s: str) -> str:
    s = unicodedata.normalize("NFC", s)
    return re.sub(r"[ \t]+", " ", s)


# ============================================================ 配置

#: 质检阈值（validate.py 使用）
DEFAULT_THRESHOLDS: Dict[str, Any] = {
    "length_ratio_min": 0.18,      # 译/原 字符数比 下限（低于则 failed）
    "length_ratio_max": 3.2,       # 上限
    "length_ratio_warn": (0.30, 2.2),
    "kana_residue_max": 0.06,      # 译文中假名占比上限（高于则 failed）
    "kana_residue_warn": 0.02,
    "line_count_warn_ratio": 0.20,  # 行数偏差 > 20% warn
    "line_count_fail_ratio": 0.45,  # 行数偏差 > 45% fail
    "same_as_src_ratio": 0.85,     # 与原文相似度高于此值 → failed
    "repeat_line_max": 3,          # 连续相同行 > 3 → warn
    "unique_char_min": 0.06,       # 去重字符率低于此值且长度>200 → failed
}


@dataclass
class Config:
    """一次翻译任务的全部可调参数（可存成 job.json 复跑）。"""

    doc_memory: bool = False          # 长文记忆（实验）：翻译前预扫原文，生成文档简报+分场台账
    memory_scene: int = 8             # 每多少块做一次分场台账

    out_flat: bool = False            # 输出去扁平化？False=out/文件名/ ，True=out 目录本身即该书目录

    # --- 服务 ---
    server_cmd: str = ""
    server_start_timeout: float = 240.0
    port_base: int = 9080
    gpus: tuple = (0,)
    workers: int = 1                  # 并发 worker 数（通常 = len(gpus)）
    model_name: str = ""              # 留空则从 /v1/models 取

    # --- 远程 API（填了 api_base 就不启本地服务）---
    api_base: str = ""                # 如 https://api.deepseek.com/v1、http://127.0.0.1:19092/v1
    api_key: str = ""                 # 留空则用环境变量 GT_API_KEY

    # --- 中控（大模型，可选）：只做判断与工具调用，不翻译 ---
    director_base: str = ""           # 如 http://127.0.0.1:19092/v1（Huihui-Qwen3.8-27B）
    director_model: str = ""          # 如 qwen3.8-27b；留空取 /models 第一个
    director_key: str = ""            # 本地引擎不需要
    director_terms: bool = True       # 专名定名走中控
    director_review: bool = True      # degraded/failed 块交中控复核
    director_proofread: bool = False  # 默认不做改稿（实测性价比低）；只保留复核
    # 需要改稿时显式开：--proofread flagged / all，或 GUI 选「只修可疑块」
    director_proofread_all: bool = False   # 全量校对（每一块都过一遍，慢）
    director_proofread_tries: int = 2      # 单块最多让中控改几次稿（防烧 token）
    translator_no_think: bool = True       # 译者请求带 enable_thinking=false（sakura/Qwen3 系省 token）
    clean_ai: bool = False                  # 清洗时让中控复核可疑行（删/留）
    line_mode_fallback: bool = False        # 行数对不上时逐行重译（保证 1:1，慢）
    proofread_patch: bool = True           # 校对只交行级补丁（False=整稿重写，旧行为）
    audit_ratio: float = 0.0               # 抽检比例：对"质检通过"的块也随机送校对（0=不抽检）
    director_veto: bool = True        # 允许中控否决“假专名”的遮蔽
    director_timeout: float = 240.0
    director_no_think: bool = False       # 给 vLLM 发 enable_thinking=false（Qwen3 系中控）

    # --- 提示词与采样 ---
    prompt: str = ""
    system_in_user: bool = False      # 指令并入 user 消息（sakura 等对 system 角色会复读的模型）
    temperature: float = 0.3
    top_p: float = 0.9
    repeat_penalty: float = 1.05
    max_tokens_per_char: float = 1.8
    max_tokens_cap: int = 2048
    request_timeout: float = 300.0    # 单块总超时
    stall_timeout: float = 90.0       # 无数据流超时（看门狗）
    retries: int = 3
    retry_split: bool = True          # 末次重试把块拆半
    erase_slot_between: bool = True   # 每块后清 KV

    # --- 清洗 / 切块 ---
    do_clean: bool = True
    lang: str = "auto"                # auto/ja/zh
    merge_wrapped_lines: Optional[bool] = None   # None=按语种自动
    drop_boilerplate: bool = True
    chunk_chars: int = 320
    chunk_hard_max: int = 520
    context_chars: int = 0            # 上文锚点长度，0=不给（给了会略增 token；80~120 可提升人称连贯）上文
    keep_scene_break: bool = True

    # --- 模式：块级 / 单元级 ---
    mode: str = "chunk"               # chunk=按块翻 / unit=完整句单元（≦20 token）
    unit_tokens: int = 20             # 单条单元 token 上限（完整句贪心打包，绝不切句）
    unit_chars_per_token: float = 1.6  # 无 tiktoken 时的估算比例
    unit_pack: int = 8                # 一个请求装多少条（1=逐条发，会慢数倍）
    unit_max_chars: int = 700         # 一个请求的字数上限（防止长句撑爆）
    unit_context: int = 80            # 单元模式的上文字数（0=不给）
    unit_check: bool = True           # 译完交中控逐批定性（ok/order/wrong）
    unit_check_pack: int = 8          # 中控一次判多少条
    unit_check_rounds: int = 2        # 最多几轮定性（含重译后复检）
    final_check: str = "all"          # 全文终检：off / fixed=只查改过条目 / all=全量
    slots: int = 1                    # 内置启动器给每个服务开多少 slot（--parallel）

    # --- 术语 / 占位符 ---
    mask_terms: bool = True
    mask_katakana: bool = False        # 是否遮蔽纯片假名专名（默认关：拟声词/外来词噪音太大）
    term_min_freq: int = 2
    placeholder_style: str = "bracket"   # bracket=⟦12⟧ double=[[12]] hash=【#12#】 brace={12}
    translate_terms: bool = True
    terms_batch: int = 25

    # --- 质检 / 失败语义 ---
    strict: bool = True               # True: 有 failed 块则文件不出成品
    allow_degraded: bool = True       # degraded 块是否计入成品
    thresholds: Dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_THRESHOLDS))

    # --- 其它 ---
    regex_rules: str = ""
    log_level: str = "info"

    # -------------------------------------------------- 序列化
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["gpus"] = list(self.gpus)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Config":
        known = {f.name for f in fields(cls)}
        kw: Dict[str, Any] = {k: v for k, v in (d or {}).items() if k in known}
        if "gpus" in kw and kw["gpus"] is not None:
            kw["gpus"] = tuple(int(x) for x in kw["gpus"])
        thr = kw.pop("thresholds", None)
        cfg = cls(**kw)
        if thr:
            cfg.thresholds.update(thr)
        return cfg

    @classmethod
    def load(cls, path: os.PathLike | str) -> "Config":
        return cls.from_dict(read_json(path, {}) or {})

    def save(self, path: os.PathLike | str) -> Path:
        return write_json_atomic(path, self.to_dict())

    # -------------------------------------------------- 派生
    def segment_fingerprint(self) -> str:
        """切块相关参数的指纹：参数变了就必须重新切块（续传失效）。"""
        keys = ["chunk_chars", "chunk_hard_max", "context_chars", "lang",
                "do_clean", "merge_wrapped_lines", "keep_scene_break",
                "drop_boilerplate", "placeholder_style", "mask_terms",
                "mask_katakana", "term_min_freq"]
        payload = {k: getattr(self, k) for k in keys}
        return hashlib.sha1(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]

    def resolve_gpus(self) -> List[int]:
        gpus = [int(g) for g in (self.gpus or (0,))]
        return gpus or [0]

    def api_key_effective(self) -> str:
        return (self.api_key or os.environ.get("GT_API_KEY", "") or "").strip()

    @property
    def use_api(self) -> bool:
        return bool((self.api_base or "").strip())

    @property
    def use_director(self) -> bool:
        return bool((self.director_base or "").strip())


#: 与旧 配置库.json 兼容的读取
def load_legacy_config_db(path: os.PathLike | str) -> Dict[str, Any]:
    return read_json(path, {}) or {}
