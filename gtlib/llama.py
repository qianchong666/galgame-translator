# -*- coding: utf-8 -*-
"""llama-server 进程管理 + HTTP 客户端。

旧版 step3 的坑（逐条修掉）
---------------------------
* `kill_llama_processes()` 用 `taskkill /im llama-server.exe`——会把别的窗口/别人正在跑的
  服务一起杀掉。现在只按**自己 spawn 的 PID** 结束，`/T` 连子进程一起收。
* 硬编码 `G:\\ai-work\\...` 模型路径，换机器就废。现在命令模板来自配置（可来自旧 配置库.json）。
* `temperature 0.8` + 无 `max_tokens` + `--parallel 1` + `--cache-type-k/v q4_0`：
  翻译任务要确定性、要长度上限、要被复用的 KV。现在默认 0.3 / 有上限 / 可配 q8_0。
* 没有卡死判定（watchdog 只看字节增量），HTTP 层也没有 read timeout 分层。现在
  connect 超时与 stall 超时分离，stall 一到就判定卡死并重启服务。
* SSE 解析不校验 finish_reason，截断的译文被当成正常结果写盘。现在记录 finish_reason，
  `length` 会触发"续写"分片重试（见 job.py）。
"""
from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

try:
    import requests
except Exception:                                     # pragma: no cover
    requests = None

from .core import Config, Logger


class LlamaError(RuntimeError):
    pass


# ------------------------------------------------------------------ 命令行
def parse_cmd(cmd: str) -> List[str]:
    """按 Windows 习惯切分命令行（支持双引号包住的路径）。"""
    out: List[str] = []
    buf: List[str] = []
    in_q = False
    for ch in (cmd or "").strip():
        if ch == '"':
            in_q = not in_q
        elif ch.isspace() and not in_q:
            if buf:
                out.append("".join(buf))
                buf = []
        else:
            buf.append(ch)
    if buf:
        out.append("".join(buf))
    return out


def _strip_flag(parts: List[str], flag: str) -> List[str]:
    out = list(parts)
    i = 0
    while i < len(out):
        if out[i] == flag:
            del out[i:i + 2]
            break
        i += 1
    return out


def build_server_cmd(base_cmd: str, port: int, gpu: Optional[int] = None,
                     extra: Optional[Sequence[str]] = None) -> List[str]:
    parts = parse_cmd(base_cmd)
    if not parts:
        raise LlamaError("server_cmd 为空：请提供 llama-server 启动命令")
    parts = _strip_flag(parts, "--port")
    parts += ["--port", str(port)]
    if gpu is not None:
        parts = _strip_flag(parts, "--main-gpu")
        parts += ["--main-gpu", str(gpu)]
    if extra:
        parts += list(extra)
    return parts


def port_free(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False


def normalize_api_base(base: str) -> str:
    """把用户填的 URL 规整成可用的 OpenAI 端点。

    * `https://api.deepseek.com`      → `https://api.deepseek.com/v1`
    * `https://api.deepseek.com/v1/`  → 去尾斜杠
    * `http://127.0.0.1:9080`         → 不动（llama-server 两种路径都支持）
    """
    b = (base or "").strip().rstrip("/")
    if not b:
        return b
    if not re.match(r"^https?://", b, re.I):
        b = "http://" + b
    if re.search(r"/v\d+$", b) or "/v1/" in b:
        return b
    m = re.match(r"^(https?://[^/]+)(/.*)?$", b, re.I)
    host = m.group(1) if m else b
    path = (m.group(2) or "").rstrip("/") if m else ""
    local = re.search(r"(127\.0\.0\.1|localhost|0\.0\.0\.0)", host, re.I) is not None
    if local and path == "":
        return host
    return host + path + "/v1"


# ------------------------------------------------------------------ 服务
class LlamaServer:
    """一个 llama-server 实例，或一个远程 OpenAI 兼容端点。

    只结束自己启动的那个 PID；远程端点不做任何进程操作。
    """

    def __init__(self, cmd: Sequence[str], port: int, log: Logger,
                 env: Optional[Dict[str, str]] = None,
                 startup_timeout: float = 240.0,
                 log_path: Optional[os.PathLike | str] = None,
                 name: str = "llama",
                 base_url: Optional[str] = None,
                 api_key: str = ""):
        self.cmd = list(cmd)
        self.port = int(port)
        self.log = log
        self.env = dict(env or {})
        self.startup_timeout = startup_timeout
        self.name = name
        self.proc: Optional[subprocess.Popen] = None
        self.owned = False            # 端口本来就被占用 → 不是我们的，不动它
        self._log_path = Path(log_path) if log_path else None
        self._tail: List[str] = []
        self._tail_lock = threading.Lock()
        self._base_url = normalize_api_base(base_url) if base_url else None
        self.api_key = api_key or ""

    @classmethod
    def remote(cls, base_url: str, log: Logger, api_key: str = "",
               name: str = "api") -> "LlamaServer":
        """连接一个已存在的 OpenAI 兼容端点（DeepSeek / OpenAI / vLLM / LM Studio…）。"""
        s = cls(cmd=[], port=0, log=log, name=name, base_url=base_url, api_key=api_key)
        s.owned = False
        return s

    @property
    def is_remote(self) -> bool:
        return self._base_url is not None

    def auth_headers(self) -> Dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    # -- 属性 --------------------------------------------------
    @property
    def base_url(self) -> str:
        if self._base_url:
            return self._base_url
        return f"http://127.0.0.1:{self.port}"

    @property
    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def tail(self, n: int = 25) -> str:
        with self._tail_lock:
            return "\n".join(self._tail[-n:])

    # -- 生命周期 ----------------------------------------------
    def start(self) -> bool:
        if self.is_remote:
            if not self.health(10.0):
                raise LlamaError(f"连不上 API 端点 {self.base_url}（检查 base_url / api_key / 网络）")
            self.log.info(f"[{self.name}] API 就绪: {self.base_url}")
            return True
        if not port_free(self.port):
            self.log.warn(f"[{self.name}] 端口 {self.port} 已被占用，复用已有服务（退出时不关闭）")
            self.owned = False
            return self.wait_ready(self.startup_timeout)

        self.log.info(f"[{self.name}] 启动: {' '.join(self.cmd[:3])} ... port={self.port}")
        env = os.environ.copy()
        env.update({k: str(v) for k, v in self.env.items()})
        if self._log_path:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.proc = subprocess.Popen(
                self.cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=env,
                creationflags=(subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0),
            )
        except Exception as e:
            raise LlamaError(f"启动失败: {e}") from e
        self.owned = True
        threading.Thread(target=self._pump, args=(self.proc,), daemon=True).start()
        ok = self.wait_ready(self.startup_timeout)
        if not ok:
            self.stop()
            raise LlamaError(
                f"llama-server 在 {self.startup_timeout:.0f}s 内未就绪；日志尾部：\n{self.tail(25)}")
        return True

    def _pump(self, proc: subprocess.Popen) -> None:
        f = None
        if self._log_path:
            try:
                f = open(self._log_path, "a", encoding="utf-8", buffering=1)
            except Exception:
                f = None
        try:
            for line in iter(proc.stdout.readline, ""):
                if not line:
                    break
                s = line.rstrip()
                with self._tail_lock:
                    self._tail.append(s)
                    if len(self._tail) > 400:
                        del self._tail[:200]
                if f:
                    try:
                        f.write(s + "\n")
                    except Exception:
                        pass
                if "error" in s.lower() and self.log:
                    self.log.debug(f"[{self.name}] {s}")
        except Exception:
            pass
        finally:
            if f:
                f.close()

    def health(self, timeout: float = 3.0) -> bool:
        if requests is None:
            raise LlamaError("缺少 requests 库")
        if self.is_remote:
            # 远程端点通常没有 /health，用 /models 探活；401/404 也算“活着”
            for path in ("/models", "/health"):
                try:
                    r = requests.get(self.base_url + path,
                                     headers=self.auth_headers(), timeout=timeout)
                    if r.status_code < 500:
                        return True
                except Exception:
                    continue
            return False
        try:
            r = requests.get(self.base_url + "/health", timeout=timeout)
            return r.status_code == 200
        except Exception:
            return False

    def wait_ready(self, timeout: float = 240.0) -> bool:
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.health(2.0):
                self.log.info(f"[{self.name}] 就绪 ({time.time() - t0:.1f}s)")
                return True
            if self.owned and self.proc is not None and self.proc.poll() is not None:
                raise LlamaError(f"llama-server 提前退出 (code={self.proc.returncode})；"
                                 f"日志尾部：\n{self.tail(25)}")
            time.sleep(0.6)
        return False

    def stop(self) -> None:
        proc = self.proc
        self.proc = None
        if proc is None or not self.owned:
            return
        try:
            if proc.poll() is None:
                if sys.platform == "win32":
                    subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                                   capture_output=True, timeout=15)
                else:
                    proc.terminate()
                try:
                    proc.wait(timeout=8)
                except Exception:
                    proc.kill()
            self.log.info(f"[{self.name}] 已关闭 (PID {proc.pid})")
        except Exception as e:
            self.log.warn(f"[{self.name}] 关闭异常: {e}")
        finally:
            self.owned = False

    def restart(self) -> bool:
        if not self.owned:
            self.log.warn(f"[{self.name}] 外部服务，无法重启，等待自行恢复…")
            return self.wait_ready(min(30.0, self.startup_timeout))
        self.log.warn(f"[{self.name}] 重启服务…")
        self.stop()
        for i in range(3):
            if port_free(self.port):
                break
            time.sleep(1.0)
        return self.start()

    def erase_slot(self, slot: int = 0) -> None:
        """清 KV 缓存（不同 build 的接口不一样，尽力而为）。"""
        if requests is None:
            return
        for url, params in (
            (f"{self.base_url}/slots/{slot}", {"action": "erase"}),
            (f"{self.base_url}/slots", {"action": "erase", "id_slot": slot}),
        ):
            try:
                r = requests.post(url, params=params, timeout=4)
                if r.status_code < 400:
                    return
            except Exception:
                continue

    def __enter__(self) -> "LlamaServer":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()


# ------------------------------------------------------------------ 客户端
@dataclass
class ChatResult:
    text: str = ""
    elapsed: float = 0.0
    prompt_tokens: int = 0
    out_tokens: int = 0
    finish: str = ""
    attempts: int = 1

    @property
    def truncated(self) -> bool:
        return self.finish == "length"


class LlamaClient:
    """OpenAI 兼容的 chat 客户端：流式读取 + 双超时 + 重试 + 服务自愈。"""

    def __init__(self, server: LlamaServer, log: Logger, cfg: Config,
                 cancel: Optional[threading.Event] = None,
                 system_prompt: str = ""):
        self.server = server
        self.log = log
        self.cfg = cfg
        self.cancel = cancel or threading.Event()
        self.system_prompt = system_prompt
        self.model = cfg.model_name or ""
        self._stats = {"requests": 0, "tokens": 0, "seconds": 0.0}

    # -- 模型名 -------------------------------------------------
    def resolve_model(self) -> str:
        if self.model:
            return self.model
        if self.server.is_remote:
            names = list_models(self.server)
            if not names:
                raise LlamaError("无法从 /models 取到模型名，请手动填写模型")
            self.model = names[0]
            return self.model
        for path in ("/v1/models", "/models"):
            try:
                r = requests.get(self.server.base_url + path, timeout=5)
                self.model = r.json()["data"][0]["id"]
                return self.model
            except Exception:
                continue
        self.model = "local"
        return self.model

    # -- 主接口 -------------------------------------------------
    def chat(self, user: str, system: Optional[str] = None,
             max_tokens: Optional[int] = None,
             temperature: Optional[float] = None,
             retries: Optional[int] = None) -> ChatResult:
        system = self.system_prompt if system is None else system
        retries = self.cfg.retries if retries is None else retries
        last_err: Optional[Exception] = None
        for attempt in range(1, retries + 1):
            if self.cancel.is_set():
                raise LlamaError("已取消")
            try:
                res = self._request(system, user, max_tokens,
                                    (temperature if temperature is not None
                                     else self.cfg.temperature) + (0.05 * (attempt - 1)))
                res.attempts = attempt
                return res
            except Exception as e:
                last_err = e
                self.log.warn(f"  请求失败 第{attempt}/{retries}次: {e}")
                if attempt == retries:
                    break
                # 服务死了就重启，否则退避重试
                if not self.server.health(2.0):
                    try:
                        self.server.restart()
                    except Exception as e2:
                        self.log.error(f"  重启失败: {e2}")
                else:
                    time.sleep(min(8.0, 1.5 * attempt))
        raise LlamaError(f"重试 {retries} 次仍失败: {last_err}")

    def _request(self, system: str, user: str, max_tokens: Optional[int],
                 temperature: float) -> ChatResult:
        if requests is None:
            raise LlamaError("缺少 requests 库")
        if max_tokens is None:
            max_tokens = int(min(self.cfg.max_tokens_cap,
                                 max(96, len(user) * self.cfg.max_tokens_per_char)))
        remote = self.server.is_remote
        if getattr(self.cfg, "system_in_user", False) and system:
            user = system + "\n\n" + user
            system = ""
        payload = {
            "model": self.resolve_model(),
            "messages": ([{"role": "system", "content": system}] if system else []) +
                        [{"role": "user", "content": user}],
            "temperature": temperature,
            "top_p": self.cfg.top_p,
            "max_tokens": max_tokens,
            "stream": True,
        }
        if not remote:
            # llama.cpp 专属字段，云端 API 可能因未知参数报 400
            payload["repeat_penalty"] = self.cfg.repeat_penalty
            payload["cache_prompt"] = False
        elif getattr(self.cfg, "translator_no_think", True):
            # vLLM/OpenAI 兼容端：Qwen3 系模板支持关思考，少烧 token
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        url = self.server.base_url + "/v1/chat/completions"
        if remote and self.server.base_url.endswith("/v1"):
            url = self.server.base_url + "/chat/completions"
        t0 = time.time()
        got_any = False
        parts: List[str] = []
        finish = ""
        usage = {}
        prompt_n = out_n = 0

        # 429/503 限流：尊重 Retry-After
        resp = None
        for attempt in range(3):
            resp = requests.post(url, json=payload, timeout=(15, self.cfg.stall_timeout),
                                 stream=True, headers=self.server.auth_headers())
            if resp.status_code in (429, 503) and attempt < 2:
                wait = 5.0 * (attempt + 1)
                try:
                    wait = min(60.0, float(resp.headers.get("Retry-After") or wait))
                except Exception:
                    pass
                resp.close()
                self.log.warn(f"  HTTP {resp.status_code} 限流，等 {wait:.0f}s 重试")
                time.sleep(wait)
                continue
            break

        with resp:
            if resp.status_code >= 400:
                body = resp.text[:400]
                raise LlamaError(f"HTTP {resp.status_code}: {body}")
            for raw in resp.iter_lines(decode_unicode=False):
                if self.cancel.is_set():
                    raise LlamaError("已取消")
                if time.time() - t0 > self.cfg.request_timeout:
                    raise LlamaError(f"总耗时超过 {self.cfg.request_timeout:.0f}s")
                if not raw:
                    continue
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except Exception:
                    continue
                ch = (obj.get("choices") or [{}])[0]
                delta = ch.get("delta") or {}
                piece = delta.get("content") or ""
                if piece:
                    got_any = True
                    parts.append(piece)
                    print_rate = len(parts) % 40 == 0
                    if print_rate:
                        self.log.debug(f"  … {len(''.join(parts))} chars")
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
                if obj.get("usage"):
                    usage = obj["usage"]
                if obj.get("timings"):
                    prompt_n = int(obj["timings"].get("prompt_n") or 0)
                    out_n = int(obj["timings"].get("predicted_n") or 0)

        text = "".join(parts)
        if not text.strip():
            raise LlamaError("返回为空" if got_any else "没有任何数据流")
        elapsed = time.time() - t0
        self._stats["requests"] += 1
        self._stats["tokens"] += out_n
        self._stats["seconds"] += elapsed
        return ChatResult(text=text, elapsed=elapsed,
                          prompt_tokens=prompt_n or int(usage.get("prompt_tokens") or 0),
                          out_tokens=out_n or int(usage.get("completion_tokens") or 0),
                          finish=finish)

    # -- 统计 ---------------------------------------------------
    @property
    def stats(self) -> Dict[str, float]:
        return dict(self._stats)


def list_models(server: LlamaServer, timeout: float = 12.0) -> List[str]:
    """列出端点上的模型名（OpenAI 兼容 /models）。"""
    if requests is None:
        return []
    for path in ("/models", "/v1/models"):
        try:
            r = requests.get(server.base_url + path, headers=server.auth_headers(),
                             timeout=timeout)
            if r.status_code >= 400:
                continue
            data = (r.json() or {}).get("data") or []
            names = [str(d.get("id")) for d in data
                     if isinstance(d, dict) and d.get("id")]
            if names:
                return names
        except Exception:
            continue
    return []


def make_servers(cfg: Config, log: Logger,
                 log_dir: Optional[os.PathLike | str] = None):
    """按配置返回 (servers, stop_fn)。

    * cfg.use_api（填了 api_base）→ 直连远程端点，不启本地进程，workers 个并发连接
    * 否则 → 按 GPU 列表拉起本地 llama-server 池
    """
    if cfg.use_api:
        srv = LlamaServer.remote(cfg.api_base, log, api_key=cfg.api_key_effective())
        srv.start()
        n = max(1, int(cfg.workers or 1))
        return [srv] * n, srv.stop
    pool = ServerPool(cfg, log, log_dir)
    return pool.start(), pool.stop


# ------------------------------------------------------------------ 服务池
class ServerPool:
    """按 GPU 列表拉起若干服务，每个 worker 独占一个（避免抢 slot）。"""

    def __init__(self, cfg: Config, log: Logger, log_dir: Optional[os.PathLike | str] = None):
        self.cfg = cfg
        self.log = log
        self.log_dir = Path(log_dir) if log_dir else None
        self.servers: List[LlamaServer] = []

    def start(self) -> List[LlamaServer]:
        gpus = self.cfg.resolve_gpus()
        n = max(1, min(int(self.cfg.workers or 1), len(gpus)))
        self.servers = []
        for i in range(n):
            gpu = gpus[i]
            port = int(self.cfg.port_base) + i
            # 用 CUDA_VISIBLE_DEVICES 限定可见卡，不再额外传 --main-gpu：
            # 当可见卡只有一张时，--main-gpu 传原卡号会直接报错。
            cmd = build_server_cmd(self.cfg.server_cmd, port,
                                   extra=(["--parallel", str(int(self.cfg.slots))]
                                          if int(self.cfg.slots or 1) > 1 else None))
            env = {"CUDA_VISIBLE_DEVICES": str(gpu)} if gpu is not None else None
            logp = (self.log_dir / f"llama_{i}_port{port}.log") if self.log_dir else None
            srv = LlamaServer(cmd, port, self.log, env=env,
                              startup_timeout=self.cfg.server_start_timeout,
                              log_path=logp, name=f"gpu{gpu}")
            srv.start()
            self.servers.append(srv)
        return self.servers

    def stop(self) -> None:
        for s in self.servers:
            s.stop()
        self.servers = []

    def __enter__(self) -> "ServerPool":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()
