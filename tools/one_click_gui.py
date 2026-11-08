# -*- coding: utf-8 -*-
"""一键翻译 GUI：选书 → 自动检查/拉起两个 vLLM → 翻译 → 复核 → 重译 → 合并。

双击 bat 打开这个窗口；也可以 `python tools/one_click_gui.py`。

设计：窗口里只放必需的东西（选书、并发、每块字数、开始），其余全自动。
服务已在跑就直接用；没跑就先拉起并等端口就绪（各自一个最小化窗口）。
"""
from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv" / "Scripts" / "python.exe"
if not PY.exists():
    PY = Path(sys.executable)

CHILD_ENV = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")

SAKURA = {
    "name": "sakura-14B（译者）",
    "port": 8090,
    "gpu": 0,
    "cmd": ("/root/venvs/sakura/bin/vllm serve /mnt/g/aimodels/"
            "Sakura-GalTransl-14B-v3.8-GPTQ-Int4 --served-model-name sakura-galtransl-14b "
            "--max-model-len 16384 --gpu-memory-utilization 0.9 --host 0.0.0.0 --port 8090"),
}
JUDGE = {
    "name": "Qwen3.5-9B-AWQ（复核）",
    "port": 9092,
    "gpu": 1,
    "cmd": ("/root/venvs/sakura/bin/vllm serve /mnt/g/aimodels/"
            "Qwen35-9B-DefiantFable-W4A16-AWQ --served-model-name qwen35-9b-awq "
            "--max-model-len 16384 --gpu-memory-utilization 0.9 --enable-auto-tool-choice "
            "--tool-call-parser hermes --host 0.0.0.0 --port 9092"),
}


def port_ok(port: int, timeout: float = 2.0) -> bool:
    import urllib.request
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.files: list[Path] = []
        self.q: "queue.Queue[str]" = queue.Queue()
        self.busy = False
        self.last_out = None

        root.title("日文小说翻译 · sakura-14B(GPU0) + Qwen3.5-9B 复核(GPU1)")
        root.geometry("880x620")

        top = ttk.Frame(root, padding=8)
        top.pack(fill=tk.X)

        ttk.Button(top, text="选择书籍（可多选）…", command=self.pick).pack(side=tk.LEFT)
        self.lbl_files = ttk.Label(top, text="未选择文件", foreground="#666")
        self.lbl_files.pack(side=tk.LEFT, padx=10)

        box = ttk.LabelFrame(root, text="参数", padding=8)
        box.pack(fill=tk.X, padx=8)
        ttk.Label(box, text="并发").grid(row=0, column=0, sticky="w")
        self.var_workers = tk.IntVar(value=3)
        ttk.Spinbox(box, from_=1, to=8, width=5, textvariable=self.var_workers).grid(row=0, column=1, padx=(2, 16))
        ttk.Label(box, text="每块字数").grid(row=0, column=2, sticky="w")
        self.var_chunk = tk.IntVar(value=400)
        ttk.Spinbox(box, from_=200, to=800, increment=50, width=6,
                    textvariable=self.var_chunk).grid(row=0, column=3, padx=(2, 16))
        ttk.Label(box, text="（并发建议 2~3；块越小越不容易并句）", foreground="#666").grid(row=0, column=4, sticky="w")

        svc = ttk.LabelFrame(root, text="服务", padding=8)
        svc.pack(fill=tk.X, padx=8, pady=6)
        self.svc_labels = {}
        for i, s in enumerate((SAKURA, JUDGE)):
            ttk.Label(svc, text=f"{s['name']}  GPU{s['gpu']} :{s['port']}").grid(row=i, column=0, sticky="w")
            lb = ttk.Label(svc, text="…", width=10)
            lb.grid(row=i, column=1, padx=8)
            self.svc_labels[s["port"]] = lb
        ttk.Button(svc, text="重新检查", command=self.check_services).grid(row=0, column=2, padx=6)

        run = ttk.Frame(root, padding=8)
        run.pack(fill=tk.X)
        self.btn = ttk.Button(run, text="▶ 开始翻译", command=self.start)
        self.btn.pack(side=tk.LEFT)
        self.progress = ttk.Progressbar(run, mode="determinate", length=320)
        self.progress.pack(side=tk.LEFT, padx=12)
        self.lbl_stage = ttk.Label(run, text="")
        self.lbl_stage.pack(side=tk.LEFT, padx=6)
        ttk.Button(run, text="打开输出目录", command=self.open_out).pack(side=tk.LEFT, padx=6)

        self.log = tk.Text(root, height=20, wrap="word")
        self.log.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))
        self.write("就绪：选好书后点「开始翻译」。服务没起会自动拉起（每个约 1~3 分钟）。\n")

        self.check_services()
        self.root.after(200, self.drain)

    # ---------------------------------------------------------- 输出
    def write(self, s: str) -> None:
        self.log.insert(tk.END, s)
        self.log.see(tk.END)

    def drain(self) -> None:
        try:
            while True:
                self.write(self.q.get_nowait())
        except queue.Empty:
            pass
        self.root.after(200, self.drain)

    def logln(self, s: str) -> None:
        self.q.put(s if s.endswith("\n") else s + "\n")

    # ---------------------------------------------------------- 交互
    def pick(self) -> None:
        paths = filedialog.askopenfilenames(title="选择要翻译的日文 txt（可多选）",
                                           filetypes=[("文本", "*.txt"), ("全部", "*.*")])
        if not paths:
            return
        self.files = [Path(p) for p in paths]
        names = "、".join(p.name for p in self.files[:3])
        more = f" 等 {len(self.files)} 个" if len(self.files) > 3 else ""
        self.lbl_files.config(text=names + more)

    def check_services(self) -> None:
        def work():
            for s in (SAKURA, JUDGE):
                ok = port_ok(s["port"])
                self.root.after(0, lambda p=s["port"], o=ok: self.svc_labels[p].config(
                    text="在线" if o else "未启动", foreground="#0a0" if o else "#c00"))
        threading.Thread(target=work, daemon=True).start()

    def ensure_service(self, s: dict) -> bool:
        if port_ok(s["port"]):
            self.logln(f"  {s['name']} 已在线，直接用（:{s['port']}）")
            return True
        self.logln(f"  {s['name']} 未启动，拉起中…（GPU{s['gpu']}，首次约 1~3 分钟）")
        subprocess.Popen(
            ["wsl.exe", "-d", "ubuntu-2404", "--", "bash", "-c",
             f"CUDA_VISIBLE_DEVICES={s['gpu']} VLLM_USE_FLASHINFER_SAMPLER=0 {s['cmd']}"],
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        for i in range(90):
            time.sleep(4)
            if port_ok(s["port"]):
                self.logln(f"  {s['name']} 就绪")
                return True
            if i % 5 == 4:
                self.logln(f"    等待中… {(i + 1) * 4}s")
        self.logln(f"  !! {s['name']} 等待超时")
        return False

    def start(self) -> None:
        if self.busy:
            return
        if not self.files:
            messagebox.showinfo("提示", "先选书")
            return
        self.busy = True
        self.btn.config(state="disabled")
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        try:
            self.logln("\n=== 检查服务 ===")
            if not self.ensure_service(SAKURA) or not self.ensure_service(JUDGE):
                self.logln("服务没起来，任务中止。")
                return
            self.check_services()
            n = len(self.files)
            for i, f in enumerate(self.files, 1):
                self.logln(f"\n=== [{i}/{n}] {f.name} ===")
                self.progress["value"] = (i - 1) / n * 100
                cmd = [str(PY), str(ROOT / "tools" / "run_all.py"), "--src", str(f),
                       "--workers", str(self.var_workers.get()),
                       "--chunk", str(self.var_chunk.get())]
                p = subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True,
                                     encoding="utf-8", errors="replace", env=CHILD_ENV)
                total_blocks = 0
                done_blocks = 0
                for line in p.stdout:            # type: ignore[union-attr]
                    line = line.rstrip()
                    if not line:
                        continue
                    import re as _re
                    m = _re.search(r"切块:\s*(\d+)\s*块", line)
                    if m:
                        total_blocks = int(m.group(1))
                    if line.strip().startswith("输出"):
                        self.last_out = line.split(None, 1)[-1].strip()
                    if "· 块" in line:
                        done_blocks += 1
                        if total_blocks:
                            pct = (i - 1 + done_blocks / total_blocks) / n * 100
                            self.root.after(0, lambda v=pct: self.progress.configure(value=v))
                            self.root.after(0, lambda d=done_blocks, t=total_blocks: self.lbl_stage.config(
                                text=f"翻译中 {d}/{t} 块"))
                        continue
                    if line.strip().startswith("["):
                        continue
                    self.logln("  " + line[:160])
                    low = line.strip()
                    if "批量复核" in low or "打回重译" in low or "汇总" in low or "批量翻译" in low:
                        self.root.after(0, lambda t=low[:40]: self.lbl_stage.config(text=t))
                p.wait()
            self.progress["value"] = 100
            self.logln("\n=== 全部完成 ===")
        finally:
            self.busy = False
            self.root.after(0, lambda: self.btn.config(state="normal"))

    def open_out(self) -> None:
        d = getattr(self, "last_out", None) or ROOT
        subprocess.Popen(["explorer", str(d)])


LOCK = ROOT / ".gui.lock"


def _already_running() -> bool:
    try:
        if not LOCK.exists():
            return False
        pid = int(LOCK.read_text().strip() or 0)
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"], capture_output=True,
                             text=True, encoding="gbk", errors="replace").stdout
        return str(pid) in out
    except Exception:
        return False


def main() -> int:
    if _already_running():
        tk.Tk().withdraw()
        messagebox.showinfo("已经在运行", "已经开着一个翻译窗口了，请用那个；多开会重复翻同一本书")