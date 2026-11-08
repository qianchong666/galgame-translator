# -*- coding: utf-8 -*-
"""一键流水线：翻译 → 批量复核 → 打回重译 → 逐行兜底 → 合并。

给 .bat 双击入口用；也可以直接命令行跑：
  python tools/run_all.py --src 原文.txt --workers 3 --chunk 400

设计口径（用户定的）：
  · 复核只判不写：判定 retry 的块打回给译者重译，不让中控代笔；
  · 第一遍不追求完美，把不合格的挑出来走第二遍；
  · 行数硬门（1:1），实在并句就 --line-mode 逐行兜底。
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv" / "Scripts" / "python.exe"
CHILD_ENV = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
if not PY.exists():
    PY = Path(sys.executable)


def run(args, title: str, stream: bool = False) -> str:
    print(f"\n===== {title} =====", flush=True)
    t0 = time.time()
    if stream:
        # GUI 要逐块进度：实时转发子进程输出，不过滤
        p = subprocess.Popen([str(PY), str(ROOT / "gt.py"), *args], cwd=str(ROOT),
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                             encoding="utf-8", errors="replace", env=CHILD_ENV)
        buf = []
        for line in p.stdout:            # type: ignore[union-attr]
            buf.append(line)
            print(line, end="", flush=True)
        p.wait()
        print(f"  （{time.time() - t0:.1f}s，退出码 {p.returncode}）", flush=True)
        return "".join(buf)
    p = subprocess.run([str(PY), str(ROOT / "gt.py"), *args], cwd=str(ROOT),
                       capture_output=True, text=True, encoding="utf-8", errors="replace",
                       env=CHILD_ENV)
    out = (p.stdout or "") + (p.stderr or "")
    tail = [ln for ln in out.splitlines() if ln.strip()][-6:]
    for ln in tail:
        print("  " + ln[:160], flush=True)
    print(f"  （{time.time() - t0:.1f}s，退出码 {p.returncode}）", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="一键翻译流水线")
    ap.add_argument("--src", required=True, help="原文 txt")
    ap.add_argument("-o", "--out", help="输出目录（默认：源文件所在目录下的 out/任务时间_文件名）")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--chunk", type=int, default=400)
    ap.add_argument("--doc-memory", action="store_true",
                    help="长文记忆（实验）：翻译前预扫原文，生成文档简报+分场台账")
    ap.add_argument("--memory-scene", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=3, help="复核-重译最多几轮")
    ap.add_argument("--api-base", default="http://127.0.0.1:8090/v1")
    ap.add_argument("--model", default="sakura-galtransl-14b")
    ap.add_argument("--director-base", default="http://127.0.0.1:9092/v1")
    ap.add_argument("--director-model", default="qwen35-9b-awq")
    ap.add_argument("--audit", type=float, default=1.0,
                    help="复核覆盖面：1.0=全批复核（默认）；0=只复核质检报警的块")
    a = ap.parse_args()

    src = Path(a.src)
    if not src.exists():
        print(f"!! 原文件不存在：{src}")
        return 2
    # 默认输出：源文件所在目录下的 out/<任务时间>_<文件名>/
    if a.out:
        out = Path(a.out)
    else:
        ts = time.strftime("%Y%m%d_%H%M")
        out = src.parent / "out" / f"{ts}_{src.stem}"

    mem_flags = (["--doc-memory", "--memory-scene", str(a.memory_scene)]
                 if a.doc_memory else [])
    common = [*mem_flags, "--api-base", a.api_base, "--model", a.model,
              "--director-base", a.director_base, "--director-model", a.director_model,
              "--director-no-think", "--clean-ai", "--chunk", str(a.chunk), "--workers", str(a.workers),
              "--style", "hash", "--system-in-user", "--temp", "0.1"]
    print("=" * 66)
    print(" 日文小说翻译（vLLM 双卡）")
    print(f"  译者   {a.model}  @ {a.api_base}")
    print(f"  复核   {a.director_model}  @ {a.director_base}")
    print(f"  并发   {a.workers}    每块 {a.chunk} 字（硬上限 {int(a.chunk * 1.3)}）")
    print(f"  输出   {out}")
    print("=" * 66, flush=True)

    # ① 批量翻译（先不做块内复核，让翻译跑满）
    run(["translate", str(src), "-o", str(out), *common, "--line-mode", "--no-director-review", "--out-flat"],
        "① 批量翻译", stream=True)

    # ② 复核 → 打回重译（循环到没有打回为止）
    for rnd in range(1, max(1, a.rounds) + 1):
        txt = run(["review", str(out), "--director-base", a.director_base,
                   "--director-model", a.director_model, "--director-no-think",
                   "--workers", str(a.workers), "--audit-ratio", str(a.audit),
                   "--tag", f"r{rnd}"], f"② 批量复核（第 {rnd} 轮）")
        m = re.search(r"打回\s*(\d+)", txt)
        retry = int(m.group(1)) if m else 0
        if retry == 0:
            print("  ✓ 没有需要重译的块，复核阶段结束", flush=True)
            break
        r = run(["translate", str(src), "-o", str(out), *common, "--line-mode", "--out-flat"],
                f"③ 打回重译（第 {rnd} 轮，{retry} 块）", stream=True)
        if "failed" in r and "ok=" in r and re.search(r"failed\s*=\s*[1-9]", r):
            print("  ! 仍有失败块，进入下一轮", flush=True)

    # ④ 汇总
    run(["merge", str(out)], "④ 汇总")

    # ⑤ 收尾：称呼统一 + 特殊字符清理
    run(["polish", str(out), "--director-base", a.director_base,
         "--director-model", a.director_model, "--director-no-think"], "⑤ 收尾（称呼/字符）")

    # ⑥ 全书一致性体检（只报告，不改文本）
    run(["consistency", str(out)], "⑥ 一致性体检")

    merged = sorted(out.glob("合并_*_成品.txt")) or sorted(out.glob("合并_*.txt"))
    print("\n" + "=" * 66)
    if merged:
        f = merged[0]
        text = f.read_text(encoding="utf-8", errors="replace")
        import re as _re
        kana = len(_re.findall(r"[ぁ-んァ-ヺ]", text))
        print(f"  成品：{f}")
        print(f"  字数 {len(text)}｜行数 {len(text.splitlines())}｜假名残留 {kana}｜占位符残渣 {text.count('【#')}")
    else:
        print("  没有生成合并稿，请查看输出目录里的 _report.json / *_FAILED.json")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
