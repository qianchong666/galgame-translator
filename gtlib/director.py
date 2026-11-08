# -*- coding: utf-8 -*-
"""中控模型（导演）：跑在另一个大模型端点上的决策层。

职责分工
--------
* **译者**（小模型，如 GalTransl-v4-4B）：只管把一块日文译成中文，不做判断。
* **中控**（大模型，如 Qwen3.8-27B @19092）：负责需要"想一下"的活，全部通过
  **工具调用**（OpenAI `tools` 协议）完成：

  | 工具 | 用途 |
  |---|---|
  | `submit_glossary` | 判定专名候选是否真为专名，并给出统一中文定名 |
  | `submit_review`   | 复核某块译文：可接受（ok）还是必须重译（retry）+ 具体修改要求 |

设计原则
--------
1. **绝不阻塞流水线**：中控不可达 / 不支持工具调用 / 输出解析失败 → 抛
   `DirectorError` 或返回 `None`，调用方回退到原来的纯小模型行为。
2. **端点方言兼容**：先试 `tools` + 强制 `tool_choice`；400/不支持 → 退成
   "只输出 JSON" 的文本协议；再不行 → 放弃。
3. 只在 `degraded` / `failed` 块上调用复核，正常块零开销（27B 慢，别乱花）。
"""
from __future__ import annotations

import json
import re
import threading
import time
from typing import Dict, List, Optional, Sequence, Tuple

from .core import Config, Logger
from .llama import LlamaServer, list_models

try:
    import requests
except Exception:                                     # pragma: no cover
    requests = None


class DirectorError(RuntimeError):
    pass


GLOSSARY_SYS = (
    "你是这部日文小说/游戏文本的术语总管，负责把专名定名后交给翻译组统一使用。\n"
    "判断标准：\n"
    "- 人名（含绰号、称呼）→ type=人名；地名/学校/建筑 → 地名；组织/团体 → 组织；\n"
    "  招式/技能/称号/特殊物品 → 术语；普通名词、动词、形容词、感叹词 → 不是专名，keep=false。\n"
    "- 中文定名要求：人名用中文习惯的音译或汉字写法（如 アンジェリカ→安洁莉卡），\n"
    "  地名意译优先（聖マリア学園→圣玛丽亚学园），同一部作品里风格统一、不要混用。\n"
    "- zh 只写译名本身：不要引号、不要括号注音、不要「译名：」前缀。\n"
    "必须调用 submit_glossary 工具提交结果，不要用普通文字回答。"
)

REVIEW_SYS = (
    "你是资深日译中校对。给你日文原文、机器译文，以及自动质检发现的问题。\n"
    "请判断这块译文是否可以放行：\n"
    "- verdict=ok：问题属于可接受的风格差异/误报，读者读起来没有问题；\n"
    "- verdict=retry：有实质错误（漏译、错译、占位符/引号丢失、假名残留、语序错乱、\n"
    "  人称/时态错、重复啰嗦），必须在 notes 里写清楚怎么改（指出具体位置或原句）。\n"
    "notes 最多 4 条，每条一句话，用中文，给译者看，要能直接照着改。\n"
    "必须调用 submit_review 工具提交结果。"
)

GLOSSARY_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_glossary",
        "description": "提交日文专名的最终中文定名",
        "parameters": {
            "type": "object",
            "properties": {
                "entries": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "term": {"type": "string", "description": "日文原词，原样照抄"},
                            "zh": {"type": "string", "description": "简体中文定名"},
                            "type": {"type": "string",
                                     "enum": ["人名", "地名", "组织", "术语", "其他"]},
                            "keep": {"type": "boolean",
                                     "description": "是否作为专名统一（普通词给 false）"},
                        },
                        "required": ["term", "zh", "keep"],
                    },
                }
            },
            "required": ["entries"],
        },
    },
}

REVIEW_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_review",
        "description": "提交这块译文的复核结论",
        "parameters": {
            "type": "object",
            "properties": {
                "verdict": {"type": "string", "enum": ["ok", "retry"]},
                "reason": {"type": "string", "description": "一句话结论"},
                "notes": {"type": "array", "items": {"type": "string"},
                          "description": "必须重译时的具体修改要求"},
            },
            "required": ["verdict"],
        },
    },
}

PROOFREAD_SYS = (
    "你是资深日译中校对员。给你日文原文和机器译文，请你**直接交出修订稿**。\n"
    "硬规则：\n"
    "1. 行数必须与原文完全一致：一行对一行，不合并、不拆分、不增删空行；\n"
    "2. 形如 ⟦数字⟧ 的占位符必须原样保留，出现次数与原文一致，不得翻译、改写、删减、扩写；\n"
    "3. 只改错的地方（错译、漏译、假名残留、语序、人称/时态、重复哨嗦、称谓不统一），\n"
    "   已经通顺的部分原封不动，不要重写风格；\n"
    "4. 不改原文的引号「」『』结构与标点层级；\n"
    "5. 提交时把修订后的全文放进 text 字段（不是 diff），不要写解释。\n"
    "必须调用 submit_proofread 工具。"
)

PROOFREAD_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_proofread",
        "description": "提交校对后的完整译文",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string",
                          "description": "修订后的全文，行数与占位符必须与原文一致"},
                "notes": {"type": "array", "items": {"type": "string"},
                          "description": "改了什么（可选）"},
                "changed": {"type": "boolean", "description": "是否做了修改"},
            },
            "required": ["text"],
        },
    },
}

#: 行级补丁校对：只交「哪几行要改」，不重抄全文（token 省一个量级、风格零漂移）
PATCH_SYS = (
    "你是资深日译中校对员。用户给出「日文原文」与「机器译文」，两边都按行编号（从 1 开始，"
    "行号一一对应）。请你只对**确实有错**的行提交补丁。\n"
    "补丁有三种动作（op 字段）：\n"
    "- replace（默认）：改掉这一行，给 line + old + new；\n"
    "- delete：删掉多余的行（典型是误抄进来的提示词、复读的行），给 line + count（删几行，默认 1）；\n"
    "- insert：补回漏掉的行，给 line（插在这一行之后）+ new。\n"
    "硬规则：\n"
    "1. 通顺的行一律不要动；只是「我会换个说法」的也不要动，宁缺毋滥；\n"
    "2. replace 的 old 必须与给出的该行中文完全一致（用于定位）；\n"
    "3. new 必须是一行，不能含换行；占位符（⟦12⟧ / 【#12#】）的数量与顺序必须与原文该行一致；\n"
    "4. 改完后译文行数必须与原文一致（删了多余的就要补回漏掉的）；\n"
    "5. 没有该改的行就提交空 patches（changed=false）。\n"
    "必须调用 submit_patches 工具。"
)

PATCH_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_patches",
        "description": "提交行级修订补丁（只改有错的行）",
        "parameters": {
            "type": "object",
            "properties": {
                "patches": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "op": {"type": "string", "enum": ["replace", "delete", "insert"],
                                   "description": "replace=改这一行（默认）；delete=删掉这一行；insert=在这一行后面插入一行"},
                            "line": {"type": "integer", "description": "1 起算的行号"},
                            "old": {"type": "string", "description": "该行当前的译文，用于定位校验"},
                            "new": {"type": "string",
                                    "description": "replace/insert 的新行内容（不得含换行）"},
                            "count": {"type": "integer",
                                      "description": "delete 时删几行（默认 1，用于删掉复读的整段）"},
                        },
                        "required": ["line"],
                    },
                },
                "notes": {"type": "array", "items": {"type": "string"},
                          "description": "改了什么（可选）"},
                "changed": {"type": "boolean", "description": "是否做了修改"},
            },
            "required": ["patches"],
        },
    },
}

#: 单元级三档判定（ok / order / wrong），一次判一批
CHECK_SYS = (
    "你是资深日译中质检员。用户会给出若干条「编号. 日文原文 ⟹ 中文译文」。逐条判定：\n"
    "- ok：意思正确、没有漏译错译，中文通顺不别扭 —— **绝大多数条目都该是 ok**；\n"
    "- order：意思正确，但中文确实读不通、语序倒置（不是「我会换个说法」这种风格差异），\n"
    "  只需要调语序 —— 请在 fix 里给出调整后的完整中文；\n"
    "- wrong：有实质错误（漏译、错译、人名/称谓错、语气或情色程度错、假名残留、\n"
    "  把两条合并或拆开），必须重译 —— 在 fix 里给出你认为正确的完整中文；\n"
    "判定要点：\n"
    "1. 只判给出的这一条，不要拿别的条目的内容去补；\n"
    "2. 条目是独立的短句，单看简单但意思是「承接上文」的，算 ok；\n"
    "3. **能用就别改**：与自己的习惯写法不同、可以更优美、加不加主语，都不算 order；\n"
    "4. fix 必须是同一条的中文（不是日文），形如 ⟦12⟧ 的占位符要原样保留；\n"
    "5. 每条都要给出结论，条数必须与输入一致，不要漏条。\n"
    "必须调用 submit_units 工具提交结果。"
)

CHECK_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_units",
        "description": "提交每条短句译文的判定结果",
        "parameters": {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "i": {"type": "integer", "description": "条目编号"},
                            "v": {"type": "string", "enum": ["ok", "order", "wrong"],
                                  "description": "判定"},
                            "fix": {"type": "string",
                                    "description": "修正后的完整中文（order/wrong 时填）"},
                            "why": {"type": "string", "description": "一句话理由"},
                        },
                        "required": ["i", "v"],
                    },
                }
            },
            "required": ["items"],
        },
    },
}


class Director:
    """中控模型客户端：只做判断，不产出大段译文。"""

    def __init__(self, cfg: Config, log: Logger,
                 cancel: Optional[threading.Event] = None):
        self.cfg = cfg
        self.log = log
        self.cancel = cancel or threading.Event()
        self.server = LlamaServer.remote(
            cfg.director_base, log,
            api_key=(cfg.director_key or cfg.api_key_effective() or ""),
            name="director")
        self.model = (cfg.director_model or "").strip()
        self.tools_ok = True                 # 端点是否支持 tools（400 后自动置 False）
        self._stats = {"calls": 0, "seconds": 0.0, "fallbacks": 0}
        self._down_until = 0.0               # 失败后冷却，避免每块都去撞墙

    # ------------------------------------------------------------ 基础
    @property
    def enabled(self) -> bool:
        return bool((self.cfg.director_base or "").strip())

    def resolve_model(self) -> str:
        if self.model:
            return self.model
        names = list_models(self.server)
        if not names:
            raise DirectorError("中控端点没有返回模型列表，请手动填 director_model")
        self.model = names[0]
        return self.model

    def available(self) -> bool:
        if not self.enabled or requests is None:
            return False
        if time.time() < self._down_until:
            return False
        return True

    def probe(self, timeout: float = 10.0) -> bool:
        """探活 + 取模型名；不可用返回 False（不抛异常）。"""
        if not self.enabled:
            return False
        try:
            if not self.server.health(timeout):
                return False
            self.resolve_model()
            return True
        except Exception:
            return False

    def _post(self, payload: dict, timeout: float) -> dict:
        if getattr(self.cfg, "director_no_think", False):
            payload = dict(payload)
            kw = dict(payload.get("chat_template_kwargs") or {})
            kw.setdefault("enable_thinking", False)
            payload["chat_template_kwargs"] = kw
        url = self.server.base_url + "/chat/completions"
        if not self.server.base_url.endswith("/v1"):
            url = self.server.base_url + "/v1/chat/completions"
        r = requests.post(url, json=payload, timeout=timeout,
                          headers=self.server.auth_headers())
        if r.status_code >= 400:
            raise DirectorError(f"HTTP {r.status_code}: {r.text[:300]}")
        return r.json()

    def _call(self, system: str, user: str, tool: dict,
              max_tokens: int = 900) -> dict:
        """一次决策调用，返回解析后的 JSON（工具参数或兜底 JSON）。"""
        if not self.available():
            raise DirectorError("中控不可用")
        model = self.resolve_model()
        t0 = time.time()
        try:
            if self.tools_ok:
                try:
                    payload = {
                        "model": model,
                        "messages": [{"role": "system", "content": system},
                                     {"role": "user", "content": user}],
                        "tools": [tool],
                        "tool_choice": {"type": "function",
                                        "function": {"name": tool["function"]["name"]}},
                        "temperature": 0.0,
                        "max_tokens": max_tokens,
                    }
                    data = self._post(payload, self.cfg.director_timeout)
                    msg = (data.get("choices") or [{}])[0].get("message") or {}
                    parsed = self._from_tool_calls(msg, tool["function"]["name"])
                    if parsed is not None:
                        self._stats["calls"] += 1
                        self._stats["seconds"] += time.time() - t0
                        return parsed
                    # 没给出工具调用 → 用同一份内容再解析一次 / 走兜底
                    txt = (msg.get("content") or "").strip()
                    parsed = _extract_json(txt)
                    if parsed is not None:
                        self._stats["calls"] += 1
                        self._stats["seconds"] += time.time() - t0
                        return parsed
                except DirectorError as e:
                    msg = str(e)
                    if re.search(r"\b400\b|tool|function|unsupported|unknown", msg, re.I):
                        self.log.warn(f"  ⚠ 中控不支持 tools，改用 JSON 文本协议（{msg[:80]}）")
                        self.tools_ok = False
                    else:
                        raise
            # ---- 兜底：纯 JSON 文本协议
            payload = {
                "model": model,
                "messages": [
                    {"role": "system",
                     "content": system + "\n\n只输出一个 JSON 对象，不要解释、不要代码块围栏。"},
                    {"role": "user",
                     "content": user + "\n\n输出字段说明：" + json.dumps(
                         tool["function"]["parameters"], ensure_ascii=False)},
                ],
                "temperature": 0.0,
                "max_tokens": max_tokens,
            }
            data = self._post(payload, self.cfg.director_timeout)
            txt = ((data.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
            parsed = _extract_json(txt)
            if parsed is None:
                self._stats["fallbacks"] += 1
                raise DirectorError(f"中控输出无法解析为 JSON: {txt[:160]}")
            self._stats["calls"] += 1
            self._stats["seconds"] += time.time() - t0
            return parsed
        except Exception as e:
            self._down_until = time.time() + 20.0     # 冷却 20s，别每块都撞
            raise DirectorError(f"{e}") from e

    @staticmethod
    def _from_tool_calls(msg: dict, name: str) -> Optional[dict]:
        for tc in (msg.get("tool_calls") or []):
            fn = tc.get("function") or {}
            if fn.get("name") not in (None, name):
                continue
            raw = fn.get("arguments")
            if isinstance(raw, dict):
                return raw
            if isinstance(raw, str) and raw.strip():
                got = _extract_json(raw)
                if got is not None:
                    return got
        return None

    @property
    def stats(self) -> dict:
        return dict(self._stats)

    # ------------------------------------------------------------ 术语定名
    def decide_terms(self, entries: Sequence, samples: Dict[str, str],
                     batch: int = 40) -> Dict[int, dict]:
        """返回 {entry.idx: {"zh":..., "type":..., "keep":...}}，失败条目自动缺席。"""
        out: Dict[int, dict] = {}
        items = list(entries)
        for i in range(0, len(items), batch):
            if self.cancel.is_set():
                break
            part = items[i:i + batch]
            lines = ["请给下面这些日文专名候选定名（每个附一句原文例句供判断语境）：", ""]
            for e in part:
                ex = (samples.get(e.term) or "").strip().replace("\n", " ")
                lines.append(f"- {e.term}（出现 {e.count} 次，启发式类型 {e.type}）"
                             + (f"  例：{ex[:80]}" if ex else ""))
            user = "\n".join(lines)
            try:
                data = self._call(GLOSSARY_SYS, user, GLOSSARY_TOOL,
                                  max_tokens=min(4000, 120 * len(part) + 200))
            except DirectorError as e:
                self.log.warn(f"  中控术语定名失败（本批 {len(part)} 条）: {e}")
                continue
            got = changed = 0
            for row in (data.get("entries") or []):
                term = str(row.get("term") or "").strip()
                zh = str(row.get("zh") or "").strip()
                keep = bool(row.get("keep", True))
                if not term:
                    continue
                for e in part:
                    if e.term != term:
                        continue
                    if not keep:
                        out[e.idx] = {"zh": "", "type": e.type, "keep": False}
                        got += 1
                    elif zh:
                        zh = _clean_name(zh)
                        if zh:
                            out[e.idx] = {"zh": zh, "type": row.get("type") or e.type,
                                          "keep": True}
                            got += 1
                            if zh != e.term:
                                changed += 1
                    break
            self.log.info(f"  中控定名 {i // batch + 1}: {got}/{len(part)} 条"
                          f"（改名 {changed}）")
        return out

    # ------------------------------------------------------------ 校对（行级补丁）
    def proofread_patches(self, src: str, dst: str, problems: Sequence[str],
                          max_tokens: int = 1600) -> Optional[dict]:
        """只让中控提交**行级补丁**，返回 {patches, notes, changed}。不重抄全文。"""
        if not self.available():
            return None
        if len(src) > 3000 or len(dst) > 3000:
            self.log.warn(f"  中控补丁跳过：块太长（原文 {len(src)} 字），请调小 chunk")
            return None
        sl = src.strip().split("\n")
        dl = dst.strip().split("\n")
        lines = ["【日文原文】"]
        lines += [f"{i}. {ln}" for i, ln in enumerate(sl, 1)]
        lines += ["", "【机器译文】"]
        lines += [f"{i}. {ln}" for i, ln in enumerate(dl, 1)]
        if problems:
            lines += ["", "【自动质检发现的问题】"] + [f"- {p}" for p in list(problems)[:8]]
        lines += ["", "请只对确实有错的行提交补丁（没有问题就提交空 patches）。"]
        try:
            data = self._call(PATCH_SYS, "\n".join(lines), PATCH_TOOL,
                              max_tokens=max_tokens)
        except DirectorError as e:
            self.log.warn(f"  中控补丁失败: {e}")
            return None
        if not isinstance(data, dict):
            return None
        patches = data.get("patches") or []
        if not isinstance(patches, list):
            return None
        notes = [str(n).strip() for n in (data.get("notes") or []) if str(n).strip()]
        return {"patches": patches, "notes": notes[:6],
                "changed": bool(data.get("changed", bool(patches)))}

    # ------------------------------------------------------------ 校对（直接改稿）
    def proofread(self, src: str, dst: str, problems: Sequence[str],
                  max_tokens: int = 1600) -> Optional[dict]:
        """让大模型直接交出修订后的译文。返回 {text, notes, changed} 或 None。"""
        if not self.available():
            return None
        if len(src) > 3000 or len(dst) > 3000:
            # 超出送审长度就跳过，不能默默截断（截断的稿子行数必然对不上）
            self.log.warn(f"  中控校对跳过：块太长（原文 {len(src)} 字），请调小 chunk")
            return None
        lines = ["【日文原文】", src.strip()[:3000], "", "【机器译文】",
                 dst.strip()[:3000], ""]
        if problems:
            lines.append("【自动质检发现的问题】")
            lines += [f"- {p}" for p in list(problems)[:8]]
            lines.append("")
        lines.append("请按硬规则提交修订稿（text 字段放全文）。")
        try:
            data = self._call(PROOFREAD_SYS, "\n".join(lines), PROOFREAD_TOOL,
                              max_tokens=max_tokens)
        except DirectorError as e:
            self.log.warn(f"  中控校对失败: {e}")
            return None
        text = _strip_wrapper(str(data.get("text") or ""))
        if not text.strip():
            return None
        notes = [str(n).strip() for n in (data.get("notes") or []) if str(n).strip()]
        return {"text": text, "notes": notes[:6],
                "changed": bool(data.get("changed", True))}

    # ------------------------------------------------------------ 单元批量判定
    def check_units(self, pairs: Sequence[Tuple[int, str, str]],
                    hint: str = "", max_tokens: int = 1400) -> Dict[int, dict]:
        """一次给一批 (编号, 日文, 中文) 定性：{编号: {v, fix, why}}。

        v 取 ok / order / wrong。调用失败返回 {}（流水线按「没检查到」处理）。
        """
        if not pairs or not self.available():
            return {}
        lines: List[str] = []
        for i, src, dst in pairs:
            lines.append(f"{i}. {(src or '').strip()[:400]} ⟹ {(dst or '(空)').strip()[:400]}")
        if hint:
            lines.append("")
            lines.append(hint)
        try:
            data = self._call(CHECK_SYS, "\n".join(lines), CHECK_TOOL,
                              max_tokens=max_tokens)
        except DirectorError as e:
            self.log.warn(f"  中控批量检查失败: {e}")
            return {}
        out: Dict[int, dict] = {}
        items = data.get("items")
        if isinstance(items, dict):
            items = [items]
        for it in (items or []):
            if not isinstance(it, dict):
                continue
            try:
                k = int(it.get("i"))
            except (TypeError, ValueError):
                continue
            v = str(it.get("v") or "").strip().lower()
            fix = _strip_wrapper(str(it.get("fix") or ""))
            if v not in ("ok", "order", "wrong"):
                v = "wrong" if fix else "ok"
            out[k] = {"v": v, "fix": fix.strip(),
                      "why": str(it.get("why") or "").strip()[:120]}
        return out

    # ------------------------------------------------------------ 复核
    def review(self, src: str, dst: str, problems: Sequence[str],
               tm=None, style_probe: str = "") -> Optional[dict]:
        """复核一块译文。返回 {"verdict": "ok"/"retry", "notes": [...], "reason": ...}。"""
        if not self.available():
            return None
        lines = ["【日文原文】", src.strip()[:2000], "", "【机器译文】",
                 (dst or "(空)").strip()[:2000], ""]
        if problems:
            lines.append("【自动质检发现的问题】")
            lines += [f"- {p}" for p in list(problems)[:8]]
        if style_probe:
            lines.append(style_probe)
        try:
            data = self._call(REVIEW_SYS, "\n".join(lines), REVIEW_TOOL, max_tokens=600)
        except DirectorError as e:
            self.log.warn(f"  中控复核失败: {e}")
            return None
        verdict = str(data.get("verdict") or "").strip().lower()
        if verdict not in ("ok", "retry"):
            verdict = "retry" if data.get("notes") else "ok"
        notes = [str(n).strip() for n in (data.get("notes") or []) if str(n).strip()]
        return {"verdict": verdict,
                "notes": notes[:4],
                "reason": str(data.get("reason") or "").strip()[:200]}


# ------------------------------------------------------------------ 工具函数
def make_director(cfg: Config, log: Logger,
                  cancel: Optional[threading.Event] = None,
                  required: bool = False) -> Optional[Director]:
    """按配置创建中控；不可达就返回 None（流水线退化成单模型）。"""
    if not cfg.use_director:
        if required:
            log.warn("  未配置中控端点（--director-base），本次按单模型跑")
        return None
    d = Director(cfg, log, cancel)
    if not d.probe():
        msg = (f"中控端点不可达：{cfg.director_base}"
               f"（先双击对应的「快启」bat，或检查端口）")
        if required:
            raise DirectorError(msg)
        log.warn("  " + msg + " → 本次按单模型跑")
        return None
    log.info(f"  中控就绪: {d.server.base_url} · 模型 {d.model}"
             f"（定名={'开' if cfg.director_terms else '关'}·"
             f"复核={'开' if cfg.director_review else '关'}）")
    return d


def _clean_name(s: str) -> str:
    s = s.strip().strip("「」『』\"'《》〈〉()（）[]【】")
    s = re.sub(r"^(译名|中文|译文)\s*[:：]\s*", "", s)
    s = re.sub(r"[（(].*?[)）]$", "", s).strip()
    return s[:24]


_FENCE = re.compile(r"```(?:json)?\s*(.+?)```", re.S)


def _strip_wrapper(text: str) -> str:
    """去掉模型包在外面的代码围栏 / “校对后译文：” 之类前缀。"""
    t = text.strip()
    m = _FENCE.fullmatch(t) or _FENCE.search(t)
    if m:
        t = m.group(1).strip()
    t = re.sub(r"^\s*(校对后|修订后|修改后)?(译文|全文)?\s*[:：]\s*", "", t)
    t = re.sub(r"^```[a-zA-Z]*\n?", "", t)
    t = re.sub(r"\n?```\s*$", "", t)
    return t.strip()


def _extract_json(txt: str) -> Optional[dict]:
    """从任意文本里抠出第一个 JSON 对象（兼容代码块围栏 / 前后废话）。"""
    if not txt:
        return None
    cands: List[str] = []
    m = _FENCE.search(txt)
    if m:
        cands.append(m.group(1))
    cands.append(txt)
    for c in cands:
        c = c.strip()
        try:
            obj = json.loads(c)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        start = c.find("{")
        while start >= 0:
            depth = 0
            for i in range(start, len(c)):
                if c[i] == "{":
                    depth += 1
                elif c[i] == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            obj = json.loads(c[start:i + 1])
                            if isinstance(obj, dict):
                                return obj
                        except Exception:
                            pass
                        break
            start = c.find("{", start + 1)
    return None
