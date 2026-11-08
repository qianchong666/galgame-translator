# -*- coding: utf-8 -*-
"""专名提取 / 过滤 / 术语表批量翻译。

旧版问题（真实产物复现过）
--------------------------
* `KANJI_HONORIFIC_RE` 贪婪吃前缀：`絶対友香`、`全部友香`、`一度友香`…… 一个角色被拆成十几个"人名"，
  遮蔽后正文语义直接崩坏；
* 短引号规则把 `是的`、`对不起`、`湿湿的` 这种口语收进术语表；
* 常用外来词 `ベッド / ドア / キス` 全被当专名遮蔽；
* min_freq=1 全量入库，1.77MB 样本出 473 个候选，绝大多数是噪声。

新版策略
--------
1. **只遮"真专名 + 设定术语"**：带敬称的人名、片假名前缀专名、地名后缀、外部术语表；
   普通汉字词、口语短句一律不收（宁可漏遮，也不破坏正文）。
2. **前缀修剪**：候选词前 1~3 字若命中高频常用词（絶対/全部/一度/本当…），自动剥掉再判定。
3. **片假名白名单机制**：常见外来词内置黑名单 + 长度/频次门槛；`アン` 这类 2 字名只在
   带敬称或高频出现时才收。
4. **置信度**：high / medium / low 三档，low 档默认不遮蔽，只进术语表供人工确认。
5. **术语翻译成批**：一次请求翻 20~30 条（带序号对齐），失败回落单条；解析不出就保留原词并标 low。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

from .placeholder import TermEntry, TermMap

# ------------------------------------------------------------------ 词表
#: 常见外来词/游戏界面词：绝不遮蔽（命中即丢）
KATAKANA_STOP = set("""
システム レベル スキル アイテム ステータス イベント ベッド ナイフ ドア メイン ショック タイミング
リアリティ コーヒー シャワー トイレ パンツ スカート ブラジャー シャツ セーター コート ブーツ タイツ
ストッキング ハイヒール アクセサリー ピアス ネックレス リング リストバンド ゴム コンドーム ピル
セックス エッチ オーラル フェラ クンニ アナル プレイ コスプレ オナニー バイブ ローター
キス ラブ ラブラブ ハグ ランチ ディナー ケーキ パン グラス カップ ボトル ワイン ビール ジュース
テレビ ラジオ パソコン スマホ ケータイ スイッチ カメラ ビデオ メール ネット ゲーム アプリ データ
テスト チェック ミス トラブル サプライズ プレゼント パーティー ピクニック デート ドライブ バイト
アルバイト サークル クラブ チーム グループ メンバー リーダー スタッフ キャンプ ホテル
マンション アパート ビル エレベーター エスカレーター バス タクシー トラック バイク タイヤ
エネルギー パワー スピード タイム サイズ シーン パターン スタイル デザイン ファッション
スマイル フェイス ボディ ヘア メイク ネイル エステ ダイエット マッサージ
アニメ マンガ コミック ドラマ ピアノ ギター ドラム ホーム ルーム キッチン リビング バスルーム
ベランダ ガレージ プール ジム サウナ カーテン ソファ テーブル チェア ランプ ライト コンセント
ドライヤー クラス ホント マジ ヤバ ウソ ウインク ハーフ タイプ ママ パパ
ペニス チンコ チンポ オチンチン マンコ ワレメ クリトリス オッパイ タマタマ ザーメン ローション
アクメ オーガズム ヌード パイパン ピストン フェラチオ パイズリ カウパー オナニ
ドロドロ ドキドキ ビクビク グチョグチョ ヌルヌル ザラザラ ベトベト ドクドク ビンビン グチュグチュ
""".split())

#: 高频常用词（出现在专名前面时剥掉）
COMMON_PREFIX = set("""
絶対 全部 一度 一度に 本当 今日 明日 昨日 多分 少し 大変 普通 全然 一緒 自分 相手 最近 最初 最後
一番 結構 必要 重要 大丈夫 無理 問題 関係 意味 気持 気持ち 感じ 時間 場所 世界 学校 会社 家族
友達 恋人 彼女 彼氏 子供 大人 男 女 先生 生徒 部屋 声 目 手 足 顔 体 心 頭 的口 今 昔 後 前
その この あの どの 例の あのさ あのね つまり やっぱり やはり たぶん きっと ぜんぜん すごく
""".split())

#: 汉字专名黑名单（名字里常见但单独成词时是普通词）
KANJI_STOP = set("""
時間 世界 学校 会社 家族 友達 恋人 自分 相手 問題 意味 気持 気持ち 関係 必要 重要 大丈夫 無理
今日 明日 昨日 本当 全部 一度 多分 少し 大変 普通 全然 一緒 最近 最初 最後 一番 結構 場所 部屋
声 目 手 足 顔 体 心 頭 男 女 子供 大人 先生 生徒 先輩 後輩 父親 母親 兄 姉 弟 妹 奥 旦那 嫁
毎日 絶倫 童貞 処女 素人 熟女 美人 巨乳 貧乳 早漏 遅漏 絶頂 絶叫 連続 大量 二人 三人 半分
""".split())


#: 拟声拟态词特征：ABAB 重叠、小假名堆叠
_MIMETIC_MIN_REPEAT = re.compile(r"^(.{2,4})\1+$")
_KANA_NOISE = set("ッーンァィゥェォャュョ・ー、。…")


def is_mimetic(term: str) -> bool:
    """グチュグチュ / ビクンビクン / ンンンッ / ドキドキ 这类拟声词。"""
    t = term.strip("・ーヵヶ")
    if len(t) < 2:
        return True
    if _MIMETIC_MIN_REPEAT.match(t):
        return True
    noise = sum(1 for c in t if c in _KANA_NOISE)
    if noise / len(t) > 0.5:
        return True
    if len(t) >= 3 and max(t.count(c) for c in set(t)) / len(t) > 0.55:
        return True
    return False

#: 敬称（长的放前面，保证 `ちゃん` 不被 `ち` 类截断）
HONORIFICS = ("ちゃん", "さん", "さま", "様", "くん", "君", "先生", "先輩", "せんぱい",
              "氏", "殿", "嬢", "坊", "ちゃん達", "たち")

KANA_RUN = re.compile(r"[ァ-ヺー・]{2,}")
KANJI_RUN = re.compile(r"[一-龯々]{2,6}")
HONORIF_RE = re.compile(
    r"([一-龯々ァ-ヺーA-Za-z・]{2,8}?)(?:" + "|".join(HONORIFICS) + r")")
SPEAKER_RE = re.compile(r"^[【\[]([^】\]]{1,8})[】\]]")
SPEAKER_QUOTE_RE = re.compile(r"^([一-龯々ァ-ヺーA-Za-z]{2,6})[「『]")
TAG_BRACKET_RE = re.compile(r"^(?:【|\[)([^】\]]{1,10})(?:】|\])$")
LOCATION_SUFFIX = ("市", "町", "村", "駅", "学園", "高校", "大学", "病院", "神社", "公園",
                   "城", "王国", "帝国", "島", "山", "川", "湖", "橋", "通り", "教会", "ギルド")
LOCATION_RE = re.compile(
    r"([一-龯々ァ-ヺー]{2,5})(?:" + "|".join(sorted(LOCATION_SUFFIX, key=len, reverse=True)) + r")")
LATIN_RE = re.compile(r"\b[A-Z][A-Za-z]{2,15}\b")


@dataclass
class Term:
    term: str
    type: str = "专名"
    count: int = 1
    conf: str = "high"
    origin: str = ""
    honor: bool = False          # 是否曾带敬称出现（决定 2 字片假名能否入库）

    def key(self) -> str:
        return f"{self.type}:{self.term}"


# ------------------------------------------------------------------ 修剪
def _trim_name(name: str) -> Optional[str]:
    """从带敬称的候选里剥掉常用词前缀，返回干净人名。"""
    s = name.strip("・ー")
    if not s:
        return None
    if s in KANJI_STOP:
        return None
    # 前 1~3 字是常用词 → 剥掉
    for n in (3, 2, 1):
        if len(s) - n >= 2 and s[:n] in COMMON_PREFIX:
            s = s[n:]
            break
    if len(s) < 2 or s in KANJI_STOP:
        return None
    if all(c in "的了是在有和就不人都一" for c in s):
        return None
    return s


def _katakana_ok(term: str, count: int, with_honorific: bool) -> bool:
    if term in KATAKANA_STOP or term.strip("ー・") in KATAKANA_STOP:
        return False
    if len(term) >= 5 and term.endswith("ー"):
        # 长音符结尾多见于普通名词（コーヒー/カレー），需更高门槛
        pass
    if with_honorific:
        return len(term) >= 2
    if len(term) >= 6:
        return count >= 1          # 超长片假名几乎只能是专名
    if len(term) >= 4:
        return count >= 2
    if len(term) == 3:
        return count >= 3
    return count >= 6      # 2 字片假名：可能是普通词，要求高频且不在黑名单


# ------------------------------------------------------------------ 提取
def extract_terms(text: str,
                  min_freq: int = 2,
                  extra_terms: Optional[Iterable[dict]] = None,
                  max_terms: int = 400,
                  mask_katakana: bool = False) -> List[Term]:
    """从文本中提取候选专名。返回按 (confidence, count) 排好序的 Term 列表。

    mask_katakana=False（默认）时，纯片假名候选只进候选表（conf=low），不参与遮蔽：
    片假名词里拟声词/普通外来词占比太高，误遮会直接破坏正文。
    """
    found: Dict[str, Term] = {}

    def add(term: str, typ: str, conf: str, origin: str, n: int = 1,
            honor: bool = False) -> None:
        term = term.strip("・ー 　")
        if not term:
            return
        cur = found.get(term)
        if cur is None:
            found[term] = Term(term=term, type=typ, count=n, conf=conf,
                               origin=origin, honor=honor)
            return
        cur.count += n
        cur.honor = cur.honor or honor
        # 置信度取高者，类型取先出现者（人名优先于专名，见优先级函数）
        order = {"high": 0, "medium": 1, "low": 2}
        if order[conf] < order[cur.conf]:
            cur.conf = conf
        if _type_rank(typ) < _type_rank(cur.type):
            cur.type = typ

    # 1. 带敬称的人名（含前缀修剪）
    for m in HONORIF_RE.finditer(text):
        name = _trim_name(m.group(1))
        if name:
            add(name, "人名", "high", "honorific")

    # 2. 说话人标签 【名字】 / 名字「
    for line in text.split("\n"):
        s = line.lstrip("　 ")
        if not s:
            continue
        m = SPEAKER_RE.match(s)
        if m:
            name = _trim_name(m.group(1))
            if name:
                add(name, "人名", "medium", "speaker_tag")
            continue
        m = SPEAKER_QUOTE_RE.match(s)
        if m:
            name = _trim_name(m.group(1))
            if name:
                add(name, "人名", "medium", "speaker_quote")

    # 3. 地名
    for m in LOCATION_RE.finditer(text):
        loc = m.group(0)
        if len(loc) >= 3 and loc not in KANJI_STOP:
            add(loc, "地名", "medium", "location_suffix")

    # 4. 片假名专名
    for m in KANA_RUN.finditer(text):
        term = m.group(0).strip("・ー")
        if len(term) < 2:
            continue
        nxt = text[m.end():m.end() + 4]
        hon = any(nxt.startswith(h) for h in ("さん", "ちゃん", "君", "様", "くん", "さま"))
        add(term, "专名", "medium", "katakana", honor=hon)

    # 5. 拉丁专名
    for m in LATIN_RE.finditer(text):
        add(m.group(0), "专名", "medium", "latin")

    # 6. 外部术语表（最高优先，无条件保留）
    for item in (extra_terms or []):
        if isinstance(item, str):
            add(item, "专名", "high", "glossary")
        else:
            add(item.get("term", ""), item.get("type", "专名"),
                item.get("conf", "high"), "glossary")

    # ---- 过滤
    result: List[Term] = []
    for t in found.values():
        if t.term in KANJI_STOP or t.term in KATAKANA_STOP:
            continue
        if is_mimetic(t.term) and t.origin != "glossary":
            continue
        if t.term.isdigit():
            continue
        if t.origin == "glossary":
            result.append(t)
            continue
        if t.origin == "katakana":
            if is_mimetic(t.term):
                continue
            if not _katakana_ok(t.term, t.count, with_honorific=t.honor):
                continue
            t.conf = "high" if (t.count >= 3 or t.honor) else "medium"
            if not mask_katakana:
                t.conf = "low"      # 只进候选表，不遮蔽
        elif t.origin == "latin":
            if t.count < 2:
                continue
        elif t.origin == "location_suffix":
            if t.count < 1:
                continue
        else:  # honorific / speaker
            if is_mimetic(t.term) or t.term in KANJI_STOP:
                continue
            if t.count < 1:
                continue
        if t.count < min_freq and t.conf != "high":
            continue
        result.append(t)

    _dedupe_substrings(result)

    result.sort(key=lambda t: (-{"high": 2, "medium": 1, "low": 0}[t.conf], -t.count))
    return result[:max_terms]


def _type_rank(typ: str) -> int:
    return {"人名": 0, "地名": 1, "术语": 2, "专名": 3}.get(typ, 9)


def _dedupe_substrings(terms: List[Term]) -> None:
    """`友香` 与 `友香さん`、`アン` 与 `アンナ` 同时出现时只保留更长的那个。"""
    drop = set()
    for a in terms:
        for b in terms:
            if a is b or a.term in drop or b.term in drop:
                continue
            if a.term != b.term and a.term in b.term and len(b.term) > len(a.term):
                # 短词是长词子串 → 丢短的（但人名优先保留）
                if a.type == "人名" and b.type != "人名":
                    continue
                if b.count >= a.count:
                    drop.add(a.term)
    terms[:] = [t for t in terms if t.term not in drop]


# ------------------------------------------------------------------ 遮蔽决策
def build_term_map(terms: List[Term], style: str = "bracket",
                   mask_low: bool = False) -> TermMap:
    """把提取结果转成遮蔽用的 TermMap（low 置信默认不进遮蔽表）。"""
    usable = [t for t in terms if t.conf != "low" or mask_low]
    usable.sort(key=lambda t: (not t.term.isascii(), -len(t.term)))
    return TermMap.build(
        [(t.term, t.type, t.conf, t.count) for t in usable], style=style)


# ------------------------------------------------------------------ 术语翻译
TERM_SYS = ("你是一个日文专名翻译工具。把你收到的每一条日文专名按序号翻译成简体中文。"
            "人名保持汉字不变或音译成常见中文名；片假名外来词译成地道中文；"
            "每行输出 `序号. 译名`，不要解释、不要空行、不要多余标点。")

TERM_EXAMPLES = (
    "示例输入：\n1. 友香\n2. アンジェリカ\n3. 聖マリア学園\n"
    "示例输出：\n1. 友香\n2. 安洁莉卡\n3. 圣玛丽亚学园\n"
)


def build_terms_prompt(batch: List[TermEntry]) -> Tuple[str, str]:
    lines = [f"{e.idx}. {e.term}" for e in batch]
    user = TERM_EXAMPLES + "\n现在翻译（只输出对应行）：\n" + "\n".join(lines)
    return TERM_SYS, user


_REPLY_LINE = re.compile(r"^\s*(\d{1,5})\s*[.、．)）:：]\s*(.+?)\s*$")


def parse_terms_reply(reply: str, batch: List[TermEntry]) -> Dict[int, str]:
    """解析 `序号. 译名` 形式的回复，只接受 batch 里出现过的序号。"""
    want = {e.idx for e in batch}
    out: Dict[int, str] = {}
    for raw in (reply or "").splitlines():
        m = _REPLY_LINE.match(raw)
        if not m:
            continue
        idx = int(m.group(1))
        val = m.group(2).strip().strip("「」『』\"'。、，, ")
        if idx in want and val and len(val) <= 24:
            out[idx] = val
    return out


def chunk_batches(items: List[TermEntry], size: int) -> List[List[TermEntry]]:
    size = max(1, size)
    return [items[i:i + size] for i in range(0, len(items), size)]
