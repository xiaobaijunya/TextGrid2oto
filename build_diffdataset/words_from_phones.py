# -*- coding: utf-8 -*-
"""从 TextGrid 的 phones 层复原 words 层。

参考 MakeDiffSinger/acoustic_forced_alignment/generate_words_from_phones.py：

    python generate_words_from_phones.py --tg <TextGrid目录> --dictionary <字典.txt> [--out 输出目录] [--overwrite]

与参考脚本相比，这里做了以下适配，便于在 wxPython GUI 中调用：
  * 可直接 import 调用 ``run()``，支持输出目录、覆盖开关；
  * 支持递归扫描子文件夹（多音阶目录）；
  * 支持 ``progress_callback`` 实时回传进度；
  * 支持 ``stop_event`` 中途停止；
  * **完全根据 phones 层重建 words 层，不读取旧 words 层的内容**；
  * 只接受名为 ``phones`` / ``phone`` 的层；两层结构时会使用非 words 的那一层；
  * 名称无法识别时直接报错，绝不把其他层当成 phones 使用；
  * 旧的 words/word 层会被整层删除，再插入新层（保留原来的层顺序）；
  * 匹配不到字典单词的音素**不合并**，直接按单个音素写入 words 层。
"""

from __future__ import annotations

import pathlib
from typing import Callable, Optional

from textgrid import IntervalTier, TextGrid

# 这些音素不参与组词，直接作为独立区间写入 words 层
SILENCE_PHONES = ("AP", "SP", "EP")


class TrieNode:
    __slots__ = ("children", "is_end_of_word", "value")

    def __init__(self):
        self.children: dict = {}
        self.is_end_of_word = False
        self.value = None


class Trie:
    """音素序列 → 单词 的前缀树。"""

    def __init__(self, dict_path=None):
        self.depth = 0
        self.root = TrieNode()
        if dict_path is not None:
            self.build_trie(dict_path)

    def insert(self, phonemes, raw_word):
        node = self.root
        for char in phonemes:
            node = node.children.setdefault(char, TrieNode())
        node.is_end_of_word = True
        node.value = raw_word

    def search(self, phonemes):
        """整段音素完全匹配时返回单词，否则返回 None。"""
        node = self.root
        for char in phonemes:
            node = node.children.get(char)
            if node is None:
                return None
        return node.value if node.is_end_of_word else None

    def has_prefix(self, phonemes):
        """该音素序列是否仍是某个词的合法前缀。"""
        node = self.root
        for char in phonemes:
            node = node.children.get(char)
            if node is None:
                return False
        return True

    def build_trie(self, dict_path):
        for phonemes, raw in self.read_dictionary(dict_path):
            self.insert(phonemes, raw)

    def read_dictionary(self, filename):
        """读取 DS 字典：``词<TAB>音素 音素 ...``。

        字典里的音素可能带语言前缀（如 ``zh/d``、``ja/k``），
        而 TextGrid 中的音素已去前缀，统一取 ``/`` 后的部分再建 trie。
        """
        with open(filename, "r", encoding="utf-8") as file:
            for line in file:
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue
                raw = parts[0]
                phonemes = [ph.split("/")[-1] for ph in parts[1].split()]
                if not phonemes:
                    continue
                self.depth = max(self.depth, len(phonemes))
                yield phonemes, raw


def _tier_name(tier) -> str:
    """统一处理层名：去空格、转小写。"""
    return str(getattr(tier, "name", "")).strip().lower()


def _is_words_tier(tier) -> bool:
    """识别 words 类层（兼容 words / word / Word 等写法）。"""
    name = _tier_name(tier)
    return name in ("words", "word") or name.startswith("word")


def _is_phones_tier(tier) -> bool:
    """识别 phones 类层（兼容 phones / phone / Phone 等写法）。"""
    name = _tier_name(tier)
    return name in ("phones", "phone") or "phone" in name


def _find_phones_tier(tg: TextGrid):
    """找到 phones 层。

    绝不会返回 words 类层：如果只有 words 层而没有 phones 层，直接报错，
    避免出现「拿 words 层当 phones 层再重建 words」的错误行为。
    """
    tiers = list(tg.tiers)

    # 1) 名称明确是 phones / phone
    for tier in tiers:
        if _is_phones_tier(tier):
            return tier

    # 2) 名称不规范时，只允许「两层且其中一层是 words」的情况：
    #    此时剩下的那层必然就是音素层（HFA / TIFA 都是这种两层结构）。
    candidates = [tier for tier in tiers if not _is_words_tier(tier)]
    if len(tiers) == 2 and len(candidates) == 1:
        return candidates[0]

    # 其余情况一律报错，绝不猜测，避免把 other 层当成 phones 使用
    names = ", ".join(repr(_tier_name(t)) for t in tiers) or "无"
    raise ValueError(f"TextGrid 中找不到 phones 层（现有层：{names}），无法重建 words 层")


def _replace_words_tier(tg: TextGrid, new_tier: IntervalTier):
    """彻底删除所有旧 words 类层，再插入新层（保留原来的位置）。

    这是「完全重建」的关键：旧 words 层的区间不会以任何方式参与新层生成。
    """
    insert_at = None
    kept = []
    for index, tier in enumerate(tg.tiers):
        if _is_words_tier(tier):
            if insert_at is None:
                insert_at = index
            continue  # 丢弃旧 words 层
        kept.append(tier)

    if insert_at is None:
        insert_at = 0
    kept.insert(min(insert_at, len(kept)), new_tier)
    tg.tiers[:] = kept


def build_words_tier(phones_tier, trie: Trie, min_time: float = 0.0, max_time=None):
    """根据 phones 层构建 words 层。

    处理规则：
      * 从当前音素开始向后匹配字典，取**最长命中**（避免 za 抢先匹配掉 zai）；
      * 匹配不到则**不合并**，直接把当前音素作为单独区间写入 words 层；
      * ``AP / SP / EP`` 不参与组词，原样写入（SP 统一写成 SP）。

    Returns:
        ``(words_tier, 未匹配并按单音素写入的数量)``
    """
    words_tier = IntervalTier(name="words", minTime=min_time, maxTime=max_time)

    # 先整理 phones 层，并去掉可能存在的语言前缀（zh/、ja/ 等）
    # 空标记保留在列表里，作为「不能跨过去组词」的屏障
    phones = []
    for interval in phones_tier:
        raw_mark = interval.mark
        phone = "" if raw_mark is None else str(raw_mark).split("/")[-1].strip()
        phones.append((interval.minTime, interval.maxTime, phone))

    fallback_phones = 0
    max_word_len = max(1, trie.depth)
    index = 0
    total = len(phones)

    while index < total:
        start, end, phone = phones[index]

        # 空标记：phones 层的空隙，不写入 words 层（写入时会自然保留为空区间）
        if not phone:
            index += 1
            continue

        # 静音类音素不参与组词，直接写入
        if phone in SILENCE_PHONES:
            words_tier.add(start, end, "SP" if phone == "SP" else phone)
            index += 1
            continue

        # 从当前位置向后尝试匹配字典中的词
        # 取**最长**匹配：字典里同时有 za(ts a) 和 zai(ts a :\i) 时，
        # 碰到 ts a :\i 必须整体命中 zai，而不是先命中 za、把 :\i 剩下
        matched = None
        seq = []
        for end_index in range(index, min(index + max_word_len, total)):
            _, _, next_phone = phones[end_index]
            if not next_phone or next_phone in SILENCE_PHONES:
                break  # 不允许跨静音或空隙组词
            seq.append(next_phone)
            if not trie.has_prefix(seq):
                break
            word = trie.search(seq)
            if word:
                matched = (end_index, word)  # 不 break，继续试更长的词

        if matched is not None:
            end_index, word = matched
            words_tier.add(start, phones[end_index][1], word)
            index = end_index + 1
        else:
            # 匹配不到：不合并，当前音素单独写入 words 层
            words_tier.add(start, end, phone)
            fallback_phones += 1
            index += 1

    return words_tier, fallback_phones


def run(
    tg_dir,
    dictionary,
    out_dir=None,
    overwrite: bool = True,
    recursive: bool = True,
    progress_callback: Optional[Callable[[str], None]] = None,
    stop_event=None,
):
    """批量从 phones 层复原 words 层。

    Args:
        tg_dir: TextGrid 所在文件夹。
        dictionary: DS 字典文件（``词<TAB>音素序列``）。
        out_dir: 输出文件夹；为 ``None`` 时原地覆盖。
        overwrite: 输出文件已存在时是否覆盖。
        recursive: 是否递归处理子文件夹。
        progress_callback: 进度回调，接收一行文本。
        stop_event: ``threading.Event``，置位后停止处理。

    Returns:
        统计字典：total / processed / skipped / failed / fallback / stopped。
    """
    tg_path_in = pathlib.Path(tg_dir)
    dict_path = pathlib.Path(dictionary)

    if not tg_path_in.is_dir():
        raise FileNotFoundError(f"TextGrid 文件夹不存在: {tg_path_in}")
    if not dict_path.is_file():
        raise FileNotFoundError(f"字典文件不存在: {dict_path}")

    trie = Trie(dict_path)
    for phone in SILENCE_PHONES:
        trie.insert([phone], phone)

    files = sorted(tg_path_in.rglob("*.TextGrid") if recursive else tg_path_in.glob("*.TextGrid"))
    out_root = pathlib.Path(out_dir) if out_dir else None

    stats = {
        "total": len(files),
        "processed": 0,
        "skipped": 0,
        "failed": 0,
        "fallback": 0,
        "stopped": False,
    }

    def log(message: str):
        if progress_callback is not None:
            progress_callback(message)
        else:
            print(message)

    if not files:
        log(f"在 {tg_path_in} 中未找到 .TextGrid 文件")
        return stats

    for index, tgfile in enumerate(files, 1):
        if stop_event is not None and stop_event.is_set():
            stats["stopped"] = True
            log("已请求停止，剩余文件未处理。")
            break

        rel_path = tgfile.relative_to(tg_path_in)
        out_file = (out_root / rel_path) if out_root else tgfile

        if out_file.exists() and not overwrite:
            stats["skipped"] += 1
            log(f"[跳过] {rel_path}（已存在）")
            continue

        try:
            tg = TextGrid()
            tg.read(str(tgfile))

            # 只读取 phones 层；旧 words 层的内容完全不参与计算
            phones_tier = _find_phones_tier(tg)
            old_words_count = sum(1 for tier in tg.tiers if _is_words_tier(tier))

            words_tier, fallback = build_words_tier(
                phones_tier, trie, min_time=tg.minTime, max_time=tg.maxTime
            )
            _replace_words_tier(tg, words_tier)

            out_file.parent.mkdir(parents=True, exist_ok=True)
            tg.write(str(out_file))

            stats["processed"] += 1
            stats["fallback"] += fallback
            suffix = f"（{fallback} 个音素未匹配到词，已按单音素写入）" if fallback else ""
            log(
                f"({index}/{len(files)}) {rel_path} → 使用 phones 层 '{phones_tier.name}'，"
                f"丢弃旧 words 层 {old_words_count} 个，重建 {len(words_tier.intervals)} 个区间{suffix}"
            )
        except Exception as exc:  # noqa: BLE001 - GUI 需要展示单文件错误后继续
            stats["failed"] += 1
            log(f"[失败] {rel_path}: {exc}")

    log(
        f"完成：共 {stats['total']} 个文件，成功 {stats['processed']}，"
        f"跳过 {stats['skipped']}，失败 {stats['failed']}，"
        f"未匹配音素（已按单音素写入） {stats['fallback']}"
    )
    return stats
