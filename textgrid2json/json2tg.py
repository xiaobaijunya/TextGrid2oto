# -*- coding: utf-8 -*-
"""由 json 生成 TextGrid（「生成 json」的反向操作）。

用于 SVDB 面板：把每个文件夹里 ``json/`` 下的标注重新写成 TextGrid，

    <folder>/json/ds_phone.json    音素级 -> phones 层（未指定字典时使用）
    <folder>/json/word_phone.json  词级   -> words 层；指定 SV 字典时，
                                          再用它把每个词切成 SV 音素写入 phones 层

输出 ``<folder>/<name>.TextGrid``，遍历方式与「生成 json」一致：
先处理根文件夹，再处理第一层子文件夹。
"""

from __future__ import annotations

import json
import pathlib
from typing import Callable, Optional

from textgrid import IntervalTier, TextGrid

PHONES_JSON = "ds_phone.json"
WORDS_JSON = "word_phone.json"

# 这些拼音在 SV 字典里通常是“元音 + 尾音”两个音素，用音尾比例切分
DIPHTHONG_WORDS = {"ai", "ei", "an", "en", "ang", "ong", "eng", "er"}


def _read_json(path: pathlib.Path) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def read_dictionary(path) -> dict:
    """读取 SV 字典：每行 ``词<TAB>音素 音素 ...``（DiffSinger 规范）。

    没有制表符的行退回「第一个空白分隔」的写法。
    """
    dic = {}
    with open(path, "r", encoding="utf-8-sig") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) < 2:
                parts = line.split(None, 1)
            if len(parts) < 2:
                continue
            dic[parts[0].strip()] = parts[1].split()
    dic.setdefault("SP", ["sil"])
    dic.setdefault("R", ["sil"])
    return dic


def _entries(item: dict) -> tuple[list[tuple[float, float, float, str]], list[str]]:
    """把 ``{"1": {xmin, middle, xmax, text}, ...}`` 转成按序号排序的区间列表。

    Praat 不支持时长 ≤ 0 的区间，json 里偶尔会有 ``xmin == xmax`` 的占位项
    （例如开头的零长度 SP），这里直接跳过，并把跳过项的序号与文字返回，便于排查。

    Returns:
        ``([(xmin, middle, xmax, text)], 被跳过的描述列表)``。
    """
    phones = item.get("phones") or {}
    result = []
    skipped = []
    for key in sorted(phones, key=lambda k: int(k) if str(k).isdigit() else 0):
        entry = phones[key]
        text = str(entry.get("text", "")).strip() if isinstance(entry, dict) else ""
        try:
            xmin = float(entry["xmin"])
            xmax = float(entry["xmax"])
            middle = float(entry.get("middle", xmin))
        except (KeyError, TypeError, ValueError):
            skipped.append(f"#{key} '{text}'（坐标无效）")
            continue
        if xmax <= xmin:
            skipped.append(f"#{key} '{text}'（{xmin:g}–{xmax:g}）")
            continue
        result.append((xmin, middle, xmax, text))
    return result, skipped


def _total(item: dict, entries: list[tuple[float, float, float, str]]) -> float:
    wav_long = item.get("wav_long") or []
    if len(wav_long) >= 2:
        try:
            return float(wav_long[1])
        except (TypeError, ValueError):
            pass
    return max((xmax for _, _, xmax, _ in entries), default=0.0)


def _split_bounds(text: str, xmin: float, middle: float, xmax: float,
                  count: int, tail_ratio: float) -> list[float]:
    """按 ``tg2svdb/tg2sv_change.word2phone`` 的规则计算切分点。"""
    if count <= 1:
        return [xmin]
    if text in DIPHTHONG_WORDS:
        return [xmin, xmax - (xmax - xmin) * tail_ratio / 100]
    if count == 2:
        return [xmin, middle]
    if count == 3:
        return [xmin, middle, xmax - (xmax - middle) * tail_ratio / 100]
    # 超过 3 个音素：均分
    step = (xmax - xmin) / count
    return [xmin + step * i for i in range(count)]


def _phones_from_words(word_entries, dictionary: dict, tail_ratio: float):
    """用 SV 字典把词切成音素区间。"""
    phones = []
    skipped = []
    for xmin, middle, xmax, text in word_entries:
        seq = dictionary.get(text)
        if not seq:
            phones.append((xmin, xmin, xmax, text))
            skipped.append(f"words '{text}'（字典未收录，保留原标签）")
            continue
        bounds = _split_bounds(text, xmin, middle, xmax, len(seq), tail_ratio)
        for i, phone in enumerate(seq):
            start = bounds[i]
            end = bounds[i + 1] if i + 1 < len(bounds) else xmax
            phones.append((start, start, end, phone))
    return phones, skipped


def build_textgrid(phones_item: Optional[dict], words_item: Optional[dict],
                   dictionary: Optional[dict] = None, tail_ratio: float = 50.0):
    """由同一段音频的 json 数据构建 TextGrid（words 层在前，phones 层在后）。

    给定 ``dictionary`` 时，phones 层由 words + 字典生成（SV 音素标签）；
    否则直接使用 ds_phone.json 里的音素。

    Returns:
        ``(TextGrid 或 None, 跳过/警告描述列表)``。
    """
    word_entries, word_skipped = _entries(words_item) if words_item else ([], [])
    skipped = [f"words {item}" for item in word_skipped]

    if dictionary and word_entries:
        phone_entries, conv_skipped = _phones_from_words(word_entries, dictionary, tail_ratio)
        skipped += conv_skipped
    else:
        phone_entries, phone_skipped = _entries(phones_item) if phones_item else ([], [])
        skipped += [f"phones {item}" for item in phone_skipped]

    if not phone_entries and not word_entries:
        return None, skipped

    total = max(_total(phones_item or {}, phone_entries),
                _total(words_item or {}, word_entries))
    if total <= 0:
        return None, skipped

    grid = TextGrid(minTime=0.0, maxTime=total)
    for name, entries in (("words", word_entries), ("phones", phone_entries)):
        if not entries:
            continue
        tier = IntervalTier(name, 0.0, total)
        for xmin, _middle, xmax, text in entries:
            start, end = min(xmin, total), min(xmax, total)
            if end <= start:  # 夹到末尾后可能退化成零长度
                skipped.append(f"{name} '{text}'（{start:g}–{end:g}，时长为 0）")
                continue
            tier.add(start, end, text)
        grid.append(tier)
    return grid, skipped


def run(folder, overwrite: bool = True, dictionary=None, tail_ratio: float = 50.0,
        progress_callback: Optional[Callable[[str], None]] = None) -> dict:
    """把 ``folder`` 及第一层子文件夹里的 json 生成 TextGrid。

    Args:
        folder: 声库根目录（每个样本目录下有 ``json/``）。
        overwrite: 目标 TextGrid 已存在时是否覆盖。
        dictionary: 可选的 SV 字典路径；给定后用词 + 字典生成 phones 层。
        tail_ratio: 音尾比例（用于 ai/an 等双音素词的切分），与 SVDB 生成一致。
        progress_callback: 进度回调。

    Returns:
        统计字典：total / written / skipped / failed / output_dir。
    """
    root = pathlib.Path(folder)
    if not root.is_dir():
        raise FileNotFoundError(f"文件夹不存在: {root}")

    dic = None
    if dictionary:
        dic_path = pathlib.Path(dictionary)
        if not dic_path.is_file():
            raise FileNotFoundError(f"字典文件不存在: {dic_path}")
        dic = read_dictionary(dic_path)

    def log(message: str):
        if progress_callback is not None:
            progress_callback(message)
        else:
            print(message)

    if dic is not None:
        log(f"音素标签：按字典 {pathlib.Path(dictionary).name} 转换（音尾比例 {tail_ratio:g}%）")
    else:
        log("音素标签：直接使用 json 中的 phones（未指定 SV 字典）")

    stats = {"total": 0, "written": 0, "skipped": 0, "failed": 0, "output_dir": str(root)}

    def process(folder_path: pathlib.Path, folder_name: str = ""):
        json_dir = folder_path / "json"
        phones_path = json_dir / PHONES_JSON
        words_path = json_dir / WORDS_JSON
        if not phones_path.is_file() and not words_path.is_file():
            return 0

        display = folder_name or str(folder_path)
        log(f"[处理] {display}")

        phones_data = _read_json(phones_path) if phones_path.is_file() else {}
        words_data = _read_json(words_path) if words_path.is_file() else {}

        written = 0
        for name in sorted(set(phones_data) | set(words_data)):
            stats["total"] += 1
            out_file = folder_path / f"{name}.TextGrid"
            if out_file.exists() and not overwrite:
                stats["skipped"] += 1
                log(f"[跳过] {out_file.name}（已存在）")
                continue
            try:
                grid, skipped = build_textgrid(
                    phones_data.get(name), words_data.get(name),
                    dictionary=dic, tail_ratio=tail_ratio,
                )
                if grid is None:
                    raise ValueError("json 中没有可用的区间")
                grid.write(str(out_file))
                stats["written"] += 1
                written += 1
                note = f"，跳过 {len(skipped)} 个零长度区间" if skipped else ""
                log(f"  → {out_file.name}（{', '.join(t.name for t in grid.tiers)}{note}）")
                for item in skipped:
                    log(f"      跳过: {item}")
            except Exception as exc:  # noqa: BLE001 - 单文件失败后继续
                stats["failed"] += 1
                log(f"[失败] {name}: {type(exc).__name__}: {exc}")
        return written

    process(root)
    for sub in sorted(p for p in root.iterdir() if p.is_dir() and p.name != "json"):
        process(sub, sub.name)

    log(f"完成：共 {stats['total']} 项，写出 {stats['written']}，"
        f"跳过 {stats['skipped']}，失败 {stats['failed']} → {root}")
    return stats
