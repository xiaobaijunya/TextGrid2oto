# -*- coding: utf-8 -*-
"""把 transcriptions.csv 反推为 TextGrid。

层名约定见 ``build_diffdataset/tg_schema.py``：

    words       词区间。优先使用 CSV 里的 word_seq（与 ph_num 一一对应）；
                否则与「从 phones 复原 words 层」完全一致：拿字典对 phones 贪心匹配组词，
                匹配不到的音素不合并，直接按单音素写入
    phones      音素区间（ph_seq / ph_dur）
    ph-num      与 ph_num 分组对齐的层，mark 为每组音素个数（可选，write_ph_num=True 时输出）
    notes       音符区间（note_seq / note_dur），尾部与 phones 对齐
    note-slur   与 notes 对齐的 0/1 连音标记（note_slur 列）

层名不写下划线（Praat 会当成下标），也不留空格，统一用连字符。

示例：

    python -m build_diffdataset.csv2tg transcriptions.csv --out-dir TextGrid
"""

from __future__ import annotations

import argparse
import csv
import math
import pathlib
from typing import Callable, Optional

from textgrid import IntervalTier, TextGrid

try:  # 包内导入
    from .tg_schema import NOTE_SLUR_TIER, PH_NUM_TIER
    from .words_from_phones import SILENCE_PHONES, Trie, build_words_tier
except ImportError:  # 直接运行脚本
    from tg_schema import NOTE_SLUR_TIER, PH_NUM_TIER
    from words_from_phones import SILENCE_PHONES, Trie, build_words_tier


def _add_interval(tier: IntervalTier, start: float, end: float, mark: str, total: float) -> float:
    """向层里加入区间，并把浮点累加误差夹回层边界。

    ``IntervalTier.addInterval`` 在 ``end > maxTime`` 时会 ``raise ValueError(maxTime)``，
    消息只有一个浮点数；逐段累加时长比 ``sum()`` 多出几个 ulp 就会触发，
    因此这里对超出总时长的微小误差做夹紧处理。返回实际写入的终点。
    """
    if end > total:
        if end - total > 1e-3:
            raise ValueError(f"区间终点 {end:.6f}s 超出总时长 {total:.6f}s")
        end = total
    if end < start:
        end = start
    tier.add(start, end, mark)
    return end


def run(
    transcription,
    out_dir=None,
    dictionary=None,
    overwrite: bool = True,
    progress_callback: Optional[Callable[[str], None]] = None,
):
    """把 transcriptions.csv 中每一行写成一个 TextGrid 文件。

    Args:
        transcription: transcriptions.csv 路径。
        out_dir: 输出目录；为空时写到 CSV 同目录。
        dictionary: 可选的两段式字典，用于恢复 words 层文字。
        overwrite: 目标 TextGrid 已存在时是否覆盖。
        progress_callback: 进度回调。

    Returns:
        统计字典：total / written / skipped / failed / output_dir。
    """
    csv_path = pathlib.Path(transcription)
    if not csv_path.is_file():
        raise FileNotFoundError(f"transcriptions.csv 不存在: {csv_path}")

    out_root = pathlib.Path(out_dir) if out_dir else csv_path.parent
    out_root.mkdir(parents=True, exist_ok=True)

    trie = None
    if dictionary:
        dictionary_path = pathlib.Path(dictionary)
        if not dictionary_path.is_file():
            raise FileNotFoundError(f"字典文件不存在: {dictionary_path}")
        trie = Trie(dictionary_path)
        # 与 words_from_phones.run() 一致：把静音音素也放进 trie，避免跨静音组词
        for phone in SILENCE_PHONES:
            trie.insert([phone], phone)

    def log(message: str):
        if progress_callback is not None:
            progress_callback(message)
        else:
            print(message)

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        items = list(reader)

    if not items:
        raise ValueError("transcriptions.csv 没有数据行")

    stats = {
        "total": len(items),
        "written": 0,
        "skipped": 0,
        "failed": 0,
        "output_dir": str(out_root),
    }

    for index, item in enumerate(items, 1):
        name = (item.get("name") or "").strip()
        if not name:
            stats["failed"] += 1
            log(f"[失败] 第 {index} 行缺少 name")
            continue

        out_file = out_root / f"{name}.TextGrid"
        if out_file.exists() and not overwrite:
            stats["skipped"] += 1
            log(f"[跳过] {out_file.name}（已存在）")
            continue

        try:
            ph_seq = (item.get("ph_seq") or "").split()
            ph_dur = [float(v) for v in (item.get("ph_dur") or "").split()]
            if len(ph_seq) != len(ph_dur):
                raise ValueError(f"ph_seq ({len(ph_seq)}) 与 ph_dur ({len(ph_dur)}) 长度不一致")

            ph_num = None
            if item.get("ph_num"):
                try:
                    ph_num = [int(v) for v in item["ph_num"].split()]
                except ValueError:
                    ph_num = None
                if ph_num is not None and sum(ph_num) != len(ph_seq):
                    log(f"[警告] {name}: ph_num 之和与音素数不一致，ph_num 已忽略")
                    ph_num = None

            note_seq = (item.get("note_seq") or "").split()
            note_dur = [float(v) for v in (item.get("note_dur") or "").split()]
            note_slur = (item.get("note_slur") or "").split()
            word_seq = (item.get("word_seq") or "").split()

            phone_total = math.fsum(ph_dur)
            note_total = math.fsum(note_dur) if note_dur else 0.0

            # 音符总时长与音素总时长存在微小误差时（例如 CSV 里 note_dur 只保留 3 位小数），
            # 把差值补到最后一个音符上，让 notes 与 phones 的尾部完全对齐
            if note_dur and abs(phone_total - note_total) <= 0.05:
                delta = phone_total - note_total
                if delta:
                    note_dur[-1] = max(0.0, note_dur[-1] + delta)
                    note_total = math.fsum(note_dur)
                    if abs(delta) >= 1e-6:  # 浮点噪声不刷日志
                        log(f"[提示] {name}: 音符总时长与音素相差 {delta:+.6f}s，已对齐到音素总时长")

            # 对齐后再计算总时长：TextGrid 写文件时每个层都会用这个 xmax，
            # 若它大于层内最后一个区间，重读时会多出尾部空区间
            total = max(phone_total, note_total)
            if total <= 0:
                raise ValueError("总时长为 0")

            tg = TextGrid(minTime=0.0, maxTime=total)

            # ── phones（必需）──
            phones_tier = IntervalTier("phones", 0.0, phone_total)
            cursor = 0.0
            for phone, duration in zip(ph_seq, ph_dur):
                cursor = _add_interval(phones_tier, cursor, cursor + duration, phone, phone_total)

            # ── words（可选）──
            # 优先用 CSV 里的 word_seq（与 ph_num 一一对应，能从原 TextGrid 无损恢复）；
            # 否则与「从 phones 复原 words 层」完全一致：字典对 phones 贪心组词，
            # 匹配不到的音素不合并，直接按单音素写入
            words_tier = None
            if word_seq and ph_num and len(word_seq) == len(ph_num):
                words_tier = IntervalTier("words", 0.0, phone_total)
                ph_index = 0
                cursor = 0.0
                for label, num in zip(word_seq, ph_num):
                    word_dur = math.fsum(ph_dur[ph_index:ph_index + num])
                    cursor = _add_interval(words_tier, cursor, cursor + word_dur, label, phone_total)
                    ph_index += num
            elif trie is not None:
                words_tier, fallback = build_words_tier(
                    phones_tier, trie, min_time=0.0, max_time=phone_total,
                )
                hint = f"，{fallback} 个音素未匹配到词（已按单音素写入）" if fallback else ""
                log(f"[词语] {name}: 由 phones 重建 {len(words_tier.intervals)} 个词区间{hint}")
            elif ph_num:
                log(f"[提示] {name}: 未提供字典，已跳过 words 层（选字典可重建 words）")

            if words_tier is not None:
                tg.append(words_tier)
            tg.append(phones_tier)

            # ── ph-num（CSV 有 ph_num 时必写）：与分组严格对齐，mark = 该组音素个数 ──
            if ph_num:
                ph_num_tier = IntervalTier(PH_NUM_TIER, 0.0, phone_total)
                ph_index = 0
                cursor = 0.0
                for num in ph_num:
                    group_dur = math.fsum(ph_dur[ph_index:ph_index + num])
                    cursor = _add_interval(ph_num_tier, cursor, cursor + group_dur, str(num), phone_total)
                    ph_index += num
                tg.append(ph_num_tier)

            # ── notes（可选）──
            has_notes = bool(note_seq) and len(note_seq) == len(note_dur)
            if note_seq and not has_notes:
                log(f"[警告] {name}: note_seq 与 note_dur 长度不一致，已跳过 notes 层")

            if has_notes:
                if abs(phone_total - note_total) > 0.01:
                    log(f"[警告] {name}: 音素总时长 {phone_total:.3f}s 与音符总时长 {note_total:.3f}s 不一致")

                notes_tier = IntervalTier("notes", 0.0, note_total)
                cursor = 0.0
                for note, duration in zip(note_seq, note_dur):
                    cursor = _add_interval(notes_tier, cursor, cursor + duration, note, note_total)
                tg.append(notes_tier)

                # ── note slur（可选）──
                if note_slur and len(note_slur) == len(note_seq) and all(v in ("0", "1") for v in note_slur):
                    slur_tier = IntervalTier(NOTE_SLUR_TIER, 0.0, note_total)
                    cursor = 0.0
                    for slur, duration in zip(note_slur, note_dur):
                        cursor = _add_interval(slur_tier, cursor, cursor + duration, slur, note_total)
                    tg.append(slur_tier)

            tg.write(str(out_file))
            stats["written"] += 1
            extras = []
            if words_tier is not None:
                extras.append(f"words={len(words_tier.intervals)}")
            if ph_num:
                extras.append(f"ph_num={len(ph_num)}")
            if has_notes:
                extras.append(f"notes={len(note_seq)}")
            suffix = ("（" + ", ".join(extras) + "）") if extras else ""
            log(f"({index}/{len(items)}) {name} → {out_file.name}{suffix}")
        except Exception as exc:  # noqa: BLE001 - 单文件失败后继续
            stats["failed"] += 1
            log(f"[失败] {name}: {type(exc).__name__}: {exc}")

    log(
        f"完成：共 {stats['total']} 行，写出 {stats['written']}，"
        f"跳过 {stats['skipped']}，失败 {stats['failed']} → {out_root}"
    )
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert transcriptions.csv back to TextGrid files")
    parser.add_argument("transcription", help="Path to transcriptions.csv")
    parser.add_argument("--out-dir", default=None, help="Output directory (default: CSV directory)")
    parser.add_argument("--dictionary", default=None, help="Optional dictionary for restoring word labels")
    parser.add_argument("--no-overwrite", action="store_true", help="Skip existing TextGrid files")
    args = parser.parse_args()

    run(
        args.transcription,
        out_dir=args.out_dir,
        dictionary=args.dictionary,
        overwrite=not args.no_overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
