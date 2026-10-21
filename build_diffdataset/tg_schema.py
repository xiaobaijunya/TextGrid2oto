# -*- coding: utf-8 -*-
"""TextGrid 层命名约定（CSV ↔ TextGrid 双向转换共用）。

    words      词区间（可选）
    phones     音素区间（必需）
    ph-num     与 ph_num 分组对齐，mark 为该组包含的音素个数（可选）
    notes      音符区间，mark 为音名（如 C4+10 / rest）
    note-slur  与 notes 对齐的连音标记，mark 为 0 / 1

层名不写下划线：Praat 会把 ``_`` 当作下标标记（``note_slur`` 会显示成
"note" + 下标 s + "lur"），也不留空格（看不明显），统一用连字符分隔。
"""

from __future__ import annotations

WORDS_TIER = "words"
PHONES_TIER = "phones"
PH_NUM_TIER = "ph-num"
NOTES_TIER = "notes"
NOTE_SLUR_TIER = "note-slur"

# CSV 列顺序（导出时按这个顺序写）
CSV_COLUMN_ORDER = (
    "name",
    "ph_seq",
    "ph_dur",
    "ph_num",
    "word_seq",
    "note_seq",
    "note_dur",
    "note_slur",
)


def tier_name(tier) -> str:
    """统一层名：去空格、转小写。"""
    return str(getattr(tier, "name", "")).strip().lower()


def find_tier(grid, name: str):
    """在 TextGrid 中按层名查找层（不区分大小写）。"""
    target = name.strip().lower()
    for tier in grid.tiers:
        if tier_name(tier) == target:
            return tier
    return None


def is_interval_tier(tier) -> bool:
    return tier is not None and hasattr(tier, "intervals")


def sample_interval_mark(tier, time_point: float) -> str:
    """取 IntervalTier 在某个时间点的 mark；找不到返回空字符串。"""
    if not is_interval_tier(tier):
        return ""
    for interval in tier.intervals:
        if interval.minTime <= time_point < interval.maxTime:
            return str(interval.mark).strip()
    return ""
