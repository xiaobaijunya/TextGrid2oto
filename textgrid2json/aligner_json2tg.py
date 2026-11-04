# -*- coding: utf-8 -*-
"""把 ``xxx_Aligner.json`` / ``xxx_ASR.json``（Qwen3-ForcedAligner 逐字时间戳）转成 Praat TextGrid。

JSON 结构（由 ``asr_lab/asr_suite.py`` 生成）::

    {
      "audio": "song.wav",
      "duration": 270.593,
      "lab": "sui yue nan de ...",
      "items": [
        {"text": "岁", "pinyin": "sui", "start": 0.16, "end": 0.64},
        ...
      ]
    }

生成的 TextGrid 默认两层：

    hanzi   汉字（items[].text）
    words   拼音（items[].pinyin）

用法::

    from textgrid2json import aligner_json2tg
    aligner_json2tg.convert_file("a_Aligner.json")
    aligner_json2tg.convert_folder(r"D:\\corpus", recursive=True)
"""

from __future__ import annotations

import json
import pathlib
from typing import Callable, Optional, Sequence

from textgrid import IntervalTier, TextGrid

# 与 asr_lab/asr_suite.py 的输出文件名保持一致；同时兼容旧版 _Aligner.json
ALIGNER_SUFFIX = "_Aligner"
ASR_SUFFIX = "_ASR"
ALIGNER_SUFFIXES = (ALIGNER_SUFFIX, ASR_SUFFIX)
ALIGNER_JSON_GLOB = f"*{ALIGNER_SUFFIX}.json"
ASR_JSON_GLOB = f"*{ASR_SUFFIX}.json"

TIER_BOTH = "both"
TIER_WORDS = "words"
TIER_PHONES = "phones"
TIER_MODES = (TIER_BOTH, TIER_WORDS, TIER_PHONES)

# 零时长 / 重叠的词会自动延长到该时长（秒），避免 Praat 因零长度区间丢词。
MIN_INTERVAL_SEC = 0.01
_EPS = 1e-4


def _read_json(path: pathlib.Path) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _normalize_entries(
    data: dict,
    min_interval: float = MIN_INTERVAL_SEC,
) -> list[tuple[float, float, str, str]]:
    """把 items 规整成 ``[(start, end, text, pinyin)]``。

    - 丢掉文本和拼音都为空的项；
    - 按开始时间排序；
    - 出现重叠时把当前项裁到上一项的结束点；
    - **起止时间相同（零时长）或裁剪后不足最小时值的项，会向后延长到
      ``min_interval``**，避免在 TextGrid 里丢词（后续项相应顺延）。

    Args:
        data: Aligner JSON 数据。
        min_interval: 最小时值（秒）。<=0 时会退化为 0.1 毫秒。
    """
    minimum = max(float(min_interval or 0.0), _EPS)
    raw_items = data.get("items") or []
    try:
        duration = float(data.get("duration") or 0.0)
    except (TypeError, ValueError):
        duration = 0.0

    parsed: list[tuple[float, float, str, str]] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        try:
            start = float(item.get("start", 0.0))
            end = float(item.get("end", 0.0))
        except (TypeError, ValueError):
            continue
        if duration > 0:
            start = min(max(start, 0.0), duration)
            end = min(max(end, 0.0), duration)
        else:
            start = max(start, 0.0)
            end = max(end, 0.0)
        text = str(item.get("text", "") or "").strip()
        pinyin = str(item.get("pinyin", "") or "").strip()
        if not text and not pinyin:
            continue
        if end < start:
            end = start
        parsed.append((start, end, text, pinyin))

    # 稳定排序：同一时间戳的多项保持原始顺序
    parsed.sort(key=lambda entry: (entry[0], entry[1]))

    result: list[tuple[float, float, str, str]] = []
    prev_end = 0.0
    for start, end, text, pinyin in parsed:
        if start < prev_end:
            start = prev_end
        if end < start:
            end = start
        if end - start < minimum:
            end = start + minimum
        result.append((start, end, text, pinyin))
        prev_end = end
    return result


def build_textgrid(
    data: dict,
    tier_mode: str = TIER_BOTH,
    min_interval: float = MIN_INTERVAL_SEC,
) -> tuple[Optional[TextGrid], int]:
    """由 Aligner JSON 数据构建 TextGrid。

    Args:
        data: Aligner JSON 数据。
        tier_mode: ``both`` / ``words`` / ``phones``。
        min_interval: 零时长项的最小时值（秒）。

    Returns:
        ``(TextGrid 或 None, 写入的区间数)``。
    """
    tier_mode = tier_mode if tier_mode in TIER_MODES else TIER_BOTH
    entries = _normalize_entries(data, min_interval=min_interval)
    if not entries:
        return None, 0

    try:
        total = float(data.get("duration") or 0.0)
    except (TypeError, ValueError):
        total = 0.0
    total = max(total, entries[-1][1])
    if total <= 0:
        return None, 0

    grid = TextGrid(minTime=0.0, maxTime=total)
    written = 0

    def add_tier(name: str, value_index: int) -> None:
        nonlocal written
        tier = IntervalTier(name, 0.0, total)
        count = 0
        for entry in entries:
            mark = entry[value_index]
            if not mark:
                continue
            tier.add(entry[0], entry[1], mark)
            count += 1
        if count:
            grid.append(tier)
            written += count

    if tier_mode in (TIER_BOTH, TIER_WORDS):
        # 汉字层
        add_tier("hanzi", 2)
    if tier_mode in (TIER_BOTH, TIER_PHONES):
        # 拼音层（层名用 words，便于 TIFA / lab_from_words 直接读取）
        add_tier("words", 3)

    if not grid.tiers:
        return None, 0
    return grid, written


def output_path_for(json_path: pathlib.Path, out_dir: Optional[pathlib.Path] = None) -> pathlib.Path:
    """``xxx_Aligner.json`` / ``xxx_ASR.json`` -> ``xxx.TextGrid``（与音频同名）。"""
    json_path = pathlib.Path(json_path)
    stem = json_path.stem
    for suffix in ALIGNER_SUFFIXES:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    name = f"{stem}.TextGrid"
    if out_dir is not None:
        return pathlib.Path(out_dir) / name
    return json_path.with_name(name)


def convert_file(
    json_path,
    out_path=None,
    tier_mode: str = TIER_BOTH,
    overwrite: bool = True,
    min_interval: float = MIN_INTERVAL_SEC,
) -> pathlib.Path:
    """转换单个 ``xxx_Aligner.json``，返回写出的 TextGrid 路径。"""
    json_path = pathlib.Path(json_path)
    target = pathlib.Path(out_path) if out_path else output_path_for(json_path)
    if target.exists() and not overwrite:
        return target

    data = _read_json(json_path)
    grid, _count = build_textgrid(data, tier_mode=tier_mode, min_interval=min_interval)
    if grid is None:
        raise ValueError(f"JSON 中没有可用的逐字时间戳：{json_path}")

    target.parent.mkdir(parents=True, exist_ok=True)
    grid.write(str(target))
    return target


def convert_folder(
    folder,
    recursive: bool = False,
    out_dir: str = "",
    tier_mode: str = TIER_BOTH,
    overwrite: bool = True,
    min_interval: float = MIN_INTERVAL_SEC,
    progress_callback: Optional[Callable[[str], None]] = None,
    cancel_checker: Optional[Callable[[], bool]] = None,
) -> dict:
    """批量把文件夹里的 ``*_Aligner.json`` 转成 TextGrid。

    Args:
        folder: 搜索根目录。
        recursive: 是否包含子文件夹（输出会保持相对目录结构）。
        out_dir: 输出目录；留空则写在 JSON 同目录。
        tier_mode: ``both`` / ``words`` / ``phones``。
        overwrite: 目标已存在时是否覆盖。
        min_interval: 零时长项的最小时值（秒）。
        progress_callback: 进度回调（接收一行文本）。
        cancel_checker: 返回 True 时中断。

    Returns:
        统计字典：total / written / skipped / failed / errors / output_dir。
    """
    root = pathlib.Path(folder)
    if not root.is_dir():
        raise FileNotFoundError(f"文件夹不存在: {root}")

    out_root = pathlib.Path(out_dir) if str(out_dir or "").strip() else None

    walker: Sequence[pathlib.Path]
    if recursive:
        walker = sorted(set(root.rglob(ALIGNER_JSON_GLOB)) | set(root.rglob(ASR_JSON_GLOB)))
    else:
        walker = sorted(set(root.glob(ALIGNER_JSON_GLOB)) | set(root.glob(ASR_JSON_GLOB)))

    def log(message: str) -> None:
        if progress_callback is not None:
            progress_callback(message)
        else:
            print(message)

    stats = {
        "total": len(walker),
        "written": 0,
        "skipped": 0,
        "failed": 0,
        "errors": [],
        "output_dir": str(out_root or root),
    }
    if not walker:
        log(f"未找到 {ALIGNER_JSON_GLOB} / {ASR_JSON_GLOB}：{root}")
        return stats

    log(f"共 {len(walker)} 个 Aligner JSON（层级：{tier_mode}）")

    for index, json_path in enumerate(walker, start=1):
        if cancel_checker is not None and cancel_checker():
            log("已停止。")
            break

        if out_root is not None:
            relative_dir = json_path.parent.relative_to(root)
            target = output_path_for(json_path, out_root / relative_dir)
        else:
            target = output_path_for(json_path)

        if target.exists() and not overwrite:
            stats["skipped"] += 1
            log(f"[{index}/{len(walker)}] 跳过（已存在）：{target.name}")
            continue

        try:
            convert_file(
                json_path,
                out_path=target,
                tier_mode=tier_mode,
                overwrite=True,
                min_interval=min_interval,
            )
            stats["written"] += 1
            log(f"[{index}/{len(walker)}] ✓ {json_path.name} -> {target.name}")
        except Exception as error:  # noqa: BLE001 - 逐个文件容错
            stats["failed"] += 1
            stats["errors"].append(f"{json_path.name}: {error}")
            log(f"[{index}/{len(walker)}] ✗ {json_path.name}：{error}")

    log(
        f"完成：成功 {stats['written']} / 跳过 {stats['skipped']} / 失败 {stats['failed']}"
        f" / 共 {stats['total']}"
    )
    return stats
