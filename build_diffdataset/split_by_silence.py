# -*- coding: utf-8 -*-
"""按 AP / SP 等静音区间把音频切成多段。

切分点在静音区间上：**相邻两段都会完整包含该静音**，所以直接拼接时不会断开
（相邻段之间只有静音重叠，不会丢内容）。

    |---- seg 1 ----|SP|
           |SP|--- seg 2 ----|AP|
                     |AP|--- seg 3 ----|

输出文件名为 ``<原名>_seg001.wav``；同时在同目录写出同名 TextGrid
（``<原名>_seg001.TextGrid``），各层按切分点裁剪并把时间轴平移回 0，
这样切分后的音频仍然带着标记。
"""

from __future__ import annotations

import pathlib
from typing import Callable, Optional

import soundfile as sf
from textgrid import IntervalTier, PointTier, TextGrid

try:  # 包内导入
    from .words_from_phones import _find_phones_tier
except ImportError:  # 直接运行脚本
    from words_from_phones import _find_phones_tier


# 切分点按采样点取整后的误差远小于 1ms；短于该值的边缘残片视为取整碎片
_EDGE_TOLERANCE = 1e-3


def _plain(mark) -> str:
    return str(mark or "").split("/")[-1].strip()


def silence_ranges(phones_tier, silences) -> list[tuple[float, float]]:
    """找出切分区间：指定的静音音素，以及**空白音素**（mark 为空）。

    首尾相连的连续切分区间合并成一个切分点。
    """
    names = {str(item).strip() for item in silences if str(item).strip()}
    ranges: list[tuple[float, float]] = []
    for interval in phones_tier:
        mark = _plain(interval.mark)
        if mark and mark not in names:
            continue  # 既不是指定静音，也不是空白
        if ranges and abs(ranges[-1][1] - interval.minTime) <= 1e-6:
            ranges[-1] = (ranges[-1][0], interval.maxTime)
        else:
            ranges.append((interval.minTime, interval.maxTime))
    return ranges


def _slice_grid(grid: TextGrid, start: float, end: float) -> TextGrid:
    """把整份 TextGrid 裁剪到 ``[start, end]``，时间轴平移到 0。

    只保留与区间有交集的区间（超出部分会被裁掉），层名与层顺序保持不变，
    这样切分出来的每段 wav 都能配上一份从 0 开始的 TextGrid。

    切分点是按采样点取整的，可能与原 TextGrid 边界差几微秒，直接裁剪会在
    段首/段尾留下「超短甚至 0 时长」的碎片区间；这里把这类边缘碎片丢掉，
    并把首尾区间吸附到 0 / 段尾，避免写出计划外的小标签。
    """
    duration = max(0.0, float(end) - float(start))
    sliced = TextGrid(minTime=0.0, maxTime=duration)

    for tier in grid.tiers:
        name = getattr(tier, "name", "")
        if hasattr(tier, "intervals"):  # IntervalTier
            pieces = []  # [start, end, mark]，时间轴已平移到 0
            for interval in tier.intervals:
                lo = max(float(interval.minTime), start)
                hi = min(float(interval.maxTime), end)
                if hi - lo <= 0.0:
                    continue
                # 被切分点裁过、且剩下的部分短于容差：认为是取整碎片，丢弃
                clipped = interval.minTime < start or interval.maxTime > end
                if clipped and hi - lo <= _EDGE_TOLERANCE:
                    continue
                pieces.append([lo - start, hi - start, str(interval.mark or "")])

            if pieces:
                # 首尾吸附：否则写文件时会自动补一个极短的空区间
                if pieces[0][0] <= _EDGE_TOLERANCE:
                    pieces[0][0] = 0.0
                if duration - pieces[-1][1] <= _EDGE_TOLERANCE:
                    pieces[-1][1] = duration

            new_tier = IntervalTier(name, 0.0, duration)
            for lo, hi, mark in pieces:
                if hi - lo <= 0.0:
                    continue
                new_tier.add(lo, hi, mark)
            sliced.append(new_tier)
        elif hasattr(tier, "points"):  # PointTier
            new_tier = PointTier(name, 0.0, duration)
            for point in tier.points:
                if point.time < start - 1e-9 or point.time > end + 1e-9:
                    continue
                when = min(max(float(point.time) - start, 0.0), duration)
                try:
                    new_tier.add(when, str(point.mark or ""))
                except ValueError:
                    pass  # 平移后落在同一时刻的点，保留第一个
            sliced.append(new_tier)

    return sliced


def segments_from_ranges(ranges: list[tuple[float, float]], duration: float) -> list[tuple[float, float]]:
    """按静音区间生成切分结果：每段前后都包含对应的静音。

    开头/结尾整段都是静音时会产生“纯静音段”，这类空段会被跳过。
    """
    if not ranges:
        return [(0.0, duration)]

    bounds = [(0.0, ranges[0][1])]
    for index in range(1, len(ranges)):
        bounds.append((ranges[index - 1][0], ranges[index][1]))
    bounds.append((ranges[-1][0], duration))

    pieces = []
    for start, end in bounds:
        start = max(0.0, min(start, duration))
        end = max(0.0, min(end, duration))
        if end - start <= 1e-6:
            continue
        if any(start >= lo - 1e-6 and end <= hi + 1e-6 for lo, hi in ranges):
            continue  # 整段都是静音，丢弃
        pieces.append((start, end))
    return pieces


def run(wav_dir, tg_dir=None, out_dir=None, silences=("SP", "AP"), recursive: bool = False,
        progress_callback: Optional[Callable[[str], None]] = None, stop_event=None,
        write_tg: bool = True) -> dict:
    """按静音音素切分 wav（可选同步生成分段 TextGrid）。

    Args:
        wav_dir: wav 文件夹。
        tg_dir: TextGrid 文件夹；为空时与 wav 同目录、同名查找。
        out_dir: 输出目录；为空时写到 ``<wav_dir>/segments``。
        silences: 静音音素名（如 SP / AP），可为字符串列表。
        recursive: 是否递归子文件夹（输出保持相对目录结构）。
        progress_callback: 进度回调。
        stop_event: ``threading.Event``，置位后停止。
        write_tg: 是否为每段音频写出同名 TextGrid（默认开启）。

    Returns:
        统计字典：total / written / tg_written / skipped / failed / stopped / output_dir。
    """
    wav_root = pathlib.Path(wav_dir)
    if not wav_root.is_dir():
        raise FileNotFoundError(f"wav 文件夹不存在: {wav_root}")
    tg_root = pathlib.Path(tg_dir) if tg_dir else wav_root
    if not tg_root.is_dir():
        raise FileNotFoundError(f"TextGrid 文件夹不存在: {tg_root}")
    out_root = pathlib.Path(out_dir) if out_dir else (wav_root / "segments")

    silences = [item for item in (silences if isinstance(silences, (list, tuple)) else [silences]) if str(item).strip()]
    if not silences:
        raise ValueError("没有可用的静音音素")

    def log(message: str):
        if progress_callback is not None:
            progress_callback(message)
        else:
            print(message)

    files = sorted(wav_root.rglob("*.wav") if recursive else wav_root.glob("*.wav"))
    stats = {"total": len(files), "written": 0, "tg_written": 0, "skipped": 0,
             "failed": 0, "stopped": False, "output_dir": str(out_root)}

    for index, wav_file in enumerate(files, 1):
        if stop_event is not None and stop_event.is_set():
            stats["stopped"] = True
            log("已请求停止，剩余文件未处理。")
            break

        rel = wav_file.relative_to(wav_root)
        tg_file = (tg_root / rel).with_suffix(".TextGrid")
        if not tg_file.is_file():
            stats["failed"] += 1
            log(f"[失败] {rel}: 找不到同名 TextGrid（{tg_file}）")
            continue

        try:
            grid = TextGrid()
            grid.read(str(tg_file))
            phones_tier = _find_phones_tier(grid)
            ranges = silence_ranges(phones_tier, silences)

            data, samplerate = sf.read(str(wav_file), dtype="float32", always_2d=True)
            duration = data.shape[0] / float(samplerate)
            pieces = segments_from_ranges(ranges, duration)
            if len(pieces) <= 1:
                stats["skipped"] += 1
                log(f"[跳过] {rel}: 没有找到 {','.join(map(str, silences))} 切分点")
                continue

            subtype = sf.info(str(wav_file)).subtype or "PCM_16"
            target_dir = out_root / rel.parent
            target_dir.mkdir(parents=True, exist_ok=True)

            written = 0
            tg_written = 0
            for seg_index, (start, end) in enumerate(pieces, 1):
                if stop_event is not None and stop_event.is_set():
                    stats["stopped"] = True
                    break
                begin = max(0, int(round(start * samplerate)))
                finish = min(data.shape[0], int(round(end * samplerate)))
                if finish <= begin:
                    continue
                stem = f"{wav_file.stem}_seg{seg_index:03d}"
                out_file = target_dir / f"{stem}.wav"
                sf.write(str(out_file), data[begin:finish], samplerate, subtype=subtype)
                written += 1

                if write_tg:
                    # 用实际写出的采样点时间裁剪，保证 TextGrid 与音频严格对齐
                    seg_start = begin / float(samplerate)
                    seg_end = finish / float(samplerate)
                    try:
                        seg_grid = _slice_grid(grid, seg_start, seg_end)
                        seg_grid.write(str(target_dir / f"{stem}.TextGrid"))
                        tg_written += 1
                    except Exception as exc:  # noqa: BLE001 - 单个 TextGrid 失败不影响音频
                        log(f"[警告] {rel} 第 {seg_index} 段 TextGrid 写入失败: "
                            f"{type(exc).__name__}: {exc}")

            stats["written"] += written
            stats["tg_written"] += tg_written
            suffix = f"，TextGrid {tg_written} 个" if write_tg else ""
            log(f"({index}/{len(files)}) {rel} → {written} 段{suffix}")
            if stats["stopped"]:
                break
        except Exception as exc:  # noqa: BLE001 - 单文件失败后继续
            stats["failed"] += 1
            log(f"[失败] {rel}: {type(exc).__name__}: {exc}")

    suffix = f"，TextGrid {stats['tg_written']} 个" if write_tg else ""
    log(f"完成：共 {stats['total']} 个 wav，写出 {stats['written']} 段{suffix}，"
        f"跳过 {stats['skipped']}，失败 {stats['failed']} → {out_root}")
    return stats
