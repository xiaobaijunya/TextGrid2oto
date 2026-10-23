# -*- coding: utf-8 -*-
"""扫描音频的平均 RMS 振幅，筛出平均音量过低、且确实没有声音的音频。

两个指标：

* **平均 RMS 振幅**：按帧（默认 50ms）计算每帧 RMS，再对帧 RMS 取平均；
* **最大振幅**：整段音频所有声道所有采样点的绝对值峰值。

任意一个指标达到阈值就说明“有声音”，只有 **两项都低于各自阈值** 时才列为
空白/过低音频：

    平均振幅达到阈值 → 没有声音（正常）
    峰值达到阈值     → 有声音（即使平均很低也不算空白）

换算关系：``dBFS = 20 * log10(振幅)``。纯静音记为 ``-120 dBFS``，日志显示 ``-inf``。
结果按阈值筛出后可直接拼成 ``a OR b OR c`` 形式，方便粘到搜索框里。
"""

from __future__ import annotations

import math
import pathlib
import shutil
from typing import Callable, Optional

import numpy as np
import soundfile as sf

# 纯静音时的替代值，避免 -inf 参与比较
_SILENCE_DB = -120.0

# 计算帧 RMS 的帧长（秒）
FRAME_SEC = 0.05

AUDIO_EXTENSIONS = (".wav", ".flac")


def linear_to_db(value: float) -> float:
    """线性振幅（0~1）→ dBFS。"""
    if value <= 0:
        return _SILENCE_DB
    return 20.0 * math.log10(value)


def db_to_linear(value: float) -> float:
    """dBFS → 线性振幅（0~1）。"""
    return 10.0 ** (float(value) / 20.0)


def measure_loudness(path, frame_sec: float = FRAME_SEC) -> dict:
    """测量单个音频的平均 RMS 振幅与最大振幅。

    平均 RMS 振幅按 ``frame_sec`` 分帧，先算每帧 RMS 再取平均（比整段一次性
    求 RMS 更能反映“平均听起来有多响”）；最大振幅取整段所有采样点的绝对值峰值。

    Returns:
        dict：rms / rms_db（平均 RMS 及其 dBFS）、peak / peak_db（最大振幅及其 dBFS）、
              duration（秒）、silent（是否完全无信号）。
    """
    file_path = pathlib.Path(path)
    block_size = 262_144  # 分块读取，长音频也不会撑爆内存

    frame_rms_sum = 0.0
    frame_count = 0
    carry_square = 0.0
    carry_count = 0
    peak = 0.0
    frames = 0

    with sf.SoundFile(str(file_path)) as handle:
        samplerate = handle.samplerate or 1
        channels = handle.channels or 1
        # 一帧覆盖的采样点数量（含所有声道）
        frame_samples = max(1, int(round(samplerate * frame_sec)) * channels)

        while True:
            block = handle.read(block_size, dtype="float32", always_2d=True)
            if block.size == 0:
                break
            frames += block.shape[0]
            flat = block.astype(np.float64, copy=False).reshape(-1)
            if flat.size:
                block_peak = float(np.max(np.abs(flat)))
                if block_peak > peak:
                    peak = block_peak

            pos = 0
            while pos < flat.size:
                take = min(frame_samples - carry_count, flat.size - pos)
                segment = flat[pos:pos + take]
                carry_square += float(np.dot(segment, segment))
                carry_count += take
                pos += take
                if carry_count >= frame_samples:
                    frame_rms_sum += math.sqrt(carry_square / frame_samples)
                    frame_count += 1
                    carry_square = 0.0
                    carry_count = 0

    if carry_count:  # 最后不足一帧的尾巴也算一帧
        frame_rms_sum += math.sqrt(carry_square / carry_count)
        frame_count += 1

    avg_rms = (frame_rms_sum / frame_count) if frame_count else 0.0
    silent = avg_rms <= 0.0 and peak <= 0.0

    return {
        "rms": avg_rms,
        "rms_db": _SILENCE_DB if avg_rms <= 0.0 else 20.0 * math.log10(avg_rms),
        "peak": peak,
        "peak_db": _SILENCE_DB if peak <= 0.0 else 20.0 * math.log10(peak),
        "duration": frames / float(samplerate),
        "silent": silent,
    }


def format_db(value: float) -> str:
    """把 dBFS 数值格式化成日志用的字符串。"""
    if value <= _SILENCE_DB:
        return "-inf"
    return f"{value:.1f}"


def format_or_list(items) -> str:
    """把文件名拼成 ``a OR b OR c``。"""
    return " OR ".join(str(item) for item in items)


def _unique_target(target: pathlib.Path) -> pathlib.Path:
    """目标已存在时自动加序号：``a.wav`` → ``a_1.wav`` → ``a_2.wav``。"""
    if not target.exists():
        return target
    index = 1
    while True:
        candidate = target.with_name(f"{target.stem}_{index}{target.suffix}")
        if not candidate.exists():
            return candidate
        index += 1


def run(wav_dir, threshold_db: float = -40.0, peak_threshold_db: float = -30.0,
        recursive: bool = False, out_dir=None, extensions=AUDIO_EXTENSIONS,
        progress_callback: Optional[Callable[[str], None]] = None,
        stop_event=None) -> dict:
    """扫描目录下所有音频，找出“平均 RMS 低且最大振幅也没起来”的文件。

    Args:
        wav_dir: 音频文件夹。
        threshold_db: 平均 RMS 振幅阈值（dBFS），低于该值视为平均音量过低。
        peak_threshold_db: 最大振幅阈值（dBFS），**达到该值说明不是空白音频**，
            不列入结果。
        recursive: 是否递归子文件夹。
        out_dir: 输出文件夹；**为空时只检测不移动文件**。设置后会把筛出的音频
            移动进去（递归时保留子目录结构，重名自动加序号）。
        extensions: 参与检测的扩展名（默认 wav / flac）。
        progress_callback: 进度回调。
        stop_event: ``threading.Event``，置位后停止。

    Returns:
        统计字典：
        total / checked / failed / stopped / threshold_db / peak_threshold_db /
        above_peak（因峰值达标而跳过）/ moved / move_failed / out_dir /
        low（[(相对路径, rms_db, peak_db), ...]，按平均 RMS 从低到高排序）。
    """
    root = pathlib.Path(wav_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"音频文件夹不存在: {root}")
    root_resolved = root.resolve()

    out_root = pathlib.Path(out_dir) if out_dir else None
    if out_root is not None:
        out_root.mkdir(parents=True, exist_ok=True)
        out_root = out_root.resolve()
        if out_root == root_resolved:
            raise ValueError("输出文件夹不能与音频文件夹相同")

    # 输出目录是否位于音频目录内部（是的话扫描时要排除它，避免重复处理）
    out_inside_root = False
    if out_root is not None:
        try:
            out_root.relative_to(root_resolved)
            out_inside_root = True
        except ValueError:
            out_inside_root = False

    def _in_output_dir(path: pathlib.Path) -> bool:
        """输出目录位于音频目录内部时，避免把自己移进去的文件再扫一遍。"""
        if not out_inside_root:
            return False
        try:
            path.resolve().relative_to(out_root)
            return True
        except ValueError:
            return False

    suffixes = {str(ext).lower() for ext in extensions}

    def log(message: str):
        if progress_callback is not None:
            progress_callback(message)
        else:
            print(message)

    candidates = root.rglob("*") if recursive else root.glob("*")
    files = sorted(p for p in candidates
                   if p.is_file() and p.suffix.lower() in suffixes and not _in_output_dir(p))

    stats = {
        "total": len(files),
        "checked": 0,
        "failed": 0,
        "stopped": False,
        "threshold_db": float(threshold_db),
        "peak_threshold_db": float(peak_threshold_db),
        "above_peak": 0,
        "moved": 0,
        "move_failed": 0,
        "out_dir": str(out_root) if out_root is not None else "",
        "low": [],
    }

    for index, audio_file in enumerate(files, 1):
        if stop_event is not None and stop_event.is_set():
            stats["stopped"] = True
            log("已请求停止，剩余文件未处理。")
            break

        rel = audio_file.relative_to(root)
        try:
            info = measure_loudness(audio_file)
        except Exception as exc:  # noqa: BLE001 - 单个文件失败不影响整体
            stats["failed"] += 1
            log(f"[失败] {rel}: {type(exc).__name__}: {exc}")
            continue

        stats["checked"] += 1
        rms_db = info["rms_db"]
        peak_db = info["peak_db"]
        below_rms = rms_db < threshold_db
        below_peak = peak_db < peak_threshold_db

        if below_rms and below_peak:
            stats["low"].append((str(rel), rms_db, peak_db))
            log(f"({index}/{len(files)}) [过低] {rel}  平均RMS {format_db(rms_db)} dBFS "
                f"| 峰值 {format_db(peak_db)} dBFS")
            if out_root is not None:
                target = _unique_target(out_root / rel)
                try:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(audio_file), str(target))
                    stats["moved"] += 1
                    log(f"        → 已移动到 {target}")
                except Exception as exc:  # noqa: BLE001 - 单个文件移动失败不影响整体
                    stats["move_failed"] += 1
                    log(f"        → 移动失败: {type(exc).__name__}: {exc}")
        elif below_rms:
            # 平均音量低，但最大振幅达标 → 有声音，不输出
            stats["above_peak"] += 1
            log(f"({index}/{len(files)}) [跳过] {rel}  平均RMS {format_db(rms_db)} dBFS "
                f"| 峰值 {format_db(peak_db)} dBFS 达到阈值，视为非空白")
        elif index % 50 == 0 or index == len(files):
            log(f"({index}/{len(files)}) 已检测 {stats['checked']} 个文件...")

    stats["low"].sort(key=lambda item: item[1])

    if out_root is None:
        move_note = "（未设置输出文件夹，仅检测，未移动文件）"
    else:
        move_note = f"；已移动 {stats['moved']} 个到 {out_root}"
        if stats["move_failed"]:
            move_note += f"（失败 {stats['move_failed']} 个）"

    log(f"完成：共 {stats['total']} 个音频，成功检测 {stats['checked']} 个，"
        f"失败 {stats['failed']} 个；平均RMS低于 {threshold_db:g} dBFS 且最大振幅低于 "
        f"{peak_threshold_db:g} dBFS 的有 {len(stats['low'])} 个"
        f"（{stats['above_peak']} 个因最大振幅达标被排除）{move_note}。")
    return stats
