# -*- coding: utf-8 -*-
"""把 wav 与最终 TextGrid 汇总为 DiffSinger 数据集（transcriptions.csv）。

对应命令：

    python build_dataset.py --wavs path/to/segments/ --tg path/to/final/textgrids/ --dataset path/to/dataset/

与 MakeDiffSinger 参考实现相比：
  * tqdm 改为可选依赖，GUI 环境无需安装；
  * 支持 ``progress_callback`` 实时回传进度；
  * 支持 ``stop_event`` 中途停止；
  * 支持递归扫描子文件夹（样本名自动加文件夹前缀，避免同名覆盖）；
  * 可选择统一音频格式（44.1kHz / 单声道 / 指定位深），或直接复制原始 wav；
  * 缺少 TextGrid / 读取失败的文件会被跳过并统计，不会中断整批任务。
"""

from __future__ import annotations

import csv
import pathlib
import shutil
from typing import Callable, Optional

import numpy as np
import soundfile
from textgrid import TextGrid

try:  # 包内导入
    from .tg_schema import (
        CSV_COLUMN_ORDER,
        NOTES_TIER,
        NOTE_SLUR_TIER,
        PHONES_TIER,
        PH_NUM_TIER,
        WORDS_TIER,
        find_tier,
        is_interval_tier,
        sample_interval_mark,
    )
except ImportError:  # 直接运行脚本
    from tg_schema import (
        CSV_COLUMN_ORDER,
        NOTES_TIER,
        NOTE_SLUR_TIER,
        PHONES_TIER,
        PH_NUM_TIER,
        WORDS_TIER,
        find_tier,
        is_interval_tier,
        sample_interval_mark,
    )

try:  # librosa 体积较大，缺失时退回到 soundfile + 线性插值
    import librosa
except ImportError:  # pragma: no cover - 仅在精简环境触发
    librosa = None


class BuildDatasetStopped(Exception):
    """任务被用户中止（保留异常类型，便于调用方区分）。"""


def _find_phones_tier(grid: TextGrid):
    tier = find_tier(grid, PHONES_TIER)
    if tier is not None:
        return tier
    if len(grid.tiers) >= 2:
        return grid.tiers[1]
    if grid.tiers:
        return grid.tiers[0]
    raise ValueError("TextGrid 中没有可用的音素层")


def _extract_extra_columns(grid: TextGrid, phones_tier) -> dict:
    """从 words / notes / note-slur / ph-num 层提取附加 CSV 列。"""
    extra: dict = {}
    if not is_interval_tier(phones_tier):
        return extra

    phones = list(phones_tier)
    midpoints = [(iv.minTime + iv.maxTime) / 2 for iv in phones]

    # ── ph-num 层 → ph_num（优先于由 words 层推导）──
    ph_num_tier = find_tier(grid, PH_NUM_TIER)
    if ph_num_tier is not None and is_interval_tier(ph_num_tier):
        values = []
        for interval in ph_num_tier:
            text = str(interval.mark).strip()
            if not text.isdigit() or int(text) <= 0:
                values = []
                break
            values.append(int(text))
        if values and sum(values) == len(phones):
            extra["ph_num"] = " ".join(str(v) for v in values)

    # ── words → ph_num / word_seq ──
    words_tier = find_tier(grid, WORDS_TIER)
    if words_tier is not None and is_interval_tier(words_tier):
        words = list(words_tier)
        ph_num = []
        valid = True
        for word in words:
            count = sum(1 for mid in midpoints if word.minTime <= mid < word.maxTime)
            if count <= 0:
                valid = False
                break
            ph_num.append(count)

        if valid and ph_num and sum(ph_num) == len(phones):
            extra.setdefault("ph_num", " ".join(str(n) for n in ph_num))
            marks = [str(word.mark).strip() for word in words]
            if len(marks) == len(ph_num) and all(marks):
                extra["word_seq"] = " ".join(marks)

    # ── notes → note_seq / note_dur ──
    notes_tier = find_tier(grid, NOTES_TIER)
    if notes_tier is not None and is_interval_tier(notes_tier):
        notes = list(notes_tier)
        if notes:
            extra["note_seq"] = " ".join(str(note.mark) for note in notes)
            extra["note_dur"] = " ".join(
                f"{note.maxTime - note.minTime:.6f}" for note in notes
            )

            # ── note-slur（可选）──
            slur_tier = find_tier(grid, NOTE_SLUR_TIER)
            if slur_tier is not None and is_interval_tier(slur_tier):
                slurs = []
                for note in notes:
                    midpoint = (note.minTime + note.maxTime) / 2
                    value = sample_interval_mark(slur_tier, midpoint)
                    if value not in ("0", "1"):
                        slurs = []
                        break
                    slurs.append(value)
                if len(slurs) == len(notes):
                    extra["note_slur"] = " ".join(slurs)

    return extra


def _load_mono(wavfile: pathlib.Path, samplerate: int = 44100) -> np.ndarray:
    """读取为 44.1kHz 单声道 float32。"""
    if librosa is not None:
        y, _ = librosa.load(str(wavfile), sr=samplerate, mono=True)
        return np.asarray(y, dtype=np.float32)

    data, sr = soundfile.read(str(wavfile), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if sr != samplerate:
        duration = mono.shape[0] / float(sr)
        target_len = max(1, int(round(duration * samplerate)))
        x_old = np.linspace(0.0, duration, num=mono.shape[0], endpoint=False)
        x_new = np.linspace(0.0, duration, num=target_len, endpoint=False)
        mono = np.interp(x_new, x_old, mono).astype(np.float32)
    return mono


def _dataset_stem(wavfile: pathlib.Path, wavs_root: pathlib.Path, recursive: bool) -> str:
    """生成数据集中的样本名。

    递归模式下用相对父目录作为前缀，例如：
      * ``A4/01.wav``     -> ``A4_01``
      * ``A4/sub/01.wav`` -> ``A4_sub_01``
    这样不同子文件夹中的同名文件不会互相覆盖。
    """
    if not recursive:
        return wavfile.stem

    parent = wavfile.relative_to(wavs_root).parent
    parts = [part for part in parent.parts if part not in ("", ".")]
    if not parts:
        return wavfile.stem
    return "_".join(parts + [wavfile.stem])


def build_dataset(
    wavs,
    tg,
    dataset,
    wav_subtype: str = "PCM_16",
    progress_callback: Optional[Callable[[str], None]] = None,
    stop_event=None,
    recursive: bool = False,
    convert_audio: bool = True,
    read_extra_tiers: bool = True,
):
    """收集音素对齐结果，生成 ``transcriptions.csv``。

    Args:
        wavs: 切片 wav 所在文件夹。
        tg: 最终 TextGrid 所在文件夹。
        dataset: 输出数据集文件夹（会创建 ``wavs/`` 与 ``transcriptions.csv``）。
        wav_subtype: 统一格式时写出的 wav 子类型，如 ``PCM_16`` / ``PCM_24`` / ``FLOAT``。
        progress_callback: 进度回调，接收一行文本。
        stop_event: ``threading.Event``，置位后停止处理并写出已收集的部分。
        recursive: 是否递归处理 wavs 子文件夹；开启后样本名自动加文件夹前缀。
        convert_audio: 为 True 时统一转为 44.1kHz 单声道并写入指定格式；
            为 False 时直接复制原始 wav，不做任何转换。
        read_extra_tiers: 为 True 时读取 TextGrid 中的 words / notes / note_slur 层，
            自动生成 ph_num、word_seq、note_seq、note_dur、note_slur 列。

    Returns:
        统计字典：total / processed / missing_tg / failed / duplicates / stopped / csv。
    """
    wavs_path = pathlib.Path(wavs)
    tg_dir = pathlib.Path(tg)
    dataset_path = pathlib.Path(dataset)

    if not wavs_path.is_dir():
        raise FileNotFoundError(f"wav 文件夹不存在: {wavs_path}")
    if not tg_dir.is_dir():
        raise FileNotFoundError(f"TextGrid 文件夹不存在: {tg_dir}")

    filelist = sorted(wavs_path.rglob("*.wav") if recursive else wavs_path.glob("*.wav"))
    stats = {
        "total": len(filelist),
        "processed": 0,
        "missing_tg": 0,
        "failed": 0,
        "duplicates": 0,
        "stopped": False,
        "csv": "",
    }

    def log(message: str):
        if progress_callback is not None:
            progress_callback(message)
        else:
            print(message)

    if not filelist:
        log(f"在 {wavs_path} 中未找到 .wav 文件")
        return stats

    dataset_path.mkdir(parents=True, exist_ok=True)
    (dataset_path / "wavs").mkdir(exist_ok=True)

    transcriptions = []
    samplerate = 44100
    used_names = set()

    for index, wavfile in enumerate(filelist, 1):
        if stop_event is not None and stop_event.is_set():
            stats["stopped"] = True
            log("已请求停止，剩余文件未处理（已写出的结果仍会保存）。")
            break

        rel_wav = wavfile.relative_to(wavs_path)
        # 递归模式下优先按相对路径匹配 TextGrid，避免不同子文件夹的同名文件串用
        rel_tgfile = tg_dir / rel_wav.with_suffix(".TextGrid")
        tgfile = rel_tgfile if rel_tgfile.exists() else tg_dir / wavfile.with_suffix(".TextGrid").name

        # 数据集样本名：递归模式下加文件夹前缀（A4/01.wav -> A4_01）
        name = _dataset_stem(wavfile, wavs_path, recursive)

        if not tgfile.exists():
            stats["missing_tg"] += 1
            log(f"[跳过] {wavfile.name}: 缺少 {tgfile.name}")
            continue

        try:
            grid = TextGrid()
            grid.read(str(tgfile))
            phones_tier = _find_phones_tier(grid)
            ph_seq = [ph.mark for ph in phones_tier]
            ph_dur = [ph.maxTime - ph.minTime for ph in phones_tier]
            extra = _extract_extra_columns(grid, phones_tier) if read_extra_tiers else {}

            if name in used_names:
                stats["duplicates"] += 1
                log(f"[警告] 样本名重复: {name}（dataset/wavs 中的同名文件会被覆盖）")
            used_names.add(name)

            destination = dataset_path / "wavs" / f"{name}.wav"
            if convert_audio:
                y = _load_mono(wavfile, samplerate)
                soundfile.write(str(destination), y, samplerate, subtype=wav_subtype)
            elif wavfile.resolve() != destination.resolve():
                # 不做任何处理，直接复制原始音频
                shutil.copy2(str(wavfile), str(destination))

            row = {
                "name": name,
                "ph_seq": " ".join(ph_seq),
                "ph_dur": " ".join(str(round(d, 6)) for d in ph_dur),
            }
            row.update(extra)
            transcriptions.append(row)
            stats["processed"] += 1
            action = "转换" if convert_audio else "复制"
            log(f"({index}/{len(filelist)}) {rel_wav} → {name}（{action}音频）")
        except Exception as exc:  # noqa: BLE001 - GUI 需要展示单文件错误后继续
            stats["failed"] += 1
            log(f"[失败] {wavfile.name}: {exc}")

    if transcriptions:
        fieldnames = [
            key for key in CSV_COLUMN_ORDER
            if any(key in row for row in transcriptions)
        ]
        # 兼容可能出现的额外列
        for row in transcriptions:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    else:
        fieldnames = ["name", "ph_seq", "ph_dur"]

    csv_path = dataset_path / "transcriptions.csv"
    with open(csv_path, "w", encoding="utf8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(transcriptions)

    stats["csv"] = str(csv_path)
    if transcriptions:
        log(f"CSV 列: {', '.join(fieldnames)}")
    log(
        f"完成：成功 {stats['processed']}，缺少 TextGrid {stats['missing_tg']}，"
        f"失败 {stats['failed']} → {csv_path}"
    )
    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Collect phoneme alignments into transcriptions.csv")
    parser.add_argument("--wavs", required=True, help="Path to the segments directory")
    parser.add_argument("--tg", required=True, help="Path to the final TextGrids directory")
    parser.add_argument("--dataset", required=True, help="Path to dataset directory")
    parser.add_argument("--wav_subtype", default="PCM_16", help="WAV subtype when converting audio")
    parser.add_argument("--recursive", action="store_true", help="Scan subfolders recursively and prefix names with folder")
    parser.add_argument("--copy_original", action="store_true",
                        help="Copy original wav files instead of converting to 44.1kHz mono")
    parser.add_argument("--no-extra-tiers", action="store_true",
                        help="Only write name/ph_seq/ph_dur (ignore words/notes/note_slur tiers)")
    args = parser.parse_args()

    build_dataset(
        args.wavs,
        args.tg,
        args.dataset,
        wav_subtype=args.wav_subtype,
        recursive=args.recursive,
        convert_audio=not args.copy_original,
        read_extra_tiers=not args.no_extra_tiers,
    )
