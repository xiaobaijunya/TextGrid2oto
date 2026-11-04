"""FoxBreatheLabeler 呼吸（AP）标注推理。

两种用法：
1. 新建模式：对音频做呼吸检测，直接生成带 AP/SP 标记的 TextGrid。
2. 合并模式：在已有 TextGrid（如 Aligner JSON 转出的 hanzi / words）上，
   把 ``SP`` 和**空白 mark** 区间拆成 ``SP`` / ``AP``，层名与层顺序全部保留。

返回值统一用 ``SP`` / ``AP``，因此后续「按 AP/SP 分割」可直接使用。

只依赖 onnxruntime + soundfile + librosa + textgrid，不需要 torch。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, List, Optional, Sequence

import numpy as np
import yaml

# ── 默认参数（与 FoxBreatheLabeler 的 CLI 默认值一致）────────────
AP_THRESHOLD = 0.4    # 呼吸概率阈值
AP_MIN_DUR = 0.08     # 最短呼吸时长（秒）
SP_MERGE_DUR = 0.1    # 短于该时长的 SP 会被吸附到相邻 AP（秒）
MAX_GAP_FRAMES = 5    # 段内允许的连续低于阈值的帧数
OTHER_MARK = "V"      # 新建模式下非呼吸区间的标记（不能是 SP/AP，否则切分会失效）

# 分块推理：每块约 40s（hop=882, sr=44100 时约 2000 帧），避免整首歌
# 的注意力矩阵过大导致内存暴涨。块之间留出上下文重叠，只取中间部分。
CHUNK_FRAMES = 2000
CHUNK_OVERLAP = 100

_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_DIR = _ROOT / "FoxBreatheLabeler" / "20ms"

_SESSION_CACHE: dict = {}


# ══════════════════════════════════════════════════════════════════
# ONNX 模型
# ══════════════════════════════════════════════════════════════════

def load_model_dir(model_dir: Optional[str | Path] = None):
    """读取模型目录下的 config.yaml 与 model.onnx，返回 (config, session)。"""
    model_dir = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
    onnx_path = model_dir / "model.onnx"
    config_path = model_dir / "config.yaml"

    if not onnx_path.is_file():
        raise FileNotFoundError(
            f"找不到 AP 模型: {onnx_path}\n"
            "请先运行 FoxBreatheLabeler/export_ap_onnx.py 导出 ONNX"
        )
    if not config_path.is_file():
        raise FileNotFoundError(f"找不到 AP 模型配置: {config_path}")

    key = str(onnx_path.resolve())
    session = _SESSION_CACHE.get(key)
    if session is None:
        import onnxruntime as ort

        providers = ort.get_available_providers()
        chosen = ["CPUExecutionProvider"] if "CPUExecutionProvider" in providers else providers
        session = ort.InferenceSession(str(onnx_path), providers=chosen)
        _SESSION_CACHE[key] = session

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config, session


# ══════════════════════════════════════════════════════════════════
# 音频 / 推理
# ══════════════════════════════════════════════════════════════════

def _load_audio(path: str | Path, target_sr: int) -> np.ndarray:
    """读取音频为单声道 float32，并重采样到模型采样率。"""
    import soundfile as sf

    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != target_sr:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr, res_type="soxr_hq")
    return np.ascontiguousarray(audio, dtype=np.float32)


def _frame_audio(audio: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    """按原始的 get_music_chunk 逻辑切帧，返回 (T, frame_length) 视图。"""
    padding = ((frame_length - hop_length) // 2,
               (frame_length - hop_length + 1) // 2)
    y = np.pad(audio, padding, mode="constant")
    return np.lib.stride_tricks.sliding_window_view(y, frame_length)[::hop_length]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def predict_probabilities(audio: np.ndarray, session, config: dict,
                          stop_event=None) -> np.ndarray:
    """整段音频的逐帧呼吸概率（分块推理，块间带上下文重叠）。"""
    frame_length = int(config["spec_win"])
    hop_length = int(config["hop_size"])

    frames = _frame_audio(audio, frame_length, hop_length)
    total = int(frames.shape[0])
    probs = np.zeros(total, dtype=np.float32)
    input_name = session.get_inputs()[0].name

    for start in range(0, total, CHUNK_FRAMES):
        if stop_event is not None and stop_event.is_set():
            break
        end = min(start + CHUNK_FRAMES, total)
        ctx_start = max(0, start - CHUNK_OVERLAP)
        ctx_end = min(total, end + CHUNK_OVERLAP)

        x = np.ascontiguousarray(frames[ctx_start:ctx_end], dtype=np.float32)[None]
        logits = session.run(None, {input_name: x})[0]
        chunk_probs = _sigmoid(logits[0, 0])

        probs[start:end] = chunk_probs[start - ctx_start:end - ctx_start]

    return probs


# ══════════════════════════════════════════════════════════════════
# 呼吸段检测（移植自 textgrid_add_ap.py）
# ══════════════════════════════════════════════════════════════════

def find_segments_dynamic(arr, time_scale, threshold=0.5, max_gap=5, min_frames=10):
    """找出概率高于阈值的连续段，返回 [(start_sec, end_sec), ...]。"""
    segments = []
    start = None
    gap_count = 0

    for i in range(len(arr)):
        if arr[i] >= threshold:
            if start is None:
                start = i
            gap_count = 0
        else:
            if start is not None:
                if gap_count < max_gap:
                    gap_count += 1
                else:
                    end = i - gap_count - 1
                    if end >= start and (end - start) >= min_frames:
                        segments.append((start * time_scale, end * time_scale))
                    start = None
                    gap_count = 0

    if start is not None and (len(arr) - start) >= min_frames:
        segments.append((start * time_scale, (len(arr) - 1) * time_scale))

    return segments


def merge_close_segments(segments, sp_dur):
    """把间隔小于 sp_dur 的相邻呼吸段合并。"""
    if not segments:
        return []
    segments = sorted(segments)
    merged = [list(segments[0])]
    for seg_start, seg_end in segments[1:]:
        if seg_start - merged[-1][1] < sp_dur:
            merged[-1][1] = max(merged[-1][1], seg_end)
        else:
            merged.append([seg_start, seg_end])
    return [(float(s), float(e)) for s, e in merged]


def find_overlapping_segments(start, end, segments, sp_dur):
    """取出与 [start, end) 重叠的呼吸段，并把间隔过近的段合并。"""
    overlapping = []
    for segment in segments:
        seg_start, seg_end = segment
        if not (seg_end <= start or seg_start >= end):
            overlapping.append(segment)
    return merge_close_segments(overlapping, sp_dur)


def detect_segments(wav_path: str | Path, session, config: dict,
                    ap_threshold: float = AP_THRESHOLD,
                    ap_min_dur: float = AP_MIN_DUR,
                    sp_dur: float = SP_MERGE_DUR,
                    stop_event=None):
    """对单个音频做呼吸检测，返回 (segments, duration)。"""
    sample_rate = int(config["audio_sample_rate"])
    hop_size = int(config["hop_size"])
    time_scale = hop_size / sample_rate

    audio = _load_audio(wav_path, sample_rate)
    duration = len(audio) / sample_rate
    probs = predict_probabilities(audio, session, config, stop_event=stop_event)

    segments = find_segments_dynamic(
        probs, time_scale,
        threshold=ap_threshold,
        max_gap=MAX_GAP_FRAMES,
        min_frames=int(ap_min_dur / time_scale),
    )
    return merge_close_segments(segments, sp_dur), duration


# ══════════════════════════════════════════════════════════════════
# TextGrid 写出
# ══════════════════════════════════════════════════════════════════

def build_ap_intervals(segments, duration: float, other_mark: str = OTHER_MARK):
    """把呼吸段补齐成 [其他标记, AP, 其他标记, AP, ...] 的完整覆盖区间。

    非呼吸区间不能用 SP：切分功能把连续的 SP/AP 当成同一块静音，
    全篇都是 SP/AP 时会没有任何切分点。
    """
    other_mark = (other_mark or OTHER_MARK).strip() or OTHER_MARK
    intervals = []
    cursor = 0.0
    for seg_start, seg_end in segments:
        seg_start = max(0.0, min(float(seg_start), duration))
        seg_end = max(0.0, min(float(seg_end), duration))
        if seg_end <= seg_start:
            continue
        if seg_start > cursor + 1e-6:
            intervals.append((cursor, seg_start, other_mark))
        intervals.append((seg_start, seg_end, "AP"))
        cursor = seg_end
    if cursor < duration - 1e-6:
        intervals.append((cursor, duration, other_mark))
    return intervals


def _new_tier(name: str, duration: float):
    import textgrid as tg

    tier = tg.IntervalTier(name=name, minTime=0.0, maxTime=duration)
    tier.strict = False
    return tier


def _new_grid(duration: float):
    import textgrid as tg

    return tg.TextGrid(minTime=0.0, maxTime=duration, strict=False)


def write_fresh_grid(out_path: str | Path, duration: float, intervals,
                     other_mark: str = OTHER_MARK) -> None:
    """直接生成带 AP 标记的 TextGrid（words + phones 两层）。"""
    grid = _new_grid(duration)
    tier_words = _new_tier("words", duration)
    tier_phones = _new_tier("phones", duration)

    for start, end, mark in intervals:
        if end <= start:
            continue
        tier_words.add(start, end, mark)
        tier_phones.add(start, end, mark)

    grid.tiers = [tier_words, tier_phones]
    grid.write(str(out_path))


def _apply_ap_to_words(words_dict: dict, segments, sp_dur: float) -> list:
    """按原始脚本的规则把 SP 段切分成 SP / AP。"""
    out = []
    for v in words_dict.values():
        if v['text'] != "SP" and v['text'] != "":
            out.append(v)
            continue

        sp_start, sp_end = v['start'], v['end']
        overlapping = find_overlapping_segments(sp_start, sp_end, segments, sp_dur)

        if len(overlapping) == 0:
            out.append(v)
            continue

        if len(overlapping) == 1:
            ap_start, ap_end = overlapping[0]
            cursor = sp_start
            if sp_start + sp_dur <= ap_start < sp_end:
                out.append({"start": cursor, "end": ap_start, "text": "SP"})
                cursor = ap_start
            if ap_end <= sp_end - sp_dur:
                out.append({"start": cursor, "end": ap_end, "text": "AP"})
                out.append({"start": ap_end, "end": sp_end, "text": "SP"})
            else:
                out.append({"start": cursor, "end": sp_end, "text": "AP"})
            continue

        cursor = sp_start
        for i, (ap_start, ap_end) in enumerate(overlapping):
            if ap_start > cursor:
                out.append({"start": cursor, "end": ap_start, "text": "SP"})
                cursor = ap_start
            if i == 0:
                if cursor < ap_start and sp_start + sp_dur <= ap_start < sp_end:
                    out.append({"start": cursor, "end": ap_start, "text": "SP"})
                    cursor = ap_start
                out.append({"start": cursor, "end": ap_end, "text": "AP"})
                cursor = ap_end
            elif i == len(overlapping) - 1:
                if ap_end <= sp_end - sp_dur:
                    out.append({"start": cursor, "end": ap_end, "text": "AP"})
                    out.append({"start": ap_end, "end": sp_end, "text": "SP"})
                else:
                    out.append({"start": cursor, "end": sp_end, "text": "AP"})
            else:
                out.append({"start": cursor, "end": ap_end, "text": "AP"})
                cursor = ap_end

    return out


def _apply_ap_to_tier(tier, segments, sp_dur: float) -> list[dict]:
    """把一层里的 SP 与**空白 mark** 区间按呼吸段拆成 SP/AP。

    其他 mark（汉字、拼音、V 等）原样保留，层名/层顺序由调用方维护。
    """
    items = {}
    for index, interval in enumerate(tier):
        items[index] = {
            "start": interval.minTime,
            "end": interval.maxTime,
            "text": interval.mark,
            "phones": [],
        }
    return _apply_ap_to_words(items, segments, sp_dur)


def merge_ap_into_grid(tg_path: str | Path, segments, duration: float,
                       sp_dur: float = SP_MERGE_DUR,
                       out_path: Optional[str | Path] = None) -> bool:
    """在已有 TextGrid 上把 SP 段和空白段改标为 AP。

    支持任意层名（如 Aligner 生成的 ``hanzi`` / ``words``）：
    对每一层分别处理，``SP`` 和空 mark 的区间都会按呼吸检测结果拆成
    ``SP`` / ``AP``，其他 mark 保持不变；层名、层顺序完整保留。

    Returns:
        是否至少写入了一个 AP。
    """
    import textgrid as tg

    tg_path = Path(tg_path)
    out_path = Path(out_path) if out_path else tg_path

    grid = tg.TextGrid()
    grid.read(str(tg_path))
    if not grid.tiers:
        return False

    tier_end = grid.maxTime or float(duration or 0.0)
    rebuilt = []
    changed = False

    for tier in grid.tiers:
        if not hasattr(tier, "intervals"):
            rebuilt.append(tier)  # PointTier 原样保留
            continue

        out = _apply_ap_to_tier(tier, segments, sp_dur)
        if any(str(item.get("text", "")) == "AP" for item in out):
            changed = True

        new_tier = _new_tier(getattr(tier, "name", ""), tier_end)
        for item in out:
            start, end = float(item["start"]), float(item["end"])
            if end <= start:
                continue
            new_tier.add(start, end, str(item.get("text", "") or ""))
        rebuilt.append(new_tier)

    if not changed:
        return False

    out_tg = tg.TextGrid(minTime=grid.minTime or 0.0, maxTime=tier_end, strict=False)
    out_tg.tiers = rebuilt
    out_tg.write(str(out_path))
    return True


# ══════════════════════════════════════════════════════════════════
# 批量入口
# ══════════════════════════════════════════════════════════════════

def _collect_wavs(folder: str | Path, recursive: bool = True) -> List[Path]:
    folder = Path(folder)
    pattern = "**/*.wav" if recursive else "*.wav"
    return sorted(p for p in folder.glob(pattern) if p.is_file())


def run(wav_folder: Optional[str | Path] = None,
        audio_files: Optional[Sequence[str | Path]] = None,
        out_dir: Optional[str | Path] = None,
        model_dir: Optional[str | Path] = None,
        ap_threshold: float = AP_THRESHOLD,
        ap_min_dur: float = AP_MIN_DUR,
        sp_dur: float = SP_MERGE_DUR,
        merge_existing: bool = False,
        other_mark: str = OTHER_MARK,
        recursive: bool = True,
        progress_callback: Optional[Callable[[str], None]] = None,
        stop_event=None) -> dict:
    """批量生成 AP 标记。

    wav_folder: 音频文件夹；audio_files: 指定的音频文件列表（优先）。
    out_dir: 输出文件夹，默认写到音频同目录。
    merge_existing: True 时合并到已有的同名 TextGrid；False 时直接新建 AP/SP 网格。
    """
    def log(msg: str):
        if progress_callback:
            progress_callback(msg)

    if audio_files:
        wav_paths = [Path(p) for p in audio_files]
    elif wav_folder:
        wav_paths = _collect_wavs(wav_folder, recursive=recursive)
    else:
        wav_paths = []

    stats = {"total": len(wav_paths), "checked": 0, "written": 0, "skipped": 0,
             "stopped": False, "failed": 0}

    if not wav_paths:
        return stats

    out_root = Path(out_dir) if out_dir else None
    if out_root:
        out_root.mkdir(parents=True, exist_ok=True)

    started = time.time()
    config, session = load_model_dir(model_dir)

    for wav_path in wav_paths:
        if stop_event is not None and stop_event.is_set():
            stats["stopped"] = True
            break

        target_dir = out_root if out_root else wav_path.parent
        tg_path = target_dir / (wav_path.stem + ".TextGrid")

        if merge_existing and not tg_path.is_file():
            stats["skipped"] += 1
            log(f"跳过（无 TextGrid）: {wav_path.name}")
            continue

        try:
            t0 = time.time()
            segments, duration = detect_segments(
                wav_path, session, config,
                ap_threshold=ap_threshold, ap_min_dur=ap_min_dur,
                sp_dur=sp_dur, stop_event=stop_event,
            )
            stats["checked"] += 1
            cost = time.time() - t0

            if not segments:
                log(f"[AP] {wav_path.name} 未检测到呼吸（{cost:.1f}s）")
                continue

            if merge_existing:
                changed = merge_ap_into_grid(tg_path, segments, duration,
                                             sp_dur=sp_dur, out_path=tg_path)
                if changed:
                    stats["written"] += 1
                    log(f"[AP] {wav_path.name} 已在原 TextGrid 上合并 AP（{cost:.1f}s）")
                else:
                    log(f"[AP] {wav_path.name} 原 TextGrid 中没有可替换的 SP/空白区间（{cost:.1f}s）")
            else:
                write_fresh_grid(tg_path, duration,
                                 build_ap_intervals(segments, duration, other_mark),
                                 other_mark=other_mark)
                stats["written"] += 1
                log(f"[AP] {wav_path.name} 已生成 {len(segments)} 段 AP（{cost:.1f}s）")
        except Exception as exc:  # 单个文件失败不影响其它文件
            stats["failed"] += 1
            log(f"[AP] {wav_path.name} 失败: {exc}")

    stats["elapsed"] = time.time() - started
    return stats
