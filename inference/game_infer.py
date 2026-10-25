# -*- coding: utf-8 -*-
"""GAME ONNX 推理：为 DiffSinger 数据集生成 MIDI 标注。

这是 GAME 官方 ``infer.py align`` 的 ONNX 版本，只依赖
``onnxruntime + numpy + librosa``，不需要 PyTorch / Lightning。

模型目录（例如 ``GAME_model/GAME-1.0.3-medium-onnx``）需要包含：

    encoder.onnx
    segmenter.onnx
    estimator.onnx
    dur2bd.onnx
    bd2dur.onnx
    config.json

处理流程（与 GAME 一致）：

    waveform ──encoder──> x_seg / x_est / maskT
    word_dur ──dur2bd──> known_boundaries
    x_seg + known_boundaries ──segmenter(D3PM)──> boundaries
    boundaries ──bd2dur──> note_dur / maskN
    x_est + boundaries ──estimator──> presence / scores(MIDI)
    scores + word_dur ──align_notes_to_words──> note_seq / note_dur / note_slur

输入 CSV 必须包含：``name, ph_seq, ph_dur, ph_num``
（其中 ``ph_num`` 可用 ``build_diffdataset/add_ph_num.py`` 生成）。
wav 路径约定为 ``CSV所在目录/wavs/<name>.wav``。

输出：在 CSV 中新增/更新 ``note_seq``、``note_dur``、``note_slur``，
并删除模型不支持的 ``note_glide``。
"""

from __future__ import annotations

import csv
import json
import math
import multiprocessing as mp
import os
import pathlib
import queue
import threading
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Callable, Iterable, Optional

import librosa
import numpy as np
import onnxruntime as ort

try:
    from .ort_utils import (
        run_session as _ort_run_session,
        create_session as _ort_create_session,
        is_device_lost as _ort_is_device_lost,
    )
except ImportError:  # 作为脚本直接运行时
    from ort_utils import (
        run_session as _ort_run_session,
        create_session as _ort_create_session,
        is_device_lost as _ort_is_device_lost,
    )

DEFAULT_UV_VOCAB = "EP,AP,SP,br,sil"


class GameInferenceStopped(Exception):
    """用户请求停止推理。"""


# ---------------------------------------------------------------------------
# ONNX Session
# ---------------------------------------------------------------------------

def _providers(device: str) -> list[str]:
    """根据设备返回 onnxruntime providers。"""
    available = ort.get_available_providers()
    device = (device or "cpu").lower()
    if device in ("dml", "directml", "gpu") and "DmlExecutionProvider" in available:
        return ["DmlExecutionProvider", "CPUExecutionProvider"]
    if device in ("cuda", "gpu") and "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def _create_session(
    path: pathlib.Path,
    device: str,
    intra_op_threads: Optional[int] = None,
) -> ort.InferenceSession:
    if not path.is_file():
        raise FileNotFoundError(f"GAME ONNX 模型不存在: {path}")
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # 长音频/大模型时减少内存池持续增长
    options.enable_mem_pattern = False
    options.enable_cpu_mem_arena = False
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    # 多个推理线程共享 session 时，限制每个 session 的算子内并行度，
    # 避免 threads x intra_op 超订导致互相抢核
    if intra_op_threads is not None and intra_op_threads > 0:
        options.intra_op_num_threads = int(intra_op_threads)
    return _ort_create_session(str(path), sess_options=options, providers=_providers(device))


# ---------------------------------------------------------------------------
# 数据集工具（对应 GAME inference/utils.py）
# ---------------------------------------------------------------------------

def _plain_phone(phone: str) -> str:
    """去掉 zh/、ja/ 等语言前缀。"""
    return str(phone).split("/")[-1].strip()


def validate_phones(ph_seq: list[str], ph_dur: list[float], ph_num: list[int]) -> None:
    if len(ph_seq) != len(ph_dur):
        raise ValueError(f"ph_seq 与 ph_dur 长度不一致: {len(ph_seq)} vs {len(ph_dur)}")
    if sum(ph_num) != len(ph_seq):
        raise ValueError(f"ph_num 之和 {sum(ph_num)} 与 ph_seq 长度 {len(ph_seq)} 不一致")


def parse_words(
    ph_seq: list[str],
    ph_dur: list[float],
    ph_num: list[int],
    uv_vocab: Optional[set[str]] = None,
    uv_cond: str = "lead",
    merge_consecutive_uv: bool = False,
) -> tuple[list[float], list[int]]:
    """音素序列 → (word_dur, word_vuv)。"""
    word_dur: list[float] = []
    word_vuv: list[int] = []
    index = 0
    for num in ph_num:
        duration = float(sum(ph_dur[index:index + num]))
        word_dur.append(duration)

        vuv = 1
        if uv_vocab is not None:
            phones = [_plain_phone(ph) for ph in ph_seq[index:index + num]]
            if uv_cond == "lead":
                vuv = 0 if phones and phones[0] in uv_vocab else 1
            else:  # "all"
                vuv = 0 if phones and all(ph in uv_vocab for ph in phones) else 1
        word_vuv.append(vuv)
        index += num

    if merge_consecutive_uv:
        word_dur, word_vuv = merge_consecutive_uv_words(word_dur, word_vuv)
    return word_dur, word_vuv


def merge_consecutive_uv_words(word_dur: list[float], word_vuv: list[int]) -> tuple[list[float], list[int]]:
    if not word_dur:
        return [], []
    merged_dur = [word_dur[0]]
    merged_vuv = [word_vuv[0]]
    for duration, vuv in zip(word_dur[1:], word_vuv[1:]):
        if vuv == 0 and merged_vuv[-1] == 0:
            merged_dur[-1] += duration
        else:
            merged_dur.append(duration)
            merged_vuv.append(vuv)
    return merged_dur, merged_vuv


def align_notes_to_words(
    word_dur: list[float],
    word_vuv: list[int],
    note_seq: list[str],
    note_dur: list[float],
    tol: float = 0.01,
    apply_word_uv: bool = False,
) -> tuple[list[str], list[float], list[int]]:
    """把音符边界对齐到词边界（GAME 原实现）。"""
    if not word_dur:
        return [], [], []
    if not note_seq or not note_dur:
        return ["rest"] * len(word_dur), [float(d) for d in word_dur], [0] * len(word_dur)

    word_start = np.cumsum([0.0] + word_dur[:-1])
    word_end = np.cumsum(word_dur)
    note_start = np.cumsum([0.0] + note_dur[:-1])
    note_end = np.cumsum(note_dur)

    new_note_seq: list[str] = []
    new_note_dur: list[float] = []
    note_slur: list[int] = []

    for word_idx in range(len(word_dur)):
        if apply_word_uv and word_vuv[word_idx] == 0:
            new_note_seq.append("rest")
            new_note_dur.append(float(word_dur[word_idx]))
            note_slur.append(0)
            continue

        note_start_idx = int(np.argmin(np.abs(note_start - word_start[word_idx])))
        if word_start[word_idx] < note_start[note_start_idx] - tol:
            note_start_idx = max(0, note_start_idx - 1)

        note_end_idx = int(np.argmin(np.abs(note_end[note_start_idx:] - word_end[word_idx]))) + note_start_idx
        if word_end[word_idx] > note_end[note_end_idx] + tol:
            note_end_idx = min(len(note_end) - 1, note_end_idx + 1)

        word_note_seq: list[str] = []
        word_note_dur: list[float] = []
        for note_idx in range(note_start_idx, note_end_idx + 1):
            start = word_start[word_idx] if note_idx == note_start_idx else note_start[note_idx]
            end = word_end[word_idx] if note_idx == note_end_idx else note_end[note_idx]
            if word_note_seq and word_note_seq[-1] == note_seq[note_idx]:
                word_note_dur[-1] += float(end - start)
            else:
                word_note_seq.append(note_seq[note_idx])
                word_note_dur.append(float(end - start))

        if not word_note_seq:
            word_note_seq = ["rest"]
            word_note_dur = [float(word_dur[word_idx])]
            note_slur.append(0)
        else:
            note_slur.extend([0] + [1] * (len(word_note_seq) - 1))

        new_note_seq.extend(word_note_seq)
        new_note_dur.extend(word_note_dur)

    return new_note_seq, new_note_dur, note_slur


def _parse_uv_vocab(value) -> Optional[set[str]]:
    if value is None:
        return None
    if isinstance(value, str):
        items = value.replace(",", " ").split()
    else:
        items = list(value)
    vocab = {_plain_phone(item) for item in items if str(item).strip()}
    return vocab or None


# ---------------------------------------------------------------------------
# ONNX 推理器
# ---------------------------------------------------------------------------

class GameOnnxAligner:
    def __init__(self, model_dir, device: str = "cpu", intra_op_threads: Optional[int] = None):
        self.model_dir = pathlib.Path(model_dir)
        config_path = self.model_dir / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"GAME 模型缺少 config.json: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        self.samplerate = int(config.get("samplerate", 44100))
        self.timestep = float(config.get("timestep", 0.01))
        self.languages: dict[str, int] = config.get("languages") or {}
        self.loop: bool = bool(config.get("loop", True))

        self.device = device or "cpu"
        self.intra_op_threads = intra_op_threads
        self._cpu_fallback = False        # 是否已因设备丢失而永久回退 CPU
        self._device_recover_count = 0    # 设备丢失后已尝试恢复的次数
        self._load_sessions(self.device)

    def _load_sessions(self, device: str):
        """（重新）创建全部 ONNX session。"""
        self.encoder = _create_session(
            self.model_dir / "encoder.onnx", device, self.intra_op_threads)
        self.segmenter = _create_session(
            self.model_dir / "segmenter.onnx", device, self.intra_op_threads)
        self.estimator = _create_session(
            self.model_dir / "estimator.onnx", device, self.intra_op_threads)
        self.dur2bd = _create_session(
            self.model_dir / "dur2bd.onnx", device, self.intra_op_threads)
        self.bd2dur = _create_session(
            self.model_dir / "bd2dur.onnx", device, self.intra_op_threads)

        # onnxruntime 实际注册的 ExecutionProvider（供日志显示）
        self.providers = list(self.encoder.get_providers())

        self.segmenter_inputs = {i.name for i in self.segmenter.get_inputs()}
        self.estimator_inputs = {i.name for i in self.estimator.get_inputs()}

    def recover_from_device_lost(self) -> bool:
        """GPU 设备被驱动重置（DXGI_ERROR_DEVICE_REMOVED / TDR）后重建 session。

        第一次重建同设备 session，再次丢失则永久回退 CPU。返回 True 表示可以重试。
        """
        if self._cpu_fallback:
            return False

        current = (self.device or "cpu").lower()
        targets = []
        if current != "cpu" and self._device_recover_count == 0:
            targets.append(current)
        targets.append("cpu")
        self._device_recover_count += 1

        for target in targets:
            try:
                self._load_sessions(target)
                self.device = target
                if target == "cpu":
                    self._cpu_fallback = True
                return True
            except Exception:
                traceback.print_exc()
        return False

    def language_id(self, language: Optional[str]) -> int:
        if not language:
            return 0
        if language not in self.languages:
            supported = ", ".join(self.languages.keys()) or "无"
            raise ValueError(f"模型不支持语言 '{language}'，可选: {supported}")
        return int(self.languages[language])

    def predict(        self,
        waveform: np.ndarray,
        known_durations: list[float],
        language_id: int = 0,
        seg_threshold: float = 0.2,
        seg_radius: float = 0.02,
        t0: float = 0.0,
        nsteps: int = 8,
        ts: Optional[Iterable[float]] = None,
        est_threshold: float = 0.2,
    ):
        """对单个片段推理，返回 (note_dur, presence, scores)。"""
        waveform = np.asarray(waveform, dtype=np.float32)
        if waveform.ndim != 1:
            raise ValueError("waveform 必须是一维单声道数组")

        duration = float(sum(known_durations))
        if duration <= 0:
            raise ValueError("known_durations 的总时长必须大于 0")

        wav_batch = waveform[None, :]
        dur_batch = np.asarray([duration], dtype=np.float32)
        x_seg, x_est, mask_t = _ort_run_session(
            self.encoder, None, {"waveform": wav_batch, "duration": dur_batch}
        )

        durations_batch = np.asarray([known_durations], dtype=np.float32)
        known_boundaries = _ort_run_session(
            self.dur2bd, None, {"durations": durations_batch, "maskT": mask_t}
        )[0]

        radius_frames = max(1, int(round(float(seg_radius) / self.timestep)))
        threshold = np.asarray(float(seg_threshold), dtype=np.float32)
        radius = np.asarray(radius_frames, dtype=np.int64)
        language = np.asarray([int(language_id)], dtype=np.int64)

        if ts is not None:
            steps = [float(v) for v in ts]
        else:
            nsteps = max(1, int(nsteps))
            step_size = (1.0 - float(t0)) / nsteps
            steps = [float(t0) + i * step_size for i in range(nsteps)]

        boundaries = known_boundaries
        max_loops = len(steps) if self.loop else 1
        for loop_idx in range(max_loops):
            inputs = {
                "x_seg": x_seg,
                "known_boundaries": known_boundaries,
                "maskT": mask_t,
                "threshold": threshold,
                "radius": radius,
            }
            if "language" in self.segmenter_inputs:
                inputs["language"] = language
            if "prev_boundaries" in self.segmenter_inputs:
                inputs["prev_boundaries"] = boundaries
            if "t" in self.segmenter_inputs:
                t_value = steps[min(loop_idx, len(steps) - 1)] if steps else 0.0
                inputs["t"] = np.asarray([t_value], dtype=np.float32)
            boundaries = _ort_run_session(self.segmenter, None, inputs)[0]

        note_durations, mask_n = _ort_run_session(
            self.bd2dur, None, {"boundaries": boundaries, "maskT": mask_t}
        )

        estimator_inputs = {
            "x_est": x_est,
            "boundaries": boundaries,
            "maskT": mask_t,
            "maskN": mask_n,
            "threshold": np.asarray(float(est_threshold), dtype=np.float32),
        }
        presence, scores = _ort_run_session(self.estimator, None, estimator_inputs)

        return note_durations[0], presence[0], scores[0]


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def _midi_to_note(midi: float, cents: bool = True) -> str:
    return librosa.midi_to_note(float(midi), unicode=False, cents=cents)


def _find_wav(csv_path: pathlib.Path, name: str) -> pathlib.Path:
    for ext in (".wav", ".flac"):
        candidate = csv_path.parent / "wavs" / f"{name}{ext}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"找不到 {name} 的音频：{csv_path.parent / 'wavs' / (name + '.wav')}"
    )


def _build_word_records(ph_num: list[int], ph_dur: list[float]) -> list[dict]:
    """根据 ph_num 把音素切成词，记录每个词对应的音素范围与时长。"""
    records = []
    index = 0
    for num in ph_num:
        duration = float(sum(ph_dur[index:index + num]))
        records.append({
            "ph_start": index,
            "ph_end": index + num,
            "dur": duration,
        })
        index += num
    return records


def _split_word_chunks(words: list[dict], word_vuv: list[int], max_chunk_sec: float) -> list[list[int]]:
    """按词边界分块；优先在无声音素（SP/AP 等）之后切开。

    Args:
        words: ``_build_word_records`` 的结果。
        word_vuv: 每个词的有声/无声标记（1=有声，0=无声）；为空表示不区分。
        max_chunk_sec: 每块最大秒数；<=0 表示不分块。
    """
    if not words:
        return []
    if max_chunk_sec is None or max_chunk_sec <= 0:
        return [list(range(len(words)))]

    chunks: list[list[int]] = []
    total_words = len(words)
    start = 0
    min_chunk_sec = max_chunk_sec * 0.5

    while start < total_words:
        total_dur = 0.0
        end = start
        last_unvoiced_end = None

        while end < total_words:
            duration = words[end]["dur"]
            if end > start and total_dur + duration > max_chunk_sec:
                break
            total_dur += duration
            end += 1
            if word_vuv and word_vuv[end - 1] == 0:
                last_unvoiced_end = end

        # 单个词就超过上限时，只能把该词单独作为一块
        if end == start:
            end = start + 1
        elif end < total_words and last_unvoiced_end is not None and last_unvoiced_end < end:
            # 在超限前存在无声词边界，优先切在那里（且不要让块太短）
            cut_dur = sum(words[i]["dur"] for i in range(start, last_unvoiced_end))
            if cut_dur >= min_chunk_sec:
                end = last_unvoiced_end

        chunks.append(list(range(start, end)))
        start = end

    return chunks


def _slice_audio(waveform: np.ndarray, samplerate: int, start_sec: float, end_sec: float) -> np.ndarray:
    """按时间切片，长度不足时补零，保证与词时长一致。"""
    start = int(round(start_sec * samplerate))
    end = int(round(end_sec * samplerate))
    expected = max(1, end - start)
    chunk = waveform[start:end]
    if len(chunk) < expected:
        chunk = np.pad(chunk, (0, expected - len(chunk)))
    elif len(chunk) > expected:
        chunk = chunk[:expected]
    return chunk


def _predict_with_retry(
    aligner: GameOnnxAligner,
    waveform: np.ndarray,
    known_dur: list[float],
    cfg: dict,
    name: str,
    emit: Optional[Callable[[str], None]] = None,
):
    """单块推理；GPU 设备丢失时尝试恢复一次后重试。"""
    attempt = 0
    while True:
        try:
            return aligner.predict(
                waveform,
                known_dur,
                language_id=cfg["language_id"],
                seg_threshold=cfg["seg_threshold"],
                seg_radius=cfg["seg_radius"],
                t0=cfg["t0"],
                nsteps=cfg["nsteps"],
                ts=cfg["ts"],
                est_threshold=cfg["est_threshold"],
            )
        except Exception as error:
            if attempt < 2 and _ort_is_device_lost(error) and aligner.recover_from_device_lost():
                attempt += 1
                if emit is not None:
                    emit(f"⚠ GPU 设备已被驱动重置（设备丢失），"
                         f"已切换到 {aligner.device.upper()}，重试 {name}")
                continue
            raise


def _align_item(
    aligner: GameOnnxAligner,
    item: dict,
    index: int,
    cfg: dict,
    stop_check: Optional[Callable[[], bool]] = None,
    emit: Optional[Callable[[str], None]] = None,
) -> str:
    """处理一条 CSV 记录，把 note_seq / note_dur / note_slur 写回 ``item``。

    只依赖传入的 ``aligner`` 与 ``cfg``，因此线程池和进程池都可以直接调用。
    Returns: 一行摘要文本（块数 / 音符数）。
    """
    name = item.get("name") or f"#{index}"
    ph_seq = item["ph_seq"].split()
    ph_dur = [float(v) for v in item["ph_dur"].split()]
    ph_num = [int(v) for v in item["ph_num"].split()]
    validate_phones(ph_seq, ph_dur, ph_num)

    words = _build_word_records(ph_num, ph_dur)
    if not words:
        raise ValueError("没有有效音素")
    word_starts = np.cumsum([0.0] + [w["dur"] for w in words])

    use_wb = bool(cfg["use_wb"])
    uv_set = cfg["uv_set"]
    uv_note_cond = cfg["uv_note_cond"]
    samplerate = cfg["samplerate"]

    # 对齐使用的词 v/uv（不合并）
    if use_wb:
        _, word_vuv = parse_words(
            ph_seq, ph_dur, ph_num,
            uv_vocab=uv_set,
            uv_cond=cfg["uv_word_cond"],
            merge_consecutive_uv=False,
        )
    else:
        word_vuv = []

    wav_path = _find_wav(cfg["csv_path"], name)
    waveform, _ = librosa.load(str(wav_path), sr=samplerate, mono=True)

    chunks = _split_word_chunks(words, word_vuv, cfg["max_chunk_sec"])
    all_note_seq: list[str] = []
    all_note_dur: list[float] = []
    all_note_slur: list[int] = []

    for chunk_indices in chunks:
        if stop_check is not None and stop_check():
            raise GameInferenceStopped()

        w0, w1 = chunk_indices[0], chunk_indices[-1]
        chunk_ph_dur = ph_dur[words[w0]["ph_start"]:words[w1]["ph_end"]]
        chunk_word_dur = [words[i]["dur"] for i in chunk_indices]
        chunk_word_vuv = [word_vuv[i] for i in chunk_indices] if use_wb else []

        start_sec = float(word_starts[w0])
        end_sec = float(word_starts[w1 + 1])
        chunk_waveform = _slice_audio(waveform, samplerate, start_sec, end_sec)

        if use_wb:
            # 模型使用的词时长：与 GAME 一致，连续无声词会合并
            if uv_set is not None:
                chunk_known_dur, _ = merge_consecutive_uv_words(chunk_word_dur, chunk_word_vuv)
            else:
                chunk_known_dur = chunk_word_dur
        else:
            chunk_known_dur = [float(sum(chunk_ph_dur))]

        note_dur_arr, presence_arr, scores_arr = _predict_with_retry(
            aligner, chunk_waveform, chunk_known_dur, cfg, name, emit)

        valid = note_dur_arr > 0
        note_dur = note_dur_arr[valid].astype(float).tolist()
        note_midi = scores_arr[valid].astype(float).tolist()
        note_vuv = presence_arr[valid].astype(bool).tolist()

        if use_wb:
            if uv_note_cond == "follow":
                note_seq = [_midi_to_note(midi) for midi in note_midi]
            else:
                note_seq = [
                    _midi_to_note(midi) if vuv else "rest"
                    for midi, vuv in zip(note_midi, note_vuv)
                ]
            note_seq, note_dur, note_slur = align_notes_to_words(
                chunk_word_dur, chunk_word_vuv, note_seq, note_dur,
                apply_word_uv=(uv_note_cond == "follow"),
            )
        else:
            note_seq = [
                _midi_to_note(midi) if vuv else "rest"
                for midi, vuv in zip(note_midi, note_vuv)
            ]
            note_slur = None

        all_note_seq.extend(note_seq)
        all_note_dur.extend(note_dur)
        if note_slur is not None:
            all_note_slur.extend(note_slur)

    item["note_seq"] = " ".join(all_note_seq)
    # 保留 6 位小数（原来是 3 位，逐个四舍五入会让总时长与音素差几毫秒）
    if all_note_dur:
        delta = float(math.fsum(ph_dur)) - float(math.fsum(all_note_dur))
        if abs(delta) <= 0.05:
            all_note_dur[-1] = max(0.0, all_note_dur[-1] + delta)
    item["note_dur"] = " ".join(f"{float(d):.6f}" for d in all_note_dur)
    if use_wb:
        item["note_slur"] = " ".join(str(v) for v in all_note_slur)
    else:
        item.pop("note_slur", None)
    item.pop("note_glide", None)

    return f"{len(chunks)} 块 / {len(all_note_seq)} 个音符"


# ---------------------------------------------------------------------------
# 多进程 worker：DirectML / CUDA 的 session 不支持并发 run（同进程多 session
# 并发同样会闪退），只能像 HubertFA 那样一个进程一份模型。
# ---------------------------------------------------------------------------

_MP_STATE: dict = {}


def _mp_init(model_dir, device, cfg, cancel_event):
    _MP_STATE["aligner"] = GameOnnxAligner(model_dir, device=device)
    _MP_STATE["cfg"] = cfg
    _MP_STATE["cancel"] = cancel_event


def _mp_task(index: int, item: dict):
    aligner = _MP_STATE["aligner"]
    cfg = _MP_STATE["cfg"]
    cancel = _MP_STATE.get("cancel")
    try:
        detail = _align_item(
            aligner, item, index, cfg,
            stop_check=(cancel.is_set if cancel is not None else None),
        )
    except GameInferenceStopped:
        return index, False, None, "已停止"
    except Exception:  # noqa: BLE001 - 回传给主进程展示
        return index, False, None, traceback.format_exc()
    return index, True, item, detail


def align_transcriptions(
    transcription,
    model_dir,
    language: Optional[str] = None,
    device: str = "cpu",
    output_path=None,
    overwrite: bool = False,
    use_wb: bool = True,
    uv_vocab=DEFAULT_UV_VOCAB,
    uv_word_cond: str = "lead",
    uv_note_cond: str = "predict",
    seg_threshold: float = 0.2,
    seg_radius: float = 0.02,
    t0: float = 0.0,
    nsteps: int = 8,
    ts=None,
    est_threshold: float = 0.2,
    max_chunk_sec: float = 30.0,
    workers: int = 1,
    progress_callback: Optional[Callable[[str], None]] = None,
    stop_event=None,
):
    """为 transcriptions.csv 添加 GAME 的 note_seq / note_dur / note_slur。

    Args:
        max_chunk_sec: 长音频按词边界分块推理的最大秒数（优先在无声音素后切开）；
            <=0 表示不分块。分块可以显著降低显存峰值。
        workers: 并行处理的条数。1 = 串行。
            - CPU：线程池共享一份 session（``session.run`` 会释放 GIL，可真正并行）；
            - DirectML / CUDA：单个 session 不能并发 ``run()``（会闪退），同进程
              多个 session 并发同样会崩，因此改用**多进程**（每个进程一份模型，
              思路同 HubertFA）。显存约为单实例的 workers 倍，进程启动时各自
              加载一次模型，小数据集收益有限。

    Returns:
        统计字典：total / updated / failed / stopped / output / providers / workers。
    """
    csv_path = pathlib.Path(transcription)
    if not csv_path.is_file():
        raise FileNotFoundError(f"transcriptions.csv 不存在: {csv_path}")
    if not use_wb and uv_note_cond == "follow":
        raise ValueError("关闭词-音符对齐时不能使用 uv_note_cond=follow")

    workers = max(1, min(int(workers or 1), 32))

    uv_set = _parse_uv_vocab(uv_vocab) if use_wb else None

    def log(message: str):
        if progress_callback is not None:
            progress_callback(message)
        else:
            print(message)

    aligner = GameOnnxAligner(model_dir, device=device)
    samplerate = aligner.samplerate
    language_id = aligner.language_id(language)

    log(f"ONNX ExecutionProvider: {', '.join(aligner.providers)}（请求设备: {device}）")

    # DirectML / CUDA 的 InferenceSession 不能并发 run（同进程就算是不同 session
    # 并发也会闪退），只能像 HubertFA 那样开多个进程、每进程一份模型。
    gpu_providers = {"DmlExecutionProvider", "CUDAExecutionProvider", "TensorrtExecutionProvider"}
    active_gpu = sorted(gpu_providers & set(aligner.providers))
    use_processes = workers > 1 and bool(active_gpu)
    if use_processes:
        log(f"{'/'.join(active_gpu)} 的 session 不能并发推理，改用多进程："
            f"{workers} 个进程各加载一份模型（显存约为单实例的 {workers} 倍，"
            f"启动时每个进程都要加载一次模型）")
    elif workers > 1:
        log(f"并行线程数: {workers}（CPU 共享同一份 session）")

    # 多线程下 worker 不直接往 GUI 写日志，统一投递到队列由主线程顺序输出
    if workers <= 1:
        def emit(message: str) -> None:
            log(message)

        def drain() -> None:
            return None
    else:
        _msg_queue: "queue.Queue[str]" = queue.Queue()

        def emit(message: str) -> None:
            _msg_queue.put(message)

        def drain() -> None:
            while True:
                try:
                    log(_msg_queue.get_nowait())
                except queue.Empty:
                    return

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        items = list(reader)

    required = {"name", "ph_seq", "ph_dur", "ph_num"}
    missing = required - set(fieldnames)
    if missing:
        raise ValueError(
            "transcriptions.csv 缺少必要列: " + ", ".join(sorted(missing)) +
            "\n请先运行「生成ph_num」。"
        )
    if not items:
        raise ValueError("transcriptions.csv 没有数据行")

    total = len(items)
    updated = 0
    failed = 0
    stopped = False

    lost_lock = threading.Lock()

    # 传给线程 / 进程 worker 的运行参数
    cfg = {
        "csv_path": csv_path,
        "use_wb": use_wb,
        "uv_set": uv_set,
        "uv_word_cond": uv_word_cond,
        "uv_note_cond": uv_note_cond,
        "language_id": language_id,
        "seg_threshold": seg_threshold,
        "seg_radius": seg_radius,
        "t0": t0,
        "nsteps": nsteps,
        "ts": ts,
        "est_threshold": est_threshold,
        "max_chunk_sec": max_chunk_sec,
        "samplerate": samplerate,
    }

    def _process(index: int, item: dict) -> tuple[bool, str]:
        """处理一条记录：只写 item 自身，不共享可变状态，可被多线程并发调用。"""
        detail = _align_item(
            aligner, item, index, cfg,
            stop_check=(stop_event.is_set if stop_event is not None else None),
            emit=emit,
        )
        return True, detail

    def _report_failure(name: str, detail: str) -> None:
        log(f"❌ {name} 生成 MIDI 失败（已跳过）：\n{detail}")

    if workers <= 1:
        for index, item in enumerate(items, 1):
            if stop_event is not None and stop_event.is_set():
                stopped = True
                log("已请求停止，剩余文件未处理。")
                break
            name = item.get("name") or f"#{index}"
            try:
                _, detail = _process(index, item)
            except GameInferenceStopped:
                stopped = True
                log("已请求停止，剩余文件未处理。")
                break
            except Exception:
                failed += 1
                _report_failure(name, traceback.format_exc())
                continue
            updated += 1
            log(f"({index}/{total}) {name} → {detail}")
    elif use_processes:
        # DirectML / CUDA：每个进程一份模型，进程间互不干扰（参考 HubertFA 的多进程）
        ctx = mp.get_context("spawn")
        cancel_event = ctx.Event()
        watcher_done = threading.Event()

        def _watch_stop() -> None:
            while not watcher_done.is_set():
                if stop_event is not None and stop_event.is_set():
                    cancel_event.set()
                    return
                watcher_done.wait(0.2)

        if stop_event is not None:
            threading.Thread(target=_watch_stop, daemon=True).start()

        try:
            with ProcessPoolExecutor(
                max_workers=workers,
                mp_context=ctx,
                initializer=_mp_init,
                initargs=(str(model_dir), str(device), cfg, cancel_event),
            ) as pool:
                future_map = {
                    pool.submit(_mp_task, index, item): index
                    for index, item in enumerate(items, 1)
                }
                pending = set(future_map)
                done = 0
                for future in as_completed(future_map):
                    pending.discard(future)
                    done += 1
                    index = future_map[future]
                    name = items[index - 1].get("name") or f"#{index}"
                    try:
                        res_index, ok, new_item, detail = future.result()
                    except Exception:  # noqa: BLE001 - 子进程崩溃/超时等
                        failed += 1
                        _report_failure(name, traceback.format_exc())
                        continue
                    if not ok:
                        if detail == "已停止":
                            stopped = True
                            log("已请求停止，剩余文件未处理。")
                            for rest in pending:
                                rest.cancel()
                            break
                        failed += 1
                        _report_failure(name, detail)
                        continue
                    items[res_index - 1] = new_item
                    updated += 1
                    log(f"({done}/{total}) {name} → {detail}")
                    if stop_event is not None and stop_event.is_set():
                        stopped = True
                        log("已请求停止，剩余文件未处理。")
                        for rest in pending:
                            rest.cancel()
                        break
        finally:
            watcher_done.set()
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            future_map = {
                pool.submit(_process, index, item): (index, item)
                for index, item in enumerate(items, 1)
            }
            pending = set(future_map)
            done = 0
            for future in as_completed(future_map):
                pending.discard(future)
                drain()
                index, item = future_map[future]
                name = item.get("name") or f"#{index}"
                done += 1
                try:
                    _, detail = future.result()
                except GameInferenceStopped:
                    stopped = True
                    for rest in pending:
                        rest.cancel()
                    break
                except Exception:
                    failed += 1
                    _report_failure(name, traceback.format_exc())
                    if stop_event is not None and stop_event.is_set():
                        stopped = True
                        for rest in pending:
                            rest.cancel()
                        break
                    continue
                updated += 1
                log(f"({done}/{total}) {name} → {detail}")
                if stop_event is not None and stop_event.is_set():
                    stopped = True
                    for rest in pending:
                        rest.cancel()
                    break
    drain()

    if stopped:
        log("已请求停止，剩余文件未处理。")

    # 输出路径
    if overwrite:
        save_path = csv_path
    elif output_path:
        save_path = pathlib.Path(output_path)
    else:
        save_path = csv_path.parent / "transcriptions-midi.csv"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # 保持原列顺序，新增列追加在末尾；note_glide 若已全部删除则不再写出
    output_fields: list[str] = []
    for item in items:
        for key in item.keys():
            if key not in output_fields:
                output_fields.append(key)
    if updated == total and "note_glide" in output_fields:
        output_fields.remove("note_glide")

    with open(save_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=output_fields)
        writer.writeheader()
        writer.writerows(items)

    log(f"完成：共处理 {updated}/{total} 条"
        + (f"，失败 {failed} 条" if failed else "")
        + f" → {save_path}")
    return {
        "total": total,
        "updated": updated,
        "failed": failed,
        "stopped": stopped,
        "workers": workers,
        "output": str(save_path),
        "providers": list(aligner.providers),
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="GAME ONNX inference: add note_seq/note_dur/note_slur to transcriptions.csv"
    )
    parser.add_argument("transcription", help="Path to transcriptions.csv")
    parser.add_argument("-m", "--model", required=True, help="GAME ONNX model directory")
    parser.add_argument("--language", default=None, help="Language code, e.g. zh / ja / en")
    parser.add_argument("--device", default="cpu", choices=["cpu", "dml", "cuda"])
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of audio files processed in parallel (CPU: 2~4 recommended)")
    parser.add_argument("--output", default=None, help="Output CSV path")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the input CSV")
    parser.add_argument("--no-wb", action="store_true", help="Disable word-note alignment")
    parser.add_argument("--uv-vocab", default=DEFAULT_UV_VOCAB, help="Unvoiced phonemes")
    parser.add_argument("--uv-word-cond", default="lead", choices=["lead", "all"])
    parser.add_argument("--uv-note-cond", default="predict", choices=["predict", "follow"])
    parser.add_argument("--seg-threshold", type=float, default=0.2)
    parser.add_argument("--seg-radius", type=float, default=0.02)
    parser.add_argument("--t0", type=float, default=0.0)
    parser.add_argument("--nsteps", type=int, default=8)
    parser.add_argument("--est-threshold", type=float, default=0.2)
    parser.add_argument("--max-chunk-sec", type=float, default=30.0,
                        help="Max seconds per chunk (0 = no chunking; splits at word/unvoiced boundaries)")
    args = parser.parse_args()

    align_transcriptions(
        args.transcription,
        args.model,
        language=args.language,
        device=args.device,
        output_path=args.output,
        overwrite=args.overwrite,
        use_wb=not args.no_wb,
        uv_vocab=args.uv_vocab,
        uv_word_cond=args.uv_word_cond,
        uv_note_cond=args.uv_note_cond,
        seg_threshold=args.seg_threshold,
        seg_radius=args.seg_radius,
        t0=args.t0,
        nsteps=args.nsteps,
        est_threshold=args.est_threshold,
        max_chunk_sec=args.max_chunk_sec,
        workers=args.workers,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
