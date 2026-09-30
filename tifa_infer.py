"""TIFA ONNX 强制对齐推理（独立实现，基于 onnxruntime）。

严格对应 TIFA 原版推理流程：

    waveform ──spectrogram.onnx──> spectrogram, maskT
    spectrogram, tokens, maskT, maskN ──model.onnx──> similarities, logits
    similarities ──Viterbi(见 modules/decoding.py::decode_alignment_flat)──> spans
    spans ──×timestep──> 秒 ──> TextGrid(words/phones 两层，首尾补 SP)

说明：
    * 本模块只实现**词典 G2P（每词单一发音）**路径，与本项目 HFA 推理的输入
      约定完全一致（``.lab`` 内为空格分隔的词序列，词典映射为音素序列）。
      这正是 TIFA 原版 ``configs/g2p.yaml`` 中 ``dictionary`` 转换器的行为。
    * TIFA 的 ``prepare/score/select`` 三张图只在**一词多音（多音字）**时才
      参与打分，单发音词典路径下候选网格无歧义（C=1），因此无需调用。
    * 批量推理：`batch_size` 越大吞吐越高，显存/内存占用也越大。
"""

import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import onnxruntime as ort

# ── 与 TIFA lib/vocabulary.py 保持一致 ──
MASK_TOKEN = 1
SPACE_TOKEN = 2
NUM_RESERVED_TOKENS = 3

# Viterbi 内部使用的负无穷（必须是真 -inf，否则加惩罚项会伪造出可达状态）
NEG_INF = np.float32(-np.inf)


class StopInference(Exception):
    """用户请求强制停止推理。"""


# ---------------------------------------------------------------------------
# 数值工具
# ---------------------------------------------------------------------------

def _log_softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    return x - x_max - np.log(np.sum(np.exp(x - x_max), axis=axis, keepdims=True))


# ---------------------------------------------------------------------------
# Viterbi 解码（移植自 TIFA modules/decoding.py::_decode_flat_single）
#
# 目标：最大化「逐帧余弦相似度之和 − 每个被跳过 token 的 skip_penalty」。
# 每个 token 至少发射一帧，否则付出 skip_penalty；退出/跳过不消耗帧。
# groups 只限制「间隙等待」，不限制零时长转移。
# ---------------------------------------------------------------------------

def _decode_flat_single(sim: np.ndarray, skip_penalty: float, gap_allowed: np.ndarray) -> np.ndarray:
    """单条序列解码。sim: [T, N] float32；返回 spans [N, 2] int64（帧索引）。"""
    T, N = sim.shape
    penalty = float(skip_penalty)

    gap = np.full(N + 1, NEG_INF, dtype=np.float32)
    token = np.full(N, NEG_INF, dtype=np.float32)
    gap_back = np.zeros((T + 1, N + 1), dtype=np.int8)
    token_back = np.zeros((T + 1, N), dtype=np.int8)

    gap[0] = 0.0
    for i in range(N):
        gap[i + 1] = gap[i] - penalty
        gap_back[0, i + 1] = 2  # skip

    idx = np.arange(N + 1, dtype=np.float32)

    for t in range(1, T + 1):
        # ── token 步：可由自身或由 gap 进入 ──
        if N:
            enter = gap[:N] > token
            token_back[t] = enter.astype(np.int8)
            best = np.where(enter, gap[:N], token)
            next_token = (best + sim[t - 1]).astype(np.float32)
        else:
            next_token = token

        # ── gap 步：wait / 从 token 退出 / 从 gap 跳过 ──
        # 参考实现（顺序 i = 0..N-1）：
        #   ng[j] = max(init[j], nt[j-1], ng[j-1] - p)
        # 展开后 ng[j] = max_{k<=j} (V[k] - p*(j-k)),  V[k] = max(init[k], nt_ext[k])
        # 于是可用「前缀最大」一次向量化：W[k] = V[k] + p*k，ng[j] = maxW(j) - p*j。
        # 若 W[j] >= maxW(j-1)（即在 j 处产生新纪录），说明参考实现会在 j 处直接
        # 采用 init/nt（来源 0/1）；否则说明是来自更早状态的跳过（来源 2）。
        init = np.where(gap_allowed, gap, NEG_INF)
        nt_ext = np.empty(N + 1, dtype=np.float32)
        nt_ext[0] = NEG_INF
        if N:
            nt_ext[1:] = next_token

        v = np.maximum(init, nt_ext)
        w = v + penalty * idx
        running = np.maximum.accumulate(w)

        prev = np.empty_like(running)
        prev[0] = NEG_INF
        if N:
            prev[1:] = running[:-1]
        record = w >= prev
        first = np.maximum.accumulate(np.where(record, idx, np.float32(-1.0))).astype(np.int64)

        next_gap = (running - penalty * idx).astype(np.float32)

        # 回溯来源：0=wait(diag)，1=token 退出，2=skip
        j = np.arange(N + 1)
        src01 = np.where(init >= nt_ext, 0, 1).astype(np.int8)
        gap_back[t] = np.where(first < j, 2, src01).astype(np.int8)

        gap = next_gap
        token = next_token

    # ── 回溯 ──
    spans = np.full((N, 2), -1, dtype=np.int64)
    t, i = T, N
    in_token = False
    while t > 0 or i > 0 or in_token:
        if in_token:
            if spans[i, 1] < 0:
                spans[i, 1] = t
            spans[i, 0] = t - 1
            entered = token_back[t, i]
            t -= 1
            if entered == 1:
                in_token = False
        else:
            source = gap_back[t, i]
            if source == 0:
                t -= 1
            elif source == 1:
                i -= 1
                in_token = True
            else:
                i -= 1
                spans[i, 0] = t
                spans[i, 1] = t

    _canonicalize_skipped_spans(spans, T, gap_allowed)
    return spans


def _canonicalize_skipped_spans(spans: np.ndarray, T: int, gap_allowed: np.ndarray) -> None:
    """把被跳过 token 的零时长区间挪到本组内首个允许的间隙处。"""
    N = len(spans)
    lo = 0
    while lo < N:
        if spans[lo, 0] != spans[lo, 1]:
            lo += 1
            continue
        hi = lo
        while hi + 1 < N and spans[hi + 1, 0] == spans[hi + 1, 1]:
            hi += 1
        left = spans[lo - 1, 1] if lo > 0 else 0
        right = spans[hi + 1, 0] if hi + 1 < N else T
        gap_index = lo
        while gap_index <= hi + 1 and not gap_allowed[gap_index]:
            gap_index += 1
        for i in range(lo, hi + 1):
            anchor = left if i < gap_index else right
            spans[i, 0] = anchor
            spans[i, 1] = anchor
        lo = hi + 1


# ---------------------------------------------------------------------------
# 音频
# ---------------------------------------------------------------------------

def _load_audio(path, sample_rate: int) -> np.ndarray:
    """读取单声道音频并重采样到 sample_rate。"""
    try:
        import librosa
        wav, _ = librosa.load(str(path), sr=sample_rate, mono=True)
        return np.asarray(wav, dtype=np.float32)
    except Exception:
        import soundfile as sf
        wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != sample_rate:
            import librosa
            wav = librosa.resample(np.asarray(wav, dtype=np.float32), orig_sr=sr, target_sr=sample_rate)
        return np.asarray(wav, dtype=np.float32)


# ---------------------------------------------------------------------------
# 主类
# ---------------------------------------------------------------------------

class TifaInference:
    """TIFA ONNX 推理。

    典型用法::

        inf = TifaInference(model_dir)
        inf.load_config()
        inf.load_model(device="cpu")
        inf.get_dataset(wav_folder, language="en", dictionary_path=dict_path)
        inf.set_progress_callback(print)
        inf.infer(batch_size=8)
        inf.export(wav_folder)
    """

    # ── 虚拟音素表 ──
    # {虚拟音素: 实际喂给模型的音素}。虚拟音素不需要存在于 vocabulary.json，
    # 但可以写在字典里；推理时用右边的「真实音素」做声学对齐，
    # 而 TextGrid 里仍然输出左边原标签（去掉语言前缀后的部分）。
    #
    # 典型场景：日语鼻濁音(ガ行鼻音)。TIFA 的日语集里没有 ja/ng，
    # 若直接写 en/ng（英语的 /ŋ/）模型会给负相似度并把它整段丢弃；
    # 写成 ja/ng 即可用 ja/g 对齐、标签仍输出 "ng"。
    VIRTUAL_PHONEMES: dict[str, str] = {
        "ja/ng": "ja/g",
    }

    # ── 直接跳过的符号 ──
    # lab 或字典里出现这些符号时不产生任何 token（词与音素两个层级都适用）。
    # UTAU 惯例：R = 休止/空白，SP = 空白音素。TIFA 词表里没有 SP，
    # 静音本来就由解码器的 gap 状态表示，所以这里直接跳过即可。
    SKIP_SYMBOLS: set[str] = {"R", "SP"}

    def __init__(self, model_dir):
        self.model_dir = Path(model_dir)
        self.cfg = None
        self.symbols = {}
        self.sessions = {}
        self.session_providers = {}
        self.device = "cpu"
        self.device_info = {}
        self.dataset = []          # [(wav_path, tokens, words, groups, phonemes, word_texts)]
        self.predictions = []      # [(wav_path, wav_length, intervals)]
        self.progress_callback = None
        self._lang = None
        self._timings = None
        self._substitutions = {}   # {虚拟音素: 出现次数}
        self._skipped_symbols = {} # {被跳过的符号: 出现次数}
        self._timers = {"spectrogram": 0.0, "model": 0.0, "viterbi": 0.0}

    # ── 回调 ──
    def set_progress_callback(self, callback):
        self.progress_callback = callback

    def _emit(self, msg: str):
        if self.progress_callback:
            self.progress_callback(msg)
        else:
            print(msg)

    # ── 配置 ──
    def load_config(self):
        cfg_path = self.model_dir / "config.json"
        vocab_path = self.model_dir / "vocabulary.json"
        if not cfg_path.exists():
            raise FileNotFoundError(f"缺少 config.json: {cfg_path}")
        if not vocab_path.exists():
            raise FileNotFoundError(f"缺少 vocabulary.json: {vocab_path}")

        with open(cfg_path, "r", encoding="utf-8") as f:
            self.cfg = json.load(f)
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        self.symbols = dict(vocab.get("symbols", {}))
        self._lang_prefixes = sorted({k.split("/")[0] for k in self.symbols if "/" in k})

        self.sample_rate = int(self.cfg["samplerate"])
        self.timestep = float(self.cfg["timestep"])
        self.hop_size = int(self.cfg["hop_size"])
        self.fft_size = int(self.cfg.get("fft_size", 2048))
        self.win_size = int(self.cfg.get("win_size", 2048))
        self.num_mels = int(self.cfg["num_mels"])
        self.vocab_size = int(self.cfg["vocab_size"])
        if not self.symbols:
            raise ValueError("vocabulary.json 中没有 symbols。")

    # ── 词表解析 ──
    def _resolve(self, phoneme: str, language):
        """先按原样查找，再尝试 ``language/phoneme``，最后遍历词表内所有语言前缀。

        TIFA 词表中的音素带语言前缀（如 ``en/aa``、``zh/l``），而字典文件里
        可能是裸音素（``aa``）或已带前缀（``zh/l``），逐级回退即可兼容。
        """
        if phoneme in self.symbols:
            return self.symbols[phoneme]
        if language:
            key = f"{language}/{phoneme}"
            if key in self.symbols:
                return self.symbols[key]
        for prefix in getattr(self, "_lang_prefixes", ()):
            key = f"{prefix}/{phoneme}"
            if key in self.symbols:
                return self.symbols[key]
        return None

    # ── 设备 / 会话 ──
    @staticmethod
    def _provider_for(device: str):
        device = (device or "cpu").lower()
        if device == "dml":
            return "DmlExecutionProvider"
        if device == "webgpu":
            return "WebGpuExecutionProvider"
        return "CPUExecutionProvider"

    def _create_session(self, onnx_path: Path):
        available = ort.get_available_providers()
        wanted = self._provider_for(self.device)
        providers = []
        if wanted in available:
            providers.append(wanted)
        # CPU 兜底，保证一定可运行
        if "CPUExecutionProvider" in available and "CPUExecutionProvider" not in providers:
            providers.append("CPUExecutionProvider")
        if not providers:
            providers = available
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        return ort.InferenceSession(str(onnx_path), sess_options=opts, providers=providers)

    def load_model(self, device: str = "cpu"):
        self.device = (device or "cpu").lower()
        need = ["spectrogram", "model"]
        self.session_providers = {}
        for name in need:
            path = self.model_dir / f"{name}.onnx"
            if not path.exists():
                raise FileNotFoundError(f"缺少 ONNX 图: {path}")
            session = self._create_session(path)
            self.sessions[name] = session
            self.session_providers[name] = list(session.get_providers())

        available = ort.get_available_providers()
        primary = self.session_providers.get("model", ["CPUExecutionProvider"])[0]
        self.device_info = {"primary_device": primary, "enabled_providers": available}

    # ── 数据集 ──
    @staticmethod
    def _load_dictionary(dict_path: Path) -> dict:
        dictionary = {}
        with open(dict_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or "\t" not in line:
                    continue
                word, phones = line.split("\t", 1)
                phones = [p for p in phones.strip().split(" ") if p]
                if phones:
                    dictionary[word.strip()] = phones
        return dictionary

    def get_dataset(self, wav_folder, language, g2p: str = "dictionary",
                    dictionary_path=None, in_format: str = "lab",
                    skip_symbols=None):
        """扫描 wav，读取同名 ``.lab`` 并转换为音素/词/组序列。"""
        if g2p != "dictionary":
            raise ValueError(f"TIFA 推理当前仅支持 dictionary G2P，收到: {g2p}")
        if dictionary_path is None or not os.path.exists(dictionary_path):
            raise FileNotFoundError(f"字典文件不存在: {dictionary_path}")

        self._lang = language
        self._substitutions = {}
        self._skipped_symbols = {}
        skip_symbols = self.SKIP_SYMBOLS if skip_symbols is None else set(skip_symbols)
        dictionary = self._load_dictionary(Path(dictionary_path))

        self.dataset = []
        skipped = 0
        for wav_path in sorted(Path(wav_folder).rglob("*.wav")):
            lab_path = wav_path.with_suffix("." + in_format)
            if not lab_path.exists():
                skipped += 1
                continue
            try:
                with open(lab_path, "r", encoding="utf-8") as f:
                    text = f.read().strip()
            except Exception as e:
                warnings.warn(f"读取失败 {lab_path}: {e}")
                skipped += 1
                continue
            if not text:
                skipped += 1
                continue

            try:
                item = self._encode_text(text, dictionary, language, skip_symbols)
            except Exception as e:
                warnings.warn(f"处理失败 {wav_path}: {e}")
                skipped += 1
                continue
            if item is None:
                skipped += 1
                continue
            self.dataset.append((wav_path, *item))

        self._emit(f"已载入 {len(self.dataset)} 个样本" + (f"（跳过 {skipped} 个）" if skipped else ""))
        if self._skipped_symbols:
            detail = ", ".join(f"{k}×{v}" for k, v in sorted(self._skipped_symbols.items()))
            self._emit(f"已跳过空白符号（不产生 token）: {detail}")
        if self._substitutions:
            detail = ", ".join(
                f"{k} → {self.VIRTUAL_PHONEMES[k]}（标签仍输出 '{k.split('/')[-1]}'）"
                for k in sorted(self._substitutions)
            )
            self._emit(f"虚拟音素替代 {sum(self._substitutions.values())} 处: {detail}")
        return len(self.dataset)

    def _encode_text(self, text, dictionary, language, skip_symbols=None):
        """空格分词的词典 G2P → (tokens, words, groups, phonemes, word_texts)。

        ``skip_symbols`` 中的符号（默认 ``{"R", "SP"}``）在词、音素两个层级
        都会被直接跳过：不产生 token，也不进入 TextGrid，静音交给解码器的
        gap 状态表示。
        """
        skip_symbols = self.SKIP_SYMBOLS if skip_symbols is None else set(skip_symbols)
        raw_words = [w for w in text.split() if w]
        tokens, words, groups, phonemes, word_texts = [], [], [], [], []
        missing = []
        for word in raw_words:
            if word in skip_symbols:
                self._skipped_symbols[word] = self._skipped_symbols.get(word, 0) + 1
                continue
            phones = dictionary.get(word)
            if not phones:
                missing.append(word)
                continue
            ids = []
            labels = []
            for ph in phones:
                if ph in skip_symbols:
                    self._skipped_symbols[ph] = self._skipped_symbols.get(ph, 0) + 1
                    continue
                tid = self._resolve(ph, language)
                if tid is None:
                    # 虚拟音素：词表里没有，改用替代音素做声学对齐，
                    # 但标签仍然输出原符号（下面按原 ph 计算）。
                    substitute = self.VIRTUAL_PHONEMES.get(ph)
                    if substitute is not None:
                        tid = self._resolve(substitute, language)
                        if tid is not None:
                            self._substitutions[ph] = self._substitutions.get(ph, 0) + 1
                if tid is None:
                    continue
                ids.append(tid)
                # 与 TIFA SaveTextGridCallback 一致：输出去掉语言前缀
                label = ph
                if "/" in ph:
                    head, tail = ph.split("/", 1)
                    if head in getattr(self, "_lang_prefixes", ()):
                        label = tail
                labels.append(label)
            if not ids:
                missing.append(word)
                continue
            word_id = len(word_texts) + 1
            tokens.extend(ids)
            words.extend([word_id] * len(ids))
            groups.extend([word_id] * len(ids))
            phonemes.extend(labels)
            word_texts.append(word)
        if missing:
            warnings.warn(f"以下词不在字典中，已跳过: {missing}")
        if not tokens:
            return None
        return (
            np.asarray(tokens, dtype=np.int64),
            np.asarray(words, dtype=np.int64),
            np.asarray(groups, dtype=np.int64),
            phonemes,
            word_texts,
        )

    # ── 推理 ──
    def infer(self, batch_size: int = 8, skip_penalty: float = 0.5, stop_event=None,
              warmup: bool = True, boundary_label: str = "SP", fill_gaps: bool = True):
        if not self.sessions:
            raise RuntimeError("请先调用 load_model()。")
        if batch_size < 1:
            raise ValueError("batch_size 必须 >= 1")

        self.predictions = []
        total = len(self.dataset)

        # ── 输出真实的 ONNX Runtime 执行提供器（EP）配置 ──
        self._emit("=" * 60)
        self._emit(f"onnxruntime 版本: {ort.__version__}")
        self._emit(f"可用执行提供器: {', '.join(ort.get_available_providers())}")
        for name in ("spectrogram", "model"):
            if name in self.session_providers:
                self._emit(f"  {name}.onnx 实际使用: {self.session_providers[name]}")
        self._emit(f"批大小: {batch_size} | 跳过惩罚: {skip_penalty} | 首尾空白标签: {boundary_label or '(空)'}")
        self._emit(f"填补中间空隙: {'是' if fill_gaps else '否'}")

        if total == 0:
            self._emit("没有可推理的样本。")
            self._emit("=" * 60)
            return

        primary = self.session_providers.get("model", ["CPUExecutionProvider"])[0]
        # if primary in ("WebGpuExecutionProvider", "DmlExecutionProvider"):
        #     self._emit(
        #         "提示: GPU EP 对这类小模型常有较大的内核调度/显存拷贝开销，未必快于 CPU；"
        #         "若某算子不被 GPU EP 支持，ORT 会自动回退到 CPU 并产生额外拷贝。"
        #         "建议用同一批音频分别测 CPU / GPU 后再决定。"
        #     )
        # self._emit("=" * 60)

        # ── 预热：初始化 ORT 内核/显存，其耗时不计入统计 ──
        if warmup:
            t_warmup = time.perf_counter()
            self._infer_batch(self.dataset[:1], skip_penalty, boundary_label, fill_gaps)
            self._emit(f"已预热: {time.perf_counter() - t_warmup:.2f}s（不计入统计）")
        self._timers = {"spectrogram": 0.0, "model": 0.0, "viterbi": 0.0}

        start = time.perf_counter()
        processed = 0
        batch_index = 0
        for begin in range(0, total, batch_size):
            if stop_event is not None and stop_event.is_set():
                raise StopInference()
            chunk = self.dataset[begin:begin + batch_size]
            batch_index += 1

            t_batch = time.perf_counter()
            results = self._infer_batch(chunk, skip_penalty, boundary_label, fill_gaps)
            batch_seconds = time.perf_counter() - t_batch

            self.predictions.extend(results)
            processed += len(chunk)

            elapsed = time.perf_counter() - start
            rate = elapsed / processed
            eta = rate * (total - processed)
            self._emit(
                f"Processing {processed}/{total}... "
                f"[批{batch_index} {len(chunk)}个 {batch_seconds:.2f}s | "
                f"累计 {elapsed:.1f}s | 均 {rate * 1000:.0f}ms/个 | 剩余≈{eta:.0f}s]"
            )

        elapsed = time.perf_counter() - start
        rate = elapsed / max(total, 1)
        self._emit("-" * 60)
        self._emit(
            f"推理完成，总耗时: {elapsed:.1f}s  (共 {total} 个文件，"
            f"平均 {rate:.3f}s/个，约 {1 / rate:.2f} 个/秒)"
        )
        self._emit_timing_breakdown(elapsed)
        self._emit("=" * 60)

    def _emit_timing_breakdown(self, elapsed: float):
        """输出各阶段耗时占比，用于判断瓶颈在频谱、模型还是解码。"""
        parts = []
        for key, label in (("spectrogram", "频谱"), ("model", "模型"), ("viterbi", "Viterbi")):
            value = self._timers.get(key, 0.0)
            pct = (value / elapsed * 100.0) if elapsed > 0 else 0.0
            parts.append(f"{label} {value:.1f}s ({pct:.0f}%)")
        other = elapsed - sum(self._timers.values())
        parts.append(f"其他(IO/组批) {other:.1f}s")
        self._emit("阶段耗时: " + " | ".join(parts))

    def _infer_batch(self, chunk, skip_penalty, boundary_label: str = "SP",
                     fill_gaps: bool = True):
        B = len(chunk)
        waveforms = []
        durations = []
        max_n = max(len(item[1]) for item in chunk)

        for wav_path, tokens, words, groups, phonemes, word_texts in chunk:
            wav = _load_audio(wav_path, self.sample_rate)
            # 反射填充要求长度足够，过短音频补一点静音
            min_len = (self.win_size - self.hop_size) + self.hop_size * 2
            if wav.shape[0] < min_len:
                wav = np.pad(wav, (0, min_len - wav.shape[0]))
            waveforms.append(wav)
            durations.append(wav.shape[0] / self.sample_rate)

        max_l = max(w.shape[0] for w in waveforms)
        wave_batch = np.zeros((B, max_l), dtype=np.float32)
        for i, w in enumerate(waveforms):
            wave_batch[i, :w.shape[0]] = w
        dur_batch = np.asarray(durations, dtype=np.float32)

        token_batch = np.zeros((B, max_n), dtype=np.int64)
        for i, item in enumerate(chunk):
            tk = item[1]
            token_batch[i, :len(tk)] = tk
        mask_n = token_batch != 0

        # ── spectrogram.onnx ──
        t_stage = time.perf_counter()
        spectrogram, mask_t = self.sessions["spectrogram"].run(
            None, {"waveform": wave_batch, "duration": dur_batch},
        )
        spectrogram = np.asarray(spectrogram, dtype=np.float32)
        mask_t = np.asarray(mask_t).astype(bool)
        self._timers["spectrogram"] += time.perf_counter() - t_stage

        # ── model.onnx ──
        t_stage = time.perf_counter()
        similarities, _logits = self.sessions["model"].run(
            None,
            {
                "spectrogram": spectrogram,
                "tokens": token_batch,
                "maskT": mask_t,
                "maskN": mask_n,
            },
        )
        similarities = np.asarray(similarities, dtype=np.float32)
        self._timers["model"] += time.perf_counter() - t_stage

        # ── 逐样本 Viterbi ──
        t_stage = time.perf_counter()
        results = []
        for b, (wav_path, tokens, words, groups, phonemes, word_texts) in enumerate(chunk):
            T_i = int(mask_t[b].sum())
            N_i = int(mask_n[b].sum())
            if N_i == 0 or T_i == 0:
                intervals = []
            else:
                sim_i = similarities[b, :T_i, :N_i]
                g = groups[:N_i]
                gap_allowed = np.ones(N_i + 1, dtype=bool)
                if N_i > 1:
                    gap_allowed[1:N_i] = g[:-1] != g[1:]
                spans = _decode_flat_single(sim_i, skip_penalty, gap_allowed)
                spans_sec = spans.astype(np.float64) * self.timestep
                intervals = self._build_intervals(
                    spans_sec, phonemes, words[:N_i], word_texts,
                    durations[b], boundary_label=boundary_label, fill_gaps=fill_gaps,
                )

            wav_length = float(durations[b])
            results.append((wav_path, wav_length, intervals))
        self._timers["viterbi"] += time.perf_counter() - t_stage
        return results

    @staticmethod
    def _build_intervals(spans_sec, phonemes, words, word_texts, wav_length,
                         boundary_label="SP", fill_gaps=True):
        """按 TIFA SaveTextGridCallback 的口径生成区间数据（words / phones 两层）。

        额外做三件事：
          * 所有未覆盖区间（含首尾、以及解码器留下的中间空白）都用
            ``boundary_label``（默认 SP）填满，使 Tier 首尾相接、完整覆盖
            [0, 音频时长]；``fill_gaps=False`` 时只补首尾（TIFA 原版行为）；
          * 不再输出冗余的 texts 层（与 words 层内容重复）。
        """
        eps = 0.001
        N = len(spans_sec)
        total = round(float(wav_length), 3)
        if N == 0:
            return {"total_duration": total, "words": [], "phones": []}

        rounded = [[round(float(s[0]), 3), round(float(s[1]), 3)] for s in spans_sec]

        # 零时长（被跳过）的 token：omit 掉
        keep = [i for i in range(N) if rounded[i][0] < rounded[i][1]]
        if not keep:
            return {"total_duration": total, "words": [], "phones": []}
        rounded = [rounded[i] for i in keep]
        labels = [phonemes[i] for i in keep]
        owners = [int(words[i]) for i in keep]
        N = len(rounded)

        phone_intervals = []
        for n in range(N):
            onset, offset = rounded[n]
            if phone_intervals and onset < phone_intervals[-1][1]:
                onset = phone_intervals[-1][1]
            if offset <= onset:
                offset = onset + eps
            phone_intervals.append((onset, offset, labels[n]))

        def grouped(owner_of, label_of):
            out = []
            i = 0
            while i < N:
                owner = owner_of[i]
                j = i + 1
                while j < N and owner_of[j] == owner:
                    j += 1
                onset = phone_intervals[i][0]
                offset = phone_intervals[j - 1][1]
                if out and onset < out[-1][1]:
                    onset = out[-1][1]
                if offset <= onset:
                    offset = onset + eps
                out.append((onset, offset, label_of(owner)))
                i = j
            return out

        words_tier = grouped(
            owners, lambda o: word_texts[o - 1] if 0 < o <= len(word_texts) else boundary_label
        )

        # ── 用 boundary_label 填满空隙 ──
        # 首尾总是补；中间空隙仅在 fill_gaps=True 时补（否则保留 TIFA 原版「中间留空」）。
        def fill(intervals):
            if not intervals:
                return [(0.0, total, boundary_label)] if total > 0 else []
            out = []
            cursor = 0.0
            for onset, offset, label in intervals:
                onset = min(max(float(onset), 0.0), total)
                offset = min(max(float(offset), onset), total)
                if onset > cursor:
                    # out 为空说明这是第一个区间 -> 首部总是补；
                    # 否则是中间空隙 -> 仅在 fill_gaps 时补。
                    if fill_gaps or not out:
                        if onset - cursor > eps:
                            out.append((cursor, onset, boundary_label))
                        else:      # 只差一点点，直接拉平，避免亚毫秒碎片
                            onset = cursor
                elif onset < cursor:
                    onset = cursor
                if offset <= onset:
                    continue
                out.append((onset, offset, label))
                cursor = offset
            if cursor < total:
                if total - cursor > eps:
                    out.append((cursor, total, boundary_label))
                elif out:
                    out[-1] = (out[-1][0], total, out[-1][2])
            return out

        phone_intervals = fill(phone_intervals)
        words_tier = fill(words_tier)

        return {
            "total_duration": total,
            "words": words_tier,
            "phones": phone_intervals,
        }

    # ── 导出 ──
    def export(self, output_folder):
        for wav_path, wav_length, data in self.predictions:
            if not data:
                continue
            tg_path = Path(wav_path).with_suffix(".TextGrid")
            self._write_textgrid(tg_path, wav_length, data)
        self._emit("已输出 TextGrid 到音频同目录。")

    @staticmethod
    def _fmt(v: float) -> str:
        return f"{v:.3f}".rstrip("0").rstrip(".") if v else "0"

    def _write_textgrid(self, path: Path, wav_length: float, data: dict):
        # xmax 取音频实际时长：下游 TextGrid2ds_json 会把前两个 xmin/xmax
        # 当作该条音频的长度（wav_long）。同时把区间裁剪进 [0, xmax]。
        total = round(float(wav_length), 3)
        lines = [
            'File type = "ooTextFile"',
            'Object class = "TextGrid"',
            "",
            "xmin = 0",
            f"xmax = {total}",
            "tiers? <exists>",
            "size = 2",
            "item []:",
        ]
        for item_index, (tier_name, intervals) in enumerate(
            (("words", data["words"]), ("phones", data["phones"])), start=1
        ):
            lines.append(f"    item [{item_index}]:")
            lines.append('        class = "IntervalTier"')
            lines.append(f'        name = "{tier_name}"')
            lines.append("        xmin = 0")
            lines.append(f"        xmax = {total}")
            lines.append(f"        intervals: size = {len(intervals)}")
            for i, (onset, offset, label) in enumerate(intervals, start=1):
                onset = min(max(float(onset), 0.0), total)
                offset = min(max(float(offset), onset), total)
                lines.append(f"        intervals [{i}]:")
                lines.append(f"            xmin = {self._fmt(onset)}")
                lines.append(f"            xmax = {self._fmt(offset)}")
                lines.append(f'            text = "{label}"')
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
