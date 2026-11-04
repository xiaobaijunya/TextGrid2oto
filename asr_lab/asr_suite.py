# coding=utf-8
"""Qwen3-ASR + Qwen3-ForcedAligner 批量推理（官方 transformers 实现）。

推理部分**完全使用官方 ``qwen_asr`` 包**（与 ``F:/coding/Qwen3-ASR`` 中同一份代码，
已内置在本项目根目录的 ``qwen_asr/`` 下）：

* ``Qwen3ASRModel``：ASR 识别，内部按官方上限自动分段；
* ``Qwen3ForcedAligner``：逐字时间戳（「只对齐」模式单独加载它）。

设计目标（与官方 ``Qwen3-ASR/app.py`` 对齐）：

* **不做用户级切片**：整条音频送进去，由模型按官方的 180 秒上限内部分段；
* **不做歌词匹配**：ASR 输出什么就用什么；
* **输出**：``{stem}.txt``、``{stem}.lab``、``{stem}_ASR.json``（含逐字时间戳）。

提示词由两部分合并而成::

    final_prompt = system_prompt + "\\n\\n" + 歌词提示块(lyrics)

``system_prompt`` 是页面上的「系统提示词」，``lyrics`` 是「歌词 / 补充提示」。
"""

from __future__ import annotations

import json
import os
import pathlib
import time
import unicodedata
import csv
import io
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional, Sequence

# ── 常量 ────────────────────────────────────────────────────────
AUDIO_SUFFIXES = (".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac", ".opus", ".aiff", ".aif")
SAMPLE_RATE = 16000
# 与官方 MAX_FORCE_ALIGN_INPUT_SECONDS 一致：只对齐模式超过这个长度要分段
OFFICIAL_ALIGN_MAX_SEC = 180.0

# 自定义分段长度（秒）：显存不足时调小（6GB 卡建议 30~60）；
# 0 表示完全交给官方分段（带时间戳 180s / 不带 1200s）。
DEFAULT_CHUNK_SEC = 60.0
MIN_CHUNK_SEC = 0.0
MAX_CHUNK_SEC = 600.0

# ── 静音跳过（避免模型在空白处凭空生成文本） ─────────────────
# 帧 RMS 低于「绝对下限」与「峰值 - rel_db」中较大者时判定为静音；
# 绝对下限防止很轻的录音被误杀，相对值保证大声录音的底噪也能被识破。
SILENCE_DBFS = -55.0
SPEECH_REL_DB = -35.0
SPEECH_MIN_SEC = 0.20        # 短于这个长度的“人声”当成噪声丢掉
SPEECH_MERGE_GAP_SEC = 5.0   # 相邻人声间隔小于此值时合并（5 秒内的停顿当正常换气）
SPEECH_PAD_SEC = 0.15        # 每段前后留一点余量，避免切掉辅音/呼吸

# 显存吃紧时（如 6GB 卡）更容易因显存碎片 OOM，开启可扩展显存段能明显缓解。
# 必须在 torch 初始化 CUDA 之前设置；setdefault 不会覆盖用户自己的配置。
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

BACKEND_TRANSFORMERS = "transformers"
BACKEND_LABELS = {BACKEND_TRANSFORMERS: "官方 Transformers（torch）"}

# ── 推理设备 / 精度（页面下拉框的取值） ────────────────────────
DEVICE_AUTO = "auto"
DEVICE_CUDA = "cuda"
DEVICE_CPU = "cpu"
DEVICES = (DEVICE_AUTO, DEVICE_CUDA, DEVICE_CPU)
DEVICE_LABELS = {DEVICE_AUTO: "自动", DEVICE_CUDA: "CUDA (GPU)", DEVICE_CPU: "CPU"}

DTYPE_AUTO = "auto"
DTYPE_BF16 = "bf16"
DTYPE_FP16 = "fp16"
DTYPE_FP32 = "fp32"
DTYPES = (DTYPE_AUTO, DTYPE_BF16, DTYPE_FP16, DTYPE_FP32)
DTYPE_LABELS = {
    DTYPE_AUTO: "自动",
    DTYPE_BF16: "bfloat16",
    DTYPE_FP16: "float16",
    DTYPE_FP32: "float32",
}

# 运行模式：
#   full  = 识别 + 对齐（两个模型同时加载）
#   asr   = 只识别（不加载 Aligner，最省显存）
#   align = 只对齐（读已有 .txt，只加载 Aligner）
RUN_MODE_FULL = "full"
RUN_MODE_ASR_ONLY = "asr"
RUN_MODE_ALIGN_ONLY = "align"
RUN_MODES = (RUN_MODE_FULL, RUN_MODE_ASR_ONLY, RUN_MODE_ALIGN_ONLY)
RUN_MODE_LABELS = {
    RUN_MODE_FULL: "识别 + 对齐",
    RUN_MODE_ASR_ONLY: "只识别（不出时间戳）",
    RUN_MODE_ALIGN_ONLY: "只对齐（用已有 txt）",
}

DEFAULT_SYSTEM_PROMPT = (
    "你是一位专业的歌词转录助手，专注于从音频中准确提取歌词文本。"
    "请专注于识别歌曲中的歌词内容，保持原有语序，不要翻译，不要添加解释。"
    "只输出音频里真正出现的内容，避免乱猜，不要补全没有唱出的句子。"
)

LYRICS_BLOCK_TEMPLATE = (
    "下面给出这首歌的参考歌词。这段音频可能只对应歌词中的一两句，"
    "也可能存在跳唱、重复、漏唱、顺序不同或与歌词不完全一致的情况。\n"
    "请结合参考歌词理解音频中实际唱出的内容，只转录音频里真正出现的人声歌词，"
    "不要补全没有唱出的部分，不要臆造。\n"
    "参考歌词：\n{lyrics}"
)

# 语言 / 输出模式
LANGUAGE_LABELS = {"zh": "中文", "ja": "日本語"}
LANGUAGE_CANONICAL = {"zh": "Chinese", "ja": "Japanese"}
OUTPUT_MODES = {
    "zh": [("pinyin", "拼音"), ("hanzi", "汉字")],
    "ja": [("romaji", "罗马音"), ("kana", "假名")],
}
OUTPUT_MODE_DEFAULT = {"zh": "pinyin", "ja": "romaji"}


# ══════════════════════════════════════════════════════════════
# 提示词
# ══════════════════════════════════════════════════════════════
def build_final_prompt(system_prompt: str, lyrics: str) -> str:
    """把「系统提示词」与「参考歌词」合并成最终传给 ASR 的 context。"""
    system_prompt = str(system_prompt or "").strip()
    lyrics = str(lyrics or "").strip()
    if lyrics:
        block = LYRICS_BLOCK_TEMPLATE.format(lyrics=lyrics)
        return f"{system_prompt}\n\n{block}".strip() if system_prompt else block
    return system_prompt


def normalize_whitespace(text: str) -> str:
    return " ".join(str(text or "").split())


# ══════════════════════════════════════════════════════════════
# 文件收集 / 文本读取
# ══════════════════════════════════════════════════════════════
def _find_audio_files(folder: pathlib.Path, recursive: bool) -> list[pathlib.Path]:
    if not folder.is_dir():
        return []
    walker: Iterable[pathlib.Path] = folder.rglob("*") if recursive else folder.iterdir()
    return sorted(
        path for path in walker
        if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES
    )


def collect_audio_paths(
    folder: str = "",
    audio_files: Optional[Sequence[str]] = None,
    recursive: bool = True,
) -> list[pathlib.Path]:
    """手动选中的音频优先，其次才是文件夹。"""
    selected = [pathlib.Path(item) for item in (audio_files or []) if item]
    selected = [path for path in selected if path.is_file()]
    if selected:
        return selected
    if folder:
        return _find_audio_files(pathlib.Path(folder), recursive)
    return []


# ══════════════════════════════════════════════════════════════
# 批量 CSV（英文表头：path,title,lyrics）
# ══════════════════════════════════════════════════════════════
CSV_COLUMNS = ("path", "title", "lyrics")
# 模板示例：第二列是标题（可为空），第三列才是歌词
CSV_TEMPLATE_ROWS = [
    ["10", "", "如果那两个字没有颤抖\n我不会发现 我难受"],
    ["segments/11.wav", "Song B", "第一句歌词\n第二句歌词"],
]
# 老模板把歌词写在第 2 列时，允许回退（仅限多行长文本，避免把标题当歌词）
_TITLE_FALLBACK_MIN_LEN = 20


def save_csv_template(csv_path: str | os.PathLike) -> pathlib.Path:
    """生成 CSV 模板（UTF-8-BOM，表头固定为 path,title,lyrics）。"""
    path = pathlib.Path(str(csv_path)).expanduser()
    if path.suffix.lower() != ".csv":
        path = path.with_suffix(".csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(CSV_COLUMNS)
        writer.writerows(CSV_TEMPLATE_ROWS)
    return path


def _detect_delimiter(sample: str) -> str:
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t").delimiter
    except csv.Error:
        return ","


def _cell(cells: Sequence[str], index: int) -> str:
    return str(cells[index]).strip() if 0 <= index < len(cells) else ""


def read_csv_manifest(csv_path: str | os.PathLike) -> tuple[list[dict], list[str]]:
    """读取批量 CSV。

    列定义（英文表头，大小写不敏感）：

    ========  =================================================
    path      音频文件或文件夹路径；相对路径按 CSV 所在目录解析
    title     标题，可空（仅用于日志）
    lyrics    歌词；该行指向文件夹时，文件夹内所有音频都用这段歌词
    ========  =================================================

    没有表头时按列顺序解析（第 1 列 path / 第 2 列 title / 第 3 列 lyrics）。

    Returns:
        ``(entries, notes)``；每个 entry 为
        ``{row, raw_path, path, title, lyrics, title_fallback}``。
    """
    path = pathlib.Path(str(csv_path)).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"CSV 文件不存在：{path}")

    text = read_text_auto(path).lstrip("\ufeff")
    delimiter = _detect_delimiter(text[:4096])
    base_dir = path.parent
    rows = list(csv.reader(io.StringIO(text), delimiter=delimiter))
    if not rows:
        return [], ["CSV 是空文件"]

    header = [str(cell).strip().lower() for cell in rows[0]]
    header_detected = "path" in header
    legacy_header = bool(
        set(header) & {"路径", "歌名", "歌词", "audio", "file", "text"}
    )
    if header_detected:
        index = {name: header.index(name) for name in CSV_COLUMNS if name in header}
        start_index = 1
    else:
        index = {"path": 0, "title": 1, "lyrics": 2}
        start_index = 1 if legacy_header else 0

    notes: list[str] = []
    if not header_detected:
        if legacy_header:
            notes.append("检测到非英文表头，已按列顺序解析（建议使用 path,title,lyrics）")
        else:
            notes.append("未检测到英文表头 path,title,lyrics，已按列顺序解析")

    entries: list[dict] = []
    for offset, cells in enumerate(rows[start_index:], start=start_index + 1):
        if not any(str(cell).strip() for cell in cells):
            continue

        raw_path = _cell(cells, index.get("path", 0))
        title = _cell(cells, index.get("title", 1))
        lyrics = _cell(cells, index.get("lyrics", 2))

        title_fallback = False
        if not lyrics and title and "\n" in title and len(title) >= _TITLE_FALLBACK_MIN_LEN:
            # 兼容旧模板：歌词被写进了第 2 列
            lyrics = title
            title_fallback = True
            notes.append(f"第 {offset} 行 lyrics 为空，已回退使用第 2 列内容")

        resolved: pathlib.Path | None = None
        if raw_path:
            expanded = os.path.expandvars(os.path.expanduser(raw_path))
            resolved = pathlib.Path(expanded)
            if not resolved.is_absolute():
                resolved = base_dir / resolved

        entries.append(
            {
                "row": offset,
                "raw_path": raw_path,
                "path": resolved,
                "title": title,
                "lyrics": lyrics,
                "title_fallback": title_fallback,
            }
        )
    return entries, notes


def collect_csv_tasks(
    csv_path: str | os.PathLike,
    recursive: bool = True,
) -> tuple[list[pathlib.Path], dict[str, str], list[str]]:
    """把 CSV 展开成 ``(音频列表, 歌词映射, 提示信息)``。

    - 一行指向文件夹时，文件夹内（可递归）所有音频都使用该行歌词；
    - 重复的音频只处理一次。
    """
    entries, notes = read_csv_manifest(csv_path)
    paths: list[pathlib.Path] = []
    lyrics_map: dict[str, str] = {}
    seen: set[str] = set()

    for entry in entries:
        row = entry["row"]
        candidate = entry.get("path")
        if candidate is None:
            notes.append(f"第 {row} 行路径为空，已跳过")
            continue
        if not candidate.exists():
            notes.append(f"第 {row} 行路径不存在：{candidate}")
            continue

        if candidate.is_dir():
            audios = _find_audio_files(candidate, recursive)
        elif candidate.suffix.lower() in AUDIO_SUFFIXES:
            audios = [candidate]
        else:
            notes.append(f"第 {row} 行不是文件夹或受支持的音频：{candidate}")
            continue

        if not audios:
            notes.append(f"第 {row} 行没有可处理的音频：{candidate}")
            continue

        for audio in audios:
            key = str(audio)
            if key in seen:
                continue
            seen.add(key)
            paths.append(audio)
            lyrics_map[key] = entry["lyrics"]

    return paths, lyrics_map, notes


def read_text_auto(path: pathlib.Path) -> str:
    """尽量按常见编码读取文本（Excel / 记事本导出的 txt 经常是 GBK）。"""
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace")


def _is_previous_output(audio_path: pathlib.Path, text: str) -> bool:
    """判断同名 .txt 是不是本程序上次生成的识别结果（避免被当成歌词提示）。"""
    json_path = audio_path.with_name(f"{audio_path.stem}_ASR.json")
    if not json_path.is_file():
        return False
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    previous = normalize_whitespace(payload.get("text", "") or "")
    return bool(previous) and previous == normalize_whitespace(text)


def _lyrics_sidecar_path(audio_path: pathlib.Path) -> pathlib.Path:
    """歌词草稿文件：{音频名}_lyrics.txt（与 ASR 输出的 {音频名}.txt 区分）。"""
    return audio_path.with_name(f"{audio_path.stem}_lyrics.txt")


def save_lyrics_sidecar(audio_path: pathlib.Path, text: str) -> pathlib.Path | None:
    """把页面里编辑的歌词写入 ``{音频名}_lyrics.txt``。

    - 空文本会删除草稿文件（这样后续仍可回退读取同名 ``.txt`` / ``.lrc``）；
    - 返回写入的路径；删除或失败时返回 ``None``。
    """
    path = _lyrics_sidecar_path(pathlib.Path(audio_path))
    content = str(text or "")
    if not content.strip():
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
        return None
    try:
        path.write_text(content, encoding="utf-8")
        return path
    except OSError:
        return None


def load_edited_lyrics(audio_path: pathlib.Path) -> str:
    """只读取页面编辑后保存的 ``{音频名}_lyrics.txt``。"""
    path = _lyrics_sidecar_path(pathlib.Path(audio_path))
    if not path.is_file():
        return ""
    try:
        return read_text_auto(path).strip()
    except OSError:
        return ""


def load_sidecar_lyrics(audio_path: pathlib.Path) -> str:
    """读取同目录下的歌词提示。

    优先级：
    1. ``{音频名}_lyrics.txt``（页面里手动编辑后自动保存的文件）
    2. ``{音频名}.txt``（旧歌词文件；如果其实是上次的识别结果则跳过）
    3. ``{音频名}.lrc``
    """
    dedicated = _lyrics_sidecar_path(audio_path)
    if dedicated.is_file():
        try:
            return read_text_auto(dedicated).strip()
        except OSError:
            pass

    for suffix in (".txt", ".lrc"):
        candidate = audio_path.with_suffix(suffix)
        if not candidate.is_file():
            continue
        try:
            text = read_text_auto(candidate).strip()
        except OSError:
            continue
        if not text:
            continue
        if suffix == ".txt" and _is_previous_output(audio_path, text):
            continue
        return text
    return ""


# ══════════════════════════════════════════════════════════════
# 逐字对齐结果规整
# ══════════════════════════════════════════════════════════════
def is_alignment_char(ch: str) -> bool:
    """标点 / 空格 / 符号不计时，只保留字母、汉字、假名、数字。"""
    if ch == "'":
        return True
    category = unicodedata.category(ch)
    return category.startswith("L") or category.startswith("N")


@dataclass(frozen=True)
class AlignItem:
    """统一后的逐字对齐结果（秒）。"""

    text: str
    start_time: float
    end_time: float


def _item_fields(item: Any) -> tuple[str, float, float]:
    if isinstance(item, dict):
        text = item.get("text", "")
        start = item.get("start_time", item.get("start", 0.0))
        end = item.get("end_time", item.get("end", 0.0))
    else:
        text = getattr(item, "text", "")
        start = getattr(item, "start_time", 0.0)
        end = getattr(item, "end_time", 0.0)
    return str(text or ""), float(start or 0.0), float(end or 0.0)


def split_items_per_char(items: Sequence[Any]) -> list[AlignItem]:
    """把对齐器的分词粒度（中文按字、日文按词、英文按词）统一压成逐字。

    多字符 item 按字符数均分时间；纯标点 item 丢弃。
    """
    result: list[AlignItem] = []
    for item in items or []:
        text, start, end = _item_fields(item)
        chars = [ch for ch in text if is_alignment_char(ch)]
        if not chars:
            continue
        if end < start:
            start, end = end, start
        step = (end - start) / len(chars)
        for index, ch in enumerate(chars):
            result.append(
                AlignItem(
                    text=ch,
                    start_time=start + step * index,
                    end_time=start + step * (index + 1),
                )
            )
    return result


# ══════════════════════════════════════════════════════════════
# G2P（复用 LyricFA 的词典）
# ══════════════════════════════════════════════════════════════
_ZH_G2P = None
_JA_G2P = None


def get_zh_g2p():
    global _ZH_G2P
    if _ZH_G2P is None:
        from asr_lab.LyricFA.tools.ZhG2p import ZhG2p

        _ZH_G2P = ZhG2p("mandarin")
    return _ZH_G2P


def get_ja_g2p():
    global _JA_G2P
    if _JA_G2P is None:
        from asr_lab.LyricFA.tools.JaG2p import JaG2p

        _JA_G2P = JaG2p()
    return _JA_G2P


def normalize_output_mode(language: str, mode: str | None) -> str:
    language = "ja" if str(language or "").lower().startswith("ja") else "zh"
    mode = str(mode or "").lower()
    aliases = {"拼音": "pinyin", "汉字": "hanzi", "罗马音": "romaji", "假名": "kana"}
    mode = aliases.get(str(mode or ""), aliases.get(str(mode or "").strip(), mode))
    valid = {value for value, _label in OUTPUT_MODES[language]}
    return mode if mode in valid else OUTPUT_MODE_DEFAULT[language]


def is_phonetic_mode(language: str, mode: str) -> bool:
    language = "ja" if str(language or "").lower().startswith("ja") else "zh"
    return (language == "zh" and mode == "pinyin") or (language == "ja" and mode == "romaji")


def display_tokens(text: str, language: str, mode: str) -> list[str]:
    """把整句文本转成 .lab 的显示 token（拼音 / 汉字 / 罗马音 / 假名）。"""
    text = str(text or "")
    if not text.strip():
        return []
    language = "ja" if str(language or "").lower().startswith("ja") else "zh"
    mode = normalize_output_mode(language, mode)

    if language == "ja":
        g2p = get_ja_g2p()
        if mode == "kana":
            return list(g2p.split_kana_no_regex(text))
        return str(g2p.convert(text, include_tone=False, convert_number=True)).split()

    g2p = get_zh_g2p()
    if mode == "pinyin":
        return str(g2p.convert(text, include_tone=False, convert_number=True)).split()
    return list(g2p.split_string_no_regex(text))


def _per_char_tokens(chars: Sequence[str], language: str, mode: str) -> list[str]:
    """逐字兜底 G2P（.lab token 数和对齐项数对不上时使用）。"""
    language = "ja" if str(language or "").lower().startswith("ja") else "zh"
    mode = normalize_output_mode(language, mode)
    if not is_phonetic_mode(language, mode):
        return [str(ch) for ch in chars]
    try:
        g2p = get_ja_g2p() if language == "ja" else get_zh_g2p()
    except Exception:
        return ["" for _ in chars]
    result: list[str] = []
    for ch in chars:
        try:
            result.append(str(g2p.convert(ch, include_tone=False, convert_number=True)).strip())
        except Exception:
            result.append("")
    return result


def join_tokens(language: str, mode: str, tokens: Sequence[str]) -> str:
    if is_phonetic_mode(language, normalize_output_mode(language, mode)):
        return " ".join(tokens)
    return "".join(tokens)


def join_transcript_chunks(texts: Sequence[str]) -> str:
    """把分段识别结果拼成整句。

    中日文直接相连；英文/数字之间补一个空格，避免拼出 ``helloworld``。
    """
    out = ""
    for raw in texts or []:
        piece = str(raw or "").strip()
        if not piece:
            continue
        if out:
            last, first = out[-1], piece[0]
            if last.isascii() and last.isalnum() and first.isascii() and first.isalnum():
                out += " "
        out += piece
    return out


def detect_speech_spans(
    wav: Any,
    sr: int = SAMPLE_RATE,
    *,
    dbfs_floor: float = SILENCE_DBFS,
    rel_db: float = SPEECH_REL_DB,
    min_speech_sec: float = SPEECH_MIN_SEC,
    merge_gap_sec: float = SPEECH_MERGE_GAP_SEC,
    pad_sec: float = SPEECH_PAD_SEC,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
) -> list[tuple[float, float]]:
    """按能量找出「有声区间」，用于跳过静音、避免模型在空白处凭空生成文本。

    阈值取「绝对下限」与「相对峰值」两者中较大者：很轻的录音不会被误杀，
    而真正的空白（含数字静音、底噪）会被跳过。

    Returns:
        ``[(start_sec, end_sec), ...]``，按时间排序，已加首尾余量。
    """
    import numpy as np

    x = np.asarray(wav, dtype=np.float32)
    if x.ndim > 1:
        x = x.mean(axis=-1)
    if x.size == 0:
        return []
    total_sec = len(x) / float(sr)

    frame = max(1, int(sr * frame_ms / 1000.0))
    hop = max(1, int(sr * hop_ms / 1000.0))
    if len(x) < frame:
        x = np.pad(x, (0, frame - len(x)))
    windows = np.lib.stride_tricks.sliding_window_view(x, frame)[::hop]
    rms = np.sqrt(np.mean(windows * windows, axis=1) + 1e-12)

    peak = float(rms.max()) if rms.size else 0.0
    if peak <= 0.0:
        return []
    threshold = max(10 ** (dbfs_floor / 20.0), peak * (10 ** (rel_db / 20.0)))

    active = np.flatnonzero(rms >= threshold)
    if active.size == 0:
        return []

    gap_frames = max(1, int(merge_gap_sec * sr / hop))
    raw_spans: list[tuple[int, int]] = []
    start_i = prev_i = int(active[0])
    for i in active[1:]:
        i = int(i)
        if i - prev_i > gap_frames:
            raw_spans.append((start_i, prev_i))
            start_i = i
        prev_i = i
    raw_spans.append((start_i, prev_i))

    min_frames = max(1, int(min_speech_sec * sr / hop))
    spans: list[tuple[float, float]] = []
    for a, b in raw_spans:
        if (b - a + 1) < min_frames:
            continue
        start_sec = max(0.0, a * hop / float(sr) - pad_sec)
        end_sec = min(total_sec, (b * hop + frame) / float(sr) + pad_sec)
        if end_sec <= start_sec:
            continue
        # pad 之后相邻段可能相连，直接合并
        if spans and start_sec <= spans[-1][1]:
            spans[-1] = (spans[-1][0], max(spans[-1][1], end_sec))
        else:
            spans.append((start_sec, end_sec))
    return spans


def offset_align_items(items: Sequence[Any], offset_sec: float) -> list[Any]:
    """把一段内的时间戳整体后移 ``offset_sec`` 秒（重建 frozen dataclass）。"""
    shifted: list[Any] = []
    for item in items or []:
        text, start, end = _item_fields(item)
        start = round(start + offset_sec, 3)
        end = round(end + offset_sec, 3)
        if isinstance(item, dict):
            shifted.append({"text": text, "start_time": start, "end_time": end})
        else:
            shifted.append(type(item)(text=text, start_time=start, end_time=end))  # type: ignore[arg-type]
    return shifted


# ══════════════════════════════════════════════════════════════
# 设置
# ══════════════════════════════════════════════════════════════
@dataclass
class AsrSettings:
    asr_model_dir: str = ""
    aligner_model_dir: str = ""
    # 推理设备：auto（有 CUDA 就用 GPU，否则 CPU）/ cuda / cpu
    device: str = DEVICE_AUTO
    # 推理精度：auto（GPU 优先 bf16）/ bf16 / fp16 / fp32
    dtype: str = DTYPE_AUTO
    language: str = "zh"
    output_mode: str = "pinyin"
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    out_dir: str = ""
    recursive: bool = True
    overwrite: bool = True
    # 官方模型的 max_inference_batch_size（对齐与识别的批大小）
    batch_size: int = 1
    save_txt: bool = True
    save_lab: bool = True
    save_json: bool = True
    # 运行模式：full（识别+对齐）/ asr（只识别）/ align（只对齐，读已有 txt）
    run_mode: str = RUN_MODE_FULL
    # 分段执行（降低峰值显存）：每段时长（秒），0 = 交给官方分段
    chunk_size_sec: float = DEFAULT_CHUNK_SEC
    # full 模式下让 Aligner 跑在 CPU 上，避免与 ASR 争显存
    aligner_on_cpu: bool = False
    # 把 audio encoder（约 0.3B）卸载到内存：显存省 ~0.6GB，速度几乎不变
    encoder_on_cpu: bool = False
    # 每段推理结束后释放 CUDA 缓存（推荐开启，代价很小）
    empty_cache_per_chunk: bool = True
    # 跳过静音段：只把有声部分送进模型，避免空白处被“脑补”出文本
    skip_silence: bool = True
    # 静音判定下限（帧 RMS 的 dBFS）
    silence_dbfs: float = SILENCE_DBFS


@dataclass
class AsrFileResult:
    audio_path: pathlib.Path
    txt_path: pathlib.Path | None = None
    lab_path: pathlib.Path | None = None
    json_path: pathlib.Path | None = None
    text: str = ""
    token_count: int = 0
    item_count: int = 0
    elapsed_sec: float = 0.0
    error: str | None = None
    # 复读截断 / 对齐失败等提示（仅记录）
    note: str = ""
    # 引擎性能统计（官方 transformers 实现目前不返回分项耗时，一般为空）
    perf: dict = field(default_factory=dict)


# ══════════════════════════════════════════════════════════════
# 后端实现
# ══════════════════════════════════════════════════════════════
def _looks_like_hf_dir(path: pathlib.Path) -> bool:
    """HF 权重目录：至少要有 config.json + 权重文件。"""
    if not path.is_dir() or not (path / "config.json").is_file():
        return False
    return any(path.glob("*.safetensors")) or any(path.glob("pytorch_model*.bin"))


def resolve_model_dir(path: str) -> pathlib.Path:
    """把「大概是模型的目录」解析成能嗂给 ``from_pretrained`` 的权重目录。

    兼容三种常见形态：

    * 直接就是权重目录（含 ``config.json`` + 权重）；
    * ModelScope 缓存布局 ``<模型>/snapshots/<revision>``（上一级也能传）；
    * 目录里恰好有子目录符合上述条件时自动下钻一层。
    """
    raw = str(path or "").strip()
    if not raw:
        raise FileNotFoundError("未指定模型目录")
    candidate = pathlib.Path(os.path.expandvars(os.path.expanduser(raw)))
    if candidate.is_file():
        candidate = candidate.parent
    if not candidate.is_dir():
        raise FileNotFoundError(f"模型目录不存在：{candidate}")
    if _looks_like_hf_dir(candidate):
        return candidate

    snapshots = candidate if candidate.name == "snapshots" else candidate / "snapshots"
    if snapshots.is_dir():
        revisions = sorted(
            (item for item in snapshots.iterdir() if item.is_dir()),
            key=lambda item: item.name,
            reverse=True,
        )
        for revision in revisions:
            if _looks_like_hf_dir(revision):
                return revision

    for child in sorted(item for item in candidate.iterdir() if item.is_dir()):
        if _looks_like_hf_dir(child):
            return child

    raise FileNotFoundError(
        "在下面的目录里找不到 HuggingFace 权重（需要 config.json + *.safetensors）：\n"
        f"{candidate}\n\n"
        "可下载：Qwen/Qwen3-ASR-1.7B 与 Qwen/Qwen3-ForcedAligner-0.6B"
    )


def is_transformers_model_dir(path: str) -> bool:
    """能否解析出可用的 transformers 权重目录。"""
    try:
        resolve_model_dir(path)
    except Exception:
        return False
    return True


def resolve_device(name: str) -> str:
    """把页面上的设备选项解析成 ``device_map`` 能用的值。"""
    choice = str(name or DEVICE_AUTO).strip().lower()
    if choice == DEVICE_CPU:
        return DEVICE_CPU
    try:
        import torch
    except ImportError as error:  # pragma: no cover - 取决于运行环境
        raise RuntimeError(
            "未安装 torch，无法使用 GPU 推理。请先安装依赖：\n"
            "    pip install -r requirements-asr.txt\n"
            f"\n原始错误：{error}"
        ) from error
    if choice == DEVICE_CUDA and not torch.cuda.is_available():
        raise RuntimeError(
            "选择了 CUDA，但当前 Python 环境里 torch.cuda.is_available() 为 False。\n"
            "请确认装的是 GPU 版 PyTorch，例如：\n"
            "    pip install torch --index-url https://download.pytorch.org/whl/cu128"
        )
    return DEVICE_CUDA if torch.cuda.is_available() and choice != DEVICE_CPU else DEVICE_CPU


def resolve_dtype(name: str, device: str):
    """把页面上的精度选项解析成 ``torch.dtype``。"""
    import torch

    choice = str(name or DTYPE_AUTO).strip().lower()
    if choice == DTYPE_BF16:
        return torch.bfloat16
    if choice == DTYPE_FP16:
        return torch.float16
    if choice == DTYPE_FP32:
        return torch.float32
    if device == DEVICE_CUDA:
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    # CPU 上用 bf16/fp16 会很慢甚至不支持算子
    return torch.float32


class _CancelStoppingCriteria:
    """每生成一个 token 检查一次用户是否点了「停止」，做到立刻中断。"""

    def __init__(self, cancel_checker: Callable[[], bool]):
        self.cancel_checker = cancel_checker
        self.cancelled = False

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        if self.cancel_checker is not None and self.cancel_checker():
            self.cancelled = True
            return True
        return False


class _TransformersBackend:
    """官方 transformers 实现：``Qwen3ASRModel``（识别）＋可选 ``Qwen3ForcedAligner``。

    显存不足时按 ``chunk_size_sec`` 把音频切成小段逐段推理（段内时间戳会自动偏移
    回整曲时间轴），每段结束可选释放 CUDA 缓存，能显著降低峰值显存。
    """

    model: Any
    progress_callback: Optional[Callable[[str], None]] = None
    stream_callback: Optional[Callable[[str], None]] = None
    cancel_checker: Optional[Callable[[], bool]] = None

    def __init__(self, settings: AsrSettings):
        try:
            import torch  # noqa: F401 - 只要装了 torch 才能用官方包
            from qwen_asr import Qwen3ASRModel
        except ImportError as error:  # pragma: no cover - 取决于运行环境
            raise RuntimeError(
                "无法导入 torch / qwen_asr，请先安装依赖：\n"
                "    pip install -r requirements-asr.txt\n"
                f"\n原始错误：{error}"
            ) from error

        self.name = BACKEND_TRANSFORMERS
        self.last_perf: dict = {}
        self.last_note: str = ""
        self.device = resolve_device(settings.device)
        self.dtype = resolve_dtype(settings.dtype, self.device)
        self.max_inference_batch_size = max(1, int(settings.batch_size or 1))
        self.chunk_size_sec = float(settings.chunk_size_sec or 0.0)
        self.empty_cache_per_chunk = bool(settings.empty_cache_per_chunk)
        self.skip_silence = bool(settings.skip_silence)
        self.silence_dbfs = float(settings.silence_dbfs or SILENCE_DBFS)
        self._stream_pending = ""
        self._stream_started = False
        self._stream_ts = 0.0
        # audio encoder（~0.3B）可以单独塞到内存里，给 6GB 卡腾出 ~0.6GB 显存
        self.encoder_on_cpu = bool(settings.encoder_on_cpu) and \
            resolve_device(settings.device) != DEVICE_CPU

        self.asr_dir = resolve_model_dir(settings.asr_model_dir)
        self.aligner_dir = None
        self.aligner_device = None
        kwargs: dict[str, Any] = {}
        # 「只识别」模式不加载 Aligner，显存只需 ASR 一份
        if settings.run_mode == RUN_MODE_FULL:
            self.aligner_dir = resolve_model_dir(settings.aligner_model_dir)
            # Aligner 可以单独放到 CPU：6GB 卡同时放两个模型容易 OOM
            self.aligner_device = DEVICE_CPU if settings.aligner_on_cpu else self.device
            aligner_dtype = (
                self.dtype if self.aligner_device == self.device
                else resolve_dtype(DTYPE_AUTO, self.aligner_device)
            )
            kwargs["forced_aligner"] = str(self.aligner_dir)
            kwargs["forced_aligner_kwargs"] = dict(
                dtype=aligner_dtype, device_map=self.aligner_device)

        self.model = Qwen3ASRModel.from_pretrained(
            str(self.asr_dir),
            dtype=self.dtype,
            device_map=self.device,
            max_inference_batch_size=self.max_inference_batch_size,
            **kwargs,
        )
        self.has_aligner = self.aligner_dir is not None
        self.aligner = getattr(self.model, "forced_aligner", None)
        if self.encoder_on_cpu:
            # 权重先正常加载到显存，再把 encoder 搬到内存（fp32）
            encoder = self.model.model.thinker.audio_tower
            encoder.to(DEVICE_CPU)
            encoder.float()
            self._release_cache()

    # ── 输入构造 / 解码 / 单次推理 ─────────────────────────
    def _build_inputs(self, wav: Any, prompt: str, lang: str) -> dict:
        processor = self.model.processor
        text_prompt = self.model._build_text_prompt(
            context=prompt or "", force_language=lang)
        inputs = processor(text=[text_prompt], audio=[wav],
                           return_tensors="pt", padding=True)

        feature_dtype = next(self.model.model.thinker.audio_tower.parameters()).dtype
        mixed: dict[str, Any] = {}
        for key, value in inputs.items():
            if key == "input_features":
                # encoder 在 CPU 时 mel 特征也留在内存；否则跟模型同设备、同精度
                mixed[key] = (value.to(feature_dtype) if self.encoder_on_cpu
                              else value.to(self.device, feature_dtype))
            elif key in ("feature_attention_mask", "audio_feature_lengths"):
                mixed[key] = value if self.encoder_on_cpu else value.to(self.device)
            else:
                mixed[key] = value.to(self.device)
        return mixed

    def _decode(self, wav: Any, prompt: str, lang: str) -> str:
        """跑 LLM 解码；设了 stream_callback 时边生成边把文本回传。"""
        import threading

        import torch

        from qwen_asr.inference.utils import parse_asr_output

        inputs = self._build_inputs(wav, prompt, lang)
        # 生成中随时可停：StoppingCriteria 在每生成一个 token 时被调用
        criteria = None
        if self.cancel_checker is not None:
            criteria = _CancelStoppingCriteria(self.cancel_checker)
            inputs = dict(inputs)
            inputs["stopping_criteria"] = [criteria]
        streamer = None
        if self.stream_callback is not None:
            from transformers import TextIteratorStreamer

            streamer = TextIteratorStreamer(
                self.model.processor.tokenizer,
                skip_prompt=True, skip_special_tokens=True)

        if streamer is None:
            with torch.inference_mode():
                generated = self.model.model.generate(**inputs, max_new_tokens=8192)
            prompt_len = int(inputs["input_ids"].shape[1])
            raw = self.model.processor.batch_decode(
                generated.sequences[:, prompt_len:],
                skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        else:
            errors: list[Exception] = []

            def _generate() -> None:
                try:
                    with torch.inference_mode():
                        self.model.model.generate(
                            **inputs, max_new_tokens=8192, streamer=streamer)
                except Exception as exc:  # noqa: BLE001 - 交给主线程抛出
                    errors.append(exc)
                    try:
                        streamer.end()   # 否则主线程会卡在迭代上
                    except Exception:
                        pass

            thread = threading.Thread(target=_generate, daemon=True)
            thread.start()
            pieces: list[str] = []
            for piece in streamer:
                if not piece:
                    continue
                pieces.append(piece)
                self._emit_stream(piece)
            thread.join()
            self._flush_stream()
            if errors:
                raise errors[0]
            raw = "".join(pieces)

        if criteria is not None and criteria.cancelled:
            raise InterruptedError("用户停止")

        _, text = parse_asr_output(raw, user_language=lang)
        return str(text or "")

    # ── 流式文本节流回传（避免把 GUI 事件队列灌满） ────────
    def _emit_stream(self, piece: str) -> None:
        if self.stream_callback is None:
            return
        self._stream_pending += piece
        now = time.time()
        if not self._stream_started:
            self.stream_callback("      > " + self._stream_pending)
            self._stream_pending = ""
            self._stream_started = True
            self._stream_ts = now
        elif now - self._stream_ts >= 0.1:
            self.stream_callback(self._stream_pending)
            self._stream_pending = ""
            self._stream_ts = now

    def _flush_stream(self) -> None:
        if self.stream_callback is not None and self._stream_started:
            if self._stream_pending:
                self.stream_callback(self._stream_pending)
            self.stream_callback("\n")
        self._stream_pending = ""
        self._stream_started = False

    def _run(self, audio: Any, prompt: str, lang: str,
             offset_sec: Optional[float] = None) -> tuple[str, list[Any]]:
        from qwen_asr.inference.utils import normalize_audio_input

        self._raise_if_cancelled()

        if isinstance(audio, tuple) and len(audio) == 2:
            wav = audio[0]
        else:
            wav = normalize_audio_input(str(audio))

        text = self._decode(wav, prompt, lang)

        items: list[Any] = []
        if self.has_aligner and self.aligner is not None and text.strip():
            if self.progress_callback:
                self.progress_callback("      正在对齐时间戳…")
            aligned = self.aligner.align(
                audio=(wav, SAMPLE_RATE), text=text, language=lang)
            if aligned:
                items = list(aligned[0].items)
                if offset_sec:
                    items = offset_align_items(items, offset_sec)
        return text, items

    def _raise_if_cancelled(self) -> None:
        """用户点了停止就立即抛 InterruptedError（run_batch 会标记为已停止）。"""
        if self.cancel_checker is not None and self.cancel_checker():
            raise InterruptedError("用户停止")

    def _release_cache(self) -> None:
        if not self.empty_cache_per_chunk:
            return
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def transcribe(self, audio_path: pathlib.Path, prompt: str, language: str) -> tuple[str, list[Any]]:
        self._raise_if_cancelled()
        lang = LANGUAGE_CANONICAL.get(language, "Chinese")
        chunk_sec = float(self.chunk_size_sec or 0.0)

        # 不跳静音 + 不分段：保持原样，整条交给模型
        if not self.skip_silence and chunk_sec <= 0:
            return self._run(str(audio_path), prompt, lang)

        from qwen_asr.inference.utils import normalize_audio_input, split_audio_into_chunks

        wav = normalize_audio_input(str(audio_path))
        total_sec = len(wav) / float(SAMPLE_RATE)

        if self.skip_silence:
            spans = detect_speech_spans(wav, SAMPLE_RATE, dbfs_floor=self.silence_dbfs)
            if not spans:
                if self.progress_callback:
                    self.progress_callback(
                        f"    ! 整条音频几乎无声（{total_sec:.1f}s），已跳过识别")
                return "", []
            kept = sum(end - start for start, end in spans)
            if self.progress_callback and total_sec - kept > 0.5:
                self.progress_callback(
                    f"    检测到 {len(spans)} 段人声，跳过静音 {total_sec - kept:.1f}s")
        else:
            spans = [(0.0, total_sec)]

        parts: list[tuple[Any, float]] = []
        for start_sec, end_sec in spans:
            start = max(0, int(start_sec * SAMPLE_RATE))
            end = min(len(wav), int(end_sec * SAMPLE_RATE))
            sub = wav[start:end]
            if sub.size == 0:
                continue
            if chunk_sec > 0 and len(sub) / float(SAMPLE_RATE) > chunk_sec:
                for piece, offset in split_audio_into_chunks(
                        wav=sub, sr=SAMPLE_RATE, max_chunk_sec=chunk_sec):
                    parts.append((piece, start_sec + offset))
            else:
                parts.append((sub, start_sec))

        return self._run_parts(parts, prompt, lang)

    def _run_parts(self, parts: Sequence[tuple[Any, float]],
                   prompt: str, lang: str) -> tuple[str, list[Any]]:
        """逐段推理并拼接结果（段间时间戳会偏移回整曲）。"""
        if not parts:
            return "", []
        if len(parts) == 1:
            chunk, offset = parts[0]
            return self._run((chunk, SAMPLE_RATE), prompt, lang, offset or None)

        texts: list[str] = []
        items: list[Any] = []
        total = len(parts)
        for index, (chunk, offset) in enumerate(parts, start=1):
            self._raise_if_cancelled()
            if self.progress_callback:
                self.progress_callback(
                    f"    分段 {index}/{total}（{offset:.1f}s 起，"
                    f"{len(chunk) / float(SAMPLE_RATE):.1f}s）")
            chunk_started = time.perf_counter()
            text, chunk_items = self._run(
                (chunk, SAMPLE_RATE), prompt, lang, offset or None)
            texts.append(text)
            items.extend(chunk_items)
            if self.progress_callback:
                self.progress_callback(
                    f"      ✓ 本段 {len(text)} 字，用时 {time.perf_counter() - chunk_started:.1f}s，"
                    f"累计 {len(join_transcript_chunks(texts))} 字")
            self._release_cache()
        return join_transcript_chunks(texts), items

    def close(self) -> None:
        self.model = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


class _AlignerOnlyBackend:
    """只加载 Qwen3-ForcedAligner：用已有 ``.txt`` 做字级对齐，不加载 ASR 模型。

    适合「先只识别（不带时间戳）→ 再只对齐」的两步流程，
    显存里只有一个 0.6B 模型。
    """

    aligner: Any

    cancel_checker: Optional[Callable[[], bool]] = None

    def __init__(self, settings: AsrSettings):
        try:
            import torch  # noqa: F401 - 只要装了 torch 才能用官方包
            from qwen_asr import Qwen3ForcedAligner
        except ImportError as error:  # pragma: no cover - 取决于运行环境
            raise RuntimeError(
                "无法导入 torch / qwen_asr，请先安装依赖：\n"
                "    pip install -r requirements-asr.txt\n"
                f"\n原始错误：{error}"
            ) from error

        self.name = RUN_MODE_ALIGN_ONLY
        self.last_perf: dict = {}
        self.last_note: str = ""
        self.device = resolve_device(settings.device)
        self.dtype = resolve_dtype(settings.dtype, self.device)
        self.chunk_size_sec = float(settings.chunk_size_sec or 0.0)
        self.empty_cache_per_chunk = bool(settings.empty_cache_per_chunk)
        self.aligner_dir = resolve_model_dir(settings.aligner_model_dir)
        self.aligner = Qwen3ForcedAligner.from_pretrained(
            str(self.aligner_dir),
            dtype=self.dtype,
            device_map=self.device,
        )

    def transcribe(self, audio_path: pathlib.Path, prompt: str, language: str) -> tuple[str, list[Any]]:
        raise RuntimeError("只对齐模式不使用 ASR 解码，请改用 align_file()")

    def align_file(self, audio_path: pathlib.Path, text: str, language: str) -> tuple[str, list[Any]]:
        """用已有文本对单个音频做对齐，返回 (text, items)。"""
        from qwen_asr.inference.qwen3_forced_aligner import ForcedAlignItem
        from qwen_asr.inference.utils import normalize_audio_input, split_audio_into_chunks

        lang = LANGUAGE_CANONICAL.get(language, "Chinese")
        wav = normalize_audio_input(str(audio_path))
        # 分段长度：页面上填得比官方上限更小时就按页面的来（省显存）
        max_chunk = OFFICIAL_ALIGN_MAX_SEC
        if 0 < float(self.chunk_size_sec or 0.0) < max_chunk:
            max_chunk = float(self.chunk_size_sec)
        parts = split_audio_into_chunks(wav=wav, sr=SAMPLE_RATE, max_chunk_sec=max_chunk)

        def _align(chunk, chunk_text: str) -> list[Any]:
            results = self.aligner.align(
                audio=(chunk, SAMPLE_RATE), text=chunk_text, language=lang)
            return list(results[0].items) if results else []

        if len(parts) == 1:
            return text, _align(parts[0][0], text)

        # 太长：按静音点分段 + 按各段时长比例切文本，随后把时间轴挪回整曲
        weights = [max(len(chunk) / float(SAMPLE_RATE), 1e-6) for chunk, _offset in parts]
        texts = split_text_proportional(text, weights)
        items: list[Any] = []
        for (chunk, offset), chunk_text in zip(parts, texts):
            if self.cancel_checker is not None and self.cancel_checker():
                raise InterruptedError("用户停止")
            if not chunk_text.strip():
                continue
            for item in _align(chunk, chunk_text):
                items.append(ForcedAlignItem(
                    text=item.text,
                    start_time=float(item.start_time) + offset,  # type: ignore[arg-type]
                    end_time=float(item.end_time) + offset,  # type: ignore[arg-type]
                ))
        return text, items

    def close(self) -> None:
        self.aligner = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def split_text_proportional(text: str, weights: Sequence[float]) -> list[str]:
    """按各段权重（时长）把文本切成 ``len(weights)`` 份。

    切点尽量落在非拉丁字符边界，避免把一个英文单词/拼音音节切成两半。
    """
    text = str(text or "")
    if not weights:
        return []
    if len(weights) == 1 or not text:
        return [text] + [""] * (len(weights) - 1)

    total = sum(float(w) for w in weights if w and w > 0) or 1.0
    cuts: list[int] = []
    acc = 0.0
    for weight in weights[:-1]:
        acc += max(float(weight), 0.0)
        cuts.append(int(round(len(text) * acc / total)))

    def is_word_char(ch: str) -> bool:
        return ch.isascii() and (ch.isalnum() or ch in "'")

    bounds = [0]
    for cut in cuts:
        pos = max(bounds[-1], min(cut, len(text)))
        # 向两侧找非单词边界，避免切断拉丁单词
        limit = len(text)
        while pos < limit and 0 < pos and is_word_char(text[pos - 1]) and is_word_char(text[pos]):
            pos += 1
        bounds.append(min(pos, limit))
    bounds.append(len(text))

    return [text[bounds[i]:bounds[i + 1]] for i in range(len(weights))]


def read_align_text(audio_path: pathlib.Path, out_dir: str = "") -> tuple[str, pathlib.Path | None]:
    """找「只对齐」模式要用的文本：优先输出目录，其次音频同目录。

    依次尝试 ``{stem}.txt`` / ``{stem}.lab``（lab 里的空白分隔歌词也能对齐）。
    """
    audio_path = pathlib.Path(audio_path)
    candidates: list[pathlib.Path] = []
    if str(out_dir or "").strip():
        base = pathlib.Path(out_dir)
        candidates += [base / f"{audio_path.stem}.txt", base / f"{audio_path.stem}.lab"]
    candidates += [audio_path.with_suffix(".txt"), audio_path.with_suffix(".lab")]
    for path in candidates:
        if not path.is_file():
            continue
        try:
            content = read_text_auto(path).strip()
        except OSError:
            continue
        if content:
            return content, path
    return "", None


_BACKEND_CACHE: dict[tuple, Any] = {}


def _backend_key(settings: AsrSettings) -> tuple:
    return (
        settings.run_mode,
        str(settings.asr_model_dir),
        str(settings.aligner_model_dir),
        settings.device,
        settings.dtype,
        int(settings.batch_size or 1),
        bool(settings.aligner_on_cpu),
        bool(settings.encoder_on_cpu),
    )


def get_backend(settings: AsrSettings):
    """按设置缓存后端实例（同一模型不重复加载）。"""
    key = _backend_key(settings)
    backend = _BACKEND_CACHE.get(key)
    if backend is None:
        # 切换模型 / 运行模式时释放旧实例，避免显存堆积
        shutdown()
        if settings.run_mode == RUN_MODE_ALIGN_ONLY:
            backend = _AlignerOnlyBackend(settings)
        else:
            backend = _TransformersBackend(settings)
        _BACKEND_CACHE[key] = backend

    # 运行期参数：改分段长度不需要重新加载模型
    backend.chunk_size_sec = float(settings.chunk_size_sec or 0.0)
    backend.empty_cache_per_chunk = bool(settings.empty_cache_per_chunk)
    backend.skip_silence = bool(settings.skip_silence)
    backend.silence_dbfs = float(settings.silence_dbfs or SILENCE_DBFS)
    return backend


def shutdown() -> None:
    """释放已缓存的后端（GUI 关闭或切换设置时调用）。"""
    for backend in list(_BACKEND_CACHE.values()):
        try:
            backend.close()
        except Exception:
            pass
    _BACKEND_CACHE.clear()


def validate_settings(settings: AsrSettings) -> None:
    run_mode = settings.run_mode if settings.run_mode in RUN_MODES else RUN_MODE_FULL

    # ① 只对齐：只需要 Aligner，读已有 .txt
    if run_mode == RUN_MODE_ALIGN_ONLY:
        if not str(settings.aligner_model_dir or "").strip():
            raise ValueError(
                "「只对齐」模式需要 Qwen3-ForcedAligner 模型目录。\n"
                "请在「Aligner 模型目录」里选择对应文件夹。"
            )
        resolve_model_dir(settings.aligner_model_dir)
        return

    needs_aligner = run_mode == RUN_MODE_FULL
    if needs_aligner and not str(settings.aligner_model_dir or "").strip():
        raise ValueError(
            "「识别 + 对齐」需要 Qwen3-ForcedAligner 模型目录（用于生成逐字时间戳）。\n"
            "请在「Aligner 模型目录」里选择对应文件夹。\n\n"
            "显存不够时不必拆开跑：把运行模式改成「只识别」→ 再改成「只对齐」分两步跑。"
        )

    resolve_model_dir(settings.asr_model_dir)
    if needs_aligner:
        resolve_model_dir(settings.aligner_model_dir)


# ══════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════
def _output_paths(audio_path: pathlib.Path, out_dir: str, stem: str) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
    base = pathlib.Path(out_dir) if str(out_dir or "").strip() else audio_path.parent
    base.mkdir(parents=True, exist_ok=True)
    return (
        base / f"{stem}.txt",
        base / f"{stem}.lab",
        base / f"{stem}_ASR.json",
    )


def _audio_duration(audio_path: pathlib.Path) -> float:
    try:
        import soundfile as sf

        return float(sf.info(str(audio_path)).duration)
    except Exception:
        return 0.0


def _write_outputs(
    audio_path: pathlib.Path,
    settings: AsrSettings,
    text: str,
    items: Sequence[AlignItem],
    *,
    prompt: str,
    lyrics: str,
    elapsed_sec: float,
    perf: Optional[dict] = None,
    write_txt: bool = True,
) -> AsrFileResult:
    language = "ja" if str(settings.language).lower().startswith("ja") else "zh"
    mode = normalize_output_mode(language, settings.output_mode)
    stem = audio_path.stem
    txt_path, lab_path, json_path = _output_paths(audio_path, settings.out_dir, stem)
    result = AsrFileResult(audio_path=audio_path)

    chars = [item.text for item in items]
    tokens = display_tokens(text, language, mode)
    matched = bool(tokens) and len(tokens) == len(chars)
    if not matched:
        tokens = _per_char_tokens(chars, language, mode) if chars else tokens
    lab_text = join_tokens(language, mode, tokens)

    if settings.save_txt and write_txt:
        txt_path.write_text(text, encoding="utf-8")
        result.txt_path = txt_path
    if settings.save_lab and lab_text:
        lab_path.write_text(lab_text, encoding="utf-8")
        result.lab_path = lab_path

    if settings.save_json:
        duration = _audio_duration(audio_path) or (items[-1].end_time if items else 0.0)
        payload = {
            "audio": str(audio_path),
            "backend": BACKEND_TRANSFORMERS,
            "engine": "qwen_asr (transformers)",
            "run_mode": settings.run_mode,
            "model": settings.asr_model_dir,
            "aligner_model": settings.aligner_model_dir,
            "device": settings.device,
            "dtype": settings.dtype,
            "chunk_sec": round(float(settings.chunk_size_sec or 0.0), 3),
            "language": language,
            "output_mode": mode,
            "duration": round(float(duration), 3),
            "count": len(items),
            "text": text,
            "lab": lab_text,
            "pinyin_matched": matched,
            "system_prompt": settings.system_prompt,
            "lyrics": lyrics,
            "prompt": prompt,
            "elapsed_sec": round(elapsed_sec, 2),
            "performance": {key: round(float(value), 3)
                            for key, value in (perf or {}).items()
                            if isinstance(value, (int, float))},
            "items": [
                {
                    "text": item.text,
                    "pinyin": tokens[index] if index < len(tokens) else "",
                    "start": round(float(item.start_time), 3),
                    "end": round(float(item.end_time), 3),
                }
                for index, item in enumerate(items)
            ],
        }
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        result.json_path = json_path

    result.text = text
    result.token_count = len(tokens)
    result.item_count = len(items)
    result.elapsed_sec = elapsed_sec
    result.perf = dict(perf or {})
    return result


def format_perf(perf: Optional[dict], elapsed_sec: float) -> str:
    """把引擎性能统计拼成一行可读文本（无数据时只给总耗时）。"""
    perf = perf or {}
    parts = []
    encode = float(perf.get("encode_time", 0.0) or 0.0)
    wait = float(perf.get("wait_time", 0.0) or 0.0)
    prefill = float(perf.get("prefill_time", 0.0) or 0.0)
    decode = float(perf.get("decode_time", 0.0) or 0.0)
    align = float(perf.get("align_time", 0.0) or 0.0)
    tokens = int(perf.get("decode_tokens", 0) or 0)
    if encode:
        extra = f"（等待 {wait:.1f}s）" if wait > 0.05 else ""
        parts.append(f"编码 {encode:.1f}s{extra}")
    if prefill:
        parts.append(f"预填 {prefill:.1f}s")
    if decode or tokens:
        speed = f"{tokens / decode:.1f} tok/s" if decode > 0 and tokens else "-"
        parts.append(f"解码 {decode:.1f}s（{tokens} tok，{speed}）")
    if align:
        parts.append(f"对齐 {align:.1f}s")
    parts.append(f"合计 {elapsed_sec:.1f}s")
    return "｜".join(parts)


def transcribe_one(
    audio_path: pathlib.Path,
    settings: AsrSettings,
    backend,
    *,
    lyrics: str = "",
    align_text: str = "",
    progress_callback: Callable[[str], None] | None = None,
    stream_callback: Callable[[str], None] | None = None,
    cancel_checker: Callable[[], bool] | None = None,
) -> AsrFileResult:
    started = time.perf_counter()
    prompt = build_final_prompt(settings.system_prompt, lyrics)
    # 让后端能把「第 x/y 段」、流式文本与停止状态写进日志（赋值即可）
    backend.progress_callback = progress_callback
    backend.stream_callback = stream_callback
    backend.cancel_checker = cancel_checker

    if settings.run_mode == RUN_MODE_ALIGN_ONLY:
        # 只对齐：用已有文本，不跑 ASR 解码
        text = str(align_text or "").strip()
        if not text:
            raise RuntimeError("找不到可用的歌词文本（需要同名 .txt 或 .lab）")
        if progress_callback:
            progress_callback("    正在对齐时间戳…")
        text, raw_items = backend.align_file(audio_path, text, settings.language)
    else:
        text, raw_items = backend.transcribe(audio_path, prompt, settings.language)

    items = split_items_per_char(raw_items)
    perf = dict(getattr(backend, "last_perf", {}) or {})
    note = str(getattr(backend, "last_note", "") or "")

    if not text.strip():
        raise RuntimeError("ASR 没有识别出任何文本")
    if not items and progress_callback:
        if settings.run_mode == RUN_MODE_ALIGN_ONLY:
            progress_callback("    ! 对齐没有返回结果：请检查 .txt 与音频是否对应")
        elif settings.run_mode == RUN_MODE_FULL:
            progress_callback("    ! 对齐器没有返回结果：.txt/.lab 仍会生成，JSON 中 items 为空")
        # 只识别模式本来就不出时间戳，这里不提示
    if note and progress_callback:
        progress_callback(f"    ! {note}")

    elapsed = time.perf_counter() - started
    result = _write_outputs(
        audio_path,
        settings,
        text,
        items,
        prompt=prompt,
        lyrics=lyrics,
        elapsed_sec=elapsed,
        perf=perf,
        write_txt=settings.run_mode != RUN_MODE_ALIGN_ONLY,
    )
    result.note = note
    return result


def run_batch(
    audio_paths: Sequence[pathlib.Path],
    settings: AsrSettings,
    *,
    lyrics_map: Optional[dict[str, str]] = None,
    notes: Sequence[str] = (),
    progress_callback: Optional[Callable[[str], None]] = None,
    stream_callback: Optional[Callable[[str], None]] = None,
    cancel_checker: Optional[Callable[[], bool]] = None,
) -> dict:
    """批量识别并落盘，返回统计字典。

    ``lyrics_map`` 以音频绝对路径（或原始字符串）为键；未命中时自动读取同名
    ``.txt`` / ``.lrc``。
    """
    paths = [pathlib.Path(item) for item in audio_paths]
    lyrics_map = dict(lyrics_map or {})
    stats = {
        "total": len(paths),
        "written": 0,
        "failed": 0,
        "stopped": False,
        "results": [],
        "errors": [],
    }

    def log(message: str) -> None:
        if progress_callback is not None:
            progress_callback(message)

    if not paths:
        log("没有可处理的音频。")
        return stats

    validate_settings(settings)
    run_mode = settings.run_mode if settings.run_mode in RUN_MODES else RUN_MODE_FULL
    align_only = run_mode == RUN_MODE_ALIGN_ONLY
    chunk_sec = float(settings.chunk_size_sec or 0.0)
    chunk_text = f"{chunk_sec:g}s/段" if chunk_sec > 0 else "官方自动分段"
    log(
        f"模式：{RUN_MODE_LABELS.get(run_mode, run_mode)}｜"
        f"推理：{BACKEND_LABELS.get(BACKEND_TRANSFORMERS, BACKEND_TRANSFORMERS)}｜"
        f"语言：{LANGUAGE_LABELS.get(settings.language, settings.language)}｜"
        f"输出模式：{normalize_output_mode(settings.language, settings.output_mode)}｜"
        f"分段：{chunk_text}｜共 {len(paths)} 个音频"
    )
    if not align_only:
        log(f"ASR 模型：{settings.asr_model_dir}")
    if run_mode != RUN_MODE_ASR_ONLY:
        log(f"Aligner 模型：{settings.aligner_model_dir}")
    if align_only:
        log("只对齐模式：读取每个音频同名的 .txt（或 .lab），不重新做 ASR 识别")
    for note in notes:
        log(f"    ! {note}")

    log("正在加载模型（首次加载 1.7B 权重需要几十秒，之后会复用）…")
    load_started = time.perf_counter()
    backend = get_backend(settings)
    load_elapsed = time.perf_counter() - load_started
    aligner_device = getattr(backend, "aligner_device", None)
    aligner_text = (f"｜Aligner 设备：{DEVICE_LABELS.get(aligner_device, aligner_device)}"
                    if aligner_device else "")
    encoder_text = "｜编码器：CPU" if getattr(backend, "encoder_on_cpu", False) else ""
    log(
        f"模型已就绪｜设备：{DEVICE_LABELS.get(getattr(backend, 'device', settings.device), settings.device)}"
        f"｜精度：{DTYPE_LABELS.get(settings.dtype, settings.dtype)}"
        f"{aligner_text}{encoder_text}｜批大小：{int(settings.batch_size or 1)}"
        f"｜加载耗时 {load_elapsed:.1f}s"
    )

    for index, audio_path in enumerate(paths, start=1):
        if cancel_checker is not None and cancel_checker():
            stats["stopped"] = True
            log("已停止。")
            break

        if not settings.overwrite:
            _txt, _lab, json_path = _output_paths(audio_path, settings.out_dir, audio_path.stem)
            if json_path.exists():
                log(f"[{index}/{len(paths)}] 跳过（已存在 {json_path.name}）：{audio_path.name}")
                continue

        lyrics = lyrics_map.get(str(audio_path)) or lyrics_map.get(audio_path.name) or ""
        # 页面手动编辑保存的 _lyrics.txt 优先级最高（清空按钮会删除它）
        edited = load_edited_lyrics(audio_path)
        if edited.strip():
            lyrics = edited
        elif not lyrics:
            lyrics = load_sidecar_lyrics(audio_path)

        align_text = ""
        if align_only:
            align_text, text_path = read_align_text(audio_path, settings.out_dir)
            log(f"[{index}/{len(paths)}] 正在对齐：{audio_path.name}")
            if align_text:
                log(f"    使用文本：{text_path}")
            else:
                stats["failed"] += 1
                stats["errors"].append(f"{audio_path.name}: 找不到同名 .txt / .lab")
                log("    ✗ 找不到同名 .txt / .lab，已跳过")
                continue
        else:
            log(f"[{index}/{len(paths)}] 正在识别：{audio_path.name}")
            if lyrics:
                log(f"    歌词提示：{normalize_whitespace(lyrics)[:60]}"
                    f"{'...' if len(normalize_whitespace(lyrics)) > 60 else ''}")

        try:
            result = transcribe_one(
                audio_path,
                settings,
                backend,
                lyrics=lyrics,
                align_text=align_text,
                progress_callback=log,
                stream_callback=stream_callback,
                cancel_checker=cancel_checker,
            )
            stats["results"].append(result)
            stats["written"] += 1
            log(
                f"    ✓ {result.item_count} 个字 / {result.token_count} 个音素，"
                f"用时 {result.elapsed_sec:.1f}s"
            )
            log(f"      ⏱ {format_perf(result.perf, result.elapsed_sec)}")
            if result.txt_path:
                log(f"      txt：{result.txt_path}")
            log(f"      lab：{result.lab_path}")
            log(f"      json：{result.json_path}")
            log(f"      文本：{normalize_whitespace(result.text)[:80]}"
                f"{'...' if len(normalize_whitespace(result.text)) > 80 else ''}")
        except InterruptedError:
            stats["stopped"] = True
            break
        except Exception as error:  # noqa: BLE001 - 单文件失败不影响整体
            stats["failed"] += 1
            stats["errors"].append(f"{audio_path.name}: {error}")
            log(f"    ✗ 失败：{error}")

    log(f"完成：成功 {stats['written']} / 失败 {stats['failed']} / 共 {stats['total']}")
    return stats
