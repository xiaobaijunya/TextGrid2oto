# coding=utf-8
"""音素审计：按音频统计异常音素，挑出「异常过多」的废录音，并支持删除。

与 ``textgrid2json.detect_short_phonemes`` 的区别：

* 按**音频**汇总（一个音频对应同目录同名 TextGrid），给出「异常处数」；
* 只有异常数 **≥ 阈值** 的音频才算问题文件，方便批量清理废录音；
* 支持一次删除问题音频及其同名 ``.lab`` / ``.TextGrid`` / ``.txt`` / ``_ASR.json`` /
  ``_lyrics.txt`` / ``.lrc``。

检查项：

* ``short``   音素时长 < ``min_phoneme_ms``（默认 30ms）
* ``zero``    时长 <= 0（标记写错或端点重叠）
* ``overlap`` 与上一个区间重叠（``xmin < prev.xmax``）

静音音素（SP / AP / EP / R）默认不参与「过短」判定——呼吸和停顿本来就很短。
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional, Sequence

from textgrid2json.detect_short_phonemes import parse_tiers, read_textgrid

AUDIO_SUFFIXES = (".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac", ".opus", ".aiff", ".aif")
TEXTGRID_SUFFIXES = (".TextGrid", ".textgrid")
SILENCE_PHONES = {"SP", "AP", "EP", "R"}

ISSUE_LABELS = {
    "short": "过短",
    "zero": "零/负时长",
    "overlap": "区间重叠",
}


@dataclass
class PhonemeIssue:
    kind: str
    tier: str
    phoneme: str
    start: float
    end: float
    duration_ms: float

    def describe(self) -> str:
        label = ISSUE_LABELS.get(self.kind, self.kind)
        name = self.phoneme or "(空)"
        return (f"{label}：{name} {self.start:.3f}–{self.end:.3f}s"
                f"（{self.duration_ms:.1f}ms，层 {self.tier}）")


@dataclass
class AudioAudit:
    audio_path: pathlib.Path
    tg_path: Optional[pathlib.Path] = None
    issues: list[PhonemeIssue] = field(default_factory=list)
    note: str = ""

    @property
    def count(self) -> int:
        return len(self.issues)

    def summary(self) -> str:
        if self.note:
            return self.note
        if not self.issues:
            return "无异常"
        by_kind: dict[str, int] = {}
        for issue in self.issues:
            by_kind[issue.kind] = by_kind.get(issue.kind, 0) + 1
        parts = [f"{ISSUE_LABELS.get(kind, kind)} {count}"
                 for kind, count in sorted(by_kind.items())]
        return f"{self.count} 处异常（" + "、".join(parts) + "）"


@dataclass
class AuditOptions:
    min_phoneme_ms: float = 30.0
    ignore_silence: bool = True
    check_overlap: bool = True


# ── 扫描 ────────────────────────────────────────────────────────
def _pick_target_tiers(tiers) -> list:
    """只挑一个最细的层来检查，避免 hanzi / words 重复计数。

    优先级：phones → 音素 → words（拼音）→ 第一个 tier。
    """
    for keyword in ("phone", "音素", "words", "word", "拼音"):
        for item in tiers:
            if keyword in item[0].lower():
                return [item]
    return tiers[:1]


def find_textgrid(audio_path: pathlib.Path) -> Optional[pathlib.Path]:
    for suffix in TEXTGRID_SUFFIXES:
        candidate = audio_path.with_suffix(suffix)
        if candidate.is_file():
            return candidate
    return None


def collect_audio_paths(folder: pathlib.Path, recursive: bool = True) -> list[pathlib.Path]:
    folder = pathlib.Path(folder)
    if not folder.is_dir():
        return []
    walker: Iterable[pathlib.Path] = folder.rglob("*") if recursive else folder.iterdir()
    return sorted(p for p in walker
                  if p.is_file() and p.suffix.lower() in AUDIO_SUFFIXES)


def audit_textgrid(
    audio_path: pathlib.Path,
    tg_path: pathlib.Path,
    options: AuditOptions,
) -> AudioAudit:
    """检查一个 TextGrid，返回该音频的异常列表。"""
    result = AudioAudit(audio_path=pathlib.Path(audio_path),
                        tg_path=pathlib.Path(tg_path))
    try:
        tiers = parse_tiers(read_textgrid(tg_path))
    except Exception as error:  # noqa: BLE001 - 单个文件失败不影响整体
        result.note = f"TextGrid 读取失败：{error}"
        return result

    if not tiers:
        result.note = "TextGrid 里没有 tier"
        return result

    # 只检查一个最细的层（phones / 音素 / words），避免重复计数
    targets = _pick_target_tiers(tiers)

    for tier_name, intervals in targets:
        prev_end: Optional[float] = None
        for xmin, xmax, text in intervals:
            phoneme = str(text).strip()
            duration_ms = (xmax - xmin) * 1000.0

            if (options.check_overlap and prev_end is not None
                    and xmin < prev_end - 1e-6):
                result.issues.append(PhonemeIssue(
                    "overlap", tier_name, phoneme, xmin, xmax, duration_ms))
            prev_end = xmax

            if not phoneme:
                continue
            if duration_ms <= 0:
                result.issues.append(PhonemeIssue(
                    "zero", tier_name, phoneme, xmin, xmax, duration_ms))
                continue
            if options.ignore_silence and phoneme.upper() in SILENCE_PHONES:
                continue
            if duration_ms < options.min_phoneme_ms:
                result.issues.append(PhonemeIssue(
                    "short", tier_name, phoneme, xmin, xmax, duration_ms))
    return result


def scan_folder(
    folder: str | pathlib.Path,
    *,
    recursive: bool = True,
    options: Optional[AuditOptions] = None,
    threshold: int = 3,
    progress: Optional[Callable[[str], None]] = None,
    stop_event=None,
    include_below: bool = False,
) -> dict:
    """扫描文件夹，返回异常数 ≥ ``threshold`` 的音频。

    Returns:
        ``{total, checked, skipped, flagged, all_results, stopped}``
    """
    options = options or AuditOptions()
    folder = pathlib.Path(folder)
    audios = collect_audio_paths(folder, recursive)

    results: list[AudioAudit] = []
    flagged: list[AudioAudit] = []
    skipped: list[pathlib.Path] = []
    stopped = False

    for index, audio in enumerate(audios, start=1):
        if stop_event is not None and stop_event.is_set():
            stopped = True
            break
        tg_path = find_textgrid(audio)
        if tg_path is None:
            skipped.append(audio)
            if progress and len(skipped) <= 5:
                progress(f"[{index}/{len(audios)}] 跳过（没有同名 TextGrid）：{audio.name}")
            continue

        audit = audit_textgrid(audio, tg_path, options)
        results.append(audit)
        if audit.count >= threshold:
            flagged.append(audit)
            if progress:
                progress(f"[{index}/{len(audios)}] ✗ {audio.name}：{audit.summary()}")
                for issue in audit.issues[:20]:
                    progress(f"      · {issue.describe()}")
                if audit.count > 20:
                    progress(f"      · …… 另有 {audit.count - 20} 处")
        elif progress and (include_below or audit.count):
            progress(f"[{index}/{len(audios)}] ✓ {audio.name}：{audit.summary()}")

    return {
        "total": len(audios),
        "checked": len(results),
        "skipped": skipped,
        "flagged": flagged,
        "all_results": results,
        "stopped": stopped,
    }


# ── 删除 ────────────────────────────────────────────────────────
def related_files(
    audio_path: str | pathlib.Path,
    *,
    remove_tg: bool = True,
    remove_lab: bool = True,
    remove_txt: bool = True,
    remove_json: bool = True,
    remove_lyrics: bool = True,
) -> list[pathlib.Path]:
    """列出与音频同名的关联文件（不含音频本身）。"""
    audio_path = pathlib.Path(audio_path)
    stem, parent = audio_path.stem, audio_path.parent
    found: list[pathlib.Path] = []

    def add(path: pathlib.Path) -> None:
        if path.is_file() and path != audio_path and path not in found:
            found.append(path)

    if remove_tg:
        for suffix in TEXTGRID_SUFFIXES:
            add(parent / f"{stem}{suffix}")
    if remove_lab:
        add(parent / f"{stem}.lab")
    if remove_txt:
        for suffix in (".txt", ".lrc"):
            add(parent / f"{stem}{suffix}")
    if remove_json:
        for name in (f"{stem}_ASR.json", f"{stem}.json"):
            add(parent / f"{name}")
    if remove_lyrics:
        add(parent / f"{stem}_lyrics.txt")
    return found


def delete_related(
    audio_paths: Sequence[str | pathlib.Path],
    *,
    remove_tg: bool = True,
    remove_lab: bool = True,
    remove_txt: bool = True,
    remove_json: bool = True,
    remove_lyrics: bool = True,
    progress: Optional[Callable[[str], None]] = None,
) -> dict:
    """删除音频及其同名关联文件。

    Returns:
        ``{audios, files, failed}``（``files`` 只统计关联文件）
    """
    deleted_audios = 0
    deleted_files = 0
    failed: list[str] = []

    for audio_path in audio_paths:
        audio_path = pathlib.Path(audio_path)
        targets = related_files(
            audio_path,
            remove_tg=remove_tg, remove_lab=remove_lab, remove_txt=remove_txt,
            remove_json=remove_json, remove_lyrics=remove_lyrics,
        )
        for target in targets:
            try:
                target.unlink()
                deleted_files += 1
                if progress:
                    progress(f"      删除 {target.name}")
            except OSError as error:
                failed.append(f"{target.name}: {error}")
        try:
            audio_path.unlink()
            deleted_audios += 1
            if progress:
                progress(f"    已删除 {audio_path.name}")
        except OSError as error:
            failed.append(f"{audio_path.name}: {error}")

    return {"audios": deleted_audios, "files": deleted_files, "failed": failed}
