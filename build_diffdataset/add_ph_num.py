# -*- coding: utf-8 -*-
"""为 transcriptions.csv 添加 ``ph_num`` 列。

参考 MakeDiffSinger/variance-temp-solution/add_ph_num.py。

分组规则（与 DiffSinger 保持一致）：
    每个元音（含 SP / AP / EP）加上它后面连续的辅音，算作一组；
    ``ph_num`` 记录每一组包含的音素数量。

例如::

    ph_seq   AP  sh  ir  zh  e
    ph_num        2       2   1

支持两种规则来源：

1. ``--dictionary``：两段式字典（每条词目为 V 或 C-V），自动收集元音/辅音；
2. ``--vowels`` + ``--consonants``：分别提供元音、辅音集合（空格/换行分隔）。

注意：
  * 只适合单音节音素体系（中文、日语等）；英语等多音节体系建议手工标注；
  * 字典音素若带 ``zh/``、``ja/`` 等语言前缀，会自动去掉后再比较；
  * 直接原地更新 ``transcriptions.csv``，其余列保持不变。
"""

from __future__ import annotations

import argparse
import csv
import pathlib
from typing import Callable, Optional

# 这些音素始终视为「元音/组起点」
DEFAULT_VOWELS = {"SP", "AP", "EP"}


def _phoneme(token: str) -> str:
    """去掉语言前缀：``zh/n`` -> ``n``。"""
    return str(token).split("/")[-1].strip()


def _load_sets_from_dictionary(path) -> tuple[set, set]:
    """从两段式字典（V 或 C-V）收集元音/辅音集合。"""
    vowels = set(DEFAULT_VOWELS)
    consonants: set = set()
    valid_lines = 0
    skipped_lines = 0

    with open(path, "r", encoding="utf-8-sig") as file:
        for line_no, raw in enumerate(file, 1):
            line = raw.strip()
            if not line:
                continue

            parts = line.split("\t")
            if len(parts) < 2:
                skipped_lines += 1
                continue

            phonemes = [_phoneme(ph) for ph in parts[1].split()]
            if not phonemes:
                skipped_lines += 1
                continue

            if len(phonemes) > 2:
                raise ValueError(
                    f"字典第 {line_no} 行包含 {len(phonemes)} 个音素，"
                    "只支持两段式字典（V 或 C-V）。\n"
                    "英语等多音节体系请手工标注 ph_num，"
                    "或改用「元音 + 辅音文件」模式。\n"
                    f"问题行：{line}"
                )

            if len(phonemes) == 1:
                vowels.add(phonemes[0])
            else:
                consonants.add(phonemes[0])
                vowels.add(phonemes[1])
            valid_lines += 1

    if valid_lines == 0:
        raise ValueError("字典中没有可用的两段式词条（请检查文件格式：词<TAB>音素）")

    return vowels, consonants


def _load_word_set(path) -> set:
    """读取空格/换行分隔的音素集合文件。"""
    with open(path, "r", encoding="utf-8-sig") as file:
        return {_phoneme(token) for token in file.read().split() if _phoneme(token)}


def _load_sets_from_files(vowels_path, consonants_path) -> tuple[set, set]:
    vowels = set(DEFAULT_VOWELS) | _load_word_set(vowels_path)
    consonants = _load_word_set(consonants_path)

    overlap = vowels & consonants
    if overlap:
        raise ValueError(
            "元音与辅音集合重叠，以下音素同时出现在两边：\n" + ", ".join(sorted(overlap))
        )
    return vowels, consonants


def _compute_ph_num(ph_seq, vowels, consonants) -> list[str]:
    """按「元音 + 后续连续辅音」切分，返回每组的音素数。"""
    unknown = sorted({ph for ph in ph_seq if ph not in vowels and ph not in consonants})
    if unknown:
        raise ValueError(
            "ph_seq 中存在字典里没有的音素：" + ", ".join(unknown) +
            "\n请确认字典/元辅音文件与当前音素体系一致。"
        )

    ph_num = []
    index = 0
    while index < len(ph_seq):
        end = index + 1
        while end < len(ph_seq) and ph_seq[end] in consonants:
            end += 1
        ph_num.append(str(end - index))
        index = end
    return ph_num


def run(
    transcription,
    dictionary=None,
    vowels=None,
    consonants=None,
    progress_callback: Optional[Callable[[str], None]] = None,
):
    """给 transcriptions.csv 原地添加/更新 ``ph_num`` 列。

    Args:
        transcription: transcriptions.csv 路径。
        dictionary: 两段式字典路径；与 vowels/consonants 二选一。
        vowels: 元音集合文件路径。
        consonants: 辅音集合文件路径。
        progress_callback: 进度回调，接收一行文本。

    Returns:
        统计字典：total / updated / csv。
    """
    csv_path = pathlib.Path(transcription)
    if not csv_path.is_file():
        raise FileNotFoundError(f"transcriptions.csv 不存在: {csv_path}")

    has_dictionary = bool(dictionary)
    has_files = bool(vowels) and bool(consonants)
    if not has_dictionary and not has_files:
        raise ValueError("请提供字典文件，或同时提供元音文件和辅音文件")

    if has_dictionary:
        vowel_set, consonant_set = _load_sets_from_dictionary(dictionary)
        source = f"字典 {dictionary}"
    else:
        vowel_set, consonant_set = _load_sets_from_files(vowels, consonants)
        source = f"元音 {vowels} + 辅音 {consonants}"

    def log(message: str):
        if progress_callback is not None:
            progress_callback(message)
        else:
            print(message)

    log(f"规则来源: {source}（元音 {len(vowel_set)} 个，辅音 {len(consonant_set)} 个）")

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = list(reader.fieldnames or [])
        if not fieldnames:
            raise ValueError("transcriptions.csv 没有表头")
        if "ph_seq" not in fieldnames:
            raise ValueError("transcriptions.csv 缺少 ph_seq 列")
        items = list(reader)

    if not items:
        log("CSV 中没有数据行")
        return {"total": 0, "updated": 0, "csv": str(csv_path)}

    total = len(items)
    for index, item in enumerate(items, 1):
        name = item.get("name") or f"#{index}"
        ph_seq = [_phoneme(ph) for ph in (item.get("ph_seq") or "").split()]
        if not ph_seq:
            raise ValueError(f"'{name}' 的 ph_seq 为空")

        item["ph_num"] = " ".join(_compute_ph_num(ph_seq, vowel_set, consonant_set))
        log(f"({index}/{total}) {name}")

    # ph_num 若原本不存在，会追加到列末尾；其余列顺序保持不变
    output_fields = list(items[0].keys())
    with open(csv_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=output_fields)
        writer.writeheader()
        writer.writerows(items)

    log(f"完成：共处理 {total} 条 → {csv_path}")
    return {"total": total, "updated": total, "csv": str(csv_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Add ph_num attribute into transcriptions.csv")
    parser.add_argument("transcription", help="Path to transcriptions.csv")
    parser.add_argument("--dictionary", help="Two-part dictionary file (V or C-V)")
    parser.add_argument("--vowels", help="Vowel set file")
    parser.add_argument("--consonants", help="Consonant set file")
    args = parser.parse_args()

    run(
        args.transcription,
        dictionary=args.dictionary,
        vowels=args.vowels,
        consonants=args.consonants,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
