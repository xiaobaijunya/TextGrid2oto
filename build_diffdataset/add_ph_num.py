# -*- coding: utf-8 -*-
"""为 transcriptions.csv 添加 ``ph_num`` 列。

参考 MakeDiffSinger/variance-temp-solution/add_ph_num.py。

分组规则（韵母 + 其后辅音）：
    一个元音（或连续元音组成的韵母）加上它后面连续的辅音算作一组；
    纯元音后面没有辅音时就自己一组。``ph_num`` 记录每组包含的音素数。
    连续的元音默认合并成一个韵母；如果提供了「元音组合字典」
    （每行一个合法韵母），则按最长匹配切分，能区分「前一字韵尾 +
    后一字零声母」这种相邻元音的情况。

例如::

    ph_seq   sil  ts  a  :\\i  w  uo  s`  @  :n  x  @U  sil  br
    ph_num    2       3        2        3      1    1    1

支持三种规则来源：

1. ``--dictionary``：两段式字典（每条词目为 V 或 C-V），自动收集元音/辅音；
2. ``--vowels`` + ``--consonants``：分别提供元音、辅音文件。**元音文件每行
   一个韵母**，一行写多个音素（如 ``iU N``）即表示它们是一个组合韵母。

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


def _load_sets_from_dictionary(path) -> tuple[set, set, list[str]]:
    """从两段式字典（V 或 C-V）收集元音/辅音集合。

    非两段式行（如 C-V-V）会被跳过并记录到返回的问题列表里，
    不中断整个流程。
    """
    vowels = set(DEFAULT_VOWELS)
    consonants: set = set()
    valid_lines = 0
    skipped_lines = 0
    issues: list[str] = []

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
                issues.append(
                    f"字典第 {line_no} 行含 {len(phonemes)} 个音素，已跳过：{line}"
                )
                skipped_lines += 1
                continue

            if len(phonemes) == 1:
                vowels.add(phonemes[0])
            else:
                consonants.add(phonemes[0])
                vowels.add(phonemes[1])
            valid_lines += 1

    if valid_lines == 0:
        raise ValueError("字典中没有可用的两段式词条（请检查文件格式：词<TAB>音素）")

    if issues:
        issues.append(
            "提示：多音素字典请改用「元音 + 辅音文件」模式"
            "（如 tg2svdb/cv_dicts/mandarin-xsampa/vowels.txt + consonants.txt）"
        )

    return vowels, consonants, issues


def _load_word_set(path) -> set:
    """读取空格/换行分隔的音素集合文件。"""
    with open(path, "r", encoding="utf-8-sig") as file:
        return {_phoneme(token) for token in file.read().split() if _phoneme(token)}


def _load_vowel_units(path) -> tuple[set, list[tuple[str, ...]]]:
    """读取元音文件：**每行一个韵母**。

    一行里有多个音素时，表示这些音素构成一个组合韵母（如 ``iU N``、
    ``a :\\i``），切分 ``ph_num`` 时会被当作一个整体；单音素行就是一个韵母。

    返回的“组合”列表包含所有行（含单音素行）：它是最长匹配的候选表，
    只有单音素韵母也在表里，``o`` + ``a :\\i`` 才能被正确地切成两个韵母，
    而不是退化成“把连续元音全部合并”。

    Returns:
        (所有元音音素集合, 韵母候选列表（按音素数降序）)
    """
    vowels: set = set(DEFAULT_VOWELS)
    compounds: set[tuple[str, ...]] = set()

    with open(path, "r", encoding="utf-8-sig") as file:
        for raw in file:
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            phones = tuple(ph for ph in (_phoneme(tok) for tok in line.split()) if ph)
            if not phones:
                continue
            vowels.update(phones)
            compounds.add(phones)

    if not vowels - DEFAULT_VOWELS:
        raise ValueError(f"元音文件为空或格式不对: {path}")
    return vowels, sorted(compounds, key=len, reverse=True)


def _load_sets_from_files(vowels_path, consonants_path) -> tuple[set, set, list[tuple[str, ...]]]:
    vowels, compounds = _load_vowel_units(vowels_path)
    consonants = _load_word_set(consonants_path)

    overlap = vowels & consonants
    if overlap:
        raise ValueError(
            "元音与辅音集合重叠，以下音素同时出现在两边：\n" + ", ".join(sorted(overlap))
        )
    return vowels, consonants, compounds


# 这些音素（静音 / 呼吸 / 停顿）不会与相邻元音合并成一个韵母
STANDALONE_PHONES = {"SP", "AP", "EP", "sil", "br", "pau"}


def _split_units(
    ph_seq: list[str],
    vowels: set,
    consonants: set,
    compounds: Optional[list[tuple[str, ...]]] = None,
) -> list[list[str]]:
    """把音素序列切成「音节内部单元」。

    * 辅音 / 静音符号 → 各自一个单元；
    * 连续的元音 → 合并成一个韵母单元（有元音组合字典时按最长匹配切分）。
    """
    unknown = sorted({
        ph for ph in ph_seq
        if ph not in vowels and ph not in consonants and ph not in STANDALONE_PHONES
    })
    if unknown:
        raise ValueError(
            "ph_seq 中存在字典里没有的音素：" + ", ".join(unknown) +
            "\n请确认字典/元辅音文件与当前音素体系一致。"
        )

    units: list[list[str]] = []
    index = 0
    total = len(ph_seq)
    while index < total:
        phone = ph_seq[index]
        if phone in STANDALONE_PHONES or phone in consonants:
            units.append([phone])
            index += 1
            continue

        matched = None
        for combo in compounds or ():
            size = len(combo)
            if size and tuple(ph_seq[index:index + size]) == combo:
                matched = list(combo)
                break
        if matched is not None:
            units.append(matched)
            index += len(matched)
            continue

        # 没有组合字典（或未命中）：把后续连续元音直接并进来
        end = index + 1
        while (end < total and ph_seq[end] in vowels
               and ph_seq[end] not in STANDALONE_PHONES):
            end += 1
        units.append(ph_seq[index:end])
        index = end
    return units


def _compute_ph_num(
    ph_seq,
    vowels,
    consonants,
    compounds: Optional[list[tuple[str, ...]]] = None,
) -> list[str]:
    """按「韵母 + 其后连续辅音」分组，返回每组音素数。

    中文音素串里一个音节的韵母后面紧跟的往往就是下一个字的声母
    （如 ``a :\\i w``），所以：**元音单元开启新组，其后的辅音并入该组**；
    纯元音后面没有辅音时就自己一组。

    连续的元音先用「元音组合字典」按最长匹配切成韵母单元，没提供字典时
    把相邻连续元音全部合并。
    """
    units = _split_units(ph_seq, vowels, consonants, compounds)
    ph_num: list[int] = []
    for unit in units:
        head = unit[0]
        is_consonant = len(unit) == 1 and head in consonants
        if is_consonant and ph_num:
            ph_num[-1] += len(unit)
        else:
            ph_num.append(len(unit))
    return [str(value) for value in ph_num]


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
        vowels: 元音文件路径。**每行一个韵母**；一行写多个音素（如``iU N``）
            表示它们是一个组合韵母，切分时当作一个整体。
        consonants: 辅音文件路径（空格 / 换行分隔）。
        progress_callback: 进度回调，接收一行文本。

    Returns:
        统计字典：total / updated / failed / errors / csv。
    """
    csv_path = pathlib.Path(transcription)
    if not csv_path.is_file():
        raise FileNotFoundError(f"transcriptions.csv 不存在: {csv_path}")

    has_dictionary = bool(dictionary)
    has_files = bool(vowels) and bool(consonants)
    if not has_dictionary and not has_files:
        raise ValueError("请提供字典文件，或同时提供元音文件和辅音文件")

    if has_dictionary:
        vowel_set, consonant_set, dict_issues = _load_sets_from_dictionary(dictionary)
        compounds: list[tuple[str, ...]] = []
        source = f"字典 {dictionary}"
    else:
        vowel_set, consonant_set, compounds = _load_sets_from_files(vowels, consonants)
        dict_issues = []
        source = f"元音 {vowels} + 辅音 {consonants}"

    def log(message: str):
        if progress_callback is not None:
            progress_callback(message)
        else:
            print(message)

    log(f"规则来源: {source}（元音 {len(vowel_set)} 个，辅音 {len(consonant_set)} 个）")
    if compounds:
        log(f"元音组合: {len(compounds)} 组（最长 {len(compounds[0])} 个音素）")

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
    errors: list[str] = []
    missing: dict[str, int] = {}
    updated = 0
    for index, item in enumerate(items, 1):
        name = item.get("name") or f"#{index}"
        ph_seq = [_phoneme(ph) for ph in (item.get("ph_seq") or "").split()]
        if not ph_seq:
            item["ph_num"] = ""
            errors.append(f"({index}/{total}) {name}: ph_seq 为空")
            continue

        unknown = sorted(
            {ph for ph in ph_seq if ph not in vowel_set and ph not in consonant_set}
        )
        if unknown:
            for ph in unknown:
                missing[ph] = missing.get(ph, 0) + 1
            item["ph_num"] = ""
            errors.append(f"({index}/{total}) {name}: 缺少音素 {', '.join(unknown)}")
            continue

        try:
            item["ph_num"] = " ".join(_compute_ph_num(ph_seq, vowel_set, consonant_set, compounds))
            updated += 1
            log(f"({index}/{total}) {name}")
        except Exception as error:  # noqa: BLE001 - 单行失败不影响整批
            item["ph_num"] = ""
            errors.append(f"({index}/{total}) {name}: {type(error).__name__}: {error}")

    # ph_num 若原本不存在，会追加到列末尾；其余列顺序保持不变
    output_fields = list(items[0].keys())
    with open(csv_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=output_fields)
        writer.writeheader()
        writer.writerows(items)

    # 错误统一放到最后一次性输出
    if dict_issues:
        log("")
        log(f"========== 字典有 {len(dict_issues)} 处问题 ==========")
        for line in dict_issues[:50]:
            log(f"  · {line}")
        if len(dict_issues) > 50:
            log(f"  ... 另有 {len(dict_issues) - 50} 行略")

    if errors:
        log("")
        log(f"========== 以下 {len(errors)} 行未能生成 ph_num ==========")
        if missing:
            log("缺少的音素（涉及行数）：")
            for ph, count in sorted(missing.items(), key=lambda kv: -kv[1]):
                log(f"  · {ph}    {count} 行")
            log("→ 请把上面这些音素补进字典/元音文件，或检查 ph_seq 是否拼写有误")
            log("")
        log("涉及的音频：")
        for line in errors[:100]:
            log(f"  ✗ {line}")
        if len(errors) > 100:
            log(f"  ... 另有 {len(errors) - 100} 行略")
        log("=" * 48)
        log("（其余行已正常写入；修正后可重新运行本步骤）")

    log(f"完成：共 {total} 条，成功 {updated}，失败 {len(errors)} → {csv_path}")
    return {
        "total": total,
        "updated": updated,
        "failed": len(errors),
        "errors": errors,
        "csv": str(csv_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Add ph_num attribute into transcriptions.csv")
    parser.add_argument("transcription", help="Path to transcriptions.csv")
    parser.add_argument("--dictionary", help="Two-part dictionary file (V or C-V)")
    parser.add_argument("--vowels", help="Vowel file, one final per line (multi-phoneme line = compound)")
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
