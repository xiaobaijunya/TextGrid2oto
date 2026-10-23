# -*- coding: utf-8 -*-
"""从 TextGrid 的 words 层生成 MFA 风格的 ``.lab``。

words 层的 mark 按顺序用空格连接成一行，写成 ``<原名>.lab``；
``ignore`` 中的 mark（默认 AP/SP/EP/R）会被跳过。
"""

from __future__ import annotations

import pathlib
from typing import Callable, Iterable, Optional

from textgrid import TextGrid

try:  # 包内导入
    from .tg_schema import WORDS_TIER, find_tier
except ImportError:  # 直接运行脚本
    from tg_schema import WORDS_TIER, find_tier


def _plain(mark) -> str:
    return str(mark or "").split("/")[-1].strip()


def words_line(grid: TextGrid, ignore: Iterable[str] = ()) -> str:
    """把 words 层的 mark 拼成一行拼音；没有 words 层时返回空字符串。"""
    tier = find_tier(grid, WORDS_TIER)
    if tier is None or not hasattr(tier, "intervals"):
        return ""
    skipped = {str(item).strip() for item in ignore if str(item).strip()}
    words = []
    for interval in tier:
        mark = _plain(interval.mark)
        if not mark or mark in skipped:
            continue
        words.append(mark)
    return " ".join(words)


def run(tg_dir, out_dir=None, ignore=(), recursive: bool = True,
        progress_callback: Optional[Callable[[str], None]] = None) -> dict:
    """批量生成 lab。

    Args:
        tg_dir: TextGrid 文件夹。
        out_dir: 输出文件夹；为空时写到 TextGrid 同目录。
        ignore: 不写入 lab 的 mark（默认为空，即不忽略任何音素）。
        recursive: 是否递归子文件夹。
        progress_callback: 进度回调。

    Returns:
        统计字典：total / written / skipped / failed / output_dir。
    """
    root = pathlib.Path(tg_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"TextGrid 文件夹不存在: {root}")
    out_root = pathlib.Path(out_dir) if out_dir else None
    ignore = [item for item in (ignore if isinstance(ignore, (list, tuple)) else [ignore]) if str(item).strip()]

    def log(message: str):
        if progress_callback is not None:
            progress_callback(message)
        else:
            print(message)

    files = sorted(root.rglob("*.TextGrid") if recursive else root.glob("*.TextGrid"))
    stats = {"total": len(files), "written": 0, "skipped": 0, "failed": 0,
             "output_dir": str(out_root or root)}

    for index, tg_file in enumerate(files, 1):
        rel = tg_file.relative_to(root)
        try:
            grid = TextGrid()
            grid.read(str(tg_file))
            line = words_line(grid, ignore)
            if not line:
                stats["skipped"] += 1
                log(f"[跳过] {rel}: 没有 words 层或 words 为空")
                continue

            out_file = (out_root / rel).with_suffix(".lab") if out_root else tg_file.with_suffix(".lab")
            out_file.parent.mkdir(parents=True, exist_ok=True)
            out_file.write_text(line + "\n", encoding="utf-8")
            stats["written"] += 1
            log(f"({index}/{len(files)}) {rel} → {out_file.name}")
        except Exception as exc:  # noqa: BLE001 - 单文件失败后继续
            stats["failed"] += 1
            log(f"[失败] {rel}: {type(exc).__name__}: {exc}")

    log(f"完成：共 {stats['total']} 个 TextGrid，写出 {stats['written']}，"
        f"跳过 {stats['skipped']}，失败 {stats['failed']} → {stats['output_dir']}")
    return stats
