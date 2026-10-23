"""TextGrid 过短音素检测

遍历指定文件夹下所有 TextGrid 文件，找出时长小于阈值（单位 ms）的音素区间。
不论音素名是什么（SP / AP / EP / 假名 / 罗马音……），只要过短都会被列出。

结果同时输出到：
  1. 命令行窗口（print）
  2. 返回的列表（供 GUI 使用）

输出末尾会额外打印一行「用 OR 连接的涉事文件名」（已去重），
可直接粘贴到 Windows 资源管理器搜索框定位对应文件。

用法（命令行）:
    python detect_short_phonemes.py <TextGrid文件夹> [阈值ms] [--ignore-edge-sp]
"""
import re
import sys
from pathlib import Path


# 被视为「静音/填充音素」、可忽略首尾的名字（不区分大小写）
EDGE_SP_NAMES = {'SP'}


# ── 读取 ────────────────────────────────────────────────────
def read_textgrid(file_path):
    """读取 TextGrid 文本内容，自动尝试 UTF-8 / UTF-16 编码"""
    for enc in ('utf-8', 'utf-16', 'utf-8-sig'):
        try:
            with open(file_path, 'r', encoding=enc) as f:
                return f.read()
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise UnicodeDecodeError(
        'detect_short_phonemes',
        b'', 0, 1,
        f'无法读取文件 {file_path}，编码格式不支持（UTF-8/UTF-16 都尝试失败）'
    )


def parse_tiers(content):
    """解析 TextGrid 内容

    返回: [(tier_name, [(xmin, xmax, text), ...]), ...]
    """
    tiers = []
    for tier in re.split(r'item \[\d+\]:', content)[1:]:
        name_match = re.search(r'name = "([^"]*)"', tier)
        tier_name = name_match.group(1) if name_match else ''
        intervals = re.findall(
            r'intervals \[\d+\]:\s*xmin = ([\d.eE+-]+)\s*xmax = ([\d.eE+-]+)\s*text = "([^"]*)"',
            tier
        )
        tiers.append((tier_name, [(float(a), float(b), c) for a, b, c in intervals]))
    return tiers


def _pick_phone_tiers(tiers):
    """优先只检测音素层（phones / phone / 音素），找不到则检测全部层"""
    phone_tiers = [t for t in tiers if 'phone' in t[0].lower() or '音素' in t[0]]
    return phone_tiers if phone_tiers else tiers


def _neighbor_texts(intervals, index):
    """取第 index 个区间前后最近的非空音素名"""
    prev_text = ''
    for i in range(index - 1, -1, -1):
        if intervals[i][2].strip():
            prev_text = intervals[i][2]
            break
    next_text = ''
    for i in range(index + 1, len(intervals)):
        if intervals[i][2].strip():
            next_text = intervals[i][2]
            break
    return prev_text, next_text


# ── 检测 ────────────────────────────────────────────────────
def detect_textgrid(file_path, threshold_ms, ignore_edge_sp=False):
    """检测单个 TextGrid 文件中时长过短的音素

    :param file_path: TextGrid 文件路径
    :param threshold_ms: 判定阈值（毫秒），时长小于该值的音素视为过短
    :param ignore_edge_sp: 是否忽略音素层开头/结尾处的 SP（中间的 SP 仍检测）
    :return: 过短音素信息列表
    """
    content = read_textgrid(file_path)
    tiers = _pick_phone_tiers(parse_tiers(content))

    results = []
    for tier_name, intervals in tiers:
        # 首个/末个「非 SP 的有效音素」位置：处于其外侧的 SP 即首尾填充 SP
        content_idx = [
            i for i, (_, _, text) in enumerate(intervals)
            if text.strip() and text.strip().upper() not in EDGE_SP_NAMES
        ]
        first_content = content_idx[0] if content_idx else None
        last_content = content_idx[-1] if content_idx else None

        for i, (xmin, xmax, text) in enumerate(intervals):
            phoneme = text.strip()
            if not phoneme:      # 空文本（无音素名）不计入
                continue
            if ignore_edge_sp and phoneme.upper() in EDGE_SP_NAMES:
                if first_content is None or i < first_content or i > last_content:
                    continue
            duration_ms = (xmax - xmin) * 1000
            if duration_ms < threshold_ms:
                prev_text, next_text = _neighbor_texts(intervals, i)
                results.append({
                    'file': Path(file_path).name,
                    'file_path': str(file_path),
                    'tier': tier_name,
                    'phoneme': phoneme,
                    'xmin': xmin,
                    'xmax': xmax,
                    'duration_ms': duration_ms,
                    'prev': prev_text,
                    'next': next_text,
                })
    return results


def build_search_line(results):
    """把检出的文件名（去重、保持检测顺序）用 " OR " 连接成一行

    方便直接粘贴到 Windows 资源管理器搜索框定位对应文件。
    """
    stems = list(dict.fromkeys(Path(item['file']).stem for item in results))
    return ' OR '.join(stems)


def run(input_dir, threshold_ms, ignore_edge_sp=False):
    """遍历文件夹检测所有 TextGrid 文件

    :param input_dir: TextGrid 所在文件夹（递归查找）
    :param threshold_ms: 判定阈值（毫秒）
    :param ignore_edge_sp: 是否忽略音素层开头/结尾处的 SP
    :return: (results, file_count)
             results 为过短音素信息列表（按文件、时间顺序排列）
    """
    threshold_ms = float(threshold_ms)
    results = []
    file_count = 0

    for file_path in sorted(Path(input_dir).rglob('*.TextGrid')):
        try:
            items = detect_textgrid(file_path, threshold_ms, ignore_edge_sp)
        except Exception as e:
            print(f"处理文件 {file_path.name} 时出错，已跳过: {e}")
            continue

        file_count += 1
        for item in items:
            results.append(item)
            print(f"[{item['file']}] {item['phoneme']}  "
                  f"{item['duration_ms']:.1f}ms "
                  f"({item['xmin']:.3f}s ~ {item['xmax']:.3f}s)"
                  f"  前:{item['prev'] or '-'}  后:{item['next'] or '-'}"
                  f"  层:{item['tier']}")

    if results:
        print(f"\n检测完成：共扫描 {file_count} 个 TextGrid 文件，"
              f"发现 {len(results)} 处时长小于 {threshold_ms:g}ms 的音素（建议复核标记）")
        # 最后一行：可直接粘贴到 Windows 搜索框的内容（文件名去重，用 OR 连接）
        print(build_search_line(results))
    else:
        print(f"\n检测完成：共扫描 {file_count} 个 TextGrid 文件，"
              f"未发现时长小于 {threshold_ms:g}ms 的音素")

    return results, file_count


if __name__ == '__main__':
    # 用法: python detect_short_phonemes.py <TextGrid文件夹> [阈值ms] [--ignore-edge-sp]
    args = [a for a in sys.argv[1:] if a != '--ignore-edge-sp']
    folder = args[0] if len(args) > 0 else r'E:\OpenUtau\Singers\白锋_02\E3\TextGrid'
    threshold = float(args[1]) if len(args) > 1 else 50
    run(folder, threshold, ignore_edge_sp='--ignore-edge-sp' in sys.argv)
