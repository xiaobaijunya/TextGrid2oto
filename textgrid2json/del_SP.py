import re
from pathlib import Path

def process_textgrid(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 按 tiers 分割内容
    tiers = re.split(r'item \[\d+\]:', content)[1:]
    processed_tiers = []

    for tier in tiers:
        # 提取所有 intervals
        intervals = re.findall(r'intervals \[\d+\]:\s+xmin = ([\d.]+)\s+xmax = ([\d.]+)\s+text = "([^"]+)"', tier)
        new_intervals = []
        i = 0
        while i < len(intervals):
            if i > 0 and i < len(intervals) - 1 and intervals[i][2] == "SP":
                # 中间的 SP 区间，删除并调整前一个区间的 xmax
                if new_intervals:
                    new_intervals[-1] = (new_intervals[-1][0], intervals[i + 1][0], new_intervals[-1][2])
                i += 1
            else:
                new_intervals.append(intervals[i])
                i += 1

        # 重新构建 tier 内容
        new_tier = re.sub(r'intervals: size = \d+', f'intervals: size = {len(new_intervals)}', tier)
        interval_texts = []
        for j, (xmin, xmax, text) in enumerate(new_intervals, start=1):
            interval_text = f'            intervals [{j}]:\n                xmin = {xmin}\n                xmax = {xmax}\n                text = "{text}"'
            interval_texts.append(interval_text)
        interval_section = '\n'.join(interval_texts)
        new_tier = re.sub(r'(intervals: size = \d+\n)(.*?)(?=item \[\d+\]|$)', f'\\1{interval_section}\n', new_tier, flags=re.DOTALL)
        processed_tiers.append(new_tier)

    # 重新组合处理后的内容
    header = re.match(r'(.*?)item \[\d+\]:', content, re.DOTALL).group(1)
    processed_content = header + ''.join([f'item [{i + 1}]:{tier}' for i, tier in enumerate(processed_tiers)])

    return processed_content
def process_all_textgrid_files(input_dir):
    """
    遍历输入文件夹下所有 TextGrid 文件，处理后直接覆盖原文件内容。

    :param input_dir: 包含 TextGrid 文件的输入文件夹路径
    """
    # 遍历输入文件夹下所有 TextGrid 文件
    for file_path in Path(input_dir).rglob('*.TextGrid'):
        try:
            # 处理文件
            processed_content = process_textgrid(file_path)

            # 直接覆盖原文件内容
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(processed_content)
            print(f"已处理并覆盖 {file_path}")
        except Exception as e:
            print(f"处理 {file_path} 时出错: {e}")

if __name__ == "__main__":
    # 指定输入文件夹路径
    input_directory = r'F:\Download\utau数据集\日语粗标\TextGrid'
    process_all_textgrid_files(input_directory)


# # 读取文件
# file_path = r'F:\Download\utau数据集\日语粗标\TextGrid\0e24a3b8a39c5b6d6503a1d1cafe5a29441089b5b4947c56b2334b69ebdcd7fc.TextGrid'
# processed_content = process_textgrid(file_path)
#
# # 保存处理后的内容
# output_file_path = 'processed_textgrid.txt'
# with open(output_file_path, 'w', encoding='utf-8') as f:
#     f.write(processed_content)