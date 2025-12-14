import re
from pathlib import Path


def process_textgrid(file_path,ignore,delete_sp):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 按 tiers 分割内容
    tiers = re.split(r'item \[\d+\]:', content)[1:]
    processed_tiers = []

    # 记录是否有SP被删除
    sp_deleted = False

    for tier in tiers:
        # 提取所有 intervals
        intervals = re.findall(r'intervals \[\d+\]:\s+xmin = ([\d.]+)\s+xmax = ([\d.]+)\s+text = "([^"]+)"', tier)
        new_intervals = []
        i = 1
        new_intervals.append(intervals[0])
        while i < len(intervals):
            if i < len(intervals)-1 and intervals[i][2] == "SP":
                # 检查前一个音和后一个音
                prev_phoneme = intervals[i-1][2]
                next_phoneme = intervals[i][2]
                if prev_phoneme == "AP":
                    print('123')
                # 如果前一个音或后一个音是AP、EP或SP，则不删除当前SP区间
                if prev_phoneme in ignore and next_phoneme in ignore:
                    new_intervals.append(intervals[i])
                    if delete_sp:
                        print(f"{file_path}已保留 SP 区间: {intervals[i][0]} - {intervals[i][1]}")
                    i += 1

                else:
                    # 中间的 SP 区间，删除并调整前一个区间的 xmax
                    if new_intervals:
                        #前音向后移动
                        # new_intervals[-1] = (new_intervals[-1][0], intervals[i + 1][0], new_intervals[-1][2])
                        #后音向前移动
                        new_intervals.append((intervals[i][0], intervals[i+1][1], intervals[i+1][2]))
                    if delete_sp:
                        print(f"{file_path}已删除 SP 区间: {intervals[i][0]} - {intervals[i][1]}")
                    i += 2

                    sp_deleted = True  # 标记有SP被删除
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
        new_tier = re.sub(r'(intervals: size = \d+\n)(.*?)(?=item \[\d+\]|$)', f'\\1{interval_section}\n', new_tier,
                          flags=re.DOTALL)
        processed_tiers.append(new_tier)

    # 重新组合处理后的内容
    header = re.match(r'(.*?)item \[\d+\]:', content, re.DOTALL).group(1)
    processed_content = header + ''.join([f'item [{i + 1}]:{tier}' for i, tier in enumerate(processed_tiers)])

    # 返回处理后的内容和是否有SP被删除的标记
    return processed_content, sp_deleted


def process_all_textgrid_files(input_dir,ignore,delete_sp):
    """
    遍历输入文件夹下所有 TextGrid 文件，处理后直接覆盖原文件内容。

    :param input_dir: 包含 TextGrid 文件的输入文件夹路径
    """
    # 记录所有被删除SP的文件名（不包含后缀）
    files_with_deleted_sp = []
    ignore = ignore.split(',')
    # 遍历输入文件夹下所有 TextGrid 文件
    for file_path in Path(input_dir).rglob('*.TextGrid'):
        if 'R' in file_path.name:
            print(f"跳过文件 {file_path.name} (文件名包含大写字母'R')")
            continue


        # 处理文件
        processed_content, sp_deleted = process_textgrid(file_path,ignore,delete_sp)

        # 如果有SP被删除，记录文件名（不包含后缀）
        if sp_deleted:
            files_with_deleted_sp.append(file_path.stem)

        # 直接覆盖原文件内容
        if delete_sp:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(processed_content)
        # print(f"已处理并覆盖 {file_path}")


    # 最后统一输出所有被删除SP的文件名
    if files_with_deleted_sp:
        print("\n以下文件中有SP被删除（开启删除sp才生效）（建议复核标记）:")
        for filename in files_with_deleted_sp:
            print(f"{filename}",end=',')
        return files_with_deleted_sp
    else:
        print("\n没有文件中有SP被删除。")
        return ['没有文件中有SP被删除','无错误']
    print(" ")


if __name__ == "__main__":
    # 指定输入文件夹路径
    input_directory = r'E:\OpenUtau\Singers\bainizh_2025.11.29\E3\TextGrid'
    ignore = 'AP,SP,EP'
    delete_sp = True
    process_all_textgrid_files(input_directory,ignore,delete_sp)

# # 读取文件
# file_path = r'F:\Download\utau数据集\日语粗标\TextGrid\0e24a3b8a39c5b6d6503a1d1cafe5a29441089b5b4947c56b2334b69ebdcd7fc.TextGrid'
# processed_content = process_textgrid(file_path)
#
# # 保存处理后的内容
# output_file_path = 'processed_textgrid.txt'
# with open(output_file_path, 'w', encoding='utf-8') as f:
#     f.write(processed_content)