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
    # 记录是否有空白被转换为 SP（新标注模型不再标记 SP）
    blank_converted = False

    for tier in tiers:
        # 提取所有 intervals
        raw_intervals = re.findall(r'intervals \[\d+\]:\s+xmin = ([\d.]+)\s+xmax = ([\d.]+)\s+text = "([^"]*)"', tier)
        # 新标注模型不再标记 SP：文本为空（或纯空白）的区间统一按 "SP" 处理，沿用原有逻辑
        if any(not text.strip() for (_, _, text) in raw_intervals):
            blank_converted = True
        intervals = [(xmin, xmax, text if text.strip() else "SP") for (xmin, xmax, text) in raw_intervals]
        new_intervals = []
        i = 1

        # if not intervals:  # 若intervals为空，直接跳过当前tier的后续处理
        #     processed_tiers.append(tier)  # 保留原始tier内容（也可根据需求跳过）
        #     print(f"{file_path}textgrid文件为空")
        #     return content, sp_deleted

        new_intervals.append(intervals[0])
        while i < len(intervals):
            # delete_sp 为 False 时（如含 R 的文件）只做空白->SP 归一化，不删除任何 SP
            if delete_sp and i < len(intervals)-1 and intervals[i][2] == "SP":
                # 检查前一个音和后一个音
                prev_phoneme = intervals[i-1][2]
                next_phoneme = intervals[i][2]
                # 如果前一个音或后一个音是AP、EP或SP，则不删除当前SP区间
                if prev_phoneme in ignore and next_phoneme in ignore:
                    new_intervals.append(intervals[i])
                    if delete_sp:
                        pass
                        # print(f"{file_path}已保留 SP 区间: {intervals[i][0]} - {intervals[i][1]}")
                    i += 1

                else:
                    # 中间的 SP 区间，删除并调整前一个区间的 xmax
                    if new_intervals:
                        #前音向后移动
                        new_intervals[-1] = (new_intervals[-1][0], intervals[i + 1][0], new_intervals[-1][2])
                        #后音向前移动
                        # new_intervals.append((intervals[i][0], intervals[i+1][1], intervals[i+1][2]))
                    if delete_sp:
                        print(f"{file_path}已删除 SP 区间: {intervals[i][0]} - {intervals[i][1]}")
                    i += 1

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

    # 返回处理后的内容、是否有SP被删除、是否有空白被转为SP
    return processed_content, sp_deleted, blank_converted


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
        # 含 R 的文件：过一遍把空白改为 SP，但不删除 SP
        is_R = 'R' in file_path.name

        try:
            # 处理文件（含 R 时关闭删除，仅做空白->SP 归一化）
            processed_content, sp_deleted, blank_converted = process_textgrid(file_path, ignore, delete_sp and not is_R)

            if is_R:
                print(f"已处理并覆盖（含R，仅将空白改为SP，不删SP）{file_path}")
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(processed_content)
                continue

            # 如果有SP被删除，记录文件名（不包含后缀）
            if sp_deleted:
                files_with_deleted_sp.append(file_path.stem)

            # 只要有改动（删除了SP，或把空白转成了SP）就覆盖原文件
            if sp_deleted or blank_converted:
                print(f"已处理并覆盖 {file_path}")
                # pass
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(processed_content)
        except Exception as e:
            # 捕获所有异常，记录错误文件并跳过
            files_with_deleted_sp.append(f"{file_path.name}: {str(e)}")
            print(f"处理文件 {file_path.name} 时出错，已跳过: {str(e)}")
            continue

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
    input_directory = r'E:\OpenUtau\Singers\白锋_02'
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