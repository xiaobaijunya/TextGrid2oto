import csv
import json
import os
import re

def create_transcriptions_csv(folder_path):
    csv_file_path = os.path.join(folder_path, 'transcriptions.csv')
    existing_data = []
    existing_names = set()

    # 如果 CSV 文件存在，读取其内容
    if os.path.exists(csv_file_path):
        with open(csv_file_path, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            headers = next(reader)  # 读取表头
            existing_data.append(headers)
            for row in reader:
                existing_data.append(row)
                existing_names.add(row[0])

    new_data = [] if existing_data else [['name', 'ph_seq', 'ph_dur', 'lab']]

    # 遍历文件夹中的所有文件
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.wav'):
                # 去除 .wav 后缀
                name = os.path.splitext(file)[0]
                if name not in existing_names:
                    # 初始化 ph_seq 和 ph_dur 列（暂时为空）
                    ph_seq = ""
                    ph_dur = ""
                    # 查找对应的 .lab 文件
                    lab_file_path = os.path.join(root, f"{name}.lab")
                    if os.path.exists(lab_file_path):
                        # 读取 .lab 文件内容
                        with open(lab_file_path, 'r', encoding='utf-8') as lab_file:
                            lab = lab_file.read().strip()
                    else:
                        lab = ""
                    # 添加一行数据到列表
                    new_data.append([name, ph_seq, ph_dur, lab])

    # 合并现有数据和新数据
    all_data = existing_data + new_data if existing_data else new_data

    # 写入或更新 transcriptions.csv 文件
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(all_data)

if __name__ == '__main__':
    # 示例调用，需替换为实际文件夹路径
    folder_path = 'your_folder_path'
    create_transcriptions_csv(folder_path)