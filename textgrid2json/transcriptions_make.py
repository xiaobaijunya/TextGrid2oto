import csv
import json
import os
import re

def ds_dict_read(ds_dictpath):
    ds_dict = {}
    with open(ds_dictpath, 'r',encoding='utf-8') as f:
        word_list = f.read().strip().split('\n')
        for word in word_list:
            word = word.split('\t')
            ds_dict[word[0]] = word[1]
    return ds_dict

def create_transcriptions_csv(folder_path,ds_dictpath):
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
                name = row[0]
                # lab = row[3]
                # 如果 lab 列为空，尝试查找对应的 .lab 文件
                # if not lab:
                #     lab_file_path = os.path.join(folder_path, f"{name}.lab")
                #     if os.path.exists(lab_file_path):
                #         with open(lab_file_path, 'r', encoding='utf-8') as lab_file:
                #             row[3] = lab_file.read().strip()
                # existing_data.append(row)
                existing_names.add(name)
    else:
        # 如果 CSV 文件不存在，创建一个空的 CSV 文件
        with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['name', 'ph_seq', 'ph_dur', 'words'])

    new_data = [] if existing_data else [['name', 'ph_seq', 'ph_dur', 'words']]

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
    ds_dict = ds_dict_read(ds_dictpath)



    # 合并现有数据和新数据
    all_data = existing_data + new_data if existing_data else new_data
    for i in range(len(all_data)):
        if all_data[i][3] == '':
            lab_file_path = os.path.join(folder_path, f"{all_data[i][0]}.lab")
            if os.path.exists(lab_file_path):
                with open(lab_file_path, 'r', encoding='utf-8') as lab_file:
                    all_data[i][3] = lab_file.read().strip()
            else:
                all_data[i][3] = ''
    for i in range(len(all_data)):
        phones = []
        if all_data[i][1] == '' and all_data[i][3] != '':
            for word in all_data[i][3].split(' '):
                if word in ds_dict:
                    phones.append(ds_dict[word])
                    # phones.append('R')
                else:
                    phones.append(word)
                    print(f"{all_data[i][0]}:{word}不存在于sofa字典中")
            phones.append('R')
            phones.append('SP')
            all_data[i][1] = ' '.join(phones)
            # print(all_data[i][1])
    # 写入或更新 transcriptions.csv 文件
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(all_data)

if __name__ == '__main__':
    # 示例调用，需替换为实际文件夹路径
    folder_path = r'F:\Download\utau数据集\有授权\中文\中文拼接数据_自有授权\baini_Chn_CVXY\E3'
    create_transcriptions_csv(folder_path,r'F:\Download\utau数据集\合并.txt')