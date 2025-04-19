import json
import os

# def ds_dict_read(ds_dictpath):
#     ds_dict = {}
#     vowels = []
#     consonant =[]
#     CV_pinyin = []
#     V_pinyin = []
#     with open(ds_dictpath, 'r') as f:
#         for line in f:
#             line = line.split()
#             if len(line) == 3:
#                 CV_pinyin.append(line[0])
#                 consonant.append(line[1])
#                 vowels.append(line[2])
#             elif len(line) == 2:
#                 V_pinyin.append(line[0])
#                 vowels.append(line[1])
#     CV_pinyin = set(CV_pinyin)
#     V_pinyin = set(V_pinyin)
#     vowels = set(vowels)
#     consonant = set(consonant)
#     # print(len(CV_pinyin),CV_pinyin)
#     # print(len(V_pinyin),V_pinyin)
#     # print(len(consonant),consonant)
#     # print(len(vowels),vowels)
#     return vowels,consonant,CV_pinyin,V_pinyin


# 新增拼音映射字典
def build_pinyin_map(ds_dictpath):
    phone_map = {}  # 音素序列 -> 拼音
    max_length = 0
    with open(ds_dictpath, 'r',encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                key = tuple(parts[1:])  # 音素组合作为元组
                phone_map[key] = parts[0]
                max_length = max(max_length, len(key))
    # print("当前字典内容:", phone_map)
    print("最大音素长度:", max_length)
    return phone_map, max_length


def phones2word(json_data,ds_dictpath):
    phone_map, max_length = build_pinyin_map(ds_dictpath)

    for audio_file, data in json_data.items():
        phones = data.get('phones', {})
        if not phones:
            continue

        sorted_phones = sorted(phones.items(), key=lambda x: int(x[0]))
        new_phones = {}
        phone_counter = 1
        i = 0

        while i < len(sorted_phones):
            current_phone = sorted_phones[i][1]
            current_text = current_phone['text']

            # 处理特殊符号（新增冒号类符号判断）
            if current_text in ('-', 'R'):
                new_phones[str(phone_counter)] = {
                    "xmin": current_phone['xmin'],
                    "middle": current_phone['xmin'],
                    "xmax": current_phone['xmax'],
                    "text": current_text
                }
                phone_counter += 1
                i += 1
                continue

            matched = False
            candidates = []
            max_possible = min(max_length, len(sorted_phones) - i)
            for k in range(max_possible, 0, -1):
                # 修改2：允许包含冒号的音素参与组合
                candidate = tuple([sorted_phones[i + m][1]['text'] for m in range(k)])
                # print(f"尝试匹配: i={i}, k={k}, 序列={candidate}")
                candidates.append((k, candidate))  # 移除过滤条件

            # 优化2：按候选长度降序处理
            for k, current_sequence in sorted(candidates, key=lambda x: -x[0]):
                # print(f"正式匹配尝试: i={i}, k={k}, 序列={current_sequence}")
                if current_sequence in phone_map:
                    # print(f"匹配成功: i={i}, k={k}, 序列={current_sequence}")
                    # 修复缩进问题
                    start_phone = sorted_phones[i][1]
                    end_phone = sorted_phones[i + k - 1][1]
                    if start_phone['xmax'] == end_phone['xmax']:
                        new_phones[str(phone_counter)] = {
                            "xmin": start_phone['xmin'],
                            "middle": start_phone['xmin'],
                            "xmax": end_phone['xmax'],
                            "text": phone_map[current_sequence]
                        }
                        phone_counter += 1
                        i += k
                        matched = True
                        break
                    new_phones[str(phone_counter)] = {
                        "xmin": start_phone['xmin'],
                        "middle": start_phone['xmax'],
                        "xmax": end_phone['xmax'],
                        "text": phone_map[current_sequence]
                    }
                    phone_counter += 1
                    i += k
                    matched = True
                    break
                    # 合并k个音素
                    start_phone = sorted_phones[i][1]
                    end_phone = sorted_phones[i + k - 1][1]
                    new_phones[str(phone_counter)] = {
                        "xmin": start_phone['xmin'],
                        "middle": start_phone['xmax'],
                        "xmax": end_phone['xmax'],
                        "text": phone_map[current_sequence]
                    }
                    phone_counter += 1
                    i += k
                    matched = True
                    break
            # ... existing special character handling ...
            if not matched:
                # 无法合并则单独处理当前音素
                print(f"无法合并的音素：{current_text}")
                current_phone = sorted_phones[i][1]
                new_phones[str(phone_counter)] = {
                    "xmin": current_phone['xmin'],
                    "middle": current_phone['xmin'],
                    "xmax": current_phone['xmax'],
                    "text": current_phone['text']
                }
                phone_counter += 1
                i += 1
            # 用新结构替换原phones数据
        data['phones'] = new_phones
        # print(new_phones)

def run(ds_dictpath,json_path):
    # valid_list = ds_dict_read(ds_dictpath)
    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    phones2word(json_data,ds_dictpath)
    new_path = os.path.dirname(json_path)+'/word_phone.json'
    with open(new_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)
    print(f'合并后的文件已保存至: {new_path}')


if __name__ == "__main__":
    ds_dictpath = 'F:/aising\SOFA_AI-main\sofa_ckpt\Mandarin_three-stage\Mandarin_three-stage_dictionary_New.txt'
    json_path = 'E:\OpenUtau\Singers\XIABAI_new_CHN_CVVC_F3_autooto\F3/TextGrid/json/ds_phone_filter.json'
    run(ds_dictpath,json_path)