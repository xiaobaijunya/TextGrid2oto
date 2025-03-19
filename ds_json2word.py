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
    cv_map = {}  # 辅音+元音 -> CV拼音
    v_map = {}  # 单独元音 -> V拼音
    with open(ds_dictpath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                cv_map[(parts[1], parts[2])] = parts[0]  # 辅音+元音映射
            elif len(parts) == 2:
                v_map[parts[1]] = parts[0]  # 单独元音映射
    print(cv_map, v_map)
    return cv_map, v_map


def phones2word(json_data,ds_dictpath):
    cv_map, v_map = build_pinyin_map(ds_dictpath)

    for audio_file, data in json_data.items():
        phones = data.get('phones', {})
        if not phones:
            continue

        sorted_phones = sorted(phones.items(), key=lambda x: int(x[0]))
        new_phones = {}
        phone_counter = 1
        i = 0

        while i < len(sorted_phones):
            key, curr = sorted_phones[i]
            if curr['text'] in ('-', 'R'):
                new_phones[str(phone_counter)] = {
                    "xmin": curr['xmin'],
                    "middle": curr['xmin'],
                    "xmax": curr['xmax'],
                    "text": curr['text']
                }
                phone_counter += 1
                i += 1
                continue
            # 处理CV组合
            if i + 1 < len(sorted_phones):
                next_key, next_phone = sorted_phones[i + 1]
                cv_key = (curr['text'], next_phone['text'])
                if cv_key in cv_map:
                    # 合并两个音素
                    new_phones[str(phone_counter)] = {
                        "xmin": curr['xmin'],
                        "middle": next_phone['xmin'],  # 元音起始时间
                        "xmax": next_phone['xmax'],
                        "text": cv_map[cv_key]
                    }
                    phone_counter += 1
                    i += 2
                    continue

            # 处理单独元音
            if curr['text'] in v_map:
                new_phones[str(phone_counter)] = {
                    "xmin": curr['xmin'],
                    "middle": curr['xmin'],  # 复制相同值
                    "xmax": curr['xmax'],
                    "text": v_map[curr['text']]
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
    ds_dictpath = 'Baimu Delta/japanese-dictionary.txt'
    json_path = 'Baimu Delta/TextGrid/json/ds_phone_filter.json'
    run(ds_dictpath,json_path)