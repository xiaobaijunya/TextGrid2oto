#主要任务，删除AP,SP以及其他DS音素中不存在的音素
#读取DS字典
#把内容转换为拼音
#分辨V和CV

import json

#音素获取
def ds_dict_read(ds_dictpath,ignore):
    vowels = []
    consonant =[]
    with open(ds_dictpath, 'r') as f:
        for line in f:
            line = line.split()
            if len(line) == 3:
                consonant.append(line[1])
                vowels.append(line[2])
            elif len(line) == 4:
                consonant.append(line[2])
                vowels.append(line[2])
                vowels.append(line[3])
            elif len(line) == 2:
                vowels.append(line[1])
    ignore = ignore.split(',')
    #忽略音素
    vowels = [vowel for vowel in vowels if vowel not in ignore]
    consonant = [con for con in consonant if con not in ignore]
    vowels = set(vowels)
    consonant = set(consonant)
    print(len(consonant),consonant)
    print(len(vowels),vowels)
    return vowels,consonant

#删除不存在的音素
#传入json数据和有效列表
def filter_json_data(json_data, valid_list):
    for key, value in json_data.items():
        phones = value.get('phones', {})
        keys_to_remove = []
        for phone_key, phone_value in phones.items():
            text = phone_value.get('text')
            if text and text not in valid_list:
                keys_to_remove.append(phone_key)
        for key_to_remove in keys_to_remove:
            del phones[key_to_remove]
    json_data=reorganize_json_data(json_data)
    return json_data

#添加-和R，首尾音素
def reorganize_json_data(json_data):
    for audio_file, data in json_data.items():
        phones = data.get('phones', {})
        wav_long = data.get('wav_long', [])

        if not phones:
            continue

        # 获取排序后的phone keys
        sorted_keys = sorted(phones.keys(), key=lambda x: int(x))

        # 处理第一个元素
        first_key = sorted_keys[0]
        first_phone = phones[first_key]
        # if float(first_phone['xmin']) != 0:
        new_phone = {
            "xmin": "0.0",
            "xmax": first_phone['xmin'],
            "text": "-"
        }
        # 创建新的有序字典
        new_phones = {"1": new_phone}
        for i, key in enumerate(sorted_keys, 2):
            new_phones[str(i)] = phones[key]
        data['phones'] = new_phones
        sorted_keys = sorted(new_phones.keys(), key=lambda x: int(x))  # 更新排序后的keys

        # 处理最后一个元素
        last_key = sorted_keys[-1]
        last_phone = data['phones'][last_key]
        wav_end = wav_long[1] if len(wav_long) >= 2 else 0
        if float(last_phone['xmax']) < wav_end:
            new_phone = {
                "xmin": last_phone['xmax'],
                "xmax": str(wav_end),
                "text": "R"
            }
            # 添加新的最后一个元素
            new_index = str(int(sorted_keys[-1]) + 1)
            data['phones'][new_index] = new_phone

    return json_data



def run(ds_dict,json_path,ignore):
    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    #音素生成
    valid_list = ds_dict_read(ds_dict,ignore)
    # print(valid_list)
    #过滤
    filtered_data = filter_json_data(json_data, valid_list[0].union(valid_list[1]))
    # 将过滤后的数据写回文件
    # print(filtered_data)
    newjson_path=json_path.split('.json')[0]+'_filter.json'
    # print(newjson_path)
    with open(newjson_path, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=4)
        print('写入成功')

if __name__ == '__main__':
    json_path = r'E:\OpenUtau\Singers\XIABAI_new_CHN_CVVC_F3_autooto\F3/TextGrid\json\ds_phone.json'
    ds_dict = 'F:/aising\SOFA_AI-main\sofa_ckpt\Mandarin_three-stage\Mandarin_three-stage_dictionary_New.txt'
    ignore = 'AP,SP'
    run(ds_dict,json_path,ignore)
