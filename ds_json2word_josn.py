#主要任务，删除AP,SP以及其他DS音素中不存在的音素
#读取DS字典
#把内容转换为拼音
#分辨V和CV

import json

if __name__ == '__main__':
    path='F3/TextGrid/json/ds_phone.json'
    ds_dict = 'opencpop-extension.txt'
    with open('ds_dict.json', 'r', encoding='utf-8') as f:
        ds_phones = json.load(f)