import json
import re
import os

#读取presamp.ini文件
def load_presamp(ini_path):
    word2phone = {}
    word_phone = {}
    with open(ini_path, 'r', encoding='utf-8') as file:
        ini_text = file.read()
        # 提取 [VOWEL] 部分
        vowel_match = re.search(r'\[VOWEL\](.*?)\[', ini_text, re.DOTALL)
        vowel_section = vowel_match.group(1).strip()
        # 提取 [CONSONANT] 部分
        consonant_match = re.search(r'\[CONSONANT\](.*?)\[', ini_text, re.DOTALL)
        consonant_section = consonant_match.group(1).strip()
    for vowel in vowel_section.split('\n'):
        vowel = vowel.split('=')
        phone = vowel[0]
        for v in vowel[2].split(','):
            word2phone.update({v:{'V':phone}})
    if consonant_section != '':
        for consonant in consonant_section.split('\n'):
            consonant2 = consonant.split('=')
            phone = consonant2[0]
            for c in consonant2[1].split(','):
                word2phone[c].update({'C': phone})
    else:
        for vowel in vowel_section.split('\n'):
            vowel = vowel.split('=')
            for c in vowel[2].split(','):
                word2phone[c].update({'C': c})
    for key, value in word2phone.items():
        if 'C' in value:
            word_phone[key]=[value['C'],value['V']]
        else:
            word_phone[key]=[value['V']]
    word_phone2 = dict(sorted(word_phone.items(), key=lambda item: len(item[0]), reverse=True))
    # print(word_phone2)
    return word_phone2


def split_pinyin_to_phones(word_data, mappings):
    """拆分单个拼音项为音素"""
    new_phones = {}
    phone_counter = 1

    for idx, (key, item) in enumerate(sorted(word_data['phones'].items(),
                                             key=lambda x: int(x[0]))):
        pinyin = item['text']

        # 保留特殊符号
        if pinyin in ('-', 'R'):
            new_phones[str(phone_counter)] = item
            phone_counter += 1
            continue

        # 获取音素分解
        phones = mappings.get(pinyin, [pinyin])

        # 时间分配逻辑
        if len(phones) == 2:  # CV结构
            # print(phones)
            # print(item)
            consonant = {

                "xmin": item['xmin'],
                "xmax": item['middle'],
                "text": phones[0]
            }
            vowel = {
                "xmin": item['middle'],
                "xmax": item['xmax'],
                "text": phones[1]
            }
            new_phones[str(phone_counter)] = consonant
            new_phones[str(phone_counter + 1)] = vowel
            phone_counter += 2
        else:  # 单个音素
            new_phones[str(phone_counter)] = {
                "xmin": item['xmin'],
                "xmax": item['xmax'],
                "text": phones[0]
            }
            phone_counter += 1

    return new_phones


def generate_utau_phone(presamp_path, word_json_path):

    word_phone = load_presamp(presamp_path)

    with open(word_json_path, 'r', encoding='utf-8') as f:
        word_data = json.load(f)

    for audio_file, data in word_data.items():
        data['phones'] = split_pinyin_to_phones(data, word_phone)
    new_path = os.path.dirname(word_json_path)+'/utau_phone.json'
    with open(new_path, 'w', encoding='utf-8') as f:
        json.dump(word_data, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    generate_utau_phone(
        presamp_path='E:\OpenUtau\Singers\空气音中文VCV_自动oto测试\中文VCV-presamp.ini',
        word_json_path='E:\OpenUtau\Singers\空气音中文VCV_自动oto测试/VCV/TextGrid/json/word_phone.json',
    )