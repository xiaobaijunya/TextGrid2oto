import json
from decimal import Decimal

def dic_read(dic_path):
    dic = {}
    with open(dic_path) as f:
        list = f.readlines()
    for i in list:
        text = i.split('\n')[0].split(' ')
        dic[text[0]]=text[1:]
        # if len(text) == 3:
        #     print(text)
    dic['R']=['sil']
    dic['SP']=['sil']
    return dic

def word_phone_json_read(json_path):
    with open(json_path,encoding='utf-8') as f:
        data = json.load(f)
    return data

def word2phone(text,dic):
    word = text['text']
    xmin = text['xmin']
    middle = text['middle']
    xmax = text['xmax']
    try:
        """
        ai	a	:\i
        ei	e	:\i
        an	a	:n
        en	@	:n
        ang	A	N
        ong	U	N
        eng	@	N
        er	a	r\`
        """
        phone = dic[word]
        if len(phone) == 1:
            return phone,[xmin]
        elif word in ['ai','ei','an','en','ang','ong','eng','er']:
            end = Decimal(xmin) + (Decimal(xmax) - Decimal(xmin)) / Decimal(2)
            return phone, [xmin, end]
        elif len(phone) == 2:
            return phone,[xmin,middle]
        elif len(phone) == 3 :
            end = Decimal(middle) + (Decimal(xmax) - Decimal(middle)) / Decimal(2)
            return phone, [xmin, middle, end]
        else:
            print('长度错误！！！')
    except:
        print('没有找到%s'%word)
        return word,[xmin]


def run(dic_path,json_path,wav_path):
    dic = dic_read(dic_path)
    data = word_phone_json_read(json_path)

    for wavname, content in data.items():
        wavname = wavname.split('.')[0]
        phonemes = []
        boundaries = []
        for i in content['phones'].values():
            j = word2phone(i, dic)
            phonemes.append(j[0])
            boundaries.append(j[1])
        boundaries.append([content['wav_long'][1]])

        phonemes_list = []
        for sub_list in phonemes:
            for num_str in sub_list:
                phonemes_list.append(num_str)
        boundaries_list = []
        for sub_list in boundaries:
            for num_str in sub_list:
                boundaries_list.append(str(num_str))
        if wavname == '_bai_fa_a_ai_nang_lia':
            print(123)
        data_svdb = {
        "filename": wavname+'.wav',
        "phonemes": phonemes_list,
        "boundaries": boundaries_list
    }
        with open(wav_path+'\\'+wavname+'.json', 'w',encoding='utf-8') as f:
            json.dump(data_svdb, f)



if __name__ == "__main__":
    dic_path = r'字典/mandarin-xsampa-dict.txt'
    json_path = r'F:\Download\SVDBCreator_Release_1.0.0\xiaxiaobai\wav\D4\TextGrid\json'+'\word_phone.json'
    wav_path = r'F:\Download\SVDBCreator_Release_1.0.0\xiaxiaobai\wav\D4'


    print('-' * 15 + '欢迎使用TG2SVDB' + '-' * 15)
    run(dic_path,json_path,wav_path)