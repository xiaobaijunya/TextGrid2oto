import json
import re


#将json转换为oto
#文件名=别名,左边界,固定,右边界（负值）,预发声,交叉
#autio_name=phone_name,left,fixed,right（负值）,Prevoice,cross
#CV(V),VC(VV)(VR),-CV（-V）

def presamp_read(presamps_path):
    V = {}
    with open(presamps_path, 'r',encoding='utf-8') as file:
        ini_text = file.read()
        # 提取 [VOWEL] 部分
        # 提取 [VOWEL] 部分
        vowel_match = re.search(r'\[VOWEL\](.*?)\[', ini_text, re.DOTALL)
        vowels = vowel_match.group(1).strip()
        # print(vowels)
        # 提取 [CONSONANT] 部分
        # consonant_match = re.search(r'\[CONSONANT\](.*?)\[', ini_text, re.DOTALL)
        # consonants = consonant_match.group(1).strip()
        # print(consonants)
    for vowel in vowels.split('\n'):
        v = vowel.split('=')[0]
        for v0 in vowel.split('=')[2].split(','):
            V[v0]=v
    # print(V)
    return V

def json2cvoto(cv_data,sum):
    oto = []
    for audio_file, data in cv_data.items():
        autio_name = audio_file
        phones = data.get('phones', {})
        long = data.get('long', [])
        if not phones:
            continue

        sorted_phones = sorted(phones.items(), key=lambda x: int(x[0]))
        i = 0

        while i < len(sorted_phones)-1:
            key, cont = sorted_phones[i]
            key1, cont2 = sorted_phones[i + 1]
            # -CV规则
            if cont['text'] in ['R', '-','SP','AP'] and cont2['text'] not in ['R','-','SP','AP']:
                phone_name = '- '+cont2['text']
                # autio_name=phone_name,left,fixed,right（负值）,Prevoice,cross
                left = float(cont2['xmin'])*1000/sum[0]

                Prevoice = (float(cont2['middle']) - float(cont2['xmin'])) * 1000 / sum[3]
                #右线占比
                right = (float(cont2['xmax'])-float(cont2['middle']))*1000/sum[2] + Prevoice
                # 固定的占比
                if sum[1] ==0:
                    fixed = Prevoice
                else:
                    fixed = Prevoice+(right-Prevoice)/sum[1]
                cross = float(Prevoice)/sum[4]
                i+=2
                # print(f"{autio_name}={phone_name},{left},{fixed},-{right},{Prevoice},{cross}\n")
                oto.append(f"{autio_name}={phone_name},{left},{fixed},-{right},{Prevoice},{cross}\n")
                continue
            # #CV规则
            # phone_name =cont['text']
            # # autio_name=phone_name,left,fixed,right（负值）,Prevoice,cross
            # left = float(cont['xmin']) * 1000 / sum[0]
            #
            # Prevoice = (float(cont['middle'])- float(cont['xmin'])) * 1000 / sum[3]
            # # 右线占比
            # right = (float(cont['xmax']) - float(cont['middle'])) * 1000 / sum[2] + Prevoice
            # # 固定的占比
            # if sum[1] == 0:
            #     fixed = Prevoice
            # else:
            #     fixed = Prevoice+(right-Prevoice)/sum[1]
            # cross = float(Prevoice) / sum[4]
            i += 1
            #
            # oto.append(f"{autio_name}={phone_name},{left},{fixed},-{right},{Prevoice},{cross}\n")

    return oto

def json2VCVoto(cv_data,CV_V,sum):
    oto = []
    for audio_file, data in cv_data.items():
        autio_name = audio_file
        phones = data.get('phones', {})
        long = data.get('long', [])
        if not phones:
            continue
        sorted_phones = sorted(phones.items(), key=lambda x: int(x[0]))
        i = 0
        while i < len(sorted_phones)-1:
            key, cont = sorted_phones[i]
            key1, cont1 = sorted_phones[i + 1]
            if cont['text'] in ['R', '-','SP','AP']:
                i+=1
                continue
            elif cont1['text'] in ['R', 'B','-','SP','AP'] and cont['text'] in CV_V:
                phone_name = CV_V[cont['text']] + ' ' + cont1['text']
                # autio_name=phone_name,left,fixed,right（负值）,Prevoice,cross
                left = float(cont["middle"]) * 1000 + ((float(cont['xmax']) - float(cont['middle'])) * 1000 / sum[0])
                # 右线占比
                Prevoice = float(cont1['middle']) * 1000 - left / sum[3]
                right = (float(cont1['xmax'])-float(cont1['middle']))*1000/sum[2] + Prevoice
                # 固定的占比
                if sum[1] ==0:
                    fixed = Prevoice
                else:
                    fixed = Prevoice + (float(cont1['xmax']) - float(cont1['middle'])) * 1000 / sum[1]
                cross = (float(cont['xmax']) - float(cont['middle'])) * 1000 / sum[4]
                i += 1
                # print(f"{autio_name}={phone_name},{left},{fixed},-{right},{Prevoice},{cross}\n")
                oto.append(f"{autio_name}={phone_name},{left},{fixed},-{right},{Prevoice},{cross}\n")
            elif  cont['text'] in CV_V and cont1['text'] in CV_V:
                phone_name = CV_V[cont['text']] + ' ' + cont1['text']
                # autio_name=phone_name,left,fixed,right（负值）,Prevoice,cross
                left = float(cont["middle"]) * 1000 + ((float(cont['xmax']) - float(cont['middle'])) * 1000 / sum[0])
                # 右线占比
                Prevoice = float(cont1['middle']) * 1000 - left / sum[3]
                right = (float(cont1['xmax'])-float(cont1['middle']))*1000/sum[2] + Prevoice
                # 固定的占比
                if sum[1] ==0:
                    fixed = Prevoice
                else:
                    fixed = Prevoice + (float(cont1['xmax']) - float(cont1['middle'])) * 1000 / sum[1]
                cross =(float(cont['xmax']) - float(cont['middle'])) *1000 / sum[4]
                i += 1
                oto.append(f"{autio_name}={phone_name},{left},{fixed},-{right},{Prevoice},{cross}\n")

            else:
                i+=1
                continue
    return oto

def run(presamp_path,utau_phone_json,word_phone_json,wav_path,cv_sum,vc_sum,vv_sum):
    CV_V = presamp_read(presamp_path)
    CV_V['R']='R'
    print(CV_V)
    # with open(utau_phone_json, 'r', encoding='utf-8') as f:
    #     vc_data = json.load(f)
    with open(word_phone_json, 'r', encoding='utf-8') as f:
        cv_data = json.load(f)
    oto = json2cvoto(cv_data,cv_sum)
    # print(oto)
    with open(wav_path+'/cv_oto.ini', 'w', encoding='utf-8') as f:
        for i in oto:
            f.write(i)
        print('cv_oto.ini生成成功')
    oto = json2VCVoto(cv_data,CV_V, vc_sum)
    # print(oto)
    with open(wav_path+'/vc_oto.ini', 'w', encoding='utf-8') as f:
        for i in oto:
            f.write(i)
        print('vc_oto.ini生成成功')


if __name__ == '__main__':
    # presamp_path = 'risku中文CVVCpresamp.ini'
    # utau_phone = 'G:/编程/utau自动标注/F3/TextGrid/json/utau_phone.json'
    # word_phone_json = 'G:/编程/utau自动标注/F3/TextGrid/json/word_phone.json'
    # wav_path = 'G:/编程/utau自动标注/F3'

    # presamp_path = 'presamp/樗儿式中文VCV-presamp.ini'
    # utau_phone = 'E:\\OpenUtau\\Singers\空气音中文VCV_自动oto测试/VCV/TextGrid/json/utau_phone.json'
    # word_phone_json = 'E:\\OpenUtau\\Singers\空气音中文VCV_自动oto测试/VCV/TextGrid/json/word_phone.json'
    # wav_path = 'E:\\OpenUtau\\Singers\空气音中文VCV_自动oto测试\VCV'

    presamp_path = r'F:\Textgrid2oto4\presamp\jp-hira-presamp.ini'
    utau_phone = r'E:\OpenUtau\Singers\BaiNi_JP_VCV_0.1\1.C3/TextGrid/json/utau_phone.json'
    word_phone_json = r'E:\OpenUtau\Singers\BaiNi_JP_VCV_0.1\1.C3/TextGrid/json/word_phone.json'
    wav_path = r'E:\OpenUtau\Singers\BaiNi_JP_VCV_0.1\1.C3'

    #-CV和CV规则：左线占比,固定的占比,右线占比,预发声不变,交叉占比
    cv_sum = [1,3,1.5,1,2]
    #VC和VV规则：左线占比,固定的占比,右线占比,预发声不变,交叉占比
    vc_sum=[1,3,1.5,1,2]
    vv_sum=[0,0,0,0,0]
    run(presamp_path,utau_phone,word_phone_json,wav_path,cv_sum,vc_sum,vv_sum)
