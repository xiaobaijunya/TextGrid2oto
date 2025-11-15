import json
import re


#将json转换为oto
#文件名=别名,左边界,固定,右边界（负值）,预发声,交叉
#autio_name=phone_name,left,fixed,right（负值）,Prevoice,cross
#CV(V),VC(VV)(VR),-CV（-V）

def presamp_read(presamps_path):
    CV_V = {}
    CV_C = {}
    V_V = []
    with open(presamps_path, 'r',encoding='utf-8') as file:
        ini_text = file.read()
        # 提取 [VOWEL] 部分
        # 提取 [VOWEL] 部分
        vowel_match = re.search(r'\[VOWEL\](.*?)\[', ini_text, re.DOTALL)
        vowels = vowel_match.group(1).strip()
        # print(vowels)
        # 提取 [CONSONANT] 部分
        consonant_match = re.search(r'\[CONSONANT\](.*?)\[', ini_text, re.DOTALL)
        consonants = consonant_match.group(1).strip()
        # print(consonants)
    for vowel in vowels.split('\n'):
        V0=(vowel.split('=')[0])
        for vowel in vowel.split('=')[2].split(','):
            CV_V[vowel]=V0
    for consonant in consonants.split('\n'):
        C0=(consonant.split('=')[0])
        for consonant in consonant.split('=')[1].split(','):
            CV_C[consonant]=C0
    V_V = [key for key in CV_V if key not in CV_C]
    print(f"CV_V:{CV_V}\nCV_C:{CV_C}\n,V_V:{V_V}")
    return CV_V, CV_C,V_V

#VV_CV_VC
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
            if cont['text'] in ['-','R']:
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
                    fixed = (float(cont2['xmax'])-float(cont2['middle']))*1000/sum[1]+ Prevoice
                cross = float(Prevoice)/sum[4]
                i+=1
                if right <0:
                    print(f'{phone_name}右边界错误')
                    continue
                oto.append(f"{autio_name}={phone_name},{left},{fixed},-{right},{Prevoice},{cross}\n")
                continue
            if cont2['text'] in ['-','R','SP']:
                phone_name =cont['text'] + 'R'
                # autio_name=phone_name,left,fixed,right（负值）,Prevoice,cross
                left = float(cont['xmin']) * 1000 / sum[0]

                Prevoice = (float(cont['middle']) - float(cont['xmin'])) * 1000 / sum[3]
                # 右线占比
                right = (float(cont['xmax']) - float(cont['middle'])) * 1000 / sum[2] + Prevoice
                # 固定的占比
                if sum[1] == 0:
                    fixed = Prevoice
                else:
                    fixed = (float(cont['xmax']) - float(cont['middle'])) * 1000 / sum[1] + Prevoice
                cross = float(Prevoice) / sum[4]
                i += 1
                if right <0:
                    print(f'{phone_name}右边界错误')
                    continue
                oto.append(f"{autio_name}={phone_name},{left},{fixed},-{right},{Prevoice},{cross}\n")
                continue
            #CV规则
            phone_name =cont['text']
            # autio_name=phone_name,left,fixed,right（负值）,Prevoice,cross
            left = float(cont['xmin']) * 1000 / sum[0]

            Prevoice = (float(cont['middle'])- float(cont['xmin'])) * 1000 / sum[3]
            # 右线占比
            right = (float(cont['xmax']) - float(cont['middle'])) * 1000 / sum[2] + Prevoice
            # 固定的占比
            if sum[1] == 0:
                fixed = Prevoice
            else:
                fixed = (float(cont['xmax'])-float(cont['middle']))*1000/sum[1]+ Prevoice
            cross = float(Prevoice) / sum[4]
            i += 1

            oto.append(f"{autio_name}={phone_name},{left},{fixed},-{right},{Prevoice},{cross}\n")
            continue

    return oto


def json2vcoto(cv_data,CV_V, CV_C,V_V , vc_sum,vv_sum):

    oto = []
    for audio_file, data in cv_data.items():
        autio_name = audio_file
        phones = data.get('phones', {})
        long = data.get('long', [])
        if not phones:
            continue

        sorted_phones = sorted(phones.items(), key=lambda x: int(x[0]))
        i = 0

        while i < len(sorted_phones) - 1:
            key, cont = sorted_phones[i]
            key1, cont1 = sorted_phones[i + 1]
            if cont['text'] in ['-','R']:
                i += 1
                continue
            elif cont1['text'] in ['R','B'] and cont['text'] in CV_V:
                phone_name = CV_V[cont['text']] + ' ' + cont1['text']
                # autio_name=phone_name,left,fixed,right（负值）,Prevoice,cross
                left = float(cont["middle"]) * 1000 + ((float(cont['xmax']) - float(cont['middle'])) * 1000 / vc_sum[0])
                # 右线占比
                Prevoice = float(cont1['middle']) * 1000 - left / vc_sum[3]
                right = (float(cont1['xmax']) - float(cont1['middle'])) * 1000 / vc_sum[2] + Prevoice
                # 固定的占比
                if vc_sum[1] == 0:
                    fixed = Prevoice
                else:
                    fixed = Prevoice + (float(cont1['xmax']) - float(cont1['middle'])) * 1000 / vc_sum[1]
                cross = (float(cont['xmax']) - float(cont['middle'])) * 1000 / vc_sum[4]
                i += 1
                # print(f"{autio_name}={phone_name},{left},{fixed},-{right},{Prevoice},{cross}\n")
                oto.append(f"{autio_name}={phone_name},{left},{fixed},-{right},{Prevoice},{cross}\n")
            elif cont1['text'] in V_V and cont['text'] in CV_V:
                phone_name = CV_V[cont['text']] + ' ' + cont1['text']
                # autio_name=phone_name,left,fixed,right（负值）,Prevoice,cross
                left = float(cont["middle"]) * 1000 + ((float(cont['xmax']) - float(cont['middle'])) * 1000 / vv_sum[0])
                # 右线占比
                Prevoice = float(cont1['middle']) * 1000 - left / vv_sum[3]
                right = (float(cont1['xmax']) - float(cont1['middle'])) * 1000 / vv_sum[2] + Prevoice
                # 固定的占比
                if vv_sum[1] == 0:
                    fixed = Prevoice
                else:
                    fixed = Prevoice + (float(cont1['xmax']) - float(cont1['middle'])) * 1000 / vv_sum[1]
                cross = (float(cont['xmax']) - float(cont['middle'])) * 1000 / vv_sum[4]
                i += 1
                oto.append(f"{autio_name}={phone_name},{left},{fixed},-{right},{Prevoice},{cross}\n")
            elif cont1['text'] in CV_C and cont['text'] in CV_V:
                phone_name = CV_V[cont['text']] + ' ' + CV_C[cont1['text']]
                # autio_name=phone_name,left,fixed,right（负值）,Prevoice,cross
                left = float(cont["middle"]) * 1000 + ((float(cont['xmax']) - float(cont['middle'])) * 1000 / vc_sum[0])
                # 右线占比
                Prevoice = float(cont1['xmin']) * 1000 - left / vc_sum[3]
                right = (float(cont1['middle']) - float(cont1['xmin'])) * 1000 / vc_sum[2] + Prevoice
                #如果预发声等于右线（当纯V被当成VC用的时候）
                if Prevoice == right:
                    if Prevoice <= 20:
                        right += 20
                    else:
                        Prevoice-=20

                # 固定的占比
                if vc_sum[1] == 0:
                    fixed = Prevoice
                else:
                    fixed = Prevoice + (float(cont1['middle']) - float(cont1['xmin'])) * 1000 / vc_sum[1]
                cross = (float(cont['xmax']) - float(cont['middle'])) * 1000 / vc_sum[4]
                i += 1
                oto.append(f"{autio_name}={phone_name},{left},{fixed},-{right},{Prevoice},{cross}\n")
            else:
                i += 1
                continue
    return oto



def run(presamp_path,utau_phone_json,word_phone_json,wav_path,cv_sum,vc_sum,vv_sum):
    CV_V, CV_C,V_V = presamp_read(presamp_path)
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
    oto = json2vcoto(cv_data,CV_V, CV_C,V_V , vc_sum,vv_sum)
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

    presamp_path = 'E:\OpenUtau\Singers\weina2\presamp.ini'
    utau_phone = 'E:\OpenUtau\Singers\weina2\F4\TextGrid\json/utau_phone.json'
    word_phone_json = 'E:\OpenUtau\Singers\weina2\F4\TextGrid\json/word_phone.json'
    wav_path = 'E:\OpenUtau\Singers\weina2\F4'

    #-CV和CV规则：左线占比,固定的占比,右线占比,预发声不变,交叉占比
    cv_sum = [1,3,1.5,1,2]
    #VC和VV规则：左线占比,固定的占比,右线占比,预发声不变,交叉占比,VV固定占比
    vc_sum=[3,0,2,1,2,3]
    vv_sum=[3,3,1.5,1,1.5]
    run(presamp_path,utau_phone,word_phone_json,wav_path,cv_sum,vc_sum,vv_sum)
