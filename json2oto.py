import json

#将json转换为oto
#文件名=别名,左边界,固定,右边界（负值）,预发声,交叉
#autio_name=phone_name,left,fixed,right（负值）,Prevoice,cross
#CV(V),VC(VV)(VR),-CV（-V）


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
            # -CV规则
            if cont['text'] == '-':
                key1, cont2 = sorted_phones[i + 1]
                phone_name = '- '+cont2['text']
                # autio_name=phone_name,left,fixed,right（负值）,Prevoice,cross
                left = float(cont2['xmin'])*1000/sum[0]
                #右线占比
                right = (float(cont2['xmax'])-float(cont2['xmin']))*1000/sum[2]
                #固定的占比
                fixed = float(right)/sum[1]
                Prevoice = (float(cont2['middle']) - float(cont2['xmin'])) * 1000 / sum[3]
                cross = float(Prevoice)/sum[4]
                i+=2
                oto.append(f"{autio_name}={phone_name},{left},{fixed},-{right},{Prevoice},{cross}\n")
                continue
            #CV规则
            phone_name =cont['text']
            # autio_name=phone_name,left,fixed,right（负值）,Prevoice,cross
            left = float(cont['xmin']) * 1000 / sum[0]
            # 右线占比
            right = (float(cont['xmax']) - float(cont['xmin'])) * 1000 / sum[2]
            # 固定的占比
            fixed = float(right) / sum[1]
            Prevoice = (float(cont['middle'])- float(cont['xmin'])) * 1000 / sum[3]
            cross = float(Prevoice) / sum[4]
            i += 1
            print(cross)
            oto.append(f"{autio_name}={phone_name},{left},{fixed},-{right},{Prevoice},{cross}\n")
            continue

    return oto






def run(utau_phone_json,word_phone_json,wav_path,sum):
    with open(utau_phone_json, 'r', encoding='utf-8') as f:
        vc_data = json.load(f)
    with open(word_phone_json, 'r', encoding='utf-8') as f:
        cv_data = json.load(f)
    oto = json2cvoto(cv_data,sum)
    # print(oto)
    with open(wav_path+'/oto.ini', 'w', encoding='utf-8') as f:
        for i in oto:
            f.write(i)
        print('oto.ini生成成功')


if __name__ == '__main__':
    utau_phone = 'G:/编程/utau自动标注/F3/TextGrid/json/utau_phone.json'
    word_phone_json = 'G:/编程/utau自动标注/F3/TextGrid/json/word_phone.json'
    wav_path = 'G:/编程/utau自动标注/F3'
    #-CV和CV规则：左线占比,固定的占比,右线占比,预发声不变,交叉占比
    sum = [1,3,1.5,1,2]
    run(utau_phone,word_phone_json,wav_path,sum)
