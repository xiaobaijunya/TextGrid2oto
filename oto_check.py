import re
import oto
def cvvc_presamp_read(presamps_path):
    V = []
    C = []
    VC = []
    VV = []
    CV=[]
    CV_C=[]
    CV_V=[]
    with open(presamps_path, 'r') as file:
        ini_text = file.read()
        # 提取 [VOWEL] 部分
        # 提取 [VOWEL] 部分
        vowel_match = re.search(r'\[VOWEL\](.*?)\[', ini_text, re.DOTALL)
        vowels = vowel_match.group(1).strip()
        for vowel in vowels.split('\n'):
            V.append(vowel.split('=')[0])
            for CV1 in vowel.split('=')[2].split(','):
                if CV1 != '':
                    CV.append(CV1)
        # print(vowels)
        # 提取 [CONSONANT] 部分
        consonant_match = re.search(r'\[CONSONANT\](.*?)\[', ini_text, re.DOTALL)
    if consonant_match != None:
        consonants = consonant_match.group(1).strip()
        # print(consonants)
        for consonant in consonants.split('\n'):
            C.append(consonant.split('=')[0])
            for CV2 in consonant.split('=')[1].split(','):
                if CV2 != '':
                    CV_C.append(CV2)

    V = set(V)
    C = set(C)
    for V1 in V:
        for C1 in C:
            VC.append(V1+' '+C1)
    VC = set(VC)
    CV = set(CV)
    CV_C = set(CV_C)
    CV_V = CV - CV_C
    for V1 in V:
        for C1 in CV_V:
            VV.append(V1+' '+C1)
    VV = set(VV)
    # CVVC = VC | CV
    # print(V,C,VV)
    return V,C,CV,VC,VV



def run(oto_path,presamps_path,pitch,vcv_mode):
    phone_name = []
    if vcv_mode == '0':
        print('音源类型：CVVC')
        V_C = cvvc_presamp_read(presamps_path)
        oto_data = oto.oto_read(oto_path)
        for byname in oto_data:
            phone_name.append(byname[1].replace(pitch, ''))
        print('缺少的CV音素：', end='')
        for phone in V_C[2]:
            if phone not in phone_name:
                print(phone,end=',')
        print('\n缺少的VC音素：', end='')
        for phone in V_C[3]:
            if phone not in phone_name:
                print(phone,end=',')
        print('\n缺少的VV音素：', end='')
        for phone in V_C[4]:
            if phone not in phone_name:
                print(phone,end=',')
        print('\n缺少的- CV音素（可忽略）：', end='')
        CV = {'- ' + c for c in V_C[2]}
        for phone in CV:
            if phone not in phone_name:
                print(phone,end=',')
    elif vcv_mode == '1':
        print('音源类型：VCV')
        V_C = cvvc_presamp_read(presamps_path)
        oto_data = oto.oto_read(oto_path)
        for byname in oto_data:
            phone_name.append(byname[1].replace(pitch, ''))
        print('缺少的VCV音素：', end='')
        for phone in V_C[4]:
            if phone not in phone_name:
                print(phone, end=',')
        print('\n缺少的- CV音素：', end='')
        CV = {'- ' + c for c in V_C[2]}
        for phone in CV:
            if phone not in phone_name:
                print(phone, end=',')
    elif vcv_mode == '2':
        print('音源类型：CVV或CVVR')
        V_C = cvvc_presamp_read(presamps_path)
        oto_data = oto.oto_read(oto_path)
        for byname in oto_data:
            phone_name.append(byname[1].replace(pitch, ''))
        print('缺少的CV音素：', end='')
        for phone in V_C[2]:
            if phone not in phone_name:
                print(phone, end=',')
        print('\n缺少的VR音素：', end='')
        for phone in V_C[3]:
            if phone not in phone_name:
                print(phone, end=',')
        print('\n缺少的_V音素：', end='')
        _V = {'_' + c for c in V_C[0]}
        for phone in _V:
            if phone not in phone_name:
                print(phone, end=',')

    print()


if __name__ == '__main__':
    print('CVVC')
    run('E:\OpenUtau\Singers\XIABAI_new_CHN_CVVC_F3_autooto\F3\oto.ini','E:\OpenUtau\Singers\XIABAI_new_CHN_CVVC_F3_autooto\presamp.ini',' F3','0')
    print('VCV')
    run('E:\OpenUtau\Singers\空气音中文VCV_自动oto测试\VCV\oto.ini','presamp/樗儿式中文VCV-presamp.ini','','1')
    print('CVVR')
    run('E:\OpenUtau\Singers\Weiyin3.0\combined\oto.ini','presamp/CVR-presamp.ini','','2')