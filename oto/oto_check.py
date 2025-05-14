import re
from oto import oto_rw
def cvvc_presamp_read(presamps_path):
    V = []
    C = []
    VC = []
    VV = []
    CV=[]
    CV_C=[]
    CV_V=[]
    with open(presamps_path, 'r',encoding='utf-8') as file:
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

def oto_read(file_path):
    oto_data=[]
    encodings = ['shift-jis','utf-8', 'gbk']
    for encoding in encodings:
        try:
            print(f'正在尝试使用 {encoding} 编码读取文件。')
            with open(file_path, 'r',encoding=encoding) as f:
                for line in f:
                    line = line.strip()
                    parts = line.split('=')
                    # print(parts)
                    parts2 = parts[1].split(',')
                    # print(parts2)
                    oto_data.append([parts[0]] + [parts2[0]] + [int(round(float(num_str))) for num_str in parts2[1:]])
                    # wav+别名+四舍五入后的数值
            print(f'成功使用 {encoding} 编码读取文件。')
            break
        except UnicodeDecodeError:
            continue
    else:
        print('无法使用尝试的编码读取oto文件，请检查文件编码。')
        input('按任意键退出')
        quit()
    print(f'oto文件解析成功：{file_path}')
    return oto_data

def run(oto_path,presamps_path,pitch,vcv_mode):
    phone_name = []
    if vcv_mode == '0':
        print('音源类型：CVVC')
        V_C = cvvc_presamp_read(presamps_path)
        oto_data = oto_read(oto_path)
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
        phone=[]
        print('音源类型：VCV')
        V_C = cvvc_presamp_read(presamps_path)
        oto_data = oto_rw.oto_read(oto_path)
        for byname in oto_data:
            phone_name.append(byname[1].replace(pitch, ''))
        print('缺少的VCV音素：', end='')
        #V,C,CV,VC,VV
        for phone0 in V_C[0]:
            for phone2 in V_C[2]:
                phone.append(phone0+' '+phone2)
        for phone3 in phone:
            if phone3 not in phone_name:
                print(phone3, end=',')
        print('\n缺少的- CV音素：', end='')
        CV = {'- ' + c for c in V_C[2]}
        for phone in CV:
            if phone not in phone_name:
                print(phone, end=',')
    elif vcv_mode == '2':
        print('音源类型：CVV或CVVR')
        V_C = cvvc_presamp_read(presamps_path)
        oto_data = oto_rw.oto_read(oto_path)
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
    run('E:\OpenUtau\Singers\空气音中文VCV_自动oto测试\VCV\oto.ini', '../presamp/樗儿式中文VCV-presamp.ini', '', '1')
    print('VCV')
    run('E:\OpenUtau\Singers\TNOT-Nottthat_VCV-TNOT-日语-VCV\VCV\oto.ini', '../presamp/jp-hira-presamp.ini', '', '1')
    print('CVVR')
    run('E:\OpenUtau\Singers\Weiyin3.0\combined\oto.ini', '../presamp/CVR中文-presamp.ini', '', '2')