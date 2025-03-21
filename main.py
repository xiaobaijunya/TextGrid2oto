import json2VCV_oto
import wavname2lab
import TextGrid2ds_json
import ds_json2filter
import ds_json2word
import word2utau_phone
import json2oto
import json2VCV_oto
import json2CV_oto
import oto
import sys
# nuitka --standalone --onefile --output-filename=TextGrid2oto_v0.1.12 main.py

def run():
    try:
        print('sofa-UTAU自动标注')
        print('1.生成lab')
        wav_path=input('请输入wav的路径：')
        cut=input('请输入分隔符默认为_')
        if not cut:
            cut = '_'
        VCV_mode=input('输入数字，选择模式：\n(默认使用CVVC模式)\nCVVC:0 \nVCV:1 \nCV(多字单独音(连单音)):2\n')
        if not VCV_mode:
            VCV_mode = '0'
        wavname2lab.run(wav_path,cut)
        print('2.生成TextGrid')
        print('需要自己前往sofa生成TextGrid')
        input('生成完成后,请输入任意键继续')
        print('3.生成json')
        TextGrid_path=input('请输入TextGrid的路径：')
        ds_dict=input('请输入sofa模型对应ds字典的路径：')
        if not TextGrid_path:
            TextGrid_path = wav_path+ '/TextGrid'
        if not ds_dict:
            ds_dict = 'opencpop-extension.txt'
        ignore=input('请输入忽略音素，多个用,分隔：')
        if not ignore:
            ignore = 'AP,SP'
        TextGrid2ds_json.run(TextGrid_path)
        ds_json2filter.run(ds_dict,TextGrid_path+'/json/ds_phone.json',ignore)
        print('3.生成word.json')
        ds_json2word.run(ds_dict,TextGrid_path+'/json/ds_phone_filter.json')
        print('4.生成utau音素')
        presamp=input('请输入presamp.ini的路径：')
        if not presamp:
            presamp = 'presamp/presamp.ini'
        word2utau_phone.generate_utau_phone(presamp,TextGrid_path+'/json/word_phone.json')
        print('5.生成utauphone_json')
        # word2utau_phone.generate_utau_phone(presamp,TextGrid_path+'/json/utau_phone.json')
        print('6.生成oto.ini')
        if VCV_mode=='1':
            # -CV和CV规则：左线占比,固定的占比,右线占比,预发声不变,交叉占比
            cv_sum = [1, 3, 1.5, 1, 2]
            # VC和VV规则：左线占比,固定的占比,右线占比,预发声不变,交叉占比,VV固定占比
            vc_sum = [3, 0, 2, 1, 2, 3]
            json2VCV_oto.run(presamp,TextGrid_path+'/json/utau_phone.json',TextGrid_path+'/json/word_phone.json',wav_path,cv_sum,vc_sum)
        elif VCV_mode=='2':
            # -CV和CV规则：左线占比,固定的占比,右线占比,预发声不变,交叉占比
            cv_sum = [1, 3, 1.5, 1, 2]
            # VC和VV规则：左线占比,固定的占比,右线占比,预发声不变,交叉占比,VV固定占比
            vc_sum = [3, 0, 2, 1, 2, 3]
            json2CV_oto.run(presamp,TextGrid_path+'/json/utau_phone.json',TextGrid_path+'/json/word_phone.json',wav_path,cv_sum,vc_sum)
        else:
            # -CV和CV规则：左线占比,固定的占比,右线占比,预发声不变,交叉占比
            cv_sum = [1, 3, 1.5, 1, 2]
            # VC和VV规则：左线占比,固定的占比,右线占比,预发声不变,交叉占比,VV固定占比
            vc_sum = [3, 0, 2, 1, 2, 3]
            json2oto.run(presamp,TextGrid_path+'/json/utau_phone.json',TextGrid_path+'/json/word_phone.json',wav_path,cv_sum,vc_sum)
        print('7.合并oto.ini')
        cv = oto.oto_read(wav_path+'/cv_oto.ini')
        vc = oto.oto_read(wav_path+'/vc_oto.ini')
        CV_repeat = input('输入CV最大重复次数：')
        VC_repeat = input('输入VC最大重复次数：')
        print('8.剔除重复并，合并oto.ini')
        cv = oto.oto_repeat(cv, int(CV_repeat))
        vc = oto.oto_repeat(vc, int(VC_repeat))
        pitch = input('请输入音阶后缀：')
        if not pitch:
            pitch = ''
        cover = input('是否覆盖原文件？y/n(默认为n)')
        if not cover:
            cover = 'n'
        oto.oto_write(wav_path+'/oto.ini',cv+vc,pitch,cover)
        print('10086.完成！')
    except Exception as e:
        import traceback
        print("\n发生错误：")
        print(traceback.format_exc())
        print("请联系开发者检查错误")
    finally:
        input('输入任意键退出')
        quit()

def auto_run(config):
    try:
        with open(config,'r',encoding='utf-8') as f:
            config = f.read().split('\n')
            # print(config)
            for i in range(len(config)):
                config[i]=config[i].strip()
            #转为字典
            config = {config[i].split('=')[0]:config[i].split('=')[1] for i in range(len(config)) if config[i]!='' and '#' not in config[i]}
            # 将字符串转换为列表并将每个元素转换为float类型
            config['cv_sum'] = [float(i) for i in config['cv_sum'].strip('[]').split(',')]
            config['vc_sum'] = [float(i) for i in config['vc_sum'].strip('[]').split(',')]
            config['cv_offset'] = [float(i) for i in config['cv_offset'].strip('[]').split(',')]
            config['vc_offset'] = [float(i) for i in config['vc_offset'].strip('[]').split(',')]
            print(config)
        print('sofa-UTAU自动标注')
        if config['lab']=='Y' or config['lab']=='y':
            print('1.生成lab')
            wavname2lab.run(config['wav_path'],config['cut'])
            print('2.生成TextGrid')
            print('需要自己前往sofa生成TextGrid')
            input('生成完成后,请输入任意键继续')
        VCV_mode=config['VCV_mode']
        if not VCV_mode:
            VCV_mode = '0'
        print('3.生成json')
        TextGrid2ds_json.run(config['TextGrid_path'])
        ds_json2filter.run(config['ds_dict'], config['TextGrid_path'] + '/json/ds_phone.json',config['ignore'])
        print('3.生成word.json')
        ds_json2word.run(config['ds_dict'], config['TextGrid_path'] + '/json/ds_phone_filter.json')
        print('4.生成utau音素')
        word2utau_phone.generate_utau_phone(config['presamp'], config['TextGrid_path'] + '/json/word_phone.json')
        print('5.生成utauphone_json')
        # word2utau_phone.generate_utau_phone(config['presamp'], config['TextGrid_path'] + '/json/utau_phone.json')
        print('6.生成oto.ini')
        # -CV和CV规则：左线占比,固定的占比,右线占比,预发声不变,交叉占比
        # VC和VV规则：左线占比,固定的占比,右线占比,预发声不变,交叉占比,VV固定占比
        if VCV_mode=='1':
            json2VCV_oto.run(config['presamp'], config['TextGrid_path'] + '/json/utau_phone.json', config['TextGrid_path'] + '/json/word_phone.json',
                         config['wav_path'], config['cv_sum'], config['vc_sum'])
        elif VCV_mode=='2':
            json2CV_oto.run(config['presamp'], config['TextGrid_path'] + '/json/utau_phone.json',
                             config['TextGrid_path'] + '/json/word_phone.json',
                             config['wav_path'], config['cv_sum'], config['vc_sum'])
        elif VCV_mode=='0':
            json2oto.run(config['presamp'], config['TextGrid_path'] + '/json/utau_phone.json', config['TextGrid_path'] + '/json/word_phone.json',
                         config['wav_path'], config['cv_sum'], config['vc_sum'])
        else:
            print('VCV_mode数值错误')
            input('退出')
            quit()
        print('7.读取CV和VC oto.ini')
        cv = oto.oto_read(config['wav_path'] + '/cv_oto.ini')
        vc = oto.oto_read(config['wav_path'] + '/vc_oto.ini')
        print('8.剔除重复项')
        cv = oto.oto_repeat(cv, int(config['CV_repeat']))
        vc = oto.oto_repeat(vc, int(config['VC_repeat']))
        print('9.偏移oto数值.ini')
        if config['cv_offset']!=[0.0, 0.0, 0.0, 0.0, 0.0]:
            cv = oto.oto_offset(cv, config['cv_offset'])
            print('9.1.偏移CV数值,运行成功')
        if config['vc_offset']!=[0.0, 0.0, 0.0, 0.0, 0.0]:
            vc = oto.oto_offset(vc, config['vc_offset'])
            print('9.1.偏移VC数值,运行成功')
        print('10.合并oto.ini')
        oto.oto_write(config['wav_path'] + '/oto.ini', cv + vc, config['pitch'], config['cover'])
        print('10086.完成！')
    except Exception as e:
        import traceback
        print("\n发生错误：")
        print(traceback.format_exc())
        print("请联系开发者检查错误")
    finally:
        input('输入任意键继续')
        return 0

if __name__ == '__main__':
    print(sys.argv)
    if len(sys.argv)>1:
        for arg in sys.argv[1:]:
            auto_run(arg)
        quit()
    run()