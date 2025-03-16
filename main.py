import wavname2lab
import TextGrid2ds_json
import ds_json2filter
import ds_json2word
import word2utau_phone
import json2oto
import oto
# nuitka --standalone --onefile --output-filename=TextGrid2oto_v0.1.2 main.py

if __name__ == '__main__':
    try:
        print('sofa-UTAU自动标注')
        print('1.生成lab')
        wav_path=input('请输入wav的路径：')
        cut=input('请输入分隔符默认为_')
        if not cut:
            cut = '_'
        wavname2lab.run(wav_path,cut)
        print('2.生成TextGrid')
        print('需要自己前往sofa生成TextGrid')
        input('生成完成后,请输入任意键继续')
        print('3.生成json')
        TextGrid_path=input('请输入TextGrid的路径：')
        ds_dict=input('请输入sofa模型对应ds字典的路径：')
        if not TextGrid_path:
            TextGrid_path = wav_path+'/TextGrid'
        if not ds_dict:
            ds_dict = 'opencpop-extension.txt'
        TextGrid2ds_json.run(TextGrid_path)
        ds_json2filter.run(ds_dict,TextGrid_path+'/json/ds_phone.json')
        print('3.生成word.json')
        ds_json2word.run(ds_dict,TextGrid_path+'/json/ds_phone_filter.json')
        print('4.生成utau音素')
        presamp=input('请输入presamp.ini的路径：')
        if not presamp:
            presamp = 'presamp.ini'
        word2utau_phone.generate_utau_phone(presamp,TextGrid_path+'/json/word_phone.json')
        print('5.生成utauphone_json')
        word2utau_phone.generate_utau_phone(presamp,TextGrid_path+'/json/utau_phone.json')
        print('6.生成oto.ini')
        # -CV和CV规则：左线占比,固定的占比,右线占比,预发声不变,交叉占比
        cv_sum = [1, 3, 1.5, 1, 2]
        # VC和VV规则：左线占比,固定的占比,右线占比,预发声不变,交叉占比,VV固定占比
        vc_sum = [3, 0, 2, 1, 2, 3]
        json2oto.run(presamp,TextGrid_path+'/json/utau_phone.json',TextGrid_path+'/json/word_phone.json',wav_path,cv_sum,vc_sum)
        print('7.合并oto.ini')
        cv = oto.oto_read(wav_path+'/cv_oto.ini')
        vc = oto.oto_read(wav_path+'/vc_oto.ini')
        pitch = input('请输入音阶后缀：')
        oto.oto_write(wav_path+'/oto.ini',cv+vc,pitch)
        print('10086.完成！')
    except Exception as e:
        import traceback
        print("\n发生错误：")
        print(traceback.format_exc())
        print("请联系开发者检查错误")
    finally:
        input('输入任意键退出')
        quit()