import wavname2lab
import TextGrid2ds_json
import ds_json2word_josn
# nuitka --standalone --onefile main.py

if __name__ == '__main__':
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
    ds_json2word_josn.run(ds_dict,TextGrid_path+'/json/ds_phone.json')

