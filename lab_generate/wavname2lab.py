import os
import re
# 定义函数来处理文件名
def process_wav_name(wav_name,cuts):
    # 移除文件扩展名
    cleaned_name = os.path.splitext(wav_name)[0]
    # 剔除下划线
    for cut in cuts:
        cleaned_name = cleaned_name.replace(cut,' ')
    # 在假名之间插入空格：小写假名（拗音/促音等）与前一个假名合并，不单独拆分成音素
    small_kana = 'ぁぃぅぇぉゃゅょっゎァィゥェォャュョッヮ'
    pattern = re.compile(r'(?<=[ぁ-ゖァ-ヺ])(?![ぁぃぅぇぉゃゅょっゎァィゥェォャュョッヮ])')
    cleaned_name = pattern.sub(' ', cleaned_name)
    cleaned_name = 'SP ' + cleaned_name + ' SP'

    cleaned_name=cleaned_name.replace('  ',' ')

    cleaned_name=cleaned_name.replace('R','SP')
    cleaned_name=cleaned_name.replace('SP SP','SP')

    if cleaned_name[0]==' ':
        cleaned_name=cleaned_name[1:]
    return cleaned_name

#传入wav路径（可选加入自定义分隔符号）
def run(path,cuts):
    # 递归查找所有 WAV 文件
    for root, dirs, files in os.walk(path):
        for wav_file in files:
            if wav_file.endswith('.wav'):
                # 处理文件名
                lab_content = process_wav_name(wav_file,cuts)
                # 生成.lab文件的文件名
                lab_file_name = os.path.splitext(wav_file)[0] + '.lab'
                # 写入.lab文件（保存在与 WAV 文件相同的目录）
                lab_file_path = os.path.join(root, lab_file_name)
                with open(lab_file_path, 'w', encoding='utf-8') as lab_file:
                    lab_file.write(lab_content)
                print(f"已生成 {lab_file_path}",end=':')
                print(lab_content)



if __name__ == '__main__':
    path = r'E:\OpenUtau\Singers\白锋_02\result'
    # 获取当前目录下所有的WAV文件
    run(path,['_','-'])