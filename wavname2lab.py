import os

# 定义函数来处理文件名
def process_wav_name(wav_name,cuts):
    # 移除文件扩展名
    cleaned_name = os.path.splitext(wav_name)[0]
    # 剔除下划线
    for cut in cuts:
        cleaned_name = cleaned_name.replace(cut,' ')
    cleaned_name=cleaned_name.replace('  ',' ')
    if cleaned_name[0]==' ':
        cleaned_name=cleaned_name[1:]
    return cleaned_name

#传入wav路径（可选加入自定义分隔符号）
def run(path,cuts):
    wav_files = [f for f in os.listdir(path) if f.endswith('.wav')]
    # 处理每个WAV文件
    for wav_file in wav_files:
        # 处理文件名
        lab_content = process_wav_name(wav_file,cuts)
        # # 生成.lab文件的内容
        # lab_content = ' '.join(pinyin_parts).strip()
        # 生成.lab文件的文件名
        lab_file_name = os.path.splitext(wav_file)[0] + '.lab'
        # 写入.lab文件
        with open(path + '/' + lab_file_name, 'w', encoding='utf-8') as lab_file:
            lab_file.write(lab_content)
        print(lab_content)
        print(f"已生成 {lab_file_name}")


if __name__ == '__main__':
    path = 'E:\OpenUtau\Singers\XIABAI_new_CHN_CVVC_F3_autooto\F3'
    # 获取当前目录下所有的WAV文件
    run(path,['_','-'])


