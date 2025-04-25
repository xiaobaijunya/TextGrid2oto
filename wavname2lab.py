import os
import re
# 定义函数来处理文件名
def process_wav_name(wav_name,cuts):
    # 移除文件扩展名
    cleaned_name = os.path.splitext(wav_name)[0]
    # 剔除下划线
    hira_list=sorted(['いぇ','うぇ','うぃ','うぉ','きゃ','きゅ','きぇ','きょ','ぎゃ','ぎゅ','ぎぇ','ぎょ','くぁ','くぃ','くぇ','くぉ','ぐぁ','ぐぃ','ぐぇ','ぐぉ','しゃ','しゅ','しぇ','しょ','じゃ','じぇ','じゅ','じょ','すぃ','ずぃ','ちゃ','ちゅ','ちぇ','ちょ','つぁ','つぃ','つぇ','つぉ','てぃ','てゅ','でぃ','でゅ','とぅ','どぅ','にゃ','にぇ','にゅ','にょ','ひゃ','ひゅ','ひぇ','ひょ','びゃ','びゅ','びぇ','びょ','ぴゃ','ぴゅ','ぴぇ','ぴょ','ふぁ','ふぃ','ふぇ','ふぉ','ぶぁ','ぶぃ','ぶぇ','ぶぉ','みゃ','みゅ','みぇ','みょ','りゃ','りゅ','りぇ','りょ','ヴぁ','ヴぃ','ヴぇ','ヴぉ','ヴ','ガ','ギ','グ','ゲ','ゴ','あ','い','う','え','お','か','が','き','ぎ','く','ぐ','け','げ','こ','ご','さ','ざ','し','じ','す','ず','せ','ぜ','そ','ぞ','た','だ','ち','つ','て','で','と','ど','な','に','ぬ','ね','の','は','ば','ぱ','ひ','び','ぴ','ふ','ぶ','ぷ','へ','べ','ぺ','ほ','ぼ','ぽ','ま','み','む','め','も','や','ゆ','よ','ら','り','る','れ','ろ','わ','ん']
                     , key=lambda x: len(x), reverse=True)

    for cut in cuts:
        cleaned_name = cleaned_name.replace(cut,' ')
    # 构建正则表达式模式（转义特殊字符）

    pattern = re.compile('|'.join(map(re.escape, hira_list)))  # 使用正则表达式优化匹配[4,7](@ref)

    # 执行替换：匹配项之间插入空格
    cleaned_name = pattern.sub(lambda m: f'{m.group()} ', cleaned_name)

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
    path = r'F:\Download\utau数据集\有授权\中文\中文拼接数据_自有授权\xiabai CVVC JIANHUA TEST\D4'
    # 获取当前目录下所有的WAV文件
    run(path,['_','-'])


