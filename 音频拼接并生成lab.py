import wave
import os

import wave
import os


def concatenate_audio_files(folder_path):
    # 创建新文件夹存放结果
    output_folder = os.path.join(folder_path, 'combined')
    os.makedirs(output_folder, exist_ok=True)

    # 获取所有wav文件并按文件名排序
    wav_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.wav')],
                       key=lambda x: x.split('.')[0])

    lab_content = []

    # 以三个为一组处理文件
    for i in range(0, len(wav_files), 3):
        group = wav_files[i:i + 3]
        # 生成组合文件名（不带扩展名）
        base_names = [os.path.splitext(f)[0] for f in group]
        combined_name = '_'.join(base_names) + '.wav'
        lab_content.append(combined_name[:-4])  # 记录用于lab文件的内容

        # 拼接音频
        output_path = os.path.join(output_folder, combined_name)
        with wave.open(output_path, 'wb') as output_wave:
            for filename in group:
                filepath = os.path.join(folder_path, filename)
                with wave.open(filepath, 'rb') as input_wave:
                    if output_wave.getnframes() == 0:
                        output_wave.setparams(input_wave.getparams())
                    output_wave.writeframes(input_wave.readframes(input_wave.getnframes()))

    # 保存lab文件到原文件夹
    # output_lab_path = os.path.join(folder_path, 'output.lab')
    # with open(output_lab_path, 'w', encoding='utf-8') as lab_file:
    #     lab_file.write(" ".join(lab_content))

    print(f"生成{len(lab_content)}个音频文件到 {output_folder}")
    # print(f".lab文件生成完成，保存为 {output_lab_path}")


# ... 剩余代码保持不变 ...

# 调用函数，将文件夹路径替换为实际的文件夹路径
folder_path = 'E:\OpenUtau\Singers\Weiyin3.0\CV'
concatenate_audio_files(folder_path)