import json
import re
def json_read(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    return json_data

def seg_change(vb_path,json_path):
    json_data = json_read(json_path)
    for audio_file, data in json_data.items():
        phones = data.get('phones', {})
        if not phones:
            continue
        seg_path = vb_path + '/'  +audio_file.split('.')[0] +'.seg'
        with open(seg_path, 'r') as f:
            text = f.read()
            n_phonemes = re.search(r'nPhonemes\s+(\d+)', text).group(1)
        if int(n_phonemes)-2 != len(phones):
            print(f'{audio_file}的nPhonemes:{n_phonemes}和phones:{len(phones)}长度不一致,但是不影响生成')

        xmin_list = [phones[key]['xmin'] for key in sorted(phones.keys(), key=int)]

        with open(seg_path, 'r') as f:
            lines = f.readlines()
            # 解析seg文件，找到音素数据开始的行
            phoneme_start_line = None
            phoneme_lines = []
            for i, line in enumerate(lines):
                line = line.strip()
                if line == '===================================================':
                    phoneme_start_line = i + 1
                    break
            if phoneme_start_line is None:
                print(f'Error: Could not find phoneme data in {seg_path}')
                continue
            # 收集音素行
            for i in range(phoneme_start_line, len(lines)):
                if lines[i].strip():  # 跳过空行
                    phoneme_lines.append(lines[i])

            # 创建修改后的音素行列表
            modified_phoneme_lines = []

            # 遍历原始音素行和phones数据
            for i, phoneme_line in enumerate(phoneme_lines):
                # 解析原始行
                parts = phoneme_line.split()
                if len(parts) < 3:
                    modified_phoneme_lines.append(phoneme_line)
                    continue

                # 保留原始音素名称和其他属性，只更新时间
                phoneme_name = parts[0]

                if i==0:
                    begin_time = '0.000000'
                    end_time = float(xmin_list[i+1]) - 0.04
                elif i == 1:
                    begin_time = float(xmin_list[i]) - 0.04
                    end_time = xmin_list[i]
                elif i <= len(xmin_list)-1:
                    begin_time = xmin_list[i-1]
                    end_time = xmin_list[i]
                elif i == len(xmin_list):
                    begin_time = xmin_list[i-1]
                    end_time = float(xmin_list[i-1]) + 0.05
                elif i == len(xmin_list)+1:
                    begin_time = float(xmin_list[i-2]) + 0.05
                    end_time = parts[2]
                else:
                    begin_time = parts[1]
                    end_time = parts[2]

                begin_time = str(begin_time)[:8]
                end_time = str(end_time)[:8]

                # 重新格式化音素行，保持原始格式
                # 注意：这里假设原始格式是固定的，实际可能需要更复杂的处理
                modified_line = f"{phoneme_name}\t\t{begin_time}\t\t{end_time}\n"
                modified_phoneme_lines.append(modified_line)

            # 构建完整的修改后的内容
            new_lines = lines[:phoneme_start_line] + modified_phoneme_lines

            # 写回seg文件
            with open(seg_path, 'w') as f:
                f.writelines(new_lines)

            print(f'已成功修改{audio_file}对应的seg文件的时值')



if __name__ == "__main__":
    vb_path = r"E:\vocaloidmake\vocaloid-dbtool-application\VOCALOID Developer\baini_CVVC_ZH\wav"
    json_path = vb_path+"\TextGrid\json\ds_phone_filter.json"
    seg_change(vb_path,json_path)


