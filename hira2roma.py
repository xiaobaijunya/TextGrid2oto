import os
import shutil

#nuitka --standalone --onefile --output-filename=hira2roma_v0.1.0 hira2roma.py

def load_conversion_dict(hira2roma_list):
    """加载平假名到罗马字的转换字典"""
    try:
        with open(hira2roma_list, "r", encoding='utf-8') as f:
            return {
                line.split(",")[0]: line.split(",")[1].strip()
                for line in f
                if len(line.strip()) > 0  # 跳过空行
            }
    except FileNotFoundError:
        print("错误：未找到 hira2roma list.txt 文件")
        exit(1)
    except UnicodeDecodeError:
        print("错误：文件编码不匹配，请尝试使用 shift-jis 编码保存文件")
        exit(1)


def convert_hiragana(text, conversion_dict):
    """转换平假名字符串为罗马字"""
    result = []
    i = 0
    while i < len(text):
        # 尝试匹配最长可能的字符（最多3个字符）
        for length in [3, 2, 1]:
            if i + length <= len(text):
                char = text[i:i + length]
                if char in conversion_dict:
                    result.append(conversion_dict[char])
                    i += length
                    break
        else:  # 未找到匹配字符
            result.append(text[i])
            i += 1
    return "".join(result)


if __name__ == "__main__":
    hira2roma_list = input('输入转换规则.txt路径：（默认hira2roma list.txt）')
    folder_path = input('输入文件夹路径：')
    if not os.path.exists(folder_path):
        hira2roma_list = "hira2roma list.txt"
    if not os.path.exists(folder_path):
        print('错误：未找到文件夹路径')
        exit(1)
    # 加载转换字典
    conversion_dict = load_conversion_dict(hira2roma_list)
    # 创建 romaji 文件夹
    romaji_folder = os.path.join(folder_path, 'romaji')
    if not os.path.exists(romaji_folder):
        os.makedirs(romaji_folder)

    # 遍历文件夹下的所有文件
    for filename in os.listdir(folder_path):
        if filename.endswith(".wav"):
            # 提取文件名（不包含扩展名）
            name_without_ext = os.path.splitext(filename)[0]
            # 根据 hira2roma 规则进行转换
            new_name = convert_hiragana(name_without_ext, conversion_dict)
            # 生成新的文件名
            new_filename = os.path.join(romaji_folder, new_name + ".wav")

            # 原文件路径
            old_file_path = os.path.join(folder_path, filename)

            # 复制文件
            shutil.copy2(old_file_path, new_filename)
            print(f"Copied {filename} to {new_filename}")