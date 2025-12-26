import json
import os
import re

def textgrid_change(textgrid_content):
    lines = textgrid_content.split('\n')
    lines = [line.strip() for line in lines if line.strip()]
    start = 0   #起始位置
    sum_indices  =[]    #间隔位置
    phones_dict = {}
    #获取长度
    long = [float(match.group(1)) for match in re.finditer(r'(?:xmin|xmax) = (\d+(?:\.\d+)?)', textgrid_content)][:2]
    for i in range(len(lines)):
        if lines[i] == 'name = "phones"':
            start = i
            break
    lines = lines[start:]   #删除无关内容

    for i in range(len(lines)):
        if lines[i].startswith('intervals ['):
            sum_indices.append(i)
    sum_indices.append(len(lines))
    for i in range(len(sum_indices)-1):
        data = lines[sum_indices[i]:sum_indices[i+1]]
        # 提取 intervals 编号
        interval_key = re.search(r'\d+', data[0]).group()
        phone_count  = {interval_key: {}}
        # 处理后续的键值对
        for item in data[1:]:
            key, value = item.split('=', 1)
            key = key.strip()
            value = value.strip()
            value = value.strip('"')
            phone_count[interval_key][key] = value
            phones_dict.update(phone_count)
    # print(phones_dict )
    return long,phones_dict

#需要传入TextGrid文件路径
def run(path_main,rec_preset):
    json_data = {}
    for root,dirs,files in os.walk(path_main):
        for file in files:
            #如果文件是TextGrid文件
            if file.endswith(".TextGrid"):
                #读取文件内容，支持UTF-8和UTF-16编码
                file_path = os.path.join(root, file)
                sum_data = {}
                data = []
                try:
                    # 先尝试UTF-8编码
                    with open(file_path, 'r', encoding='utf-8') as f:
                        textgrid_content = f.read()
                except UnicodeDecodeError:
                    try:
                        # 如果UTF-8失败，尝试UTF-16编码
                        with open(file_path, 'r', encoding='utf-16') as f:
                            textgrid_content = f.read()
                    except UnicodeDecodeError:
                        # 如果都失败，打印错误信息并跳过该文件
                        print(f"无法读取文件 {file_path}，编码格式不支持（UTF-8/UTF-16都尝试失败）")
                        continue
                # 转换为JSON
                data = textgrid_change(textgrid_content)
                # json_data = dict(wav=file)
                sum_data[file.split('.')[0]] = dict(wav_long=data[0],phones=data[1])
                json_data.update(sum_data)
    # print(json_data)
    if rec_preset is not None:
        try:
            rec_lines_miss = []
            rec_list_redundant = []
            with open(rec_preset, 'r', encoding='utf-8') as f:
                rec_lines = f.readlines()
            # 解析rec_preset文件，获取文件名列表
            rec_file_list = []
            for line in rec_lines:
                line = line.strip()
                if line:
                    rec_file_list.append(line)
            # 根据rec_file_list的顺序对json_data进行排序
            sorted_json_data = {}
            for file_name in rec_file_list:
                if file_name in json_data:
                    sorted_json_data[file_name] = json_data[file_name]
                else:
                    rec_lines_miss.append(file_name)
                    print(f"rec_preset中包含的文件 {file_name} 不在json_data中（漏录了该条目）")
            else:
                rec_list_redundant.append(file_name)
            # 添加不在rec_preset中的剩余文件（保持原顺序）
            for file_name in json_data:
                if file_name not in sorted_json_data:
                    sorted_json_data[file_name] = json_data[file_name]

            json_data = sorted_json_data
            print(f"rec_preset中包含的文件 {rec_lines_miss} 不在json_data中（漏录了这些条目）")
            print(f"rec_preset中包含的文件 {rec_list_redundant} 不在json_data中（多录了这些条目）")
        except Exception as e:
            print(f"读取或解析rec_preset文件时出错: {e}")



    json_string = json.dumps(json_data, indent=4, ensure_ascii=False,separators=(',', ':'))
    json_dir = os.path.join(path_main, 'json')

    # 2. 创建目录（如果不存在）
    os.makedirs(json_dir, exist_ok=True)
    # 将JSON字符串写入文件
    with open(os.path.join(json_dir, 'ds_phone.json'), 'w', encoding='utf-8') as f:
        f.write(json_string)
    print(f"转换完成，结果已保存为{json_dir}\ds_phone.json")

if __name__ == '__main__':
    # 读取TextGrid文件内容
    #设置工作目录
    path_main = r"E:\OpenUtau\Singers\白锋_02\D4"
    rec_preset = r"E:\OpenUtau\Singers\白锋_02\普通6字表.txt"
    #遍历目录下的所有文件
    run(path_main+'/TextGrid',rec_preset)
