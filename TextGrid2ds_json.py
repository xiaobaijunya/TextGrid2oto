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
            key, value = item.split(' = ', 1)
            value = value.strip('"')
            phone_count[interval_key][key] = value
            phones_dict.update(phone_count)
    # print(phones_dict )
    return long,phones_dict

#需要传入TextGrid文件路径
def run(path_main):
    json_data = {}
    for root,dirs,files in os.walk(path_main):
        for file in files:
            #如果文件是TextGrid文件
            if file.endswith(".TextGrid"):
                #读取文件内容
                with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                    sum_data = {}
                    data = []
                    textgrid_content = f.read()
                    # 转换为JSON
                    data = textgrid_change(textgrid_content)
                    # json_data = dict(wav=file)
                    sum_data[file.replace('.TextGrid','.wav')] = dict(wav_long=data[0],phones=data[1])
                    json_data.update(sum_data)
    # print(json_data)
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
    path_main = "G:/编程/utau自动标注/F3"
    #遍历目录下的所有文件
    run(path_main+'/TextGrid')
