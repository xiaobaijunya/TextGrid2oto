import os
import wave
import time
#oto格式
#文件名=别名,左边界,固定,右边界（负值）,预发声,交叉

# 定义颜色代码
BLACK = '\033[30m'
RED = '\033[31m'
GREEN = '\033[32m'
YELLOW = '\033[33m'
BLUE = '\033[34m'
MAGENTA = '\033[35m'
CYAN = '\033[36m'
WHITE = '\033[37m'
RESET = '\033[0m'  # 用于恢复默认颜色

#文件存在验证
def check_file(file_path):
    if os.path.exists(file_path):
        print(f'{GREEN}文件存在：{file_path}{RESET}')
        return False
    else:
        input(f'{RED}文件不存在：{file_path}{RESET}')
        return True

# 此函数用于检查文件是否存在，如果存在则添加数字后缀以生成唯一的文件路径。
def oto_path(file_path):
    new_file_path=file_path.strip(".ini")
    for i in range (1,100):
        if not os.path.exists(file_path):
            return file_path
        file_path = f"{new_file_path}_{i:02d}.ini"
        # print(file_path)

#预发声转为负数
def oto_check(file_path,oto_data):
    file_path=file_path.strip("oto.ini")
    new_oto_data=[]
    for oto in oto_data:
        if int(oto[4]) >0:
            with wave.open(file_path+oto[0], 'rb') as f:
                duration = f.getnframes() / f.getframerate() *1000
            oto[4] =int(duration - int(oto[2]) - int(oto[4]))*-1
            if oto[4]>0:
                print(f'{RED}右边界错误：{oto}wav:{duration}s{RESET}')
        new_oto_data.append(oto)
    return new_oto_data

    # 解析 oto 文件
def oto_read(file_path):
    if check_file(file_path):return False
    oto_data=[]
    encodings = ['utf-8', 'shift-jis','gbk']
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                for line in f:
                    line = line.strip()
                    parts = line.split('=')
                    # print(parts)
                    parts2 = parts[1].split(',')
                    # print(parts2)
                    oto_data.append([parts[0]] + [parts2[0]] + [int(round(float(num_str))) for num_str in parts2[1:]])
                    # wav+别名+四舍五入后的数值
            print(f'成功使用 {encoding} 编码读取文件。')
            break
        except UnicodeDecodeError:
            continue
    else:
        print('无法使用尝试的编码读取oto文件，请检查文件编码。')
        input('按任意键退出')
        quit()
    print(f'{GREEN}oto文件解析成功：{file_path}{RESET}')
    return oto_check(file_path,oto_data)
#没写完
def oto_repeat(oto_data,repeat):
    phone_count = {}
    new_oto_data = []
    for oto in oto_data:
        phone = oto[1]
        if phone not in phone_count:
            phone_count[phone] = 1
            new_oto_data.append(oto)
        else:
            count = phone_count[phone]
            if count < repeat:
                oto[1] = f"{phone}_{count}"
                new_oto_data.append(oto)
            phone_count[phone] += 1
            # print(count)
    return new_oto_data

#oto数值偏移函数
def oto_offset(oto_data,offset):
    new_oto_data = []
    oto_sum=[]
    # print(oto_data)
    # 遍历 oto_data 列表中的每个元素
    for oto in oto_data:
        # 0文件名=1别名,2左边界,3固定,4右边界（负值）,5预发声,6交叉
        oto_sum=[oto[2] + offset[0],oto[3] + offset[1],oto[4]*-1 + offset[2],oto[5] + offset[3],oto[6] + offset[4]]

        if oto_sum[0] >= 0:oto[2] = oto_sum[0]  # 左
        else:print(f'{oto[1]}修改后的左边界为{oto_sum[0]}，跳过偏移操作')
        if oto_sum[4] >= 0:oto[6] = oto_sum[4]#交叉
        else:
            oto[6] = 20
            print(f'{oto[1]}错误的交叉：{oto_sum[4]}，交叉设为20')
        if oto_sum[3] >= 0:oto[5] = oto_sum[3]
        else:print(f'{oto[1]}错误的预发声:{oto[5] + offset[0]}，跳过偏移操作')
        if oto_sum[1] >= oto[5]:oto[3] = oto_sum[1]
        else:
            oto[3] = oto[5]
            print(f'{oto[1]}错误的固定:{oto[3] + offset[1]}，设为{oto[5]}')
        if oto_sum[2] >= oto[3]:oto[4] = oto_sum[2]*-1
        else:
            oto[4] = (oto[3]+10)*-1
            print(f'{oto[1]}错误的右边界:{oto_sum[2]}，设为{oto[4]}')

        new_oto_data.append(oto)
    return new_oto_data


def oto_write(file_path,oto_data,pitch,cover):
    #写入 oto 文件
    if cover=='y' or cover=='Y':
        print(f'{GREEN}覆盖原文件：{file_path}{RESET}')
        new_file_path = file_path
    else:
        print(f'{GREEN}不覆盖原文件：{file_path}{RESET}')
        new_file_path=oto_path(file_path)
    encodings = ['shift-jis', 'utf-8']
    for encoding in encodings:
        try:
            with open(new_file_path, 'w', encoding=encoding) as f:
                for oto in oto_data:
                    f.write(f'{oto[0]}={oto[1]}{pitch},{oto[2]},{oto[3]},{oto[4]},{oto[5]},{oto[6]}\n')
            print(f'{GREEN}新oto以 {encoding} 编码写入成功：{new_file_path}{RESET}')
            break
        except UnicodeEncodeError:
            if encoding == 'shift-jis':
                print(f'{YELLOW}无法使用 shift-jis 编码写入，尝试使用 utf-8 编码...{RESET}')
            else:
                print(f'{RED}无法使用 utf-8 编码写入，请检查数据。{RESET}')
    else:
        print(f'{RED}所有尝试的编码都失败，无法写入文件。{RESET}')

def run(oto_path):
    oto=oto_read(oto_path)
    oto_write(oto_path,oto)

if __name__ == '__main__':
    start_time = time.time()

    path = "G:\编程\\UU2VV\E3 - 副本\oto.ini"
    oto=oto_read(path)
    oto_write("G:\编程\\UU2VV\F3\oto.ini",oto)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"代码运行耗时: {elapsed_time*1000} 毫秒")