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
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            parts = line.split('=')
            # print(parts)
            parts2 = parts[1].split(',')
            # print(parts2)
            oto_data.append([parts[0]]+[parts2[0]]+[int(round(float(num_str))) for num_str in parts2[1:]])
            #wav+别名+四舍五入后的数值
    print(f'{GREEN}oto文件解析成功：{file_path}{RESET}')
    return oto_check(file_path,oto_data)

def oto_write(file_path,oto_data):
    #写入 oto 文件
    new_file_path=oto_path(file_path)
    with open(new_file_path, 'w') as f:
        for oto in oto_data:
            f.write(f'{oto[0]}={oto[1]},{oto[2]},{oto[3]},{oto[4]},{oto[5]},{oto[6]}\n')
    print(f'{GREEN}新oto写入成功：{new_file_path}{RESET}')

def run(oto_path):
    oto=oto_read(oto_path)
    oto_write(oto_path,oto)

if __name__ == '__main__':
    start_time = time.time()

    path = "G:\编程\\UU2VV\E3 - 副本\oto.ini"
    oto=oto_read(path)
    oto_write("G:\编程\\UU2VV\E3\oto.ini",oto)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"代码运行耗时: {elapsed_time*1000} 毫秒")