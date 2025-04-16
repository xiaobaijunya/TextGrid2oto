import os
import shutil


def main_process():
    # 自定义转换表路径
    default_table = "sofa-dic/hira2roma_list.txt"
    table_path = input(f"请输入转换表路径（回车使用默认值 {default_table}）: ") or default_table

    # 加载转换表
    print(f"\n正在加载转换表：{table_path}")
    hira_dict = {}
    try:
        with open(table_path, "r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, 1):
                line = line.strip()
                if line and not line.startswith("#"):
                    try:
                        k, v = line.split(",", 1)
                        hira_dict[k] = v
                    except ValueError:
                        print(f"警告：第 {line_number} 行格式错误 - {line}")
        print(f"成功加载 {len(hira_dict)} 个转换规则")
    except FileNotFoundError:
        print(f"错误：转换表文件 {table_path} 不存在")
        return

    # 按长度降序排序
    sorted_keys = sorted(hira_dict.keys(), key=lambda x: -len(x))

    # 编码转换选项
    convert_encoding = input("需要将文件名编码从Shift_JIS转换为GBK吗？(y/n): ").lower() == "y"

    # 源目录输入
    src_root = input("请输入WAV目录路径: ")
    dest_root = os.path.join(src_root, "new_wav")
    print(f"目标目录设置为：{dest_root}")

    # 初始化统计
    total_files = 0
    converted_files = 0
    encoding_errors = 0

    # 遍历处理
    for root, dirs, files in os.walk(src_root):
        if "new_wav" in root:
            continue

        for filename in files:
            if filename.lower().endswith(".wav"):
                total_files += 1
                src_path = os.path.join(root, filename)

                # 编码转换
                if convert_encoding:
                    try:
                        # 修正为：GBK → Unicode → Shift_JIS（根据实际需求调整）
                        decoded = filename.encode('gbk').decode('shiftjis')
                        filename_gbk = decoded.encode('shiftjis', 'replace').decode('shiftjis')
                    except UnicodeDecodeError:
                        # 二次尝试：直接处理常见日文字符
                        try:
                            filename_gbk = filename.encode('shiftjis', 'replace').decode('shiftjis')
                        except Exception as e:
                            encoding_errors += 1
                            print(f"编码转换失败：{filename}，错误：{str(e)}")
                            filename_gbk = filename
                    except Exception as e:
                        encoding_errors += 1
                        print(f"编码转换失败：{filename}，错误：{str(e)}")
                        filename_gbk = filename
                else:
                    filename_gbk=filename

                # 构建目标路径
                rel_path = os.path.relpath(root, src_root)
                dest_dir = os.path.join(dest_root, rel_path)
                os.makedirs(dest_dir, exist_ok=True)

                # 罗马字替换
                new_name = filename_gbk
                for k in sorted_keys:
                    new_name = new_name.replace(k, hira_dict[k])

                # 记录转换
                if new_name != filename_gbk:
                    converted_files += 1
                    print(f"重命名：{filename[:30]}... → {new_name[:30]}...")

                # 复制文件
                dest_path = os.path.join(dest_dir, new_name)
                try:
                    shutil.copy2(src_path, dest_path)
                except Exception as e:
                    print(f"文件复制失败：{src_path} → {str(e)}")

    # 输出统计
    print(f"\n处理完成！")
    print(f"扫描文件总数：{total_files}")
    print(f"执行重命名数：{converted_files}")
    print(f"编码转换失败数：{encoding_errors}")
    print(f"输出目录：{dest_root}")


if __name__ == "__main__":
    main_process()