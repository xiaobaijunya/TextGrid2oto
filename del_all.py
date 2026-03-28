import os

# ====================== 【请修改这里为你的目标路径】 ======================
target_path = r"F:\Download\SVDBCreator_Release_1.0.0\baini_JP\wav"  # Windows示例


# target_path = "/home/你的/目标/文件夹/路径"  # Linux/Mac示例
# =========================================================================

def clean_directory(path):
    """
    清理指定目录：
    1. 删除所有非.wav后缀的文件
    2. 删除所有空文件夹
    """
    # 检查路径是否存在
    if not os.path.exists(path):
        print(f"❌ 错误：路径 {path} 不存在！")
        return

    # 第一步：遍历所有文件，删除非wav文件
    print("🔍 开始删除非wav文件...")
    for root, dirs, files in os.walk(path, topdown=False):
        # 处理文件
        for file in files:
            file_path = os.path.join(root, file)
            # 判断文件后缀是否不是.wav（不区分大小写）
            if not file.lower().endswith(".wav"):
                try:
                    os.remove(file_path)
                    print(f"✅ 已删除文件：{file_path}")
                except Exception as e:
                    print(f"❌ 删除文件失败 {file_path}：{str(e)}")

    # 第二步：遍历所有文件夹，删除空文件夹
    print("\n🔍 开始删除空文件夹...")
    for root, dirs, files in os.walk(path, topdown=False):
        # 处理文件夹
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            # 判断文件夹是否为空
            if len(os.listdir(dir_path)) == 0:
                try:
                    os.rmdir(dir_path)
                    print(f"✅ 已删除空文件夹：{dir_path}")
                except Exception as e:
                    print(f"❌ 删除文件夹失败 {dir_path}：{str(e)}")

    print("\n🎉 清理完成！仅保留.wav文件，所有空文件夹已删除。")


if __name__ == "__main__":
    # 二次确认，防止误运行
    confirm = input(f"⚠️ 确认要清理路径 {target_path} 吗？此操作不可恢复！(y/n)：")
    if confirm.lower() == "y":
        clean_directory(target_path)
    else:
        print("🚫 已取消清理操作。")