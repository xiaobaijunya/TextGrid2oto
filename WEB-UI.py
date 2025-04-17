import webview
import app
import threading
import queue
# nuitka --standalone --windows-icon-from-ico=img/TextGrid2oto.ico --output-dir=output --output-filename=TextGrid2oto-WEBUI WEB-UI.py

# 创建一个队列用于线程间通信
file_dialog_queue = queue.Queue()

def run_app():
    app.run()

# 修改 app.py 中的 select_file 函数，使其使用队列进行通信
# 这里假设可以修改 app.py，添加以下代码到 app.py 中
# 在 app.py 顶部添加
import threading
import queue
file_dialog_queue = queue.Queue()

def select_file():
    def show_dialog():
        from tkinter import Tk, filedialog
        root = Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(
            title="选择文本或INI文件",
            filetypes=[("Text Files", "*.txt;*.ini")]
        )
        file_dialog_queue.put(file_path)
    threading.Thread(target=show_dialog).start()
    return file_dialog_queue.get()


# 创建一个线程来运行 app
app_thread = threading.Thread(target=run_app)
app_thread.daemon = True  # 将线程设置为守护线程，这样主线程退出时，子线程也会退出
app_thread.start()


print("启动 PyWebview 窗口...")
window = webview.create_window(
    title='TextGrid2oto',
    url='http://127.0.0.1:7860',
    width=850,
    height=540
)

# 启动 PyWebview，只保留一次调用
webview.start()
webview.start()