import webview
import app
import threading
import queue
# nuitka --standalone --windows-icon-from-ico=img/TextGrid2oto.ico --output-dir=output --output-filename=TextGrid2oto-WEBUI WEB-UI.py

def run_app():
    app.run()

# 创建一个线程来运行 app
app_thread = threading.Thread(target=run_app)
app_thread.daemon = True  # 将线程设置为守护线程，这样主线程退出时，子线程也会退出
app_thread.start()
w=960*1.2
h=640*1.2
print("启动 PyWebview 窗口...")
window = webview.create_window(
    title='TextGrid2oto',
    url='http://127.0.0.1:7860',
    width=int(w),
    height=int(h),
)

webview.start()
