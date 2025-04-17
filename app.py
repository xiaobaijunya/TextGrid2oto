import gradio as gr
import main
from tkinter import Tk, filedialog  # 导入文件对话框库

def run():
    demo.launch(server_port=7860, show_error=True, inbrowser=False)

def generate_config(
        wav_path, ds_dict, presamp, cut, ignore,
        VCV_mode, lab, cv_sum, vc_sum, vv_sum,
        cv_offset, vc_offset, pitch, CV_repeat,
        VC_repeat, cover, sofa_model
):
    config = (f"""wav_path={wav_path}\nds_dict={ds_dict}\npresamp={presamp}\ncut={cut}\nignore={ignore}\n
              VCV_mode={VCV_mode}\nlab={lab}\ncv_sum={cv_sum}\nvc_sum={vc_sum}\nvv_sum={vv_sum}\ncv_offset={cv_offset}\n
              vc_offset={vc_offset}\npitch={pitch}\nCV_repeat={CV_repeat}\nVC_repeat={VC_repeat}\ncover={cover}\n
              sofa_model={sofa_model}""")
    with open('config.txt', 'w', encoding='utf-8') as f:
        f.write(config)

    yield "✅ 配置文件已生成，开始执行主程序..."
    main.auto_run('config.txt')
    yield "🎉 任务完成！最终结果：..."

# 定义文件夹选择函数
def select_folder():
    root = Tk()
    root.withdraw()  # 隐藏主窗口
    folder_path = filedialog.askdirectory(title="选择文件夹")  # 打开文件夹选择对话框
    root.destroy()
    return folder_path


def select_file():
    root = Tk()
    root.withdraw()  # 隐藏主窗口

    file_path = filedialog.askopenfilename(
        title="选择文本或INI文件",
        filetypes=[("Text Files", "*.txt;*.ini")]  # 限制文件类型[4,6](@ref)
    )
    return file_path

def model_file():
    root = Tk()
    root.withdraw()  # 隐藏主窗口
    file_path = filedialog.askopenfilename(
        title="选择模型文件",
        filetypes=[("Model Files", "*.ckpt;*.onnx")]  # 限制文件类型[4,6](@ref)
    )
    return file_path

def update_params(voice_type):
    if voice_type == 0:
        return "1,3,1.5,1,2", "3,0,2,1,2", "3,3,1.5,1,1.5", "0,0,0,0,0", "0,0,0,0,0"
    elif voice_type == 1:
        return "1,3,1.5,1,2", "3,3,1.5,1,3,3", "0,0,0,0,0", "0,0,0,0,0", "0,0,0,0,0"
    elif voice_type == 2:
        return "1,3,1,1,2", "5,0,2,1,2", "0,0,0,0,0", "0,0,0,0,0", "0,0,0,0,0"
    elif voice_type == 3:
        return "0,0,0,0,0", "0,0,0,0,0", "0,0,0,0,0", "0,0,0,0,0", "0,0,0,0,0"
    else:
        return "0,0,1.5,1,2", "3,0,2,1,2", "3,3,1.5,1,1.5", "0,0,0,0,0", "0,0,0,0,0"

with gr.Blocks(title="UTAU 参数生成器") as demo:
    gr.Markdown("### 必填参数配置")
    with gr.Row(equal_height=True):
        with gr.Row(equal_height=True):
            with gr.Column(scale=3,min_width=150):
                wav_path = gr.Textbox(label="音源wav路径", placeholder="输入文件夹路径")
            with gr.Column(scale=2,min_width=150):
                folder_btn = gr.Button("选择文件夹", variant="primary")
        with gr.Row(equal_height=True):
            with gr.Column(scale=3,min_width=150):
                presamp = gr.Textbox(label="presamp.ini路径",placeholder="输入文件路径")
            with gr.Column(scale=2,min_width=150):
                presamp_btn = gr.Button("选择文件", variant="primary")
    with gr.Row(equal_height=True):
        with gr.Row(equal_height=True):
            with gr.Column(scale=3,min_width=150):
                ds_dict = gr.Textbox(label="sofa字典路径",placeholder="输入模型路径")
            with gr.Column(scale=2,min_width=150):
                ds_dict_btn = gr.Button("选择文件", variant="primary")
        with gr.Row(equal_height=True):
            with gr.Column(scale=3,min_width=150):
                sofa_model = gr.Textbox(label="sofa模型路径(暂不支持)",placeholder="输入文件路径")
            with gr.Column(scale=2,min_width=150):
                model_btn = gr.Button("选择文件", variant="primary")

    with gr.Row():
        VCV_mode = gr.Radio(
            choices=[("CVVC", 0), ("VCV", 1), ("CV", 2), ("Test", 3)],  # (显示文本, 实际值)
            value=0,  # 默认选中值
            label="音源类型"
        )
        pitch = gr.Textbox(label="音阶后缀",placeholder="例如： F3")

    with gr.Row():
        lab = gr.Radio(choices=["Y", "N"], value="Y", label="生成lab文件")
        cover = gr.Radio(choices=["Y", "N"], value="Y", label="覆盖oto")
    with gr.Row():
        cut = gr.Textbox(label="字符分隔符", value="_,-")
        ignore = gr.Textbox(label="忽略的sofa音素", value="AP,SP")

    gr.Markdown("### 可选参数配置")
    with gr.Accordion("高级参数配置", open=False):
        gr.Markdown("**规则参数（逗号分隔数值）​**​")
        with gr.Row():
            cv_sum = gr.Textbox(label="CV规则比例", value="1,3,1.5,1,2")
            vc_sum = gr.Textbox(label="VC规则比例", value="3,3,1.5,1,3,3")
            vv_sum = gr.Textbox(label="VV规则比例", value="0,0,0,0,0")

            cv_offset = gr.Textbox(label="CV数值偏移量", value="0,0,0,0,0")
            vc_offset = gr.Textbox(label="VC数值偏移量", value="0,0,0,0,0")
        with gr.Row():
            CV_repeat = gr.Textbox(label="CV重复次数", value="1")
            VC_repeat = gr.Textbox(label="VC重复次数", value="1")
    # 定义更新参数的函数

    # 按钮点击事件绑定
    folder_btn.click(
        fn=select_folder,  # 调用文件夹选择函数
        outputs=wav_path,  # 将结果传递给文本框
    )

    # 按钮点击事件绑定
    presamp_btn.click(
        fn=select_file,  # 调用文件夹选择函数
        outputs=presamp,  # 将结果传递给文本框
    )

    ds_dict_btn.click(
        fn=select_file,  # 调用文件夹选择函数
        outputs=ds_dict,  # 将结果传递给文本框
    )

    model_btn.click(
        fn=model_file,  # 调用文件夹选择函数
        outputs=sofa_model,  # 将结果传递给文本框
    )

    VCV_mode.change(
        fn=update_params,
        inputs=VCV_mode,
        outputs=[cv_sum, vc_sum, vv_sum, cv_offset, vc_offset]
    )

    btn = gr.Button("生成配置", variant="primary")
    output = gr.Textbox(label="生成结果",lines=10)

    btn.click(
        fn=generate_config,
        inputs=[
            wav_path, ds_dict, presamp, cut, ignore,
            VCV_mode, lab, cv_sum, vc_sum, vv_sum,
            cv_offset, vc_offset, pitch, CV_repeat,
            VC_repeat, cover,sofa_model
        ],
        outputs=output
    )

if __name__ == "__main__":
    run()