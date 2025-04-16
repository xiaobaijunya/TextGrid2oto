import gradio as gr

def run():
    demo.launch(server_port=7860, show_error=True, inbrowser=False)

def generate_config(
        wav_path, ds_dict, presamp, cut, ignore,
        VCV_mode, lab, cv_sum, vc_sum, vv_sum,
        cv_offset, vc_offset, pitch, CV_repeat,
        VC_repeat, cover, sofa_model
):
    # 这里可以添加参数处理逻辑
    return "配置生成成功！"

def update_params(voice_type):
    if voice_type == "0":
        return "1,3,1.5,1,2", "3,0,2,1,2", "3,3,1.5,1,1.5", "0,0,0,0,0", "0,0,0,0,0"
    elif voice_type == "1":
        return "1,3,1.5,1,2", "3,3,1.5,1,3,3", "0,0,0,0,0", "0,0,0,0,0", "0,0,0,0,0"
    elif voice_type == "2":
        return "1,3,1,1,2", "5,0,2,1,2", "0,0,0,0,0", "0,0,0,0,0", "0,0,0,0,0"
    else:
        return "1,3,1.5,1,2", "3,0,2,1,2", "3,3,1.5,1,1.5", "0,0,0,0,0", "0,0,0,0,0"


with gr.Blocks(title="UTAU 参数生成器") as demo:
    gr.Markdown("### 必填参数配置")
    with gr.Row():
        wav_path = gr.Textbox(label="音源wav路径", placeholder="输入文件夹路径")
        presamp = gr.Textbox(label="presamp.ini路径",placeholder="输入文件路径")
    with gr.Row():
        ds_dict = gr.Textbox(label="sofa字典路径",placeholder="输入模型路径")
        sofa_model = gr.Textbox(label="sofa模型路径",placeholder="输入文件路径")

    with gr.Row():
        pitch = gr.Textbox(label="音阶后缀",placeholder="例如： F3")
        VCV_mode = gr.Radio(
            choices=[("CVVC", 0), ("VCV", 1), ("CV", 2), ("Test", 3)],  # (显示文本, 实际值)
            value=0,  # 默认选中值
            label="音源类型"
        )
    with gr.Row():
        lab = gr.Radio(choices=["Y", "N"], value="Y", label="生成lab文件")
        cover = gr.Radio(choices=["Y", "N"], value="Y", label="覆盖oto")
    with gr.Row():
        cut = gr.Textbox(label="字符分隔符", value="_,-")
        ignore = gr.Textbox(label="忽略的sofa音素", value="AP,SP")

    gr.Markdown("### 可选参数配置")
    with gr.Accordion("高级参数配置", open=False):
        with gr.Row():
            gr.Markdown("**规则参数（逗号分隔数值）​**​")
            cv_sum = gr.Textbox(label="CV规则比例", value="1,3,1.5,1,2")
            vc_sum = gr.Textbox(label="VC规则比例", value="3,3,1.5,1,3,3")
            vv_sum = gr.Textbox(label="VV规则比例", value="0,0,0,0,0")

            cv_offset = gr.Textbox(label="CV数值偏移量", value="0,0,0,0,0")
            vc_offset = gr.Textbox(label="VC数值偏移量", value="0,0,0,0,0")
        with gr.Row():
            CV_repeat = gr.Textbox(label="CV重复次数", value="1")
            VC_repeat = gr.Textbox(label="VC重复次数", value="1")
    # 定义更新参数的函数

    # 将函数与组件绑定
    VCV_mode.change(
        fn=update_params,
        inputs=VCV_mode,
        outputs=[cv_sum, vc_sum, vv_sum, cv_offset, vc_offset]
    )

    btn = gr.Button("生成配置", variant="primary")
    output = gr.Textbox(label="生成结果")

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