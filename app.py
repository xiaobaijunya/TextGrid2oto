import gradio as gr
from tkinter import filedialog
import wavname2lab
from textgrid2json import ds_json2filter, word2utau_phone, TextGrid2ds_json, ds_json2word
from json2oto import json2CV_oto, json2oto, json2VCV_oto
from oto import oto_check
from oto import oto_rw
import sys
import os
import subprocess
import click

def run():
    demo.launch(server_port=7860, show_error=True, inbrowser=False)

def generate_config(
        wav_path, ds_dict, presamp, cut, ignore,
        VCV_mode, lab, cv_sum, vc_sum, vv_sum,
        cv_offset, vc_offset, pitch, CV_repeat,
        VC_repeat, cover, sofa_model,SOFA_mode,SOFA_type,progress=gr.Progress()):
    config1 = (
        f"wav_path={wav_path}\n"
        f"ds_dict={ds_dict}\n"
        f"presamp={presamp}\n"
        f"cut={cut}\n"
        f"ignore={ignore}\n"
        f"VCV_mode={VCV_mode}\n"
        f"lab={lab}\n"
        f"cv_sum={cv_sum}\n"
        f"vc_sum={vc_sum}\n"
        f"vv_sum={vv_sum}\n"
        f"cv_offset={cv_offset}\n"
        f"vc_offset={vc_offset}\n"
        f"pitch={pitch}\n"
        f"CV_repeat={CV_repeat}\n"
        f"VC_repeat={VC_repeat}\n"
        f"cover={cover}\n"
        f"sofa_model={sofa_model}"
        f"SOFA_mode={SOFA_mode}\n"
        f"SOFA_type={SOFA_type}\n"
    )
    with open('config.txt', 'w', encoding='utf-8') as f:
        f.write(config1)
        f.write(f'#python infer.py --folder {wav_path} --dictionary {os.path.abspath(ds_dict)} --ckpt {os.path.abspath(sofa_model)} --out_formats textgrid --save_confidence')
    progress(0, desc="✅ 配置文件已生成，开始执行主程序...")
    with open('config.txt', 'r', encoding='utf-8') as f:
        config = f.read().split('\n')
        for i in range(len(config)):
            config[i] = config[i].strip()
        # 转为字典
        config = {config[i].split('=')[0]: config[i].split('=')[1] for i in range(len(config)) if
                  config[i] != '' and not config[i].startswith('#')}  # 修改判断条件为跳过以#开头的行
        # 将字符串转换为列表并将每个元素转换为float类型
        config['cut'] = config['cut'].split(',')
        config['cv_sum'] = [float(i) for i in config['cv_sum'].strip('[]').split(',')]
        config['vc_sum'] = [float(i) for i in config['vc_sum'].strip('[]').split(',')]
        config['vv_sum'] = [float(i) for i in config['vv_sum'].strip('[]').split(',')]
        config['cv_offset'] = [float(i) for i in config['cv_offset'].strip('[]').split(',')]
        config['vc_offset'] = [float(i) for i in config['vc_offset'].strip('[]').split(',')]
        config['TextGrid_path'] = config['wav_path'] + '/TextGrid'
    progress(0.1,'1.配置文件读取成功')
    if config['lab'] == 'Y' or config['lab'] == 'y':
        progress(0.2,'1.生成lab')
        wavname2lab.run(config['wav_path'], config['cut'])
    if SOFA_mode == 0:
        progress(0.3, '2.正在前往sofa生成TextGrid')
        sys.path.append('SOFA')
        from SOFA import infer
        print(f'--folder {wav_path} --dictionary {os.path.abspath(ds_dict)} --ckpt {os.path.abspath(sofa_model)} --out_formats textgrid --save_confidence')
        sys.argv = f'--folder {wav_path} --dictionary {os.path.abspath(ds_dict)} --ckpt {os.path.abspath(sofa_model)} --out_formats textgrid --save_confidence'
        # infer.main(ckpt=os.path.abspath(sofa_model),mode='force',ap_detector='LoudnessSpectralcentroidAPDetector',g2p=os.path.abspath(ds_dict),folder=wav_path,out_formats='textgrid',in_format='lab',save_confidence=True)
        with click.Context(infer.main) as ctx:
            result = ctx.invoke(
                infer.main,
                ckpt=os.path.abspath(sofa_model),
                folder=wav_path,
                dictionary=os.path.abspath(ds_dict),
                out_formats='textgrid',
                save_confidence=True
            )
    VCV_mode = config['VCV_mode']
    if not VCV_mode:
        VCV_mode = '0'
    progress(0.4,'3.生成json')
    TextGrid2ds_json.run(config['TextGrid_path'])
    ds_json2filter.run(config['ds_dict'], config['TextGrid_path'] + '/json/ds_phone.json', config['ignore'])
    progress(0.5,'3.生成word.json')
    ds_json2word.run(config['ds_dict'], config['TextGrid_path'] + '/json/ds_phone_filter.json')
    progress(0.6,'6.生成oto.ini')
    if VCV_mode == '1':
        progress(0.6,'生成模式：VCV')
        json2VCV_oto.run(config['presamp'], config['TextGrid_path'] + '/json/utau_phone.json',
                         config['TextGrid_path'] + '/json/word_phone.json',
                         config['wav_path'], config['cv_sum'], config['vc_sum'], config['vv_sum'])
    elif VCV_mode == '2':
        progress(0.6,'生成模式：CVV')
        json2CV_oto.run(config['presamp'], config['TextGrid_path'] + '/json/utau_phone.json',
                        config['TextGrid_path'] + '/json/word_phone.json',
                        config['wav_path'], config['cv_sum'], config['vc_sum'], config['vv_sum'])
    elif VCV_mode == '0':
        progress(0.6,'生成模式：CVVC')
        json2oto.run(config['presamp'], config['TextGrid_path'] + '/json/utau_phone.json',
                     config['TextGrid_path'] + '/json/word_phone.json',
                     config['wav_path'], config['cv_sum'], config['vc_sum'], config['vv_sum'])
    else:
        progress(0.6,'VCV_mode数值错误')
        input(0.6,'退出')
        return 'VCV_mode数值错误'
    progress(0.7,'7.读取CV和VC oto.ini')
    cv = oto_rw.oto_read(config['wav_path'] + '/cv_oto.ini')
    vc = oto_rw.oto_read(config['wav_path'] + '/vc_oto.ini')
    progress(0.7,'8.剔除重复项')
    cv = oto_rw.oto_repeat(cv, int(config['CV_repeat']))
    vc = oto_rw.oto_repeat(vc, int(config['VC_repeat']))
    progress(0.7,'9.偏移oto数值.ini')
    if config['cv_offset'] != [0.0, 0.0, 0.0, 0.0, 0.0]:
        cv = oto_rw.oto_offset(cv, config['cv_offset'])
        progress(0.7,'9.1.偏移CV数值,运行成功')
    if config['vc_offset'] != [0.0, 0.0, 0.0, 0.0, 0.0]:
        vc = oto_rw.oto_offset(vc, config['vc_offset'])
        progress(0.7,'9.1.偏移VC数值,运行成功')
    progress(0.8,'10.合并oto.ini')
    oto_rw.oto_write(config['wav_path'] + '/oto.ini', cv + vc, config['pitch'], config['cover'])
    progress(0.9,'11.检测缺少的音素')
    oto_check.run(config['wav_path'] + '/oto.ini', config['presamp'], config['pitch'], config['VCV_mode'])
    progress(1,"🎉 任务完成！最终结果：")
    return "🎉 任务完成！最终结果：去命令行窗口查看"


# 定义文件夹选择函数
def select_folder():
    folder_path = filedialog.askdirectory(title="选择文件夹")  # 打开文件夹选择对话框
    return folder_path


def select_file():
    file_path = filedialog.askopenfilename(
        title="选择文本或INI文件",
        filetypes=[("Text Files", "*.txt;*.ini")]
    )
    return file_path

def model_file():
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

def scan_model_folder():
    model_dir = "model"
    if os.path.exists(model_dir) and os.path.isdir(model_dir):
        sub_folders = [d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))]
        return sub_folders
    return []

def scan_presamp_folder():
    presamp = "presamp"
    if presamp:
        all_files = [f for f in os.listdir(presamp) if os.path.isfile(os.path.join(presamp, f))]
        print(all_files)
        return all_files
    return []

# 定义选择文件夹后更新 ds_dict 和 sofa_model 的函数
def update_model_paths(selected_folder):
    folder_path = os.path.join("model", selected_folder)
    txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    ckpt_files = [f for f in os.listdir(folder_path) if f.endswith('.ckpt')]
    ds_dict_path = os.path.join(folder_path, txt_files[0]) if txt_files else ""
    sofa_model_path = os.path.join(folder_path, ckpt_files[0]) if ckpt_files else ""
    return ds_dict_path, sofa_model_path

def update_presamp_paths(selected_folder):
    folder_path = os.path.join("presamp", selected_folder)
    return folder_path

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
        SOFA_mode = gr.Radio(
            choices=[("开启", 0), ("关闭", 1)],  # (显示文本, 实际值)
            value=0,  # 默认选中值
            label="是否生成TextGrid"
        )
        SOFA_type = gr.Radio(
            choices=[("SOFA", 0), ("HubertFA", 1)],  # (显示文本, 实际值)
            value=0,  # 默认选中值
            label="选择标记程序（目前仅sofa模型）"
        )
        # 替换原按钮为下拉框
        model_folders = scan_model_folder()
        model_folder_selector = gr.Dropdown(choices=model_folders, label="选择自带模型",value='')
        model_presamp = scan_presamp_folder()
        model_presamp_selector = gr.Dropdown(choices=model_presamp, label="选择自带presamp",value='')

    with gr.Row(equal_height=True):
        with gr.Row(equal_height=True):
            with gr.Column(scale=3,min_width=150):
                ds_dict = gr.Textbox(label="sofa字典路径",placeholder="输入模型路径")
            with gr.Column(scale=2,min_width=150):
                ds_dict_btn = gr.Button("选择文件", variant="primary")
        with gr.Row(equal_height=True):
            with gr.Column(scale=3,min_width=150):
                sofa_model = gr.Textbox(label="sofa模型路径",placeholder="输入文件路径")
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
            vv_sum = gr.Textbox(label="VV规则比例", value="3,3,1.5,1,1.5")

            cv_offset = gr.Textbox(label="CV数值偏移量", value="0,0,0,0,0")
            vc_offset = gr.Textbox(label="VC数值偏移量", value="0,0,0,0,0")
        with gr.Row():
            CV_repeat = gr.Textbox(label="CV重复次数", value="1")
            VC_repeat = gr.Textbox(label="VC重复次数", value="1")
    # 定义更新参数的函数

    model_folder_selector.change(
        fn=update_model_paths,
        inputs=model_folder_selector,
        outputs=[ds_dict, sofa_model]
    )
    model_presamp_selector.change(
        fn=update_presamp_paths,
        inputs=model_presamp_selector,
        outputs=presamp
    )
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
            VC_repeat, cover,sofa_model,SOFA_mode,SOFA_type
        ],
        outputs=output
    )

if __name__ == "__main__":
    run()