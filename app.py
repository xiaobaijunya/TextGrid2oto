import os
import time
start=time.time()
print('hello')
import certifi
os.environ["SSL_CERT_FILE"] = certifi.where()

# 添加torch GPU检测代码
print("=== Torch 环境检测 ===")
try:
    import torch
    has_torch = True
    print(f"PyTorch 版本: {torch.__version__}")

    # 检测是否有可用的CUDA
    cuda_available = torch.cuda.is_available()
    print(f"CUDA 可用: {cuda_available}")

    if cuda_available:
        # 获取GPU数量和详细信息
        gpu_count = torch.cuda.device_count()
        print(f"GPU 数量: {gpu_count}")

        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f"GPU {i}: {gpu_name}")
            print(f"  总显存: {gpu_mem:.2f} GB")
            print(f"  CUDA 能力: {torch.cuda.get_device_capability(i)}")
        print("当前使用的设备: CUDA")
    else:
        # 检查是否有MPS支持 (MacOS)
        try:
            mps_available = torch.backends.mps.is_available()
            print(f"MPS 可用: {mps_available}")
            if mps_available:
                print("当前使用的设备: MPS (Apple Silicon)")
            else:
                print("当前使用的设备: CPU")
        except:
            print("当前使用的设备: CPU")

    print("=== Torch 环境检测完成 ===")
except ImportError:
    print("PyTorch 未安装")
    print("当前使用的设备: CPU")
    print("=== Torch 环境检测完成 ===")

import gradio as gr

from tkinter import filedialog
import wavname2lab
from textgrid2json import ds_json2filter, word2utau_phone, TextGrid2ds_json, ds_json2word,transcriptions_make,del_SP
from json2oto import json2CV_oto, json2oto, json2VCV_oto ,json2test
from oto import oto_check
from oto import oto_rw
import sys
import shutil
import pathlib
import click
from build_diffdataset import build_dataset

def run():
    demo.launch(
        server_port=7861,
        show_error=True,
        inbrowser=False,
        # share=False,  # 关闭分享功能
        # debug=False,  # 关闭调试模式
        # auth=None,  # 关闭认证功能
        # favicon_path=None,  # 不加载favicon，减少请求
    )
print(f'耗时: {time.time()-start:.4f}s')


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


#多音阶或单音阶分配任务
def config_generator_dispatcher(
        wav_path, ds_dict, presamp, cut, ignore,
        VCV_mode, lab, cv_sum, vc_sum, vv_sum,
        cv_offset, vc_offset, pitch, CV_repeat,
        VC_repeat, clear_tg_cache, cover, sofa_model, SOFA_mode, SOFA_type, multi_pitch_mode, delete_sp,rec_preset,oto_preset,oto_encoding, progress=gr.Progress()):
    """
    根据多音阶模式开关状态，决定调用哪个配置生成函数
    """
    # 如果开启了多音阶模式，调用专门的多音阶处理函数
    if multi_pitch_mode == 1:
        return generate_config_multi_pitch(
            wav_path, ds_dict, presamp, cut, ignore,
            VCV_mode, lab, cv_sum, vc_sum, vv_sum,
            cv_offset, vc_offset, pitch, CV_repeat,
            VC_repeat, clear_tg_cache, cover, sofa_model, SOFA_mode, SOFA_type, delete_sp,rec_preset,oto_preset,oto_encoding, progress
        )
    else:
        # 否则调用单音阶处理函数
        return generate_config(
            wav_path, ds_dict, presamp, cut, ignore,
            VCV_mode, lab, cv_sum, vc_sum, vv_sum,
            cv_offset, vc_offset, pitch, CV_repeat,
            VC_repeat, clear_tg_cache, cover, sofa_model, SOFA_mode, SOFA_type, delete_sp,rec_preset,oto_preset,oto_encoding, progress
        )

#多音阶启动
def generate_config_multi_pitch(
        wav_path, ds_dict, presamp, cut, ignore,
        VCV_mode, lab, cv_sum, vc_sum, vv_sum,
        cv_offset, vc_offset, pitch, CV_repeat,
        VC_repeat, clear_tg_cache, cover, sofa_model, SOFA_mode, SOFA_type, delete_sp,rec_preset,oto_preset,oto_encoding, progress=gr.Progress()):
    # 获取主文件夹下的所有子文件夹
    subfolders = [f for f in os.listdir(wav_path) if os.path.isdir(os.path.join(wav_path, f))]
    print(f'获取到以下音阶文件夹：{subfolders}')
    if not subfolders:
        return "错误：未找到子文件夹，请检查路径是否正确"

    # 为每个子文件夹生成结果
    results = []
    for i, subfolder in enumerate(subfolders):
        subfolder_path = os.path.join(wav_path, subfolder)

        # 设置当前子文件夹的音阶后缀为空格+文件夹名
        # current_pitch = f" {subfolder}"
        current_pitch = f"{subfolder}"
        print(f"处理子文件夹 {subfolder} (音阶: {current_pitch})")

        # 调用原始处理逻辑，但使用子文件夹路径和当前音阶
        result = generate_config(
            subfolder_path, ds_dict, presamp, cut, ignore,
            VCV_mode, lab, cv_sum, vc_sum, vv_sum,
            cv_offset, vc_offset, current_pitch, CV_repeat,
            VC_repeat, clear_tg_cache, cover, sofa_model, SOFA_mode, SOFA_type, delete_sp, rec_preset, oto_preset,
            oto_encoding, progress=gr.Progress())

        if result and "错误" in result:
            return f"处理子文件夹 {subfolder} 失败: {result}"
        results.append(f"子文件夹 {subfolder} 处理完成")

    return f"🎉 多音阶模式任务完成！已成功处理 {len(subfolders)} 个子文件夹\n" + "\n".join(results)

#核心运行代码
def generate_config(
        wav_path, ds_dict, presamp, cut, ignore,
        VCV_mode, lab, cv_sum, vc_sum, vv_sum,
        cv_offset, vc_offset, pitch, CV_repeat,
        VC_repeat, clear_tg_cache,cover, sofa_model,SOFA_mode,SOFA_type,delete_sp,rec_preset,oto_preset,oto_encoding,progress=gr.Progress()):
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
        f"clear_tg_cache={clear_tg_cache}\n"
        f"cover={cover}\n"
        f"sofa_model={sofa_model}\n"
        f"SOFA_mode={SOFA_mode}\n"
        f"SOFA_type={SOFA_type}\n"
        f"delete_sp={delete_sp}\n"
        f"rec_preset={rec_preset}\n"
        f"oto_preset={oto_preset}\n"
        f"oto_encoding={oto_encoding}\n"
    )
    deleted_sp_list = []
    with open('config.txt', 'w', encoding='utf-8') as f:
        f.write(config1)
        if SOFA_type == 0:
            f.write(f'#python infer.py --folder {wav_path} --dictionary {os.path.abspath(ds_dict)} --ckpt {os.path.abspath(sofa_model)} --out_formats textgrid --save_confidence')
        elif SOFA_type == 1:
            if sofa_model.split('.')[-1] == 'onnx':
                f.write(f'#python onnx_infer.py --onnx_path {os.path.dirname(os.path.abspath(sofa_model))} --wav_folder {wav_path} --language {ds_dict.split('\\')[-1].split('/')[-1].split('.')[0].split('-')[0]} --dictionary {os.path.abspath(ds_dict)}')
            else:
                f.write(f'#python infer.py --ckpt {os.path.abspath(sofa_model)} --folder {wav_path} --language {ds_dict.split('\\')[-1].split('/')[-1].split('.')[0]} --dictionary {os.path.abspath(ds_dict)} --save_confidence')
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
    print(config['wav_path'])
    progress(0.1,'1.配置文件读取成功')
    if config['lab'] == 'Y' or config['lab'] == 'y':
        progress(0.2,'1.生成lab')
        wavname2lab.run(config['wav_path'], config['cut'])
        transcriptions_make.create_transcriptions_csv(config['wav_path'], config['ds_dict'])
    if config['clear_tg_cache']== 'Y' or config['clear_tg_cache'] == 'y':
        progress(0.2,'1.1 删除TextGrid')
        print('正在删除TextGrid')
        textgrid_path2 = config['wav_path'] + '/TextGrid'
        if os.path.exists(textgrid_path2):
            shutil.rmtree(textgrid_path2)
        else:
            print('TextGrid文件夹不存在')
        # 删除confidence目录（如果存在）
        confidence_path2 = config['wav_path'] + '/transcriptions.csv'
        if os.path.exists(confidence_path2):
            os.remove(confidence_path2)
        else:
            print('transcriptions.csv不存在')
    if SOFA_mode == 0:
        if SOFA_type == 0:
            progress(0.3, '2.正在前往sofa生成TextGrid')
            print( '2.正在前往sofa生成TextGrid')
            sys.path.append('SOFA')
            # from SOFA import infer
            # print(f'--folder {wav_path} --dictionary {os.path.abspath(ds_dict)} --ckpt {os.path.abspath(sofa_model)} --out_formats textgrid --save_confidence')
            # sys.argv = f'--folder {wav_path} --dictionary {os.path.abspath(ds_dict)} --ckpt {os.path.abspath(sofa_model)} --out_formats textgrid --save_confidence'
            # # infer.main(ckpt=os.path.abspath(sofa_model),mode='force',ap_detector='LoudnessSpectralcentroidAPDetector',g2p=os.path.abspath(ds_dict),folder=wav_path,out_formats='textgrid',in_format='lab',save_confidence=True)
            # with click.Context(infer.main) as ctx:
            #     result = ctx.invoke(
            #         infer.main,
            #         ckpt=os.path.abspath(sofa_model),
            #         folder=wav_path,
            #         dictionary=pathlib.Path(os.path.abspath(ds_dict)),
            #         out_formats='textgrid',
            #         save_confidence=True
            #     )
        elif SOFA_type == 1:
            if sofa_model.split('.')[-1] == 'ckpt':
                progress(1, "❌ 已经不支持.ckpt模型，请使用.onnx模型！")
                print("❌ 已经不支持.ckpt模型，请使用.onnx模型！")
                return "❌ 已经不支持.ckpt模型，请使用.onnx模型！"
                # progress(0.3, '2.正在前往HubertFA生成TextGrid')
                # print('2.正在前往HubertFA生成TextGrid')
                # sys.path.append('HubertFA')
                # from HubertFA import infer
                # print(f'--ckpt {os.path.abspath(sofa_model)} --folder {wav_path} --language {ds_dict.split('\\')[-1].split('/')[-1].split('.')[0].split('-')[0]} --dictionary {os.path.abspath(ds_dict)} --save_confidence')
                # with click.Context(infer.infer) as ctx:
                #     result = ctx.invoke(
                #         infer.infer,
                #         nll_path=pathlib.Path(os.path.abspath(sofa_model)),  # 使用与fa_path相同的路径作为nll_path
                #         fa_path=pathlib.Path(os.path.abspath(sofa_model)),
                #         wav_folder=pathlib.Path(wav_path),
                #         language=ds_dict.split('\\')[-1].split('/')[-1].split('.')[0].split('-')[0],#忽略-以后的内容
                #         dictionary=pathlib.Path(os.path.abspath(ds_dict)),
                #         encoder=pathlib.Path('dependencies')
                #     )
            else:
                progress(0.3, '2.正在前往HubertFA生成TextGrid')
                print('2.正在前往HubertFA生成TextGrid')
                sys.path.append('HubertFA')
                from HubertFA import onnx_infer
                print(f'--onnx_path {os.path.dirname(os.path.abspath(sofa_model))} --wav_folder {wav_path} --language {ds_dict.split('\\')[-1].split('/')[-1].split('.')[0].split('-')[0]} --dictionary {os.path.abspath(ds_dict)}')
                with click.Context(onnx_infer.infer) as ctx:
                    result = ctx.invoke(
                        onnx_infer.infer,
                        onnx_path=pathlib.Path(sofa_model),  # 修改这里：使用onnx_folder而不是ckpt
                        wav_folder=pathlib.Path(wav_path),
                        language=ds_dict.split('\\')[-1].split('/')[-1].split('.')[0].split('-')[0],  # 忽略-以后的内容
                        dictionary=pathlib.Path(os.path.abspath(ds_dict)),
                    )
            print('已执行HubertFA')
            if config['delete_sp'] == "Y" or config['delete_sp'] == "y":
                delete_sp_switch = True
                print('删除错误的SP标记')
            else:
                delete_sp_switch = False
                print('跳过删除SP标记')
            deleted_sp_list = del_SP.process_all_textgrid_files(wav_path + '/TextGrid', config['ignore'],delete_sp_switch)
    VCV_mode = config['VCV_mode']
    if not VCV_mode:
        VCV_mode = '0'
    progress(0.4,'3.生成json')
    TextGrid2ds_json.run(config['TextGrid_path'],config['rec_preset'])
    ds_json2filter.run(config['ds_dict'], config['TextGrid_path'] + '/json/ds_phone.json', config['ignore'])
    progress(0.5,'3.生成word.json')
    ds_json2word.run(config['ds_dict'], config['TextGrid_path'] + '/json/ds_phone_filter.json')
    progress(0.6,'6.生成oto.ini')
    if VCV_mode == '1':
        progress(0.6,'生成模式：VCV')
        json2VCV_oto.run(config['presamp'], config['TextGrid_path'] + '/json/utau_phone.json',
                         config['TextGrid_path'] + '/json/word_phone.json',
                         config['wav_path'], config['cv_sum'], config['vc_sum'], config['vv_sum'], config['ignore'])
    elif VCV_mode == '2':
        progress(0.6,'生成模式：CVV')
        json2CV_oto.run(config['presamp'], config['TextGrid_path'] + '/json/utau_phone.json',
                        config['TextGrid_path'] + '/json/word_phone.json',
                        config['wav_path'], config['cv_sum'], config['vc_sum'], config['vv_sum'], config['ignore'])
    elif VCV_mode == '0':
        progress(0.6,'生成模式：CVVC')
        json2oto.run(config['presamp'], config['TextGrid_path'] + '/json/utau_phone.json',
                     config['TextGrid_path'] + '/json/word_phone.json',
                     config['wav_path'], config['cv_sum'], config['vc_sum'], config['vv_sum'],config['ignore'])
    elif VCV_mode == '3':
        progress(0.6,'生成模式：Test')
        json2test.run(config['presamp'], config['TextGrid_path'] + '/json/utau_phone.json',
                     config['TextGrid_path'] + '/json/word_phone.json',
                     config['wav_path'], config['cv_sum'], config['vc_sum'], config['vv_sum'], config['ignore'])

    else:
        progress(0.6,'VCV_mode数值错误')
        input(0.6,'退出')
        return 'VCV_mode数值错误'
    progress(0.7,'7.读取CV和VC oto.ini')
    cv = oto_rw.oto_read(config['wav_path'] + '/cv_oto.ini')
    vc = oto_rw.oto_read(config['wav_path'] + '/vc_oto.ini')
    progress(0.7,'8.剔除重复项')
    cv = oto_rw.oto_repeat(cv, int(config['CV_repeat']), config['oto_preset'])
    vc = oto_rw.oto_repeat(vc, int(config['VC_repeat']), config['oto_preset'])
    progress(0.7,'9.偏移oto数值.ini')
    if config['cv_offset'] != [0.0, 0.0, 0.0, 0.0, 0.0]:
        cv = oto_rw.oto_offset(cv, config['cv_offset'])
        progress(0.7,'9.1.偏移CV数值,运行成功')
    if config['vc_offset'] != [0.0, 0.0, 0.0, 0.0, 0.0]:
        vc = oto_rw.oto_offset(vc, config['vc_offset'])
        progress(0.7,'9.1.偏移VC数值,运行成功')
    progress(0.8,'10.合并oto.ini')
    oto_rw.oto_write(config['wav_path'] + '/oto.ini', cv + vc, config['pitch'], config['cover'],  config['oto_encoding'])
    progress(0.9,'11.检测缺少的音素')
    oto_check.run(config['wav_path'] + '/oto.ini', config['presamp'], config['pitch'], config['VCV_mode'])
    if deleted_sp_list:
        print(f"以下音频标记可能有错误，请检查tg标记：")
        if deleted_sp_list:
            for filename in deleted_sp_list:
                print(f"{filename}", end=',')
    progress(1,"🎉 任务完成！最终结果：")
    return "🎉 任务完成！最终结果：去命令行窗口查看。"



def update_params(voice_type):
    #CVVC
    if voice_type == 0:
        return "1,3,1.5,1,4", "3,0,2,1,3", "3,3,1.5,1,3", "0,0,0,0,0", "0,0,0,0,0"
    #VCV
    elif voice_type == 1:
        return "1,3,1.5,1,2", "2.5,3,1.5,1,3", "0,0,0,0,0", "0,0,0,0,0", "0,0,0,0,0"
    #CV
    elif voice_type == 2:
        return "1,3,1,1,3", "5,0,2,1,3", "0,0,0,0,6", "0,0,0,0,0", "0,0,0,0,0"
    #test
    elif voice_type == 3:
        return "1,8,1.5,1,4", "3,0,2,1,2", "3,3,1.5,1,2", "0,0,0,0,0", "0,0,0,0,0"
    else:
        return "0,0,1.5,1,2", "3,0,2,1,3", "3,3,1.5,1,3", "0,0,0,0,0", "0,0,0,0,0"

def scan_model_folder(SOFA_type):
    model_dir = "HubertFA_model"
    if SOFA_type == 0:
        model_dir = "SOFA_model"
    elif SOFA_type == 1:
        model_dir = "HubertFA_model"
    if os.path.exists(model_dir) and os.path.isdir(model_dir):
        sub_folders = [d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))]
        print(sub_folders)
        return gr.Dropdown(choices=sub_folders)
    return []

def scan_presamp_folder():
    presamp = "presamp"
    if presamp:
        all_files = [f for f in os.listdir(presamp) if os.path.isfile(os.path.join(presamp, f))]
        print(all_files)
        return all_files
    return []

# 定义选择文件夹后更新 ds_dict 和 sofa_model 的函数
def update_model_paths(SOFA_type, selected_folder):
    model_dir = "HubertFA_model"
    if SOFA_type == 0:
        model_dir = "SOFA_model"
    elif SOFA_type == 1:
        model_dir = "HubertFA_model"

    folder_path = os.path.join(model_dir, selected_folder)
    txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    ckpt_files = [f for f in os.listdir(folder_path) if f.endswith('.ckpt') or f.endswith('.onnx')]

    ds_dict_path = os.path.join(folder_path, txt_files[0]) if txt_files else ""
    sofa_model_path = os.path.join(folder_path, ckpt_files[0]) if ckpt_files else ""
    ds_dict_path = os.path.abspath(ds_dict_path)
    sofa_model_path = os.path.abspath(sofa_model_path)
    # 使用 update() 方法更新下拉选项
    return (
        ds_dict_path,
        sofa_model_path,
        gr.Dropdown(choices=txt_files, value=txt_files[0]),
        gr.Dropdown(choices=ckpt_files, value=ckpt_files[0])
    )

def update_dict_paths(sofa_model,dict_folders):
    folder_path = os.path.dirname(sofa_model)
    folder_path = os.path.join(folder_path, dict_folders)
    folder_path = os.path.abspath(folder_path)
    return folder_path

def update_model_version_paths(sofa_model,model_version_folder_selector):
    folder_path = os.path.dirname(sofa_model)
    folder_path = os.path.join(folder_path, model_version_folder_selector)
    folder_path = os.path.abspath(folder_path)
    return folder_path

def update_presamp_paths(selected_folder):
    folder_path = os.path.join("presamp", selected_folder)
    folder_path = os.path.abspath(folder_path)
    return folder_path

with gr.Blocks(title="UTAU oto生成器") as demo:
    # gr.Markdown("<h1 style='text-align: center;'>UTAU 参数生成器</h1>")
    #
    # # 添加顶部选项卡
    with gr.Tabs(elem_classes=["custom-tabs"]):
        with gr.TabItem("oto生成"):
    #     # 原有的主配置界面
            gr.Markdown("### 必填参数配置")
            with gr.Row(equal_height=True):
                with gr.Row(equal_height=True):
                    with gr.Column(scale=3,min_width=150):
                        wav_path = gr.Textbox(label="音源wav路径", placeholder="输入文件夹路径")
                    with gr.Column(scale=2,min_width=150):
                        folder_btn = gr.Button("选择文件夹", variant="primary")
                with gr.Row(equal_height=True):
                    with gr.Column(scale=3,min_width=150):
                        presamp = gr.Textbox(label="presamp.ini路径",placeholder="输入文件路径",value='\\presamp.ini')
                    with gr.Column(scale=2,min_width=150):
                        presamp_btn = gr.Button("选择文件", variant="primary")
            with gr.Row(equal_height=True):
                SOFA_mode = gr.Radio(
                    choices=[("开启", 0), ("关闭", 1)],  # (显示文本, 实际值)
                    value=0,  # 默认选中值
                    label="是否生成TextGrid"
                )
                # SOFA_type = gr.Radio(
                #     choices=[("SOFA", 0), ("HubertFA", 1)],  # (显示文本, 实际值)
                #     value=1,  # 默认选中值
                #     label="选择标记程序"
                # )
                SOFA_type = gr.Radio(
                    choices=[ ("HubertFA", 1)],  # (显示文本, 实际值)
                    value=1,  # 默认选中值
                    label="选择标记程序"
                )
                model_folder_selector = gr.Dropdown(choices=[], label="选择模型文件夹",value='')
                model_version_folder_selector = gr.Dropdown(choices=[], label="选择模型",value='')
                dict_folders_selector = gr.Dropdown(choices=[], label="选择模型字典",value='')
                model_presamp = scan_presamp_folder()
                model_presamp_selector = gr.Dropdown(choices=model_presamp, label="选择presamp(优先使用录音表提供的)",value='')

            with gr.Row(equal_height=True):
                with gr.Row(equal_height=True):
                    with gr.Column(scale=3,min_width=150):
                        sofa_model = gr.Textbox(label="fa模型路径",placeholder="输入文件路径")
                    with gr.Column(scale=2,min_width=150):
                        model_btn = gr.Button("选择文件", variant="primary")
                with gr.Row(equal_height=True):
                    with gr.Column(scale=3,min_width=150):
                        ds_dict = gr.Textbox(label="fa字典路径",placeholder="输入模型路径")
                    with gr.Column(scale=2,min_width=150):
                        ds_dict_btn = gr.Button("选择文件", variant="primary")

            with gr.Row():
                VCV_mode = gr.Radio(
                    choices=[("CVVC", 0), ("VCV", 1), ("CV", 2), ("Test", 3)],  # (显示文本, 实际值)
                    value=0,  # 默认选中值
                    label="音源类型"
                )
                # 添加多音阶模式开关
                multi_pitch_mode = gr.Radio(
                    choices=[("关闭", 0), ("开启", 1)],  # (显示文本, 实际值)
                    value=0,  # 默认关闭
                    label="多音阶模式（音源目录设置为根目录）[音阶后缀现已经直接为文件夹名，不添加空格]"
                )
                pitch = gr.Textbox(label="音阶后缀", placeholder="例如： F3")

            with gr.Row():
                lab = gr.Radio(choices=["Y", "N"], value="Y", label="生成lab文件")
                delete_sp = gr.Radio(choices=["Y", "N"], value="Y", label="处理错误的SP标记")
                clear_tg_cache = gr.Radio(choices=["Y", "N"], value="N", label="清空TextGrid标注及缓存")
                #生成后清空所有杂项文件
                #还没做
                cover = gr.Radio(choices=["Y", "N"], value="Y", label="覆盖oto")
                #编码格式
                oto_encoding = gr.Radio(choices=["utf-8", "shift-jis"], value="utf-8", label="oto编码格式")

            with gr.Row():
                cut = gr.Textbox(label="字符分隔符", value="_,-")
                ignore = gr.Textbox(label="视为间隔音素R的音素", value="AP,SP,EP,R,-,B")

            gr.Markdown("### 可选参数配置")
            with gr.Accordion("高级参数配置（建议修改）", open=True):
                gr.Markdown("**规则参数（逗号分隔数值）**")
                with gr.Row():
                    cv_sum = gr.Textbox(label="CV规则比例(左线占比,固定的占比,右线占比,预发声不变,交叉占比)", value="1,3,1.5,1,4")
                    vc_sum = gr.Textbox(label="VC规则比例(左线占比,固定的占比,右线占比,预发声不变,交叉占比)", value="3,0,2,1,2")
                    vv_sum = gr.Textbox(label="VV规则比例(左线占比,固定的占比,右线占比,预发声不变,交叉占比(母音结合))", value="3,3,1.5,1,2")
                    cv_offset = gr.Textbox(label="CV数值偏移量(左线偏移,固定偏移,右线偏移,预发声偏移,交叉偏移)", value="0,0,0,0,0")
                    vc_offset = gr.Textbox(label="VC数值偏移量(左线偏移,固定偏移,右线偏移,预发声偏移,交叉偏移)", value="0,0,0,0,0")
                with gr.Row():
                    rec_preset = gr.Textbox(label="优先按录音表顺序生成oto（只读utf-8）",placeholder="输入录音表文件路径",value="")
                    oto_preset = gr.Textbox(label="(暂不支持)优先使用模板中的oto（只读utf-8）",placeholder="输入oto模板路径",value="")
                with gr.Row():
                    CV_repeat = gr.Textbox(label="CV重复次数(无上限)", value="1")
                    VC_repeat = gr.Textbox(label="VC重复次数(无上限)", value="1")
            # 定义更新参数的函数

            demo.load(
                fn=scan_model_folder,
                inputs=SOFA_type,
                outputs=model_folder_selector
            )

            SOFA_type.change(
                fn=scan_model_folder,
                inputs=SOFA_type,
                outputs=model_folder_selector
            )

            model_folder_selector.change(
                fn=update_model_paths,
                inputs=[SOFA_type,model_folder_selector],
                outputs=[ds_dict, sofa_model,dict_folders_selector,model_version_folder_selector]
            )
            dict_folders_selector.change(
                fn=update_dict_paths,
                inputs= [sofa_model,dict_folders_selector],
                outputs=ds_dict
            )
            model_version_folder_selector.change(
                fn=update_model_version_paths,
                inputs=[sofa_model,model_version_folder_selector],
                outputs=sofa_model
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
                fn=config_generator_dispatcher,
                inputs=[
                    wav_path, ds_dict, presamp, cut, ignore,
                    VCV_mode, lab, cv_sum, vc_sum, vv_sum,
                    cv_offset, vc_offset, pitch, CV_repeat,
                    VC_repeat, clear_tg_cache, cover, sofa_model, SOFA_mode, SOFA_type, multi_pitch_mode, delete_sp,rec_preset,oto_preset,oto_encoding
                ],
                outputs=output
            )

        # 新增标签页：Transcriptions 生成
        with gr.TabItem("transcriptions 生成"):
            gr.Markdown("### 选择路径并生成 transcriptions 文件")

            with gr.Row(equal_height=True):
                with gr.Column(scale=3, min_width=150):
                    transcriptions_path = gr.Textbox(label="选择路径", placeholder="请选择目标文件夹路径")
                with gr.Column(scale=2, min_width=150):
                    transcriptions_folder_btn = gr.Button("选择文件夹", variant="primary")

            transcriptions_generate_btn = gr.Button("生成完整的Transcriptions", variant="primary")
            transcriptions_output = gr.Textbox(label="生成结果", lines=5)

            # 按钮点击事件绑定
            transcriptions_folder_btn.click(
                fn=select_folder,  # 调用文件夹选择函数
                outputs=transcriptions_path  # 将结果传递给文本框
            )


            def generate_transcriptions(full_path):
                """
                根据指定路径生成完整的 transcriptions.csv 文件
                """
                try:
                    wavs = full_path
                    dataset = full_path
                    tg = full_path + "\\TextGrid"
                    skip_silence_insertion = True
                    wav_subtype = "PCM_16"
                    # 调用已有逻辑生成 transcriptions.csv
                    build_dataset.build_dataset(wavs, tg, dataset, skip_silence_insertion, wav_subtype)
                    return f"✅ 成功生成 Transcriptions 文件：{full_path}/transcriptions.csv"
                except Exception as e:
                    return f"❌ 生成失败：{str(e)}"


            transcriptions_generate_btn.click(
                fn=generate_transcriptions,
                inputs=transcriptions_path,
                outputs=transcriptions_output
            )

        with gr.TabItem("tg2svdb"):
            gr.Markdown("### 生成svR1所使用的标记格式")
            gr.Markdown("请先运行oto生成，再运行这里")
            db_path = gr.Textbox(label="选择路径", placeholder="输入wav文件夹路径")
            dic_path = gr.Radio(
                choices=[("中文",'tg2svdb/字典/mandarin-xsampa-dict.txt'), ("日语", 'tg2svdb/字典/japanese-romaji-dict.txt'), ("英语",'tg2svdb/字典/eng.txt')],  # (显示文本, 实际值)
                value='tg2svdb/字典/mandarin-xsampa-dict.txt',  # 默认选中值
                label="语言"
            )
            # 添加按钮
            tg2svdb_btn = gr.Button("生成标记", variant="primary")

            def svdb_run(db_path, dic_path):
                import tg2svdb.tg2sv_change
                folders = [f for f in os.listdir(db_path) if os.path.isdir(os.path.join(db_path, f))]
                print(folders)
                for folder in folders:
                    json_path = os.path.join(db_path, folder, 'TextGrid', 'json', 'word_phone.json')
                    wav_path = os.path.join(db_path, folder)
                    print(json_path, wav_path)
                    tg2svdb.tg2sv_change.run(dic_path, json_path, wav_path)

            # 添加事件绑定
            tg2svdb_btn.click(
                fn=svdb_run,  # 需要定义处理函数
                inputs=[db_path, dic_path]
            )

        with gr.TabItem("tg2V3db"):
            gr.Markdown("### 生成V3DB所使用的粗切割标记")
            gr.Markdown("请先运行oto生成，再运行这里")




if __name__ == "__main__":
    run()