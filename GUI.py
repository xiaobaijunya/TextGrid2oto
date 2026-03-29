import wx
import os
import sys
import threading
import io
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import wavname2lab
import onnx_infer
from pathlib import Path
from textgrid2json import del_SP, TextGrid2ds_json, ds_json2filter, ds_json2word
from json2oto import json2CV_oto, json2oto, json2VCV_oto, json2test
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tg2svdb'))
from tg2svdb import tg2sv_change

class TextRedirector:
    def __init__(self, text_ctrl):
        self.text_ctrl = text_ctrl
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
    
    def write(self, text):
        self.original_stdout.write(text)
        if self.text_ctrl:
            wx.CallAfter(self.text_ctrl.AppendText, text)
    
    def flush(self):
        self.original_stdout.flush()
    
    def __enter__(self):
        sys.stdout = self
        sys.stderr = self
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr

class MainFrame(wx.Frame):
    def __init__(self):
        super().__init__(None, title="textgrid-to-多引擎标记转换器", size=(800, 600))
        
        icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "img", "TextGrid2oto.ico")
        icon = wx.Icon(icon_path, wx.BITMAP_TYPE_ICO)
        self.SetIcon(icon)
        
        panel = wx.Panel(self)
        
        notebook = wx.Notebook(panel)
        
        lab_panel = wx.Panel(notebook)
        lab_sizer = wx.BoxSizer(wx.VERTICAL)
        
        lab_title = wx.StaticText(lab_panel, label="LAB生成")
        lab_sizer.Add(lab_title, 0, wx.ALL | wx.CENTER, 10)
        
        path_sizer = wx.BoxSizer(wx.HORIZONTAL)
        path_label = wx.StaticText(lab_panel, label="音源文件夹：")
        self.path_text = wx.TextCtrl(lab_panel, size=(400, -1))
        browse_btn = wx.Button(lab_panel, label="选择文件夹")
        browse_btn.Bind(wx.EVT_BUTTON, lambda event: self.on_browse_folder(event, self.path_text))
        path_sizer.Add(path_label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        path_sizer.Add(self.path_text, 1, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        path_sizer.Add(browse_btn, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        lab_sizer.Add(path_sizer, 0, wx.EXPAND | wx.ALL, 10)
        
        separator_sizer = wx.BoxSizer(wx.HORIZONTAL)
        separator_label = wx.StaticText(lab_panel, label="音素分隔符：")
        self.separator_text = wx.TextCtrl(lab_panel, value="_,-", size=(200, -1))
        separator_sizer.Add(separator_label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        separator_sizer.Add(self.separator_text, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        lab_sizer.Add(separator_sizer, 0, wx.ALL, 10)
        
        generate_btn = wx.Button(lab_panel, label="生成LAB文件")
        generate_btn.Bind(wx.EVT_BUTTON, self.on_generate_lab)
        lab_sizer.Add(generate_btn, 0, wx.ALL | wx.CENTER, 10)
        
        # 结果显示文本框
        lab_result_label = wx.StaticText(lab_panel, label="处理结果：")
        lab_sizer.Add(lab_result_label, 0, wx.ALL | wx.LEFT, 10)
        
        self.lab_result_text = wx.TextCtrl(lab_panel, style=wx.TE_MULTILINE | wx.TE_READONLY, size=(-1, 400))
        lab_sizer.Add(self.lab_result_text, 0, wx.EXPAND | wx.ALL, 10)
        
        lab_panel.SetSizer(lab_sizer)
        
        textgrid_panel = wx.Panel(notebook)
        textgrid_sizer = wx.BoxSizer(wx.VERTICAL)

        textgrid_title = wx.StaticText(textgrid_panel, label="TextGrid推理")
        textgrid_sizer.Add(textgrid_title, 0, wx.ALL | wx.CENTER, 10)

        folder_sizer = wx.BoxSizer(wx.HORIZONTAL)
        folder_label = wx.StaticText(textgrid_panel, label="音源文件夹：")
        self.textgrid_folder_text = wx.TextCtrl(textgrid_panel, size=(400, -1))
        browse_folder_btn = wx.Button(textgrid_panel, label="选择文件夹")
        browse_folder_btn.Bind(wx.EVT_BUTTON, lambda event: self.on_browse_folder(event, self.textgrid_folder_text))
        folder_sizer.Add(folder_label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        folder_sizer.Add(self.textgrid_folder_text, 1, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        folder_sizer.Add(browse_folder_btn, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        textgrid_sizer.Add(folder_sizer, 0, wx.EXPAND | wx.ALL, 10)

        model_folder_sizer = wx.BoxSizer(wx.HORIZONTAL)
        model_folder_label = wx.StaticText(textgrid_panel, label="模型文件夹：")
        self.model_folder_choice = wx.Choice(textgrid_panel, size=(300, -1))
        self.model_folder_choice.Bind(wx.EVT_CHOICE, self.on_model_folder_selected)
        model_folder_sizer.Add(model_folder_label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        model_folder_sizer.Add(self.model_folder_choice, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        textgrid_sizer.Add(model_folder_sizer, 0, wx.ALL, 10)

        model_file_sizer = wx.BoxSizer(wx.HORIZONTAL)
        model_file_label = wx.StaticText(textgrid_panel, label="选择模型：")
        self.model_file_choice = wx.Choice(textgrid_panel, size=(300, -1))
        model_file_sizer.Add(model_file_label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        model_file_sizer.Add(self.model_file_choice, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        textgrid_sizer.Add(model_file_sizer, 0, wx.ALL, 10)

        dict_sizer = wx.BoxSizer(wx.HORIZONTAL)
        dict_label = wx.StaticText(textgrid_panel, label="选择字典：")
        self.dict_choice = wx.Choice(textgrid_panel, size=(300, -1))
        dict_sizer.Add(dict_label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        dict_sizer.Add(self.dict_choice, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        textgrid_sizer.Add(dict_sizer, 0, wx.ALL, 10)

        infer_btn = wx.Button(textgrid_panel, label="开始推理")
        infer_btn.Bind(wx.EVT_BUTTON, self.on_infer)
        textgrid_sizer.Add(infer_btn, 0, wx.ALL | wx.CENTER, 10)

        # 结果显示文本框
        infer_result_label = wx.StaticText(textgrid_panel, label="处理结果：")
        textgrid_sizer.Add(infer_result_label, 0, wx.ALL | wx.LEFT, 10)

        self.infer_result_text = wx.TextCtrl(textgrid_panel, style=wx.TE_MULTILINE | wx.TE_READONLY, size=(-1, 400))
        textgrid_sizer.Add(self.infer_result_text, 0, wx.EXPAND | wx.ALL, 10)

        textgrid_panel.SetSizer(textgrid_sizer)

        # JSON生成面板
        json_panel = wx.Panel(notebook)
        json_sizer = wx.BoxSizer(wx.VERTICAL)
        
        json_title = wx.StaticText(json_panel, label="JSON生成")
        json_sizer.Add(json_title, 0, wx.ALL | wx.CENTER, 10)
        
        # WAV路径框
        json_path_sizer = wx.BoxSizer(wx.HORIZONTAL)
        json_path_label = wx.StaticText(json_panel, label="WAV文件夹：")
        self.json_path_text = wx.TextCtrl(json_panel, size=(400, -1))
        json_browse_btn = wx.Button(json_panel, label="选择文件夹")
        json_browse_btn.Bind(wx.EVT_BUTTON, lambda event: self.on_browse_folder(event, self.json_path_text))
        json_path_sizer.Add(json_path_label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        json_path_sizer.Add(self.json_path_text, 1, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        json_path_sizer.Add(json_browse_btn, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        json_sizer.Add(json_path_sizer, 0, wx.EXPAND | wx.ALL, 10)

        json_folder_sizer = wx.BoxSizer(wx.HORIZONTAL)
        json_folder_label = wx.StaticText(json_panel, label="模型文件夹：")
        self.json_folder_choice = wx.Choice(json_panel, size=(300, -1))
        self.json_folder_choice.Bind(wx.EVT_CHOICE, self.on_json_folder_selected)
        json_folder_sizer.Add(json_folder_label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        json_folder_sizer.Add(self.json_folder_choice, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        json_sizer.Add(json_folder_sizer, 0, wx.ALL, 10)

        # 模型字典选择
        json_dict_sizer = wx.BoxSizer(wx.HORIZONTAL)
        json_dict_label = wx.StaticText(json_panel, label="模型字典：")
        self.json_dict_choice = wx.Choice(json_panel, size=(300, -1))
        self.json_dict_choice.Bind(wx.EVT_CHOICE, self.on_json_dict_selected)
        json_dict_sizer.Add(json_dict_label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        json_dict_sizer.Add(self.json_dict_choice, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        json_sizer.Add(json_dict_sizer, 0, wx.ALL, 10)
        
        # 忽略音素
        json_ignore_sizer = wx.BoxSizer(wx.HORIZONTAL)
        json_ignore_label = wx.StaticText(json_panel, label="忽略音素：")
        self.json_ignore_text = wx.TextCtrl(json_panel, value="AP,SP,EP,R", size=(200, -1))
        json_ignore_sizer.Add(json_ignore_label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        json_ignore_sizer.Add(self.json_ignore_text, 1, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        json_sizer.Add(json_ignore_sizer, 0, wx.ALL, 10)
        
        # 生成JSON按钮
        generate_json_btn = wx.Button(json_panel, label="生成JSON")
        generate_json_btn.Bind(wx.EVT_BUTTON, self.on_generate_json)
        json_sizer.Add(generate_json_btn, 0, wx.ALL | wx.CENTER, 10)
        
        # 结果显示文本框
        json_result_label = wx.StaticText(json_panel, label="处理结果：")
        json_sizer.Add(json_result_label, 0, wx.ALL | wx.LEFT, 10)
        
        self.json_result_text = wx.TextCtrl(json_panel, style=wx.TE_MULTILINE | wx.TE_READONLY, size=(-1, 400))
        json_sizer.Add(self.json_result_text, 0, wx.EXPAND | wx.ALL, 10)
        
        json_panel.SetSizer(json_sizer)

        # TextGrid清洗面板
        clean_panel = wx.Panel(notebook)
        clean_sizer = wx.BoxSizer(wx.VERTICAL)

        clean_title = wx.StaticText(clean_panel, label="TextGrid清洗")
        clean_sizer.Add(clean_title, 0, wx.ALL | wx.CENTER, 10)

        # WAV路径框
        clean_path_sizer = wx.BoxSizer(wx.HORIZONTAL)
        clean_path_label = wx.StaticText(clean_panel, label="WAV文件夹：")
        self.clean_path_text = wx.TextCtrl(clean_panel, size=(400, -1))
        clean_browse_btn = wx.Button(clean_panel, label="选择文件夹")
        clean_browse_btn.Bind(wx.EVT_BUTTON, lambda event: self.on_browse_folder(event, self.clean_path_text))
        clean_path_sizer.Add(clean_path_label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        clean_path_sizer.Add(self.clean_path_text, 1, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        clean_path_sizer.Add(clean_browse_btn, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        clean_sizer.Add(clean_path_sizer, 0, wx.EXPAND | wx.ALL, 10)

        # 清理SP按钮
        clean_sp_btn = wx.Button(clean_panel, label="清理SP")
        clean_sp_btn.Bind(wx.EVT_BUTTON, self.on_clean_sp)
        clean_sizer.Add(clean_sp_btn, 0, wx.ALL | wx.CENTER, 10)

        # 结果显示文本框
        clean_result_label = wx.StaticText(clean_panel, label="处理结果：")
        clean_sizer.Add(clean_result_label, 0, wx.ALL | wx.LEFT, 10)

        self.clean_result_text = wx.TextCtrl(clean_panel, style=wx.TE_MULTILINE | wx.TE_READONLY, size=(-1, 400))
        clean_sizer.Add(self.clean_result_text, 0, wx.EXPAND | wx.ALL, 10)

        clean_panel.SetSizer(clean_sizer)

        mark_panel = wx.Panel(notebook)
        mark_sizer = wx.BoxSizer(wx.VERTICAL)

        mark_notebook = wx.Notebook(mark_panel)

        oto_panel = wx.Panel(mark_notebook)
        oto_sizer = wx.BoxSizer(wx.VERTICAL)
        
        oto_title = wx.StaticText(oto_panel, label="OTO生成")
        oto_sizer.Add(oto_title, 0, wx.ALL | wx.CENTER, 10)
        
        # WAV路径框
        oto_path_sizer = wx.BoxSizer(wx.HORIZONTAL)
        oto_path_label = wx.StaticText(oto_panel, label="音源文件夹：")
        self.oto_path_text = wx.TextCtrl(oto_panel, size=(400, -1))
        oto_browse_btn = wx.Button(oto_panel, label="选择文件夹")
        oto_browse_btn.Bind(wx.EVT_BUTTON, lambda event: self.on_browse_folder(event, self.oto_path_text))
        oto_path_sizer.Add(oto_path_label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        oto_path_sizer.Add(self.oto_path_text, 1, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        oto_path_sizer.Add(oto_browse_btn, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        oto_sizer.Add(oto_path_sizer, 0, wx.EXPAND | wx.ALL, 10)
        
        # Presamp路径和预设
        presamp_sizer = wx.BoxSizer(wx.HORIZONTAL)
        presamp_path_label = wx.StaticText(oto_panel, label="Presamp路径：")
        self.oto_presamp_path_text = wx.TextCtrl(oto_panel, size=(300, -1))
        presamp_browse_btn = wx.Button(oto_panel, label="浏览")
        presamp_browse_btn.Bind(wx.EVT_BUTTON, lambda event: self.on_browse_file(event, self.oto_presamp_path_text))
        presamp_preset_label = wx.StaticText(oto_panel, label="预设：")
        self.oto_presamp_choice = wx.Choice(oto_panel, size=(150, -1))
        self.oto_presamp_choice.Bind(wx.EVT_CHOICE, self.on_oto_presamp_selected)
        presamp_sizer.Add(presamp_path_label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        presamp_sizer.Add(self.oto_presamp_path_text, 1, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        presamp_sizer.Add(presamp_browse_btn, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        presamp_sizer.Add(presamp_preset_label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        presamp_sizer.Add(self.oto_presamp_choice, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        oto_sizer.Add(presamp_sizer, 0, wx.EXPAND | wx.ALL, 10)
        
        # 生成模式和编码
        mode_encoding_sizer = wx.BoxSizer(wx.HORIZONTAL)
        oto_mode_label = wx.StaticText(oto_panel, label="生成模式：")
        self.oto_mode_choice = wx.Choice(oto_panel, choices=["CVVC", "VCV", "CVV", "Test"], size=(100, -1))
        self.oto_mode_choice.SetSelection(0)
        self.oto_mode_choice.Bind(wx.EVT_CHOICE, self.on_oto_mode_changed)
        oto_encoding_label = wx.StaticText(oto_panel, label="OTO编码：")
        self.oto_encoding_choice = wx.Choice(oto_panel, choices=["utf-8", "shift-jis", "gbk"], size=(100, -1))
        self.oto_encoding_choice.SetSelection(0)
        self.oto_cover_checkbox = wx.CheckBox(oto_panel, label="覆盖现有oto.ini")
        self.oto_cover_checkbox.SetValue(True)
        mode_encoding_sizer.Add(oto_mode_label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        mode_encoding_sizer.Add(self.oto_mode_choice, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        mode_encoding_sizer.Add(oto_encoding_label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        mode_encoding_sizer.Add(self.oto_encoding_choice, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        mode_encoding_sizer.Add(self.oto_cover_checkbox, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        oto_sizer.Add(mode_encoding_sizer, 0, wx.ALL, 10)
        
        # 参数组1：CV参数、VC参数、VV参数
        params1_sizer = wx.BoxSizer(wx.HORIZONTAL)
        oto_cv_sum_sizer = wx.BoxSizer(wx.VERTICAL)
        oto_cv_sum_label = wx.StaticText(oto_panel, label="CV参数：")
        self.oto_cv_sum_text = wx.TextCtrl(oto_panel, value="1,3,1.5,1,4", size=(120, -1))
        oto_cv_sum_sizer.Add(oto_cv_sum_label, 0, wx.ALL, 2)
        oto_cv_sum_sizer.Add(self.oto_cv_sum_text, 0, wx.ALL, 2)
        
        oto_vc_sum_sizer = wx.BoxSizer(wx.VERTICAL)
        oto_vc_sum_label = wx.StaticText(oto_panel, label="VC参数：")
        self.oto_vc_sum_text = wx.TextCtrl(oto_panel, value="3,0,2,1,3", size=(120, -1))
        oto_vc_sum_sizer.Add(oto_vc_sum_label, 0, wx.ALL, 2)
        oto_vc_sum_sizer.Add(self.oto_vc_sum_text, 0, wx.ALL, 2)
        
        oto_vv_sum_sizer = wx.BoxSizer(wx.VERTICAL)
        oto_vv_sum_label = wx.StaticText(oto_panel, label="VV参数：")
        self.oto_vv_sum_text = wx.TextCtrl(oto_panel, value="3,3,1.5,1,3", size=(120, -1))
        oto_vv_sum_sizer.Add(oto_vv_sum_label, 0, wx.ALL, 2)
        oto_vv_sum_sizer.Add(self.oto_vv_sum_text, 0, wx.ALL, 2)
        
        params1_sizer.Add(oto_cv_sum_sizer, 0, wx.ALL, 5)
        params1_sizer.Add(oto_vc_sum_sizer, 0, wx.ALL, 5)
        params1_sizer.Add(oto_vv_sum_sizer, 0, wx.ALL, 5)
        oto_sizer.Add(params1_sizer, 0, wx.ALL, 10)
        
        # 参数组2：CV偏移、VC偏移、音阶后缀

        oto_cv_offset_sizer = wx.BoxSizer(wx.VERTICAL)
        oto_cv_offset_label = wx.StaticText(oto_panel, label="CV偏移：")
        self.oto_cv_offset_text = wx.TextCtrl(oto_panel, value="0,0,0,0,0", size=(120, -1))
        oto_cv_offset_sizer.Add(oto_cv_offset_label, 0, wx.ALL, 2)
        oto_cv_offset_sizer.Add(self.oto_cv_offset_text, 0, wx.ALL, 2)
        
        oto_vc_offset_sizer = wx.BoxSizer(wx.VERTICAL)
        oto_vc_offset_label = wx.StaticText(oto_panel, label="VC偏移：")
        self.oto_vc_offset_text = wx.TextCtrl(oto_panel, value="0,0,0,0,0", size=(120, -1))
        oto_vc_offset_sizer.Add(oto_vc_offset_label, 0, wx.ALL, 2)
        oto_vc_offset_sizer.Add(self.oto_vc_offset_text, 0, wx.ALL, 2)

        params1_sizer.Add(oto_cv_offset_sizer, 0, wx.ALL, 5)
        params1_sizer.Add(oto_vc_offset_sizer, 0, wx.ALL, 5)

        oto_cv_repeat_sizer = wx.BoxSizer(wx.VERTICAL)
        oto_cv_repeat_label = wx.StaticText(oto_panel, label="CV重复次数：")
        self.oto_cv_repeat_text = wx.TextCtrl(oto_panel, value="1", size=(80, -1))
        oto_cv_repeat_sizer.Add(oto_cv_repeat_label, 0, wx.ALL, 2)
        oto_cv_repeat_sizer.Add(self.oto_cv_repeat_text, 0, wx.ALL, 2)
        
        oto_vc_repeat_sizer = wx.BoxSizer(wx.VERTICAL)
        oto_vc_repeat_label = wx.StaticText(oto_panel, label="VC重复次数：")
        self.oto_vc_repeat_text = wx.TextCtrl(oto_panel, value="1", size=(80, -1))
        oto_vc_repeat_sizer.Add(oto_vc_repeat_label, 0, wx.ALL, 2)
        oto_vc_repeat_sizer.Add(self.oto_vc_repeat_text, 0, wx.ALL, 2)
        
        oto_ignore_sizer = wx.BoxSizer(wx.VERTICAL)
        oto_ignore_label = wx.StaticText(oto_panel, label="忽略音素：")
        self.oto_ignore_text = wx.TextCtrl(oto_panel, value="AP,SP,EP,R", size=(130, -1))
        oto_ignore_sizer.Add(oto_ignore_label, 0, wx.ALL, 2)
        oto_ignore_sizer.Add(self.oto_ignore_text, 0, wx.ALL, 2)

        params2_sizer = wx.BoxSizer(wx.HORIZONTAL)
        oto_pitch_sizer = wx.BoxSizer(wx.VERTICAL)
        oto_pitch_label = wx.StaticText(oto_panel, label="音阶后缀：")
        self.oto_pitch_text = wx.TextCtrl(oto_panel, value="", size=(100, -1))
        oto_pitch_sizer.Add(oto_pitch_label, 0, wx.ALL, 2)
        oto_pitch_sizer.Add(self.oto_pitch_text, 0, wx.ALL, 2)

        oto_preset_sizer = wx.BoxSizer(wx.VERTICAL)
        oto_preset_label = wx.StaticText(oto_panel, label="录音表路径（暂不可用）：")
        self.oto_preset_text = wx.TextCtrl(oto_panel, value="", size=(210, -1))
        oto_preset_sizer.Add(oto_preset_label, 0, wx.ALL, 2)
        oto_preset_sizer.Add(self.oto_preset_text, 0, wx.ALL, 2)

        params2_sizer.Add(oto_cv_repeat_sizer, 0, wx.ALL, 5)
        params2_sizer.Add(oto_vc_repeat_sizer, 0, wx.ALL, 5)
        params2_sizer.Add(oto_ignore_sizer, 0, wx.ALL, 5)
        params2_sizer.Add(oto_pitch_sizer, 0, wx.ALL, 5)
        params2_sizer.Add(oto_preset_sizer, 0, wx.ALL, 5)
        oto_sizer.Add(params2_sizer, 0, wx.ALL, 10)


        

        
        # 生成OTO按钮
        oto_generate_btn = wx.Button(oto_panel, label="生成OTO")
        oto_generate_btn.Bind(wx.EVT_BUTTON, self.on_generate_oto)
        oto_sizer.Add(oto_generate_btn, 0, wx.ALL | wx.CENTER, 10)
        
        # 结果显示文本框
        oto_result_label = wx.StaticText(oto_panel, label="处理结果：")
        oto_sizer.Add(oto_result_label, 0, wx.ALL | wx.LEFT, 10)
        
        self.oto_result_text = wx.TextCtrl(oto_panel, style=wx.TE_MULTILINE | wx.TE_READONLY, size=(-1, 400))
        oto_sizer.Add(self.oto_result_text, 0, wx.EXPAND | wx.ALL, 10)
        
        oto_panel.SetSizer(oto_sizer)

        svdb_panel = wx.Panel(mark_notebook)
        svdb_sizer = wx.BoxSizer(wx.VERTICAL)
        
        svdb_title = wx.StaticText(svdb_panel, label="SVR1DB生成")
        svdb_sizer.Add(svdb_title, 0, wx.ALL | wx.CENTER, 10)
        
        svdb_path_sizer = wx.BoxSizer(wx.HORIZONTAL)
        svdb_path_label = wx.StaticText(svdb_panel, label="声库文件夹：")
        self.svdb_path_text = wx.TextCtrl(svdb_panel, size=(400, -1))
        svdb_browse_btn = wx.Button(svdb_panel, label="选择文件夹")
        svdb_browse_btn.Bind(wx.EVT_BUTTON, lambda event: self.on_browse_folder(event, self.svdb_path_text))
        svdb_path_sizer.Add(svdb_path_label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        svdb_path_sizer.Add(self.svdb_path_text, 1, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        svdb_path_sizer.Add(svdb_browse_btn, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        svdb_sizer.Add(svdb_path_sizer, 0, wx.EXPAND | wx.ALL, 10)
        
        svdb_dict_sizer = wx.BoxSizer(wx.HORIZONTAL)
        svdb_dict_label = wx.StaticText(svdb_panel, label="选择字典：")
        self.svdb_dict_choice = wx.Choice(svdb_panel, size=(300, -1))
        svdb_dict_sizer.Add(svdb_dict_label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        svdb_dict_sizer.Add(self.svdb_dict_choice, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        svdb_sizer.Add(svdb_dict_sizer, 0, wx.ALL, 10)
        
        svdb_generate_btn = wx.Button(svdb_panel, label="生成json标记")
        svdb_generate_btn.Bind(wx.EVT_BUTTON, self.on_generate_svdb)
        svdb_sizer.Add(svdb_generate_btn, 0, wx.ALL | wx.CENTER, 10)
        
        svdb_result_label = wx.StaticText(svdb_panel, label="处理结果：")
        svdb_sizer.Add(svdb_result_label, 0, wx.ALL | wx.LEFT, 10)
        
        self.svdb_result_text = wx.TextCtrl(svdb_panel, style=wx.TE_MULTILINE | wx.TE_READONLY, size=(-1, 400))
        svdb_sizer.Add(self.svdb_result_text, 0, wx.EXPAND | wx.ALL, 10)
        
        svdb_panel.SetSizer(svdb_sizer)

        v3db_panel = wx.Panel(mark_notebook)
        v3db_sizer = wx.BoxSizer(wx.VERTICAL)
        v3db_text = wx.StaticText(v3db_panel, label="V3DB")
        v3db_sizer.Add(v3db_text, 0, wx.ALL | wx.CENTER, 20)
        v3db_panel.SetSizer(v3db_sizer)

        mark_notebook.AddPage(oto_panel, "OTO生成")
        mark_notebook.AddPage(svdb_panel, "SVR1DB")
        mark_notebook.AddPage(v3db_panel, "V3DB")

        mark_sizer.Add(mark_notebook, 1, wx.EXPAND | wx.ALL, 5)
        mark_panel.SetSizer(mark_sizer)

        notebook.AddPage(lab_panel, "1.LAB生成")
        notebook.AddPage(textgrid_panel, "2.TextGrid推理")
        notebook.AddPage(clean_panel, "3.TextGrid清洗")
        notebook.AddPage(json_panel, "4.JSON生成")
        notebook.AddPage(mark_panel, "5.标记生成")

        main_sizer = wx.BoxSizer(wx.VERTICAL)
        main_sizer.Add(notebook, 1, wx.EXPAND)
        panel.SetSizer(main_sizer)
        
        # 在所有控件创建完成后加载模型
        self.load_models()
        self.load_svdb_dicts()
        self.load_presamp()
        self.load_presamp()

    def load_models(self):
        hubert_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'HubertFA_model')
        if os.path.exists(hubert_path):
            model_folders = [d for d in os.listdir(hubert_path) if os.path.isdir(os.path.join(hubert_path, d))]
            
            # 填充 TextGrid 推理面板的模型文件夹选择框
            self.model_folder_choice.Clear()
            self.model_folder_choice.AppendItems(model_folders)
            if model_folders:
                self.model_folder_choice.SetSelection(0)
                self.on_model_folder_selected(None)
            
            # 填充 JSON 生成面板的模型文件夹选择框
            self.json_folder_choice.Clear()
            self.json_folder_choice.AppendItems(model_folders)
            if model_folders:
                self.json_folder_choice.SetSelection(0)
                self.on_json_folder_selected(None)

    def on_model_folder_selected(self, event):
        selected_folder = self.model_folder_choice.GetStringSelection()
        if selected_folder:
            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'HubertFA_model', selected_folder)
            if os.path.exists(model_path):
                onnx_files = [f for f in os.listdir(model_path) if f.endswith('.onnx')]
                self.model_file_choice.Clear()
                self.model_file_choice.AppendItems(onnx_files)
                if onnx_files:
                    self.model_file_choice.SetSelection(0)

                dict_files = [f for f in os.listdir(model_path) if f.endswith('.txt')]
                self.dict_choice.Clear()
                self.dict_choice.AppendItems(dict_files)
                if dict_files:
                    self.dict_choice.SetSelection(0)
                
                self.json_dict_choice.Clear()
                self.json_dict_choice.AppendItems(dict_files)
                if dict_files:
                    self.json_dict_choice.SetSelection(0)

    def on_json_folder_selected(self, event):
        selected_folder = self.json_folder_choice.GetStringSelection()
        if selected_folder:
            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'HubertFA_model', selected_folder)
            if os.path.exists(model_path):
                dict_files = [f for f in os.listdir(model_path) if f.endswith('.txt')]
                self.json_dict_choice.Clear()
                self.json_dict_choice.AppendItems(dict_files)
                if dict_files:
                    self.json_dict_choice.SetSelection(0)

    def on_json_dict_selected(self, event):
        pass
    
    def load_svdb_dicts(self):
        dict_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tg2svdb', '字典')
        if os.path.exists(dict_dir):
            dict_files = [f for f in os.listdir(dict_dir) if f.endswith('.txt')]
            self.svdb_dict_choice.Clear()
            self.svdb_dict_choice.AppendItems(dict_files)
            if dict_files:
                self.svdb_dict_choice.SetSelection(0)
    
    def on_generate_svdb(self, event):
        svdb_path = self.svdb_path_text.GetValue().strip()
        dict_file = self.svdb_dict_choice.GetStringSelection()
        
        if not svdb_path:
            wx.MessageBox("请选择声库文件夹", "错误", wx.OK | wx.ICON_ERROR)
            return
        
        if not os.path.exists(svdb_path):
            wx.MessageBox("声库文件夹不存在", "错误", wx.OK | wx.ICON_ERROR)
            return
        
        if not dict_file:
            wx.MessageBox("请选择字典文件", "错误", wx.OK | wx.ICON_ERROR)
            return
        
        def generate_svdb_thread():
            try:
                wx.CallAfter(self.svdb_result_text.Clear)
                wx.CallAfter(self.svdb_result_text.AppendText, "开始生成SVDB...\n")
                
                dict_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tg2svdb', '字典', dict_file)
                
                def process_folder(folder_path, folder_name=""):
                    json_path = os.path.join(folder_path, 'json', 'word_phone.json')
                    wav_files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
                    
                    if wav_files and os.path.exists(json_path):
                        display_name = folder_name if folder_name else folder_path
                        wx.CallAfter(self.svdb_result_text.AppendText, f"正在处理: {display_name}\n")
                        tg2sv_change.run(dict_path, json_path, folder_path)
                        wx.CallAfter(self.svdb_result_text.AppendText, f"完成: {display_name}\n")
                        return True
                    return False
                
                processed_count = 0
                
                if process_folder(svdb_path):
                    processed_count += 1
                
                subfolders = [f for f in os.listdir(svdb_path) if os.path.isdir(os.path.join(svdb_path, f))]
                for folder in subfolders:
                    folder_path = os.path.join(svdb_path, folder)
                    if process_folder(folder_path, folder):
                        processed_count += 1
                
                wx.CallAfter(self.svdb_result_text.AppendText, f"\nSVDB生成完成！共处理 {processed_count} 个文件夹\n")
                wx.CallAfter(wx.MessageBox, f"SVDB生成完成！共处理 {processed_count} 个文件夹", "成功", wx.OK | wx.ICON_INFORMATION)
            except Exception as e:
                error_msg = f"生成失败：{str(e)}"
                wx.CallAfter(self.svdb_result_text.AppendText, error_msg + "\n")
                wx.CallAfter(wx.MessageBox, error_msg, "错误", wx.OK | wx.ICON_ERROR)
        
        thread = threading.Thread(target=generate_svdb_thread)
        thread.start()
    
    def load_presamp(self):
        presamp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'presamp')
        if os.path.exists(presamp_dir):
            presamp_files = [f for f in os.listdir(presamp_dir) if os.path.isfile(os.path.join(presamp_dir, f))]
            self.oto_presamp_choice.Clear()
            self.oto_presamp_choice.AppendItems(presamp_files)
            if presamp_files:
                self.oto_presamp_choice.SetSelection(0)
    
    def on_oto_presamp_selected(self, event):
        presamp_file = self.oto_presamp_choice.GetStringSelection()
        if presamp_file:
            presamp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'presamp')
            presamp_full_path = os.path.join(presamp_dir, presamp_file)
            self.oto_presamp_path_text.SetValue(presamp_full_path)
    
    def on_oto_mode_changed(self, event):
        mode = self.oto_mode_choice.GetSelection()
        if mode == 0:  # CVVC
            self.oto_cv_sum_text.SetValue("1,3,1.5,1,4")
            self.oto_vc_sum_text.SetValue("3,0,2,1,3")
            self.oto_vv_sum_text.SetValue("3,3,1.5,1,3")
            self.oto_cv_offset_text.SetValue("0,0,0,0,0")
            self.oto_vc_offset_text.SetValue("0,0,0,0,0")
        elif mode == 1:  # VCV
            self.oto_cv_sum_text.SetValue("1,3,1.5,1,2")
            self.oto_vc_sum_text.SetValue("2.5,3,1.5,1,3")
            self.oto_vv_sum_text.SetValue("0,0,0,0,0")
            self.oto_cv_offset_text.SetValue("0,0,0,0,0")
            self.oto_vc_offset_text.SetValue("0,0,0,0,0")
        elif mode == 2:  # CVV
            self.oto_cv_sum_text.SetValue("1,3,1,1,3")
            self.oto_vc_sum_text.SetValue("5,0,2,1,3")
            self.oto_vv_sum_text.SetValue("0,0,0,0,6")
            self.oto_cv_offset_text.SetValue("0,0,0,0,0")
            self.oto_vc_offset_text.SetValue("0,0,0,0,0")
        elif mode == 3:  # Test
            self.oto_cv_sum_text.SetValue("1,8,1.5,1,4")
            self.oto_vc_sum_text.SetValue("3,0,2,1,2")
            self.oto_vv_sum_text.SetValue("3,3,1.5,1,2")
            self.oto_cv_offset_text.SetValue("0,0,0,0,0")
            self.oto_vc_offset_text.SetValue("0,0,0,0,0")
    
    def on_generate_oto(self, event):
        wav_path = self.oto_path_text.GetValue().strip()
        presamp_path = self.oto_presamp_path_text.GetValue().strip()
        presamp_file = self.oto_presamp_choice.GetStringSelection()
        vcv_mode = str(self.oto_mode_choice.GetSelection())
        
        if not wav_path:
            wx.MessageBox("请选择音源文件夹", "错误", wx.OK | wx.ICON_ERROR)
            return
        
        if not os.path.exists(wav_path):
            wx.MessageBox("音源文件夹不存在", "错误", wx.OK | wx.ICON_ERROR)
            return
        
        if not presamp_path:
            wx.MessageBox("请选择Presamp路径", "错误", wx.OK | wx.ICON_ERROR)
            return
        
        if not os.path.exists(presamp_path):
            wx.MessageBox("Presamp路径不存在", "错误", wx.OK | wx.ICON_ERROR)
            return
        
        try:
            cv_sum = [float(x) for x in self.oto_cv_sum_text.GetValue().split(',')]
            vc_sum = [float(x) for x in self.oto_vc_sum_text.GetValue().split(',')]
            vv_sum = [float(x) for x in self.oto_vv_sum_text.GetValue().split(',')]
            cv_offset = [float(x) for x in self.oto_cv_offset_text.GetValue().split(',')]
            vc_offset = [float(x) for x in self.oto_vc_offset_text.GetValue().split(',')]
        except ValueError:
            wx.MessageBox("参数格式错误，请检查逗号分隔的数字", "错误", wx.OK | wx.ICON_ERROR)
            return
        
        pitch = self.oto_pitch_text.GetValue().strip()
        cv_repeat = self.oto_cv_repeat_text.GetValue().strip()
        vc_repeat = self.oto_vc_repeat_text.GetValue().strip()
        ignore = self.oto_ignore_text.GetValue().strip()
        oto_preset = self.oto_preset_text.GetValue().strip()
        oto_encoding = self.oto_encoding_choice.GetStringSelection()
        cover_bool = self.oto_cover_checkbox.GetValue()
        cover = 'y' if cover_bool else 'n'
        
        if not cv_repeat:
            cv_repeat = "1"
        if not vc_repeat:
            vc_repeat = "1"
        
        textgrid_path = os.path.join(wav_path, 'TextGrid')
        
        if not os.path.exists(textgrid_path):
            wx.MessageBox("TextGrid文件夹不存在，请先生成TextGrid", "错误", wx.OK | wx.ICON_ERROR)
            return
        
        utau_phone_path = os.path.join(textgrid_path, 'json', 'utau_phone.json')
        word_phone_path = os.path.join(textgrid_path, 'json', 'word_phone.json')
        
        if not os.path.exists(utau_phone_path) or not os.path.exists(word_phone_path):
            wx.MessageBox("JSON文件不存在，请先生成JSON", "错误", wx.OK | wx.ICON_ERROR)
            return
        
        def generate_oto_thread():
            try:
                wx.CallAfter(self.oto_result_text.Clear)
                wx.CallAfter(self.oto_result_text.AppendText, "开始生成OTO...\n")
                
                from oto import oto_rw
                from oto import oto_check
                
                with TextRedirector(self.oto_result_text):
                    wx.CallAfter(self.oto_result_text.AppendText, f"生成模式: {['CVVC', 'VCV', 'CVV', 'Test'][int(vcv_mode)]}\n")
                    
                    if vcv_mode == '1':
                        wx.CallAfter(self.oto_result_text.AppendText, "生成模式：VCV\n")
                        json2VCV_oto.run(presamp_path, utau_phone_path, word_phone_path,
                                         wav_path, cv_sum, vc_sum, vv_sum, ignore)
                    elif vcv_mode == '2':
                        wx.CallAfter(self.oto_result_text.AppendText, "生成模式：CVV\n")
                        json2CV_oto.run(presamp_path, utau_phone_path, word_phone_path,
                                        wav_path, cv_sum, vc_sum, vv_sum, ignore)
                    elif vcv_mode == '0':
                        wx.CallAfter(self.oto_result_text.AppendText, "生成模式：CVVC\n")
                        json2oto.run(presamp_path, utau_phone_path, word_phone_path,
                                     wav_path, cv_sum, vc_sum, vv_sum, ignore)
                    elif vcv_mode == '3':
                        wx.CallAfter(self.oto_result_text.AppendText, "生成模式：Test\n")
                        json2test.run(presamp_path, utau_phone_path, word_phone_path,
                                      wav_path, cv_sum, vc_sum, vv_sum, ignore)
                    
                    wx.CallAfter(self.oto_result_text.AppendText, "读取CV和VC oto.ini\n")
                    cv = oto_rw.oto_read(os.path.join(wav_path, 'cv_oto.ini'))
                    vc = oto_rw.oto_read(os.path.join(wav_path, 'vc_oto.ini'))
                    
                    wx.CallAfter(self.oto_result_text.AppendText, "剔除重复项\n")
                    cv = oto_rw.oto_repeat(cv, int(cv_repeat), oto_preset)
                    vc = oto_rw.oto_repeat(vc, int(vc_repeat), oto_preset)
                    
                    wx.CallAfter(self.oto_result_text.AppendText, "偏移oto数值.ini\n")
                    if cv_offset != [0.0, 0.0, 0.0, 0.0, 0.0]:
                        cv = oto_rw.oto_offset(cv, cv_offset)
                        wx.CallAfter(self.oto_result_text.AppendText, "偏移CV数值,运行成功\n")
                    if vc_offset != [0.0, 0.0, 0.0, 0.0, 0.0]:
                        vc = oto_rw.oto_offset(vc, vc_offset)
                        wx.CallAfter(self.oto_result_text.AppendText, "偏移VC数值,运行成功\n")
                    
                    wx.CallAfter(self.oto_result_text.AppendText, "合并oto.ini\n")
                    oto_rw.oto_write(os.path.join(wav_path, 'oto.ini'), cv + vc, pitch, cover, oto_encoding)
                    
                    wx.CallAfter(self.oto_result_text.AppendText, "检测缺少的音素\n")
                    oto_check.run(os.path.join(wav_path, 'oto.ini'), presamp_path, pitch, vcv_mode)
                
                wx.CallAfter(self.oto_result_text.AppendText, "\nOTO生成完成！\n")
                wx.CallAfter(wx.MessageBox, "OTO生成完成！", "成功", wx.OK | wx.ICON_INFORMATION)
            except Exception as e:
                error_msg = f"生成失败：{str(e)}"
                wx.CallAfter(self.oto_result_text.AppendText, error_msg + "\n")
                wx.CallAfter(wx.MessageBox, error_msg, "错误", wx.OK | wx.ICON_ERROR)
        
        thread = threading.Thread(target=generate_oto_thread)
        thread.start()
    
    def load_presamp(self):
        presamp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'presamp')
        if os.path.exists(presamp_dir):
            presamp_files = [f for f in os.listdir(presamp_dir) if os.path.isfile(os.path.join(presamp_dir, f))]
            self.oto_presamp_choice.Clear()
            self.oto_presamp_choice.AppendItems(presamp_files)
            if presamp_files:
                self.oto_presamp_choice.SetSelection(0)
    
    def on_oto_presamp_selected(self, event):
        presamp_file = self.oto_presamp_choice.GetStringSelection()
        if presamp_file:
            presamp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'presamp')
            presamp_full_path = os.path.join(presamp_dir, presamp_file)
            self.oto_presamp_path_text.SetValue(presamp_full_path)
    
    def on_oto_mode_changed(self, event):
        mode = self.oto_mode_choice.GetSelection()
        if mode == 0:  # CVVC
            self.oto_cv_sum_text.SetValue("1,3,1.5,1,4")
            self.oto_vc_sum_text.SetValue("3,0,2,1,3")
            self.oto_vv_sum_text.SetValue("3,3,1.5,1,3")
            self.oto_cv_offset_text.SetValue("0,0,0,0,0")
            self.oto_vc_offset_text.SetValue("0,0,0,0,0")
        elif mode == 1:  # VCV
            self.oto_cv_sum_text.SetValue("1,3,1.5,1,2")
            self.oto_vc_sum_text.SetValue("2.5,3,1.5,1,3")
            self.oto_vv_sum_text.SetValue("0,0,0,0,0")
            self.oto_cv_offset_text.SetValue("0,0,0,0,0")
            self.oto_vc_offset_text.SetValue("0,0,0,0,0")
        elif mode == 2:  # CVV
            self.oto_cv_sum_text.SetValue("1,3,1,1,3")
            self.oto_vc_sum_text.SetValue("5,0,2,1,3")
            self.oto_vv_sum_text.SetValue("0,0,0,0,6")
            self.oto_cv_offset_text.SetValue("0,0,0,0,0")
            self.oto_vc_offset_text.SetValue("0,0,0,0,0")
        elif mode == 3:  # Test
            self.oto_cv_sum_text.SetValue("1,8,1.5,1,4")
            self.oto_vc_sum_text.SetValue("3,0,2,1,2")
            self.oto_vv_sum_text.SetValue("3,3,1.5,1,2")
            self.oto_cv_offset_text.SetValue("0,0,0,0,0")
            self.oto_vc_offset_text.SetValue("0,0,0,0,0")
    
    def on_generate_oto(self, event):
        wav_path = self.oto_path_text.GetValue().strip()
        presamp_path = self.oto_presamp_path_text.GetValue().strip()
        presamp_file = self.oto_presamp_choice.GetStringSelection()
        vcv_mode = str(self.oto_mode_choice.GetSelection())
        
        if not wav_path:
            wx.MessageBox("请选择音源文件夹", "错误", wx.OK | wx.ICON_ERROR)
            return
        
        if not os.path.exists(wav_path):
            wx.MessageBox("音源文件夹不存在", "错误", wx.OK | wx.ICON_ERROR)
            return
        
        if not presamp_path:
            wx.MessageBox("请选择Presamp路径", "错误", wx.OK | wx.ICON_ERROR)
            return
        
        if not os.path.exists(presamp_path):
            wx.MessageBox("Presamp路径不存在", "错误", wx.OK | wx.ICON_ERROR)
            return
        
        try:
            cv_sum = [float(x) for x in self.oto_cv_sum_text.GetValue().split(',')]
            vc_sum = [float(x) for x in self.oto_vc_sum_text.GetValue().split(',')]
            vv_sum = [float(x) for x in self.oto_vv_sum_text.GetValue().split(',')]
            cv_offset = [float(x) for x in self.oto_cv_offset_text.GetValue().split(',')]
            vc_offset = [float(x) for x in self.oto_vc_offset_text.GetValue().split(',')]
        except ValueError:
            wx.MessageBox("参数格式错误，请检查逗号分隔的数字", "错误", wx.OK | wx.ICON_ERROR)
            return
        
        pitch = self.oto_pitch_text.GetValue().strip()
        cv_repeat = self.oto_cv_repeat_text.GetValue().strip()
        vc_repeat = self.oto_vc_repeat_text.GetValue().strip()
        ignore = self.oto_ignore_text.GetValue().strip()
        oto_preset = self.oto_preset_text.GetValue().strip()
        oto_encoding = self.oto_encoding_choice.GetStringSelection()
        cover_bool = self.oto_cover_checkbox.GetValue()
        cover = 'y' if cover_bool else 'n'
        
        if not cv_repeat:
            cv_repeat = "1"
        if not vc_repeat:
            vc_repeat = "1"
        
        textgrid_path = os.path.join(wav_path, 'TextGrid')
        
        if not os.path.exists(textgrid_path):
            wx.MessageBox("TextGrid文件夹不存在，请先生成TextGrid", "错误", wx.OK | wx.ICON_ERROR)
            return
        
        utau_phone_path = os.path.join(textgrid_path, 'json', 'utau_phone.json')
        word_phone_path = os.path.join(textgrid_path, 'json', 'word_phone.json')
        
        if not os.path.exists(utau_phone_path) or not os.path.exists(word_phone_path):
            wx.MessageBox("JSON文件不存在，请先生成JSON", "错误", wx.OK | wx.ICON_ERROR)
            return
        
        def generate_oto_thread():
            try:
                wx.CallAfter(self.oto_result_text.Clear)
                wx.CallAfter(self.oto_result_text.AppendText, "开始生成OTO...\n")
                
                from oto import oto_rw
                from oto import oto_check
                
                with TextRedirector(self.oto_result_text):
                    wx.CallAfter(self.oto_result_text.AppendText, f"生成模式: {['CVVC', 'VCV', 'CVV', 'Test'][int(vcv_mode)]}\n")
                    
                    if vcv_mode == '1':
                        wx.CallAfter(self.oto_result_text.AppendText, "生成模式：VCV\n")
                        json2VCV_oto.run(presamp_path, utau_phone_path, word_phone_path,
                                         wav_path, cv_sum, vc_sum, vv_sum, ignore)
                    elif vcv_mode == '2':
                        wx.CallAfter(self.oto_result_text.AppendText, "生成模式：CVV\n")
                        json2CV_oto.run(presamp_path, utau_phone_path, word_phone_path,
                                        wav_path, cv_sum, vc_sum, vv_sum, ignore)
                    elif vcv_mode == '0':
                        wx.CallAfter(self.oto_result_text.AppendText, "生成模式：CVVC\n")
                        json2oto.run(presamp_path, utau_phone_path, word_phone_path,
                                     wav_path, cv_sum, vc_sum, vv_sum, ignore)
                    elif vcv_mode == '3':
                        wx.CallAfter(self.oto_result_text.AppendText, "生成模式：Test\n")
                        json2test.run(presamp_path, utau_phone_path, word_phone_path,
                                      wav_path, cv_sum, vc_sum, vv_sum, ignore)
                    
                    wx.CallAfter(self.oto_result_text.AppendText, "读取CV和VC oto.ini\n")
                    cv = oto_rw.oto_read(os.path.join(wav_path, 'cv_oto.ini'))
                    vc = oto_rw.oto_read(os.path.join(wav_path, 'vc_oto.ini'))
                    
                    wx.CallAfter(self.oto_result_text.AppendText, "剔除重复项\n")
                    cv = oto_rw.oto_repeat(cv, int(cv_repeat), oto_preset)
                    vc = oto_rw.oto_repeat(vc, int(vc_repeat), oto_preset)
                    
                    wx.CallAfter(self.oto_result_text.AppendText, "偏移oto数值.ini\n")
                    if cv_offset != [0.0, 0.0, 0.0, 0.0, 0.0]:
                        cv = oto_rw.oto_offset(cv, cv_offset)
                        wx.CallAfter(self.oto_result_text.AppendText, "偏移CV数值,运行成功\n")
                    if vc_offset != [0.0, 0.0, 0.0, 0.0, 0.0]:
                        vc = oto_rw.oto_offset(vc, vc_offset)
                        wx.CallAfter(self.oto_result_text.AppendText, "偏移VC数值,运行成功\n")
                    
                    wx.CallAfter(self.oto_result_text.AppendText, "合并oto.ini\n")
                    oto_rw.oto_write(os.path.join(wav_path, 'oto.ini'), cv + vc, pitch, cover, oto_encoding)
                    
                    wx.CallAfter(self.oto_result_text.AppendText, "检测缺少的音素\n")
                    oto_check.run(os.path.join(wav_path, 'oto.ini'), presamp_path, pitch, vcv_mode)
                
                wx.CallAfter(self.oto_result_text.AppendText, "\nOTO生成完成！\n")
                wx.CallAfter(wx.MessageBox, "OTO生成完成！", "成功", wx.OK | wx.ICON_INFORMATION)
            except Exception as e:
                error_msg = f"生成失败：{str(e)}"
                wx.CallAfter(self.oto_result_text.AppendText, error_msg + "\n")
                wx.CallAfter(wx.MessageBox, error_msg, "错误", wx.OK | wx.ICON_ERROR)
        
        thread = threading.Thread(target=generate_oto_thread)
        thread.start()
    
    def load_presamp(self):
        presamp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'presamp')
        if os.path.exists(presamp_dir):
            presamp_files = [f for f in os.listdir(presamp_dir) if os.path.isfile(os.path.join(presamp_dir, f))]
            self.oto_presamp_choice.Clear()
            self.oto_presamp_choice.AppendItems(presamp_files)
            if presamp_files:
                self.oto_presamp_choice.SetSelection(0)
    
    def on_oto_presamp_selected(self, event):
        presamp_file = self.oto_presamp_choice.GetStringSelection()
        if presamp_file:
            presamp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'presamp')
            presamp_full_path = os.path.join(presamp_dir, presamp_file)
            self.oto_presamp_path_text.SetValue(presamp_full_path)
    
    def on_oto_mode_changed(self, event):
        mode = self.oto_mode_choice.GetSelection()
        if mode == 0:  # CVVC
            self.oto_cv_sum_text.SetValue("1,3,1.5,1,4")
            self.oto_vc_sum_text.SetValue("3,0,2,1,3")
            self.oto_vv_sum_text.SetValue("3,3,1.5,1,3")
            self.oto_cv_offset_text.SetValue("0,0,0,0,0")
            self.oto_vc_offset_text.SetValue("0,0,0,0,0")
        elif mode == 1:  # VCV
            self.oto_cv_sum_text.SetValue("1,3,1.5,1,2")
            self.oto_vc_sum_text.SetValue("2.5,3,1.5,1,3")
            self.oto_vv_sum_text.SetValue("0,0,0,0,0")
            self.oto_cv_offset_text.SetValue("0,0,0,0,0")
            self.oto_vc_offset_text.SetValue("0,0,0,0,0")
        elif mode == 2:  # CVV
            self.oto_cv_sum_text.SetValue("1,3,1,1,3")
            self.oto_vc_sum_text.SetValue("5,0,2,1,3")
            self.oto_vv_sum_text.SetValue("0,0,0,0,6")
            self.oto_cv_offset_text.SetValue("0,0,0,0,0")
            self.oto_vc_offset_text.SetValue("0,0,0,0,0")
        elif mode == 3:  # Test
            self.oto_cv_sum_text.SetValue("1,8,1.5,1,4")
            self.oto_vc_sum_text.SetValue("3,0,2,1,2")
            self.oto_vv_sum_text.SetValue("3,3,1.5,1,2")
            self.oto_cv_offset_text.SetValue("0,0,0,0,0")
            self.oto_vc_offset_text.SetValue("0,0,0,0,0")
    
    def on_generate_oto(self, event):
        wav_path = self.oto_path_text.GetValue().strip()
        presamp_path = self.oto_presamp_path_text.GetValue().strip()
        vcv_mode = str(self.oto_mode_choice.GetSelection())
        
        if not wav_path:
            wx.MessageBox("请选择音源文件夹", "错误", wx.OK | wx.ICON_ERROR)
            return
        
        if not os.path.exists(wav_path):
            wx.MessageBox("音源文件夹不存在", "错误", wx.OK | wx.ICON_ERROR)
            return
        
        if not presamp_path:
            wx.MessageBox("请选择Presamp路径", "错误", wx.OK | wx.ICON_ERROR)
            return
        
        if not os.path.exists(presamp_path):
            wx.MessageBox("Presamp路径不存在", "错误", wx.OK | wx.ICON_ERROR)
            return
        
        try:
            cv_sum = [float(x) for x in self.oto_cv_sum_text.GetValue().split(',')]
            vc_sum = [float(x) for x in self.oto_vc_sum_text.GetValue().split(',')]
            vv_sum = [float(x) for x in self.oto_vv_sum_text.GetValue().split(',')]
            cv_offset = [float(x) for x in self.oto_cv_offset_text.GetValue().split(',')]
            vc_offset = [float(x) for x in self.oto_vc_offset_text.GetValue().split(',')]
        except ValueError:
            wx.MessageBox("参数格式错误，请检查逗号分隔的数字", "错误", wx.OK | wx.ICON_ERROR)
            return
        
        pitch = self.oto_pitch_text.GetValue().strip()
        cv_repeat = self.oto_cv_repeat_text.GetValue().strip()
        vc_repeat = self.oto_vc_repeat_text.GetValue().strip()
        ignore = self.oto_ignore_text.GetValue().strip()
        oto_preset = self.oto_preset_text.GetValue().strip()
        oto_encoding = self.oto_encoding_choice.GetStringSelection()
        cover_bool = self.oto_cover_checkbox.GetValue()
        cover = 'y' if cover_bool else 'n'
        
        if not cv_repeat:
            cv_repeat = "1"
        if not vc_repeat:
            vc_repeat = "1"
        
        textgrid_path = wav_path
        
        if not os.path.exists(textgrid_path):
            wx.MessageBox("TextGrid文件夹不存在，请先生成TextGrid", "错误", wx.OK | wx.ICON_ERROR)
            return

        word_phone_path = os.path.join(textgrid_path, 'json', 'word_phone.json')
        
        if not os.path.exists(word_phone_path):
            wx.MessageBox("JSON文件不存在，请先生成JSON", "错误", wx.OK | wx.ICON_ERROR)
            return
        
        def generate_oto_thread():
            try:
                wx.CallAfter(self.oto_result_text.Clear)
                wx.CallAfter(self.oto_result_text.AppendText, "开始生成OTO...\n")
                
                from oto import oto_rw
                from oto import oto_check
                
                with TextRedirector(self.oto_result_text):
                    wx.CallAfter(self.oto_result_text.AppendText, f"生成模式: {['CVVC', 'VCV', 'CVV', 'Test'][int(vcv_mode)]}\n")
                    
                    if vcv_mode == '1':
                        wx.CallAfter(self.oto_result_text.AppendText, "生成模式：VCV\n")
                        json2VCV_oto.run(presamp_path, word_phone_path,
                                         wav_path, cv_sum, vc_sum, vv_sum, ignore)
                    elif vcv_mode == '2':
                        wx.CallAfter(self.oto_result_text.AppendText, "生成模式：CVV\n")
                        json2CV_oto.run(presamp_path, word_phone_path,
                                        wav_path, cv_sum, vc_sum, vv_sum, ignore)
                    elif vcv_mode == '0':
                        wx.CallAfter(self.oto_result_text.AppendText, "生成模式：CVVC\n")
                        json2oto.run(presamp_path, word_phone_path,
                                     wav_path, cv_sum, vc_sum, vv_sum, ignore)
                    elif vcv_mode == '3':
                        wx.CallAfter(self.oto_result_text.AppendText, "生成模式：Test\n")
                        json2test.run(presamp_path, word_phone_path,
                                      wav_path, cv_sum, vc_sum, vv_sum, ignore)
                    
                    wx.CallAfter(self.oto_result_text.AppendText, "读取CV和VC oto.ini\n")
                    cv = oto_rw.oto_read(os.path.join(wav_path, 'cv_oto.ini'))
                    vc = oto_rw.oto_read(os.path.join(wav_path, 'vc_oto.ini'))
                    
                    wx.CallAfter(self.oto_result_text.AppendText, "剔除重复项\n")
                    cv = oto_rw.oto_repeat(cv, int(cv_repeat), oto_preset)
                    vc = oto_rw.oto_repeat(vc, int(vc_repeat), oto_preset)
                    
                    wx.CallAfter(self.oto_result_text.AppendText, "偏移oto数值.ini\n")
                    if cv_offset != [0.0, 0.0, 0.0, 0.0, 0.0]:
                        cv = oto_rw.oto_offset(cv, cv_offset)
                        wx.CallAfter(self.oto_result_text.AppendText, "偏移CV数值,运行成功\n")
                    if vc_offset != [0.0, 0.0, 0.0, 0.0, 0.0]:
                        vc = oto_rw.oto_offset(vc, vc_offset)
                        wx.CallAfter(self.oto_result_text.AppendText, "偏移VC数值,运行成功\n")
                    
                    wx.CallAfter(self.oto_result_text.AppendText, "合并oto.ini\n")
                    oto_rw.oto_write(os.path.join(wav_path, 'oto.ini'), cv + vc, pitch, cover, oto_encoding)
                    
                    wx.CallAfter(self.oto_result_text.AppendText, "检测缺少的音素\n")
                    oto_check.run(os.path.join(wav_path, 'oto.ini'), presamp_path, pitch, vcv_mode)
                
                wx.CallAfter(self.oto_result_text.AppendText, "\nOTO生成完成！\n")
                wx.CallAfter(wx.MessageBox, "OTO生成完成！", "成功", wx.OK | wx.ICON_INFORMATION)
            except Exception as e:
                error_msg = f"生成失败：{str(e)}"
                wx.CallAfter(self.oto_result_text.AppendText, error_msg + "\n")
                wx.CallAfter(wx.MessageBox, error_msg, "错误", wx.OK | wx.ICON_ERROR)
        
        thread = threading.Thread(target=generate_oto_thread)
        thread.start()

    def on_browse_folder(self, event, text_ctrl):
        dialog = wx.DirDialog(None, "选择文件夹", style=wx.DD_DEFAULT_STYLE)
        if dialog.ShowModal() == wx.ID_CANCEL:
            return
        path = dialog.GetPath()
        text_ctrl.SetValue(path)
        dialog.Destroy()
    
    def on_browse_file(self, event, text_ctrl):
        dialog = wx.FileDialog(None, "选择文件", style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)
        if dialog.ShowModal() == wx.ID_CANCEL:
            return
        path = dialog.GetPath()
        text_ctrl.SetValue(path)
        dialog.Destroy()
    
    def on_browse_file(self, event, text_ctrl):
        dialog = wx.FileDialog(None, "选择文件", style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)
        if dialog.ShowModal() == wx.ID_CANCEL:
            return
        path = dialog.GetPath()
        text_ctrl.SetValue(path)
        dialog.Destroy()
    
    def on_browse_file(self, event, text_ctrl):
        dialog = wx.FileDialog(None, "选择文件", style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)
        if dialog.ShowModal() == wx.ID_CANCEL:
            return
        path = dialog.GetPath()
        text_ctrl.SetValue(path)
        dialog.Destroy()
    
    def on_generate_lab(self, event):
        path = self.path_text.GetValue().strip()
        separator_str = self.separator_text.GetValue().strip()
        
        if not path:
            wx.MessageBox("请选择音源路径", "错误", wx.OK | wx.ICON_ERROR)
            return
        
        if not os.path.exists(path):
            wx.MessageBox("音源路径不存在", "错误", wx.OK | wx.ICON_ERROR)
            return
        
        cuts = [s.strip() for s in separator_str.split(',') if s.strip()]
        
        def generate_lab_thread():
            try:
                wx.CallAfter(self.lab_result_text.Clear)
                wx.CallAfter(self.lab_result_text.AppendText, "正在生成LAB文件...\n")

                with TextRedirector(self.lab_result_text):
                    wavname2lab.run(path, cuts)

                wx.CallAfter(self.lab_result_text.AppendText, "LAB文件生成完成！\n")
                wx.CallAfter(wx.MessageBox, "LAB文件生成完成", "成功", wx.OK | wx.ICON_INFORMATION)
            except Exception as e:
                error_msg = f"生成失败：{str(e)}"
                wx.CallAfter(self.lab_result_text.AppendText, error_msg + "\n")
                wx.CallAfter(wx.MessageBox, error_msg, "错误", wx.OK | wx.ICON_ERROR)

        thread = threading.Thread(target=generate_lab_thread)
        thread.start()

    def on_infer(self, event):
        wav_folder = self.textgrid_folder_text.GetValue().strip()
        model_folder = self.model_folder_choice.GetStringSelection()
        model_file = self.model_file_choice.GetStringSelection()
        dict_file = self.dict_choice.GetStringSelection()

        if not wav_folder:
            wx.MessageBox("请选择音源文件夹", "错误", wx.OK | wx.ICON_ERROR)
            return

        if not os.path.exists(wav_folder):
            wx.MessageBox("音源文件夹不存在", "错误", wx.OK | wx.ICON_ERROR)
            return

        if not model_folder:
            wx.MessageBox("请选择模型文件夹", "错误", wx.OK | wx.ICON_ERROR)
            return

        if not model_file:
            wx.MessageBox("请选择模型文件", "错误", wx.OK | wx.ICON_ERROR)
            return

        if not dict_file:
            wx.MessageBox("请选择字典文件", "错误", wx.OK | wx.ICON_ERROR)
            return

        def infer_thread():
            try:
                wx.CallAfter(self.infer_result_text.Clear)
                wx.CallAfter(self.infer_result_text.AppendText, "正在加载模型...\n")


                model_path = Path(os.path.dirname(os.path.abspath(__file__))) / 'HubertFA_model' / model_folder / model_file
                dict_path = Path(os.path.dirname(os.path.abspath(__file__))) / 'HubertFA_model' / model_folder / dict_file

                language = dict_file.split('.')[0].split('-')[0]
                language = language[0] if len(language) == 1 else language

                wx.CallAfter(self.infer_result_text.AppendText, f"模型: {model_file}\n")
                wx.CallAfter(self.infer_result_text.AppendText, f"字典: {dict_file}\n")
                wx.CallAfter(self.infer_result_text.AppendText, f"语言: {language}\n")

                inference = onnx_infer.InferenceOnnx(model_path)
                wx.CallAfter(self.infer_result_text.AppendText, "加载配置...\n")
                inference.load_config()
                wx.CallAfter(self.infer_result_text.AppendText, "加载模型...\n")
                inference.load_model()
                wx.CallAfter(self.infer_result_text.AppendText, "初始化解码器...\n")
                inference.init_decoder()

                def progress_callback(msg):
                    wx.CallAfter(self.infer_result_text.AppendText, msg + "\n")
                    wx.CallAfter(self.infer_result_text.ShowPosition, self.infer_result_text.GetLastPosition())

                inference.set_progress_callback(progress_callback)

                wx.CallAfter(self.infer_result_text.AppendText, "加载数据集...\n")
                inference.get_dataset(wav_folder, language=language, g2p="dictionary", dictionary_path=str(dict_path), in_format="lab")
                wx.CallAfter(self.infer_result_text.AppendText, "开始推理...\n")
                inference.infer(non_lexical_phonemes="AP", pad_times=1, pad_length=5)
                wx.CallAfter(self.infer_result_text.AppendText, "导出结果...\n")
                inference.export(wav_folder)

                wx.CallAfter(self.infer_result_text.AppendText, "TextGrid推理完成！\n")
                wx.CallAfter(wx.MessageBox, "TextGrid推理完成", "成功", wx.OK | wx.ICON_INFORMATION)
            except Exception as e:
                error_msg = f"推理失败：{str(e)}"
                wx.CallAfter(self.infer_result_text.AppendText, error_msg + "\n")
                wx.CallAfter(wx.MessageBox, error_msg, "错误", wx.OK | wx.ICON_ERROR)

        thread = threading.Thread(target=infer_thread)
        thread.start()

    def on_clean_sp(self, event):
        wav_folder = self.clean_path_text.GetValue().strip()
        
        if not wav_folder:
            wx.MessageBox("请选择WAV文件夹", "错误", wx.OK | wx.ICON_ERROR)
            return
        
        if not os.path.exists(wav_folder):
            wx.MessageBox("WAV文件夹不存在", "错误", wx.OK | wx.ICON_ERROR)
            return
        
        def clean_sp_thread():
            try:
                wx.CallAfter(self.clean_result_text.Clear)
                wx.CallAfter(self.clean_result_text.AppendText, "正在清理SP标记...\n")
                
                ignore = 'AP,SP,EP'
                delete_sp = True
                
                with TextRedirector(self.clean_result_text):
                    deleted_files = del_SP.process_all_textgrid_files(wav_folder, ignore, delete_sp)
                
                if deleted_files and deleted_files[0] != '没有文件中有SP被删除':
                    result_msg = "以下文件中有SP被删除（建议复核标记）：\n"
                    result_msg += ', '.join(deleted_files)
                    wx.CallAfter(self.clean_result_text.AppendText, result_msg + "\n")
                    wx.CallAfter(wx.MessageBox, "SP清理完成，请查看结果", "成功", wx.OK | wx.ICON_INFORMATION)
                else:
                    wx.CallAfter(self.clean_result_text.AppendText, "没有文件中有SP被删除。\n")
                    wx.CallAfter(wx.MessageBox, "没有文件中有SP被删除", "提示", wx.OK | wx.ICON_INFORMATION)
                    
            except Exception as e:
                error_msg = f"清理SP失败：{str(e)}"
                wx.CallAfter(self.clean_result_text.AppendText, error_msg + "\n")
                wx.CallAfter(wx.MessageBox, error_msg, "错误", wx.OK | wx.ICON_ERROR)

        thread = threading.Thread(target=clean_sp_thread)
        thread.start()

    def on_generate_json(self, event):
        wav_folder = self.json_path_text.GetValue().strip()
        dict_file = self.json_dict_choice.GetStringSelection()
        ignore = self.json_ignore_text.GetValue().strip()

        if not wav_folder:
            wx.MessageBox("请选择WAV文件夹", "错误", wx.OK | wx.ICON_ERROR)
            return
        
        if not os.path.exists(wav_folder):
            wx.MessageBox("WAV文件夹不存在", "错误", wx.OK | wx.ICON_ERROR)
            return

        if not dict_file:
            wx.MessageBox("请选择模型字典", "错误", wx.OK | wx.ICON_ERROR)
            return
        
        def generate_json_thread():
            try:
                wx.CallAfter(self.json_result_text.Clear)
                wx.CallAfter(self.json_result_text.AppendText, "正在生成JSON文件...\n")

                # 获取字典路径
                selected_folder = self.json_folder_choice.GetStringSelection()
                dict_full_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'HubertFA_model', selected_folder, dict_file)

                if not os.path.exists(dict_full_path):
                    wx.CallAfter(wx.MessageBox, "字典文件不存在", "错误", wx.OK | wx.ICON_ERROR)
                    return

                def process_folder(folder_path, folder_name=""):
                    textgrid_files = [f for f in os.listdir(folder_path) if f.endswith('.TextGrid')]
                    
                    if textgrid_files:
                        display_name = folder_name if folder_name else folder_path
                        wx.CallAfter(self.json_result_text.AppendText, f"正在处理: {display_name}\n")
                        
                        # 生成ds_phone.json
                        rec_preset = None
                        with TextRedirector(self.json_result_text):
                            TextGrid2ds_json.run(folder_path, rec_preset)
                        
                        # 过滤音素
                        json_path = os.path.join(folder_path, 'json', 'ds_phone.json')
                        if os.path.exists(json_path):
                            with TextRedirector(self.json_result_text):
                                ds_json2filter.run(dict_full_path, json_path, ignore)
                            
                            # 生成word.json
                            filter_json_path = os.path.join(folder_path, 'json', 'ds_phone_filter.json')
                            if os.path.exists(filter_json_path):
                                with TextRedirector(self.json_result_text):
                                    ds_json2word.run(dict_full_path, filter_json_path)
                        
                        wx.CallAfter(self.json_result_text.AppendText, f"完成: {display_name}\n")
                        return True
                    return False
                
                processed_count = 0
                
                # 处理主文件夹
                if process_folder(wav_folder):
                    processed_count += 1
                
                # 遍历第一层子文件夹
                subfolders = [f for f in os.listdir(wav_folder) if os.path.isdir(os.path.join(wav_folder, f))]
                for folder in subfolders:
                    folder_path = os.path.join(wav_folder, folder)
                    if process_folder(folder_path, folder):
                        processed_count += 1
                
                wx.CallAfter(self.json_result_text.AppendText, f"\nJSON文件生成完成！共处理 {processed_count} 个文件夹\n")
                wx.CallAfter(wx.MessageBox, f"JSON文件生成完成！共处理 {processed_count} 个文件夹", "成功", wx.OK | wx.ICON_INFORMATION)

            except Exception as e:
                error_msg = f"生成JSON失败：{str(e)}"
                wx.CallAfter(self.json_result_text.AppendText, error_msg + "\n")
                wx.CallAfter(wx.MessageBox, error_msg, "错误", wx.OK | wx.ICON_ERROR)

        thread = threading.Thread(target=generate_json_thread)
        thread.start()

if __name__ == "__main__":
    app = wx.App()
    frame = MainFrame()
    frame.Show()
    app.MainLoop()