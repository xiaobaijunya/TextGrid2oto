import wx
import os
import sys
import time
import traceback
import threading
from pathlib import Path

# ✅ 万能获取程序根目录（开发/打包通用）
def get_app_root() -> Path:
    if getattr(sys, 'frozen', False):
        # 打包后运行：返回 exe 所在目录
        return Path(sys.executable).parent
    else:
        # 开发时运行：返回当前 .py 文件所在目录
        return Path(__file__).parent

# 全局变量，直接用
ROOT = get_app_root()

sys.path.append(str(ROOT))
import lab_generate.wavname2lab as wavname2lab
import lab_generate.index2lab as index2lab
import onnx_infer
from textgrid2json import del_SP, TextGrid2ds_json, ds_json2filter, ds_json2word
from json2oto import json2CV_oto, json2oto, json2VCV_oto, json2test,json2arpasing_oto
sys.path.append(str(ROOT / 'tg2svdb'))
from tg2svdb import tg2sv_change
from i18n import _
from oto.oto_param_panel import OtoParamPanel

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
        self.original_stderr.flush()
    
    def __enter__(self):
        sys.stdout = self
        sys.stderr = self
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr

class MainFrame(wx.Frame):
    def __init__(self):
        super().__init__(None, title=_('window.title'), size=(850, 760))
        from i18n import register
        register(self, 'window.title', 'title')
        
        icon_path = str(ROOT / "img" / "TextGrid2oto.ico")
        icon = wx.Icon(icon_path, wx.BITMAP_TYPE_ICO)
        self.SetIcon(icon)
        
        panel = wx.Panel(self)
        
        notebook = wx.Notebook(panel)
        
        lab_panel = wx.Panel(notebook)
        lab_sizer = wx.BoxSizer(wx.VERTICAL)
        
        lab_notebook = wx.Notebook(lab_panel)
        
        # 第一个标签页：根据wav名生成lab（常规）
        lab_wav_panel = wx.Panel(lab_notebook)
        lab_wav_sizer = wx.BoxSizer(wx.VERTICAL)

        # ── 音源设置 ──
        wav_box = wx.StaticBox(lab_wav_panel, label=_('lab.wavname.settings'))
        wav_box_sizer = wx.StaticBoxSizer(wav_box, wx.VERTICAL)
        register(wav_box, 'lab.wavname.settings', 'label')

        path_sizer = wx.BoxSizer(wx.HORIZONTAL)
        path_label = wx.StaticText(lab_wav_panel, label=_('lab.wavname.folder'))
        register(path_label, 'lab.wavname.folder')
        self.path_text = wx.TextCtrl(lab_wav_panel, size=(400, -1))
        browse_btn = wx.Button(lab_wav_panel, label=_('lab.wavname.browse_folder'))
        register(browse_btn, 'lab.wavname.browse_folder', 'button')
        browse_btn.Bind(wx.EVT_BUTTON, lambda event: self.on_browse_folder(event, self.path_text))
        path_sizer.Add(path_label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        path_sizer.Add(self.path_text, 1, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        path_sizer.Add(browse_btn, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        wav_box_sizer.Add(path_sizer, 0, wx.EXPAND | wx.ALL, 3)

        separator_sizer = wx.BoxSizer(wx.HORIZONTAL)
        separator_label = wx.StaticText(lab_wav_panel, label=_('lab.wavname.separator'))
        register(separator_label, 'lab.wavname.separator')
        self.separator_text = wx.TextCtrl(lab_wav_panel, value="_,-", size=(200, -1))
        separator_sizer.Add(separator_label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        separator_sizer.Add(self.separator_text, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        wav_box_sizer.Add(separator_sizer, 0, wx.ALL, 3)

        generate_btn = wx.Button(lab_wav_panel, label=_('lab.wavname.generate'))
        register(generate_btn, 'lab.wavname.generate', 'button')
        generate_btn.Bind(wx.EVT_BUTTON, self.on_generate_lab)
        wav_box_sizer.Add(generate_btn, 0, wx.ALL | wx.CENTER, 5)

        lab_wav_sizer.Add(wav_box_sizer, 0, wx.EXPAND | wx.ALL, 5)

        # 结果显示文本框
        lab_result_label = wx.StaticText(lab_wav_panel, label=_('lab.wavname.result'))
        register(lab_result_label, 'lab.wavname.result')
        lab_wav_sizer.Add(lab_result_label, 0, wx.ALL | wx.LEFT, 5)

        self.lab_result_text = wx.TextCtrl(lab_wav_panel, style=wx.TE_MULTILINE | wx.TE_READONLY, size=(-1, 400))
        lab_wav_sizer.Add(self.lab_result_text, 1, wx.EXPAND | wx.ALL, 5)

        lab_wav_panel.SetSizer(lab_wav_sizer)
        
        # 第二个标签页：根据index生成lab
        lab_index_panel = wx.Panel(lab_notebook)
        lab_index_sizer = wx.BoxSizer(wx.VERTICAL)

        # ── 音源设置 ──
        idx_box = wx.StaticBox(lab_index_panel, label=_('lab.index.settings'))
        idx_box_sizer = wx.StaticBoxSizer(idx_box, wx.VERTICAL)
        register(idx_box, 'lab.index.settings', 'label')

        # WAV路径框
        lab_wav_path_sizer = wx.BoxSizer(wx.HORIZONTAL)
        lab_wav_path_label = wx.StaticText(lab_index_panel, label=_('lab.index.wav_path'))
        register(lab_wav_path_label, 'lab.index.wav_path')
        self.lab_wav_path_text = wx.TextCtrl(lab_index_panel, size=(400, -1))
        lab_wav_browse_btn = wx.Button(lab_index_panel, label=_('lab.index.browse_wav'))
        register(lab_wav_browse_btn, 'lab.index.browse_wav', 'button')
        lab_wav_browse_btn.Bind(wx.EVT_BUTTON, lambda event: self.on_browse_folder(event, self.lab_wav_path_text))
        lab_wav_path_sizer.Add(lab_wav_path_label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        lab_wav_path_sizer.Add(self.lab_wav_path_text, 1, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        lab_wav_path_sizer.Add(lab_wav_browse_btn, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        idx_box_sizer.Add(lab_wav_path_sizer, 0, wx.EXPAND | wx.ALL, 3)

        # Index路径框
        lab_index_path_sizer = wx.BoxSizer(wx.HORIZONTAL)
        lab_index_path_label = wx.StaticText(lab_index_panel, label=_('lab.index.index_path'))
        register(lab_index_path_label, 'lab.index.index_path')
        self.lab_index_path_text = wx.TextCtrl(lab_index_panel, size=(400, -1))
        lab_index_browse_btn = wx.Button(lab_index_panel, label=_('lab.index.browse_file'))
        register(lab_index_browse_btn, 'lab.index.browse_file', 'button')
        lab_index_browse_btn.Bind(wx.EVT_BUTTON, lambda event: self.on_browse_file(event, self.lab_index_path_text))
        lab_index_path_sizer.Add(lab_index_path_label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        lab_index_path_sizer.Add(self.lab_index_path_text, 1, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        lab_index_path_sizer.Add(lab_index_browse_btn, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        idx_box_sizer.Add(lab_index_path_sizer, 0, wx.EXPAND | wx.ALL, 3)

        # 分隔符输入框
        lab_separator_sizer = wx.BoxSizer(wx.HORIZONTAL)
        lab_separator_label = wx.StaticText(lab_index_panel, label=_('lab.index.separator'))
        register(lab_separator_label, 'lab.index.separator')
        self.lab_separator_text = wx.TextCtrl(lab_index_panel, value="_,-", size=(200, -1))
        lab_separator_sizer.Add(lab_separator_label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        lab_separator_sizer.Add(self.lab_separator_text, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        idx_box_sizer.Add(lab_separator_sizer, 0, wx.ALL, 3)

        lab_index_generate_btn = wx.Button(lab_index_panel, label=_('lab.index.generate'))
        register(lab_index_generate_btn, 'lab.index.generate', 'button')
        lab_index_generate_btn.Bind(wx.EVT_BUTTON, self.on_generate_lab_from_index)
        idx_box_sizer.Add(lab_index_generate_btn, 0, wx.ALL | wx.CENTER, 5)

        lab_index_sizer.Add(idx_box_sizer, 0, wx.EXPAND | wx.ALL, 5)

        # 结果显示文本框
        lab_index_result_label = wx.StaticText(lab_index_panel, label=_('lab.index.result'))
        register(lab_index_result_label, 'lab.index.result')
        lab_index_sizer.Add(lab_index_result_label, 0, wx.ALL | wx.LEFT, 5)

        self.lab_index_result_text = wx.TextCtrl(lab_index_panel, style=wx.TE_MULTILINE | wx.TE_READONLY, size=(-1, 400))
        lab_index_sizer.Add(self.lab_index_result_text, 1, wx.EXPAND | wx.ALL, 5)

        lab_index_panel.SetSizer(lab_index_sizer)
        
        # 添加两个标签页到notebook
        lab_notebook.AddPage(lab_wav_panel, _('lab.tab.wavname'))
        register(lab_notebook, 'lab.tab.wavname', 'notebook_tab', 0)
        lab_notebook.AddPage(lab_index_panel, _('lab.tab.index'))
        register(lab_notebook, 'lab.tab.index', 'notebook_tab', 1)
        
        lab_sizer.Add(lab_notebook, 1, wx.EXPAND | wx.ALL, 5)
        lab_panel.SetSizer(lab_sizer)
        
        textgrid_panel = wx.Panel(notebook)
        textgrid_sizer = wx.BoxSizer(wx.VERTICAL)

        textgrid_title = wx.StaticText(textgrid_panel, label=_('textgrid.title'))
        register(textgrid_title, 'textgrid.title')
        textgrid_sizer.Add(textgrid_title, 0, wx.ALL | wx.CENTER, 5)

        # ── 音源文件夹 ──
        folder_box = wx.StaticBox(textgrid_panel, label=_('textgrid.folder'))
        folder_box_sizer = wx.StaticBoxSizer(folder_box, wx.HORIZONTAL)
        register(folder_box, 'textgrid.folder', 'label')
        self.textgrid_folder_text = wx.TextCtrl(textgrid_panel, size=(500, -1))
        browse_folder_btn = wx.Button(textgrid_panel, label=_('textgrid.browse_folder'))
        register(browse_folder_btn, 'textgrid.browse_folder', 'button')
        browse_folder_btn.Bind(wx.EVT_BUTTON, lambda event: self.on_browse_folder(event, self.textgrid_folder_text))
        folder_box_sizer.Add(self.textgrid_folder_text, 1, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        folder_box_sizer.Add(browse_folder_btn, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        textgrid_sizer.Add(folder_box_sizer, 0, wx.EXPAND | wx.ALL, 5)

        # 主内容区域：左右两栏
        main_content_sizer = wx.BoxSizer(wx.HORIZONTAL)

        # 左侧栏：模型选择
        left_model_box = wx.StaticBox(textgrid_panel, label=_('textgrid.model_section'))
        left_sizer = wx.StaticBoxSizer(left_model_box, wx.VERTICAL)
        register(left_model_box, 'textgrid.model_section', 'label')

        model_folder_sizer = wx.BoxSizer(wx.HORIZONTAL)
        model_folder_label = wx.StaticText(textgrid_panel, label=_('textgrid.model_folder'))
        register(model_folder_label, 'textgrid.model_folder')
        self.model_folder_choice = wx.Choice(textgrid_panel, size=(250, -1))
        self.model_folder_choice.Bind(wx.EVT_CHOICE, self.on_model_folder_selected)
        model_folder_sizer.Add(model_folder_label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        model_folder_sizer.Add(self.model_folder_choice, 1, wx.EXPAND | wx.ALL, 3)
        left_sizer.Add(model_folder_sizer, 0, wx.EXPAND | wx.ALL, 3)

        model_file_sizer = wx.BoxSizer(wx.HORIZONTAL)
        model_file_label = wx.StaticText(textgrid_panel, label=_('textgrid.model_file'))
        register(model_file_label, 'textgrid.model_file')
        self.model_file_choice = wx.Choice(textgrid_panel, size=(250, -1))
        model_file_sizer.Add(model_file_label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        model_file_sizer.Add(self.model_file_choice, 1, wx.EXPAND | wx.ALL, 3)
        left_sizer.Add(model_file_sizer, 0, wx.EXPAND | wx.ALL, 3)

        dict_sizer = wx.BoxSizer(wx.HORIZONTAL)
        dict_label = wx.StaticText(textgrid_panel, label=_('textgrid.dict'))
        register(dict_label, 'textgrid.dict')
        self.dict_choice = wx.Choice(textgrid_panel, size=(250, -1))
        dict_sizer.Add(dict_label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        dict_sizer.Add(self.dict_choice, 1, wx.EXPAND | wx.ALL, 3)
        left_sizer.Add(dict_sizer, 0, wx.EXPAND | wx.ALL, 3)

        main_content_sizer.Add(left_sizer, 1, wx.EXPAND | wx.ALL, 5)

        # 中间分隔线
        separator = wx.StaticLine(textgrid_panel, style=wx.LI_VERTICAL)
        main_content_sizer.Add(separator, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 5)

        # 右侧栏：推理设置
        right_infer_box = wx.StaticBox(textgrid_panel, label=_('textgrid.infer_section'))
        right_sizer = wx.StaticBoxSizer(right_infer_box, wx.VERTICAL)
        register(right_infer_box, 'textgrid.infer_section', 'label')

        # 设备选择
        device_sizer = wx.BoxSizer(wx.HORIZONTAL)
        device_label = wx.StaticText(textgrid_panel, label=_('textgrid.device'))
        register(device_label, 'textgrid.device')
        self.device_choice = wx.Choice(textgrid_panel, size=(400, -1))
        self.device_choice.Append(_('textgrid.device.cpu'), "cpu")
        self.device_choice.Append(_('textgrid.device.dml'), "dml")
        self.device_choice.Append(_('textgrid.device.webgpu'), "webgpu")
        self.device_choice.SetSelection(0)
        register(self.device_choice, '', 'choice', [('textgrid.device.cpu', 'cpu'), ('textgrid.device.dml', 'dml'), ('textgrid.device.webgpu', 'webgpu')])
        device_sizer.Add(device_label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        device_sizer.Add(self.device_choice, 1, wx.EXPAND | wx.ALL, 3)
        right_sizer.Add(device_sizer, 0, wx.EXPAND | wx.ALL, 3)

        # 并行工作进程数选择（DML / WebGPU）
        worker_sizer = wx.BoxSizer(wx.HORIZONTAL)
        worker_label = wx.StaticText(textgrid_panel, label=_('textgrid.workers'))
        register(worker_label, 'textgrid.workers')
        self.worker_choice = wx.Choice(textgrid_panel, size=(400, -1))
        self.worker_choice.SetToolTip(_('textgrid.workers.tooltip'))
        register(self.worker_choice, 'textgrid.workers.tooltip', 'tooltip')
        self.worker_choice.Append(_('textgrid.workers.1'), 1)
        self.worker_choice.Append(_('textgrid.workers.2'), 2)
        self.worker_choice.Append(_('textgrid.workers.3'), 3)
        self.worker_choice.Append(_('textgrid.workers.4'), 4)
        self.worker_choice.SetSelection(0)
        register(self.worker_choice, '', 'choice', [('textgrid.workers.1', 1), ('textgrid.workers.2', 2), ('textgrid.workers.3', 3), ('textgrid.workers.4', 4)])
        worker_sizer.Add(worker_label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        worker_sizer.Add(self.worker_choice, 1, wx.EXPAND | wx.ALL, 3)
        right_sizer.Add(worker_sizer, 0, wx.EXPAND | wx.ALL, 3)

        # pad_times选择
        pad_times_sizer = wx.BoxSizer(wx.HORIZONTAL)
        pad_times_label = wx.StaticText(textgrid_panel, label=_('textgrid.pad_times'))
        register(pad_times_label, 'textgrid.pad_times')
        self.pad_times_choice = wx.Choice(textgrid_panel, size=(400, -1))
        self.pad_times_choice.SetToolTip(_('textgrid.pad_times.tooltip'))
        register(self.pad_times_choice, 'textgrid.pad_times.tooltip', 'tooltip')
        self.pad_times_choice.Append(_('textgrid.pad_times.1'), 1)
        self.pad_times_choice.Append(_('textgrid.pad_times.3'), 3)
        self.pad_times_choice.Append(_('textgrid.pad_times.5'), 5)
        self.pad_times_choice.SetSelection(1)
        register(self.pad_times_choice, '', 'choice', [('textgrid.pad_times.1', 1), ('textgrid.pad_times.3', 3), ('textgrid.pad_times.5', 5)])
        pad_times_sizer.Add(pad_times_label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        pad_times_sizer.Add(self.pad_times_choice, 1, wx.EXPAND | wx.ALL, 3)
        right_sizer.Add(pad_times_sizer, 0, wx.EXPAND | wx.ALL, 3)

        # pad_length选择
        pad_length_sizer = wx.BoxSizer(wx.HORIZONTAL)
        pad_length_label = wx.StaticText(textgrid_panel, label=_('textgrid.pad_length'))
        register(pad_length_label, 'textgrid.pad_length')
        self.pad_length_choice = wx.Choice(textgrid_panel, size=(400, -1))
        self.pad_length_choice.SetToolTip(_('textgrid.pad_length.tooltip'))
        register(self.pad_length_choice, 'textgrid.pad_length.tooltip', 'tooltip')
        self.pad_length_choice.Append(_('textgrid.pad_length.3'), 3)
        self.pad_length_choice.Append(_('textgrid.pad_length.5'), 5)
        self.pad_length_choice.Append(_('textgrid.pad_length.7'), 7)
        self.pad_length_choice.Append(_('textgrid.pad_length.10'), 10)
        self.pad_length_choice.SetSelection(1)
        register(self.pad_length_choice, '', 'choice', [('textgrid.pad_length.3', 3), ('textgrid.pad_length.5', 5), ('textgrid.pad_length.7', 7), ('textgrid.pad_length.10', 10)])
        pad_length_sizer.Add(pad_length_label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        pad_length_sizer.Add(self.pad_length_choice, 1, wx.EXPAND | wx.ALL, 3)
        right_sizer.Add(pad_length_sizer, 0, wx.EXPAND | wx.ALL, 3)

        # 合并重复音素选项
        merge_phonemes_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.merge_phonemes_checkbox = wx.CheckBox(textgrid_panel, label=_('textgrid.merge_phonemes'))
        self.merge_phonemes_checkbox.SetToolTip(_('textgrid.merge_phonemes.tooltip'))
        register(self.merge_phonemes_checkbox, 'textgrid.merge_phonemes.tooltip', 'tooltip')
        register(self.merge_phonemes_checkbox, 'textgrid.merge_phonemes', 'checkbox')
        self.merge_phonemes_checkbox.SetValue(True)
        merge_phonemes_sizer.Add(self.merge_phonemes_checkbox, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        right_sizer.Add(merge_phonemes_sizer, 0, wx.ALL, 3)

        main_content_sizer.Add(right_sizer, 1, wx.EXPAND | wx.ALL, 5)

        textgrid_sizer.Add(main_content_sizer, 0, wx.EXPAND | wx.ALL, 10)

        infer_btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        infer_btn = wx.Button(textgrid_panel, label=_('textgrid.infer'))
        register(infer_btn, 'textgrid.infer', 'button')
        infer_btn.Bind(wx.EVT_BUTTON, self.on_infer)
        infer_btn_sizer.Add(infer_btn, 0, wx.ALL | wx.CENTER, 5)
        self.stop_infer_btn = wx.Button(textgrid_panel, label=_('textgrid.stop_infer'))
        register(self.stop_infer_btn, 'textgrid.stop_infer', 'button')
        self.stop_infer_btn.Bind(wx.EVT_BUTTON, self.on_stop_infer)
        self.stop_infer_btn.Disable()
        infer_btn_sizer.Add(self.stop_infer_btn, 0, wx.ALL | wx.CENTER, 5)
        textgrid_sizer.Add(infer_btn_sizer, 0, wx.ALL | wx.CENTER, 5)

        # 结果显示文本框
        infer_result_label = wx.StaticText(textgrid_panel, label=_('textgrid.result'))
        register(infer_result_label, 'textgrid.result')
        textgrid_sizer.Add(infer_result_label, 0, wx.ALL | wx.LEFT, 5)

        self.infer_result_text = wx.TextCtrl(textgrid_panel, style=wx.TE_MULTILINE | wx.TE_READONLY, size=(-1, 400))
        textgrid_sizer.Add(self.infer_result_text, 1, wx.EXPAND | wx.ALL, 5)

        textgrid_panel.SetSizer(textgrid_sizer)

        # JSON生成面板
        json_panel = wx.Panel(notebook)
        json_sizer = wx.BoxSizer(wx.VERTICAL)

        # ── 音源设置 ──
        json_src_box = wx.StaticBox(json_panel, label=_('json.src_section'))
        json_src_sizer = wx.StaticBoxSizer(json_src_box, wx.VERTICAL)
        register(json_src_box, 'json.src_section', 'label')

        json_path_sizer = wx.BoxSizer(wx.HORIZONTAL)
        json_path_label = wx.StaticText(json_panel, label=_('json.wav_folder'))
        register(json_path_label, 'json.wav_folder')
        self.json_path_text = wx.TextCtrl(json_panel, size=(400, -1))
        json_browse_btn = wx.Button(json_panel, label=_('json.browse_folder'))
        register(json_browse_btn, 'json.browse_folder', 'button')
        json_browse_btn.Bind(wx.EVT_BUTTON, lambda event: self.on_browse_folder(event, self.json_path_text))
        json_path_sizer.Add(json_path_label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        json_path_sizer.Add(self.json_path_text, 1, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        json_path_sizer.Add(json_browse_btn, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        json_src_sizer.Add(json_path_sizer, 0, wx.EXPAND | wx.ALL, 3)

        json_recording_sizer = wx.BoxSizer(wx.HORIZONTAL)
        json_recording_label = wx.StaticText(json_panel, label=_('json.recording'))
        register(json_recording_label, 'json.recording')
        self.json_recording_text = wx.TextCtrl(json_panel, value="", size=(300, -1))
        json_recording_browse_btn = wx.Button(json_panel, label=_('json.browse'), size=(60, -1))
        register(json_recording_browse_btn, 'json.browse', 'button')
        json_recording_browse_btn.Bind(wx.EVT_BUTTON, self.on_browse_json_recording)
        json_recording_sizer.Add(json_recording_label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        json_recording_sizer.Add(self.json_recording_text, 1, wx.EXPAND | wx.ALL, 3)
        json_recording_sizer.Add(json_recording_browse_btn, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        json_src_sizer.Add(json_recording_sizer, 0, wx.EXPAND | wx.ALL, 3)

        json_sizer.Add(json_src_sizer, 0, wx.EXPAND | wx.ALL, 5)

        # ── 模型与过滤 ──
        json_model_box = wx.StaticBox(json_panel, label=_('json.model_section'))
        json_model_sizer = wx.StaticBoxSizer(json_model_box, wx.VERTICAL)
        register(json_model_box, 'json.model_section', 'label')

        json_folder_sizer = wx.BoxSizer(wx.HORIZONTAL)
        json_folder_label = wx.StaticText(json_panel, label=_('json.model_folder'))
        register(json_folder_label, 'json.model_folder')
        self.json_folder_choice = wx.Choice(json_panel, size=(300, -1))
        self.json_folder_choice.Bind(wx.EVT_CHOICE, self.on_json_folder_selected)
        json_folder_sizer.Add(json_folder_label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        json_folder_sizer.Add(self.json_folder_choice, 1, wx.EXPAND | wx.ALL, 3)
        json_model_sizer.Add(json_folder_sizer, 0, wx.EXPAND | wx.ALL, 3)

        json_dict_sizer = wx.BoxSizer(wx.HORIZONTAL)
        json_dict_label = wx.StaticText(json_panel, label=_('json.model_dict'))
        register(json_dict_label, 'json.model_dict')
        self.json_dict_choice = wx.Choice(json_panel, size=(300, -1))
        self.json_dict_choice.Bind(wx.EVT_CHOICE, self.on_json_dict_selected)
        json_dict_sizer.Add(json_dict_label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        json_dict_sizer.Add(self.json_dict_choice, 1, wx.EXPAND | wx.ALL, 3)
        json_model_sizer.Add(json_dict_sizer, 0, wx.EXPAND | wx.ALL, 3)

        # 忽略音素
        json_ignore_sizer = wx.BoxSizer(wx.HORIZONTAL)
        json_ignore_label = wx.StaticText(json_panel, label=_('json.ignore'))
        register(json_ignore_label, 'json.ignore')
        self.json_ignore_text = wx.TextCtrl(json_panel, value="AP,SP,EP,R", size=(200, -1))
        json_ignore_sizer.Add(json_ignore_label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        json_ignore_sizer.Add(self.json_ignore_text, 1, wx.EXPAND | wx.ALL, 3)
        json_model_sizer.Add(json_ignore_sizer, 0, wx.EXPAND | wx.ALL, 3)

        json_sizer.Add(json_model_sizer, 0, wx.EXPAND | wx.ALL, 5)

        # 生成JSON按钮
        generate_json_btn = wx.Button(json_panel, label=_('json.generate'))
        register(generate_json_btn, 'json.generate', 'button')
        generate_json_btn.Bind(wx.EVT_BUTTON, self.on_generate_json)
        json_sizer.Add(generate_json_btn, 0, wx.ALL | wx.CENTER, 5)

        # 结果显示文本框
        json_result_label = wx.StaticText(json_panel, label=_('json.result'))
        register(json_result_label, 'json.result')
        json_sizer.Add(json_result_label, 0, wx.ALL | wx.LEFT, 5)

        self.json_result_text = wx.TextCtrl(json_panel, style=wx.TE_MULTILINE | wx.TE_READONLY, size=(-1, 400))
        json_sizer.Add(self.json_result_text, 1, wx.EXPAND | wx.ALL, 5)

        json_panel.SetSizer(json_sizer)

        mark_panel = wx.Panel(notebook)
        mark_sizer = wx.BoxSizer(wx.VERTICAL)

        mark_notebook = wx.Notebook(mark_panel)

        oto_panel = wx.Panel(mark_notebook)
        oto_sizer = wx.BoxSizer(wx.VERTICAL)
        
        oto_title = wx.StaticText(oto_panel, label=_('mark.oto.title'))
        register(oto_title, 'mark.oto.title')
        oto_sizer.Add(oto_title, 0, wx.ALL | wx.CENTER, 5)
        
        # ── 音源文件夹 ──
        oto_folder_box = wx.StaticBox(oto_panel, label=_('mark.oto.folder'))
        oto_folder_sizer = wx.StaticBoxSizer(oto_folder_box, wx.HORIZONTAL)
        register(oto_folder_box, 'mark.oto.folder', 'label')
        self.oto_path_text = wx.TextCtrl(oto_panel, size=(400, -1))
        oto_browse_btn = wx.Button(oto_panel, label=_('mark.oto.browse_folder'))
        register(oto_browse_btn, 'mark.oto.browse_folder', 'button')
        oto_browse_btn.Bind(wx.EVT_BUTTON, lambda event: self.on_browse_folder(event, self.oto_path_text))
        oto_folder_sizer.Add(self.oto_path_text, 1, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        oto_folder_sizer.Add(oto_browse_btn, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        oto_sizer.Add(oto_folder_sizer, 0, wx.EXPAND | wx.ALL, 5)

        # ── 共享 OTO 参数面板 ──
        self.oto_params = OtoParamPanel(oto_panel)
        oto_sizer.Add(self.oto_params, 0, wx.EXPAND | wx.ALL, 5)

        # 生成OTO按钮
        oto_generate_btn = wx.Button(oto_panel, label=_('mark.oto.generate'))
        register(oto_generate_btn, 'mark.oto.generate', 'button')
        oto_generate_btn.Bind(wx.EVT_BUTTON, self.on_generate_oto)
        oto_sizer.Add(oto_generate_btn, 0, wx.ALL | wx.CENTER, 5)
        
        # 结果显示文本框
        oto_result_label = wx.StaticText(oto_panel, label=_('mark.oto.result'))
        register(oto_result_label, 'mark.oto.result')
        oto_sizer.Add(oto_result_label, 0, wx.ALL | wx.LEFT, 0)
        
        self.oto_result_text = wx.TextCtrl(oto_panel, style=wx.TE_MULTILINE | wx.TE_READONLY, size=(-1, 400))
        oto_sizer.Add(self.oto_result_text, 1, wx.EXPAND | wx.ALL, 10)
        
        oto_panel.SetSizer(oto_sizer)

        svdb_panel = wx.Panel(mark_notebook)
        svdb_sizer = wx.BoxSizer(wx.VERTICAL)

        # ── 声库设置 ──
        svdb_box = wx.StaticBox(svdb_panel, label=_('mark.svdb.settings'))
        svdb_box_sizer = wx.StaticBoxSizer(svdb_box, wx.VERTICAL)
        register(svdb_box, 'mark.svdb.settings', 'label')

        svdb_path_sizer = wx.BoxSizer(wx.HORIZONTAL)
        svdb_path_label = wx.StaticText(svdb_panel, label=_('mark.svdb.folder'))
        register(svdb_path_label, 'mark.svdb.folder')
        self.svdb_path_text = wx.TextCtrl(svdb_panel, size=(400, -1))
        svdb_browse_btn = wx.Button(svdb_panel, label=_('mark.svdb.browse_folder'))
        register(svdb_browse_btn, 'mark.svdb.browse_folder', 'button')
        svdb_browse_btn.Bind(wx.EVT_BUTTON, lambda event: self.on_browse_folder(event, self.svdb_path_text))
        svdb_path_sizer.Add(svdb_path_label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        svdb_path_sizer.Add(self.svdb_path_text, 1, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        svdb_path_sizer.Add(svdb_browse_btn, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        svdb_box_sizer.Add(svdb_path_sizer, 0, wx.EXPAND | wx.ALL, 3)

        svdb_dict_sizer = wx.BoxSizer(wx.HORIZONTAL)
        svdb_dict_label = wx.StaticText(svdb_panel, label=_('mark.svdb.dict'))
        register(svdb_dict_label, 'mark.svdb.dict')
        self.svdb_dict_choice = wx.Choice(svdb_panel, size=(300, -1))
        svdb_dict_sizer.Add(svdb_dict_label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        svdb_dict_sizer.Add(self.svdb_dict_choice, 1, wx.EXPAND | wx.ALL, 3)
        svdb_box_sizer.Add(svdb_dict_sizer, 0, wx.EXPAND | wx.ALL, 3)

        svdb_tail_ratio_sizer = wx.BoxSizer(wx.HORIZONTAL)
        svdb_tail_ratio_label = wx.StaticText(svdb_panel, label=_('mark.svdb.tail_ratio'))
        register(svdb_tail_ratio_label, 'mark.svdb.tail_ratio')
        self.svdb_tail_ratio_text = wx.TextCtrl(svdb_panel, value="50", size=(40, -1))
        svdb_tail_ratio_percent = wx.StaticText(svdb_panel, label=_('mark.svdb.tail_percent'))
        svdb_tail_ratio_sizer.Add(svdb_tail_ratio_label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        svdb_tail_ratio_sizer.Add(self.svdb_tail_ratio_text, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        svdb_tail_ratio_sizer.Add(svdb_tail_ratio_percent, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        svdb_box_sizer.Add(svdb_tail_ratio_sizer, 0, wx.ALL, 3)

        svdb_generate_btn = wx.Button(svdb_panel, label=_('mark.svdb.generate'))
        register(svdb_generate_btn, 'mark.svdb.generate', 'button')
        svdb_generate_btn.Bind(wx.EVT_BUTTON, self.on_generate_svdb)
        svdb_box_sizer.Add(svdb_generate_btn, 0, wx.ALL | wx.CENTER, 5)

        svdb_sizer.Add(svdb_box_sizer, 0, wx.EXPAND | wx.ALL, 5)

        svdb_result_label = wx.StaticText(svdb_panel, label=_('mark.svdb.result'))
        register(svdb_result_label, 'mark.svdb.result')
        svdb_sizer.Add(svdb_result_label, 0, wx.ALL | wx.LEFT, 5)

        self.svdb_result_text = wx.TextCtrl(svdb_panel, style=wx.TE_MULTILINE | wx.TE_READONLY, size=(-1, 400))
        svdb_sizer.Add(self.svdb_result_text, 1, wx.EXPAND | wx.ALL, 5)

        svdb_panel.SetSizer(svdb_sizer)

        # ── 多音阶 OTO 面板 ──
        multi_oto_panel = wx.Panel(mark_notebook)
        multi_oto_sizer = wx.BoxSizer(wx.VERTICAL)

        multi_oto_title = wx.StaticText(multi_oto_panel, label=_('mark.multi_oto.title'))
        register(multi_oto_title, 'mark.multi_oto.title')
        multi_oto_sizer.Add(multi_oto_title, 0, wx.ALL | wx.CENTER, 5)

        # ── 根文件夹 ──
        multi_oto_root_box = wx.StaticBox(multi_oto_panel, label=_('mark.multi_oto.root_folder'))
        multi_oto_root_sizer = wx.StaticBoxSizer(multi_oto_root_box, wx.HORIZONTAL)
        register(multi_oto_root_box, 'mark.multi_oto.root_folder', 'label')
        self.multi_oto_root_text = wx.TextCtrl(multi_oto_panel, size=(400, -1))
        self.multi_oto_root_text.SetToolTip(_('mark.multi_oto.root_folder.tooltip'))
        register(self.multi_oto_root_text, 'mark.multi_oto.root_folder.tooltip', 'tooltip')
        multi_oto_root_browse = wx.Button(multi_oto_panel, label=_('mark.multi_oto.browse_folder'))
        register(multi_oto_root_browse, 'mark.multi_oto.browse_folder', 'button')
        multi_oto_root_browse.Bind(wx.EVT_BUTTON, lambda event: self.on_browse_folder(event, self.multi_oto_root_text))
        multi_oto_root_sizer.Add(self.multi_oto_root_text, 1, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        multi_oto_root_sizer.Add(multi_oto_root_browse, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        multi_oto_sizer.Add(multi_oto_root_sizer, 0, wx.EXPAND | wx.ALL, 5)

        # ── 共享 OTO 参数面板 ──
        self.multi_oto_params = OtoParamPanel(multi_oto_panel, show_pitch=False)
        multi_oto_sizer.Add(self.multi_oto_params, 0, wx.EXPAND | wx.ALL, 5)

        # 生成按钮
        multi_oto_generate_btn = wx.Button(multi_oto_panel, label=_('mark.multi_oto.generate'))
        register(multi_oto_generate_btn, 'mark.multi_oto.generate', 'button')
        multi_oto_generate_btn.Bind(wx.EVT_BUTTON, self.on_generate_multi_oto)
        multi_oto_sizer.Add(multi_oto_generate_btn, 0, wx.ALL | wx.CENTER, 5)

        # 结果文本框
        multi_oto_result_label = wx.StaticText(multi_oto_panel, label=_('mark.multi_oto.result'))
        register(multi_oto_result_label, 'mark.multi_oto.result')
        multi_oto_sizer.Add(multi_oto_result_label, 0, wx.ALL | wx.LEFT, 0)

        self.multi_oto_result_text = wx.TextCtrl(multi_oto_panel, style=wx.TE_MULTILINE | wx.TE_READONLY, size=(-1, 400))
        multi_oto_sizer.Add(self.multi_oto_result_text, 1, wx.EXPAND | wx.ALL, 10)

        multi_oto_panel.SetSizer(multi_oto_sizer)

        mark_notebook.AddPage(oto_panel, _('mark.oto.title'))
        register(mark_notebook, 'mark.oto.title', 'notebook_tab', 0)
        mark_notebook.AddPage(svdb_panel, _('mark.svdb.title'))
        register(mark_notebook, 'mark.svdb.title', 'notebook_tab', 1)
        mark_notebook.AddPage(multi_oto_panel, _('mark.multi_oto.title'))
        register(mark_notebook, 'mark.multi_oto.title', 'notebook_tab', 2)

        mark_sizer.Add(mark_notebook, 1, wx.EXPAND | wx.ALL, 5)
        mark_panel.SetSizer(mark_sizer)

        notebook.AddPage(lab_panel, _('notebook.lab'))
        register(notebook, 'notebook.lab', 'notebook_tab', 0)
        notebook.AddPage(textgrid_panel, _('notebook.textgrid'))
        register(notebook, 'notebook.textgrid', 'notebook_tab', 1)
        notebook.AddPage(json_panel, _('notebook.json'))
        register(notebook, 'notebook.json', 'notebook_tab', 2)
        notebook.AddPage(mark_panel, _('notebook.mark'))
        register(notebook, 'notebook.mark', 'notebook_tab', 3)

        # 底部：语言选择栏
        lang_sizer = wx.BoxSizer(wx.HORIZONTAL)
        lang_label = wx.StaticText(panel, label=_('lang.label'))
        register(lang_label, 'lang.label')
        self.lang_choice = wx.Choice(panel, size=(120, -1))
        # 填充可用语言
        from i18n import LANGUAGES, current_lang
        lang_codes = []
        for code, name in LANGUAGES.items():
            self.lang_choice.Append(name, code)
            lang_codes.append(code)
        current = current_lang()
        if current in lang_codes:
            self.lang_choice.SetSelection(lang_codes.index(current))
        self.lang_choice.Bind(wx.EVT_CHOICE, self.on_language_switch)
        lang_sizer.Add(lang_label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        lang_sizer.Add(self.lang_choice, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        lang_sizer.AddStretchSpacer()
        from i18n import APP_VERSION
        lang_hint_text = wx.StaticText(panel, label=f"TextGrid2oto {APP_VERSION}")
        lang_sizer.Add(lang_hint_text, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)

        main_sizer = wx.BoxSizer(wx.VERTICAL)
        main_sizer.Add(notebook, 1, wx.EXPAND)
        main_sizer.Add(lang_sizer, 0, wx.EXPAND | wx.ALL, 5)
        panel.SetSizer(main_sizer)
        
        # 在所有控件创建完成后加载模型
        # 推理强制停止相关状态
        self._infer_stop = threading.Event()
        self._infer_processes = []
        self._infer_stopped = False
        self.load_models()
        self.load_svdb_dicts()
        # load_presamp 已由 OtoParamPanel 内部处理

    def load_models(self):
        hubert_path = str(ROOT / 'HubertFA_model')
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
            model_path = str(ROOT / 'HubertFA_model' / selected_folder)
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
            model_path = str(ROOT / 'HubertFA_model' / selected_folder)
            if os.path.exists(model_path):
                dict_files = [f for f in os.listdir(model_path) if f.endswith('.txt')]
                self.json_dict_choice.Clear()
                self.json_dict_choice.AppendItems(dict_files)
                if dict_files:
                    self.json_dict_choice.SetSelection(0)

    def on_json_dict_selected(self, event):
        pass
    
    def load_svdb_dicts(self):
        dict_dir = str(ROOT / 'tg2svdb' / '字典')
        if os.path.exists(dict_dir):
            dict_files = [f for f in os.listdir(dict_dir) if f.endswith('.txt')]
            self.svdb_dict_choice.Clear()
            self.svdb_dict_choice.AppendItems(dict_files)
            if dict_files:
                self.svdb_dict_choice.SetSelection(0)
    
    def on_generate_svdb(self, event):
        svdb_path = self.svdb_path_text.GetValue().strip()
        dict_file = self.svdb_dict_choice.GetStringSelection()
        tail_ratio_str = self.svdb_tail_ratio_text.GetValue().strip()
        
        if not svdb_path:
            wx.MessageBox(_('msg.err.select_folder'), _('msg.error'), wx.OK | wx.ICON_ERROR)
            return
        
        if not os.path.exists(svdb_path):
            wx.MessageBox(_('msg.err.folder_not_exist'), _('msg.error'), wx.OK | wx.ICON_ERROR)
            return
        
        if not dict_file:
            wx.MessageBox(_('msg.err.select_dict'), _('msg.error'), wx.OK | wx.ICON_ERROR)
            return
        
        try:
            num = float(tail_ratio_str)
            if num <= 0 or num >= 100:
                wx.MessageBox(_('msg.err.tail_ratio_range'), _('msg.error'), wx.OK | wx.ICON_ERROR)
                return
        except ValueError:
            wx.MessageBox(_('msg.err.tail_ratio_invalid'), _('msg.error'), wx.OK | wx.ICON_ERROR)
            return
        
        def generate_svdb_thread():
            try:
                wx.CallAfter(self.svdb_result_text.Clear)
                wx.CallAfter(self.svdb_result_text.AppendText, _('log.start_generate_svdb'))
                
                dict_path = str(ROOT / 'tg2svdb' / '字典' / dict_file)
                
                def process_folder(folder_path, folder_name=""):
                    json_path = os.path.join(folder_path, 'json', 'word_phone.json')
                    wav_files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
                    
                    if wav_files and os.path.exists(json_path):
                        display_name = folder_name if folder_name else folder_path
                        wx.CallAfter(self.svdb_result_text.AppendText, _('log.processing').format(folder=display_name))
                        tg2sv_change.run(dict_path, json_path, folder_path,num)
                        wx.CallAfter(self.svdb_result_text.AppendText, _('log.completed').format(folder=display_name))
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
                
                wx.CallAfter(self.svdb_result_text.AppendText, _('log.svdb_complete').format(count=processed_count))
                wx.CallAfter(wx.MessageBox, _('msg.ok.svgdb_complete').format(count=processed_count), _('msg.success'), wx.OK | wx.ICON_INFORMATION)
            except Exception:
                tb = traceback.format_exc()
                wx.CallAfter(self.svdb_result_text.AppendText, _('log.failed').format(error=tb))
                wx.CallAfter(wx.MessageBox, _('log.failed').format(error=tb), _('msg.error'), wx.OK | wx.ICON_ERROR)
        
        thread = threading.Thread(target=generate_svdb_thread)
        thread.start()

    
    def on_language_switch(self, event):
        """切换语言（实时生效）"""
        lang_code = self.lang_choice.GetClientData(self.lang_choice.GetSelection())
        from i18n import switch_lang, update_all
        switch_lang(lang_code)
        update_all()

    # ── 模式/预设切换已由 OtoParamPanel 内部处理 ──
    
    def on_generate_oto(self, event):
        wav_path = self.oto_path_text.GetValue().strip()

        if not wav_path:
            wx.MessageBox(_('msg.err.select_folder'), _('msg.error'), wx.OK | wx.ICON_ERROR)
            return
        if not os.path.exists(wav_path):
            wx.MessageBox(_('msg.err.folder_not_exist'), _('msg.error'), wx.OK | wx.ICON_ERROR)
            return

        # 从共享参数面板获取所有设置
        try:
            p = self.oto_params.get_params()
        except ValueError:
            wx.MessageBox(_('msg.err.params_format'), _('msg.error'), wx.OK | wx.ICON_ERROR)
            return

        presamp_path = p["presamp_path"]
        if not presamp_path:
            wx.MessageBox(_('msg.err.select_presamp'), _('msg.error'), wx.OK | wx.ICON_ERROR)
            return
        if not os.path.exists(presamp_path):
            wx.MessageBox(_('msg.err.presamp_not_exist'), _('msg.error'), wx.OK | wx.ICON_ERROR)
            return

        vcv_mode = str(p["mode_index"])
        cv_sum, vc_sum, vv_sum = p["cv_sum"], p["vc_sum"], p["vv_sum"]
        cv_offset, vc_offset = p["cv_offset"], p["vc_offset"]
        pitch, ignore = p["pitch"], p["ignore"]
        cv_repeat, vc_repeat = p["cv_repeat"], p["vc_repeat"]
        oto_preset = p["template_path"]
        oto_encoding = p["encoding"]
        cover = p["cover"]

        word_phone_path = os.path.join(wav_path, 'json', 'word_phone.json')
        if not os.path.exists(word_phone_path):
            wx.MessageBox(_('msg.err.json_not_exist'), _('msg.error'), wx.OK | wx.ICON_ERROR)
            return

        if oto_preset and not os.path.exists(oto_preset):
            wx.MessageBox(_('msg.err.template_not_exist'), _('msg.error'), wx.OK | wx.ICON_ERROR)
            return
        
        def generate_oto_thread():
            try:
                wx.CallAfter(self.oto_result_text.Clear)
                wx.CallAfter(self.oto_result_text.AppendText, _('log.start_generate_oto'))
                
                from oto import oto_rw
                from oto import oto_check
                
                with TextRedirector(self.oto_result_text):
                    wx.CallAfter(self.oto_result_text.AppendText, _('log.generate_mode').format(mode=['CVVC', 'VCV', 'CVV','ARPAsing', 'Test'][int(vcv_mode)]))
                    
                    if vcv_mode == '1':
                        wx.CallAfter(self.oto_result_text.AppendText, _('log.generate_mode').format(mode='VCV'))
                        json2VCV_oto.run(presamp_path, word_phone_path,
                                         wav_path, cv_sum, vc_sum, vv_sum, ignore)
                    elif vcv_mode == '3':
                        wx.CallAfter(self.oto_result_text.AppendText, _('log.generate_mode').format(mode='ARPAsing'))
                        json2arpasing_oto.run(presamp_path, word_phone_path,
                                         wav_path, cv_sum, vc_sum, vv_sum, ignore)
                    elif vcv_mode == '2':
                        wx.CallAfter(self.oto_result_text.AppendText, _('log.generate_mode').format(mode='CVV'))
                        json2CV_oto.run(presamp_path, word_phone_path,
                                        wav_path, cv_sum, vc_sum, vv_sum, ignore)
                    elif vcv_mode == '0':
                        wx.CallAfter(self.oto_result_text.AppendText, _('log.generate_mode').format(mode='CVVC'))
                        json2oto.run(presamp_path, word_phone_path,
                                     wav_path, cv_sum, vc_sum, vv_sum, ignore)
                    elif vcv_mode == '4':
                        wx.CallAfter(self.oto_result_text.AppendText, _('log.generate_mode').format(mode='Test'))
                        json2test.run(presamp_path, word_phone_path,
                                      wav_path, cv_sum, vc_sum, vv_sum, ignore)
                    
                    wx.CallAfter(self.oto_result_text.AppendText, _('log.read_cv_vc'))
                    cv = oto_rw.oto_read(os.path.join(wav_path, 'cv_oto.ini'))
                    vc = oto_rw.oto_read(os.path.join(wav_path, 'vc_oto.ini'))
                    
                    if not os.path.exists(oto_preset) or oto_preset == "":
                        wx.CallAfter(self.oto_result_text.AppendText, _('log.remove_duplicates'))
                        cv = oto_rw.oto_repeat(cv, int(cv_repeat))
                        vc = oto_rw.oto_repeat(vc, int(vc_repeat))
                    
                    wx.CallAfter(self.oto_result_text.AppendText, _('log.apply_offset'))
                    if cv_offset != [0.0, 0.0, 0.0, 0.0, 0.0]:
                        cv = oto_rw.oto_offset(cv, cv_offset)
                        wx.CallAfter(self.oto_result_text.AppendText, _('log.offset_cv'))
                    if vc_offset != [0.0, 0.0, 0.0, 0.0, 0.0]:
                        vc = oto_rw.oto_offset(vc, vc_offset)
                        wx.CallAfter(self.oto_result_text.AppendText, _('log.offset_vc'))
                    
                    wx.CallAfter(self.oto_result_text.AppendText, _('log.merge_auto_oto'))
                    oto_rw.oto_write(os.path.join(wav_path, 'auto_oto.ini'), cv + vc, '', cover, oto_encoding)
                    
                    oto_data = oto_rw.oto_read(os.path.join(wav_path, 'auto_oto.ini'))
                    if os.path.exists(oto_preset):
                        wx.CallAfter(self.oto_result_text.AppendText, _('log.apply_template').format(path=oto_preset))
                        oto_data = oto_rw.oto_apply_template(oto_data, oto_preset,pitch)
                    oto_rw.oto_write(os.path.join(wav_path, 'oto.ini'), oto_data, pitch, cover, oto_encoding)
                    
                    
                    
                    wx.CallAfter(self.oto_result_text.AppendText, _('log.check_missing'))
                    oto_check.run(os.path.join(wav_path, 'oto.ini'), presamp_path, pitch, vcv_mode)
                
                wx.CallAfter(self.oto_result_text.AppendText, _('log.oto_complete'))
                wx.CallAfter(wx.MessageBox, _('msg.ok.oto_complete'), _('msg.success'), wx.OK | wx.ICON_INFORMATION)
            except Exception:
                tb = traceback.format_exc()
                wx.CallAfter(self.oto_result_text.AppendText, _('log.failed').format(error=tb))
                wx.CallAfter(wx.MessageBox, _('log.failed').format(error=tb), _('msg.error'), wx.OK | wx.ICON_ERROR)
        
        thread = threading.Thread(target=generate_oto_thread)
        thread.start()

    def on_generate_multi_oto(self, event):
        """多音阶 OTO 生成：遍历根目录下所有子文件夹，以文件夹名作为音阶后缀"""
        root_path = self.multi_oto_root_text.GetValue().strip()

        if not root_path:
            wx.MessageBox(_('msg.err.select_folder'), _('msg.error'), wx.OK | wx.ICON_ERROR)
            return
        if not os.path.exists(root_path):
            wx.MessageBox(_('msg.err.folder_not_exist'), _('msg.error'), wx.OK | wx.ICON_ERROR)
            return

        # 从共享参数面板获取所有设置
        try:
            p = self.multi_oto_params.get_params()
        except ValueError:
            wx.MessageBox(_('msg.err.params_format'), _('msg.error'), wx.OK | wx.ICON_ERROR)
            return

        presamp_path = p["presamp_path"]
        if not presamp_path:
            wx.MessageBox(_('msg.err.select_presamp'), _('msg.error'), wx.OK | wx.ICON_ERROR)
            return
        if not os.path.exists(presamp_path):
            wx.MessageBox(_('msg.err.presamp_not_exist'), _('msg.error'), wx.OK | wx.ICON_ERROR)
            return

        vcv_mode = str(p["mode_index"])
        cv_sum, vc_sum, vv_sum = p["cv_sum"], p["vc_sum"], p["vv_sum"]
        cv_offset, vc_offset = p["cv_offset"], p["vc_offset"]
        ignore = p["ignore"]
        cv_repeat, vc_repeat = p["cv_repeat"], p["vc_repeat"]
        oto_preset = p["template_path"]
        oto_encoding = p["encoding"]
        cover = p["cover"]

        # 获取所有子文件夹
        subfolders = sorted([d for d in os.listdir(root_path)
                             if os.path.isdir(os.path.join(root_path, d))])
        if not subfolders:
            wx.MessageBox(_('msg.err.multi_oto.no_subfolders'), _('msg.error'), wx.OK | wx.ICON_ERROR)
            return

        def generate_multi_oto_thread():
            try:
                wx.CallAfter(self.multi_oto_result_text.Clear)
                wx.CallAfter(self.multi_oto_result_text.AppendText,
                             _('log.multi_oto.start').format(count=len(subfolders)))

                from oto import oto_rw
                from oto import oto_check

                success_count = 0
                for folder_name in subfolders:
                    sub_path = os.path.join(root_path, folder_name)
                    word_phone_path = os.path.join(sub_path, 'json', 'word_phone.json')

                    if not os.path.exists(word_phone_path):
                        wx.CallAfter(self.multi_oto_result_text.AppendText,
                                     _('log.multi_oto.skip').format(folder=folder_name))
                        continue

                    pitch = folder_name  # 音阶后缀 = 文件夹名
                    wx.CallAfter(self.multi_oto_result_text.AppendText,
                                 _('log.multi_oto.processing').format(folder=folder_name, pitch=pitch))

                    with TextRedirector(self.multi_oto_result_text):
                        wx.CallAfter(self.multi_oto_result_text.AppendText,
                                     _('log.generate_mode').format(
                                         mode=['CVVC', 'VCV', 'CVV', 'ARPAsing', 'Test'][int(vcv_mode)]))

                        if vcv_mode == '1':
                            json2VCV_oto.run(presamp_path, word_phone_path,
                                             sub_path, cv_sum, vc_sum, vv_sum, ignore)
                        elif vcv_mode == '3':
                            json2arpasing_oto.run(presamp_path, word_phone_path,
                                                  sub_path, cv_sum, vc_sum, vv_sum, ignore)
                        elif vcv_mode == '2':
                            json2CV_oto.run(presamp_path, word_phone_path,
                                            sub_path, cv_sum, vc_sum, vv_sum, ignore)
                        elif vcv_mode == '0':
                            json2oto.run(presamp_path, word_phone_path,
                                         sub_path, cv_sum, vc_sum, vv_sum, ignore)
                        elif vcv_mode == '4':
                            json2test.run(presamp_path, word_phone_path,
                                          sub_path, cv_sum, vc_sum, vv_sum, ignore)

                        cv = oto_rw.oto_read(os.path.join(sub_path, 'cv_oto.ini'))
                        vc = oto_rw.oto_read(os.path.join(sub_path, 'vc_oto.ini'))

                        if not os.path.exists(oto_preset) or oto_preset == "":
                            cv = oto_rw.oto_repeat(cv, int(cv_repeat))
                            vc = oto_rw.oto_repeat(vc, int(vc_repeat))

                        if cv_offset != [0.0, 0.0, 0.0, 0.0, 0.0]:
                            cv = oto_rw.oto_offset(cv, cv_offset)
                        if vc_offset != [0.0, 0.0, 0.0, 0.0, 0.0]:
                            vc = oto_rw.oto_offset(vc, vc_offset)

                        oto_rw.oto_write(os.path.join(sub_path, 'auto_oto.ini'), cv + vc, '', cover, oto_encoding)

                        oto_data = oto_rw.oto_read(os.path.join(sub_path, 'auto_oto.ini'))
                        if os.path.exists(oto_preset):
                            oto_data = oto_rw.oto_apply_template(oto_data, oto_preset)
                        oto_rw.oto_write(os.path.join(sub_path, 'oto.ini'), oto_data, pitch, cover, oto_encoding)

                        oto_check.run(os.path.join(sub_path, 'oto.ini'), presamp_path, pitch, vcv_mode)

                    success_count += 1
                    wx.CallAfter(self.multi_oto_result_text.AppendText,
                                 _('log.multi_oto.done').format(folder=folder_name))

                wx.CallAfter(self.multi_oto_result_text.AppendText,
                             _('log.multi_oto.complete').format(success=success_count, total=len(subfolders)))
                wx.CallAfter(wx.MessageBox,
                             _('msg.ok.multi_oto_complete').format(success=success_count, total=len(subfolders)),
                             _('msg.success'), wx.OK | wx.ICON_INFORMATION)
            except Exception:
                tb = traceback.format_exc()
                wx.CallAfter(self.multi_oto_result_text.AppendText, _('log.failed').format(error=tb))
                wx.CallAfter(wx.MessageBox, _('log.failed').format(error=tb),
                             _('msg.error'), wx.OK | wx.ICON_ERROR)

        thread = threading.Thread(target=generate_multi_oto_thread)
        thread.start()

    def on_browse_folder(self, event, text_ctrl):
        dialog = wx.DirDialog(None, _('msg.select_folder'), style=wx.DD_DEFAULT_STYLE)
        if dialog.ShowModal() == wx.ID_CANCEL:
            return
        path = dialog.GetPath()
        text_ctrl.SetValue(path)
        dialog.Destroy()
    
    def on_browse_file(self, event, text_ctrl):
        dialog = wx.FileDialog(None, _('msg.select_file'), style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)
        if dialog.ShowModal() == wx.ID_CANCEL:
            return
        path = dialog.GetPath()
        text_ctrl.SetValue(path)
        dialog.Destroy()
    
    def on_browse_json_recording(self, event):
        dialog = wx.FileDialog(None, _('msg.select_recording'), style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)
        if dialog.ShowModal() == wx.ID_CANCEL:
            return
        path = dialog.GetPath()
        self.json_recording_text.SetValue(path)
    
    def on_generate_lab(self, event):
        path = self.path_text.GetValue().strip()
        separator_str = self.separator_text.GetValue().strip()
        
        if not path:
            wx.MessageBox(_('msg.err.select_folder'), _('msg.error'), wx.OK | wx.ICON_ERROR)
            return
        
        if not os.path.exists(path):
            wx.MessageBox(_('msg.err.folder_not_exist'), _('msg.error'), wx.OK | wx.ICON_ERROR)
            return
        
        cuts = [s.strip() for s in separator_str.split(',') if s.strip()]
        
        def generate_lab_thread():
            try:
                wx.CallAfter(self.lab_result_text.Clear)
                wx.CallAfter(self.lab_result_text.AppendText, _('log.start_generate_lab'))

                with TextRedirector(self.lab_result_text):
                    wavname2lab.run(path, cuts)

                wx.CallAfter(self.lab_result_text.AppendText, _('log.lab_complete'))
                wx.CallAfter(wx.MessageBox, _('msg.ok.generate_complete'), _('msg.success'), wx.OK | wx.ICON_INFORMATION)
            except Exception:
                tb = traceback.format_exc()
                wx.CallAfter(self.lab_result_text.AppendText, _('log.failed').format(error=tb))
                wx.CallAfter(wx.MessageBox, _('log.failed').format(error=tb), _('msg.error'), wx.OK | wx.ICON_ERROR)

        thread = threading.Thread(target=generate_lab_thread)
        thread.start()

    def on_generate_lab_from_index(self, event):
        wav_path = self.lab_wav_path_text.GetValue().strip()
        index_path = self.lab_index_path_text.GetValue().strip()
        separator_str = self.lab_separator_text.GetValue().strip()
        
        if not wav_path:
            wx.MessageBox(_('msg.err.select_wav_folder'), _('msg.error'), wx.OK | wx.ICON_ERROR)
            return
        
        if not os.path.exists(wav_path):
            wx.MessageBox(_('msg.err.wav_folder_not_exist'), _('msg.error'), wx.OK | wx.ICON_ERROR)
            return
        
        if not index_path:
            wx.MessageBox(_('msg.err.select_index'), _('msg.error'), wx.OK | wx.ICON_ERROR)
            return
        
        if not os.path.exists(index_path):
            wx.MessageBox(_('msg.err.index_not_exist'), _('msg.error'), wx.OK | wx.ICON_ERROR)
            return
        
        cuts = [s.strip() for s in separator_str.split(',') if s.strip()]
        
        def generate_lab_from_index_thread():
            try:
                wx.CallAfter(self.lab_index_result_text.Clear)
                wx.CallAfter(self.lab_index_result_text.AppendText, _('log.start_generate_lab_index'))

                with TextRedirector(self.lab_index_result_text):
                    index2lab.run(wav_path, index_path, cuts)

                wx.CallAfter(self.lab_index_result_text.AppendText, _('log.lab_index_complete'))
                wx.CallAfter(wx.MessageBox, _('msg.ok.generate_complete'), _('msg.success'), wx.OK | wx.ICON_INFORMATION)
            except Exception:
                tb = traceback.format_exc()
                wx.CallAfter(self.lab_index_result_text.AppendText, _('log.failed').format(error=tb))
                wx.CallAfter(wx.MessageBox, _('log.failed').format(error=tb), _('msg.error'), wx.OK | wx.ICON_ERROR)

        thread = threading.Thread(target=generate_lab_from_index_thread)
        thread.start()

    def on_infer(self, event):
        wav_folder = self.textgrid_folder_text.GetValue().strip()
        model_folder = self.model_folder_choice.GetStringSelection()
        model_file = self.model_file_choice.GetStringSelection()
        dict_file = self.dict_choice.GetStringSelection()

        if not wav_folder:
            wx.MessageBox(_('msg.err.select_folder'), _('msg.error'), wx.OK | wx.ICON_ERROR)
            return

        if not os.path.exists(wav_folder):
            wx.MessageBox(_('msg.err.folder_not_exist'), _('msg.error'), wx.OK | wx.ICON_ERROR)
            return

        if not model_folder:
            wx.MessageBox(_('msg.err.select_model'), _('msg.error'), wx.OK | wx.ICON_ERROR)
            return

        if not model_file:
            wx.MessageBox(_('msg.err.select_model_file'), _('msg.error'), wx.OK | wx.ICON_ERROR)
            return

        if not dict_file:
            wx.MessageBox(_('msg.err.select_dict'), _('msg.error'), wx.OK | wx.ICON_ERROR)
            return

        def infer_thread():
            try:
                wx.CallAfter(self.infer_result_text.Clear)
                # 重置停止状态并启用停止按钮
                self._infer_stop.clear()
                self._infer_processes = []
                self._infer_stopped = False
                wx.CallAfter(self.stop_infer_btn.Enable)

                # 获取用户选择的设备
                device_selection = self.device_choice.GetSelection()
                device_data = self.device_choice.GetClientData(device_selection)
                device = device_data if device_data else 'cpu'
                
                model_path = ROOT / 'HubertFA_model' / model_folder / model_file
                dict_path = ROOT / 'HubertFA_model' / model_folder / dict_file

                language = dict_file.split('.')[0].split('-')[0]
                language = language[0] if len(language) == 1 else language

                wx.CallAfter(self.infer_result_text.AppendText, _('log.model_info').format(model=model_file))
                wx.CallAfter(self.infer_result_text.AppendText, _('log.dict_info').format(dict=dict_file))
                wx.CallAfter(self.infer_result_text.AppendText, _('log.lang_info').format(lang=language))
                wx.CallAfter(self.infer_result_text.AppendText, _('log.device_info').format(device=device.upper()))

                # 获取用户选择的推理参数
                pad_times_selection = self.pad_times_choice.GetSelection()
                pad_times = self.pad_times_choice.GetClientData(pad_times_selection)
                pad_length_selection = self.pad_length_choice.GetSelection()
                pad_length = self.pad_length_choice.GetClientData(pad_length_selection)
                merge_phonemes = self.merge_phonemes_checkbox.GetValue()

                wx.CallAfter(self.infer_result_text.AppendText, _('log.start_infer'))
                wx.CallAfter(self.infer_result_text.AppendText, _('log.infer_times').format(times=pad_times))
                wx.CallAfter(self.infer_result_text.AppendText, _('log.pad_length').format(len=pad_length))
                wx.CallAfter(self.infer_result_text.AppendText, _('log.merge_phonemes').format(merge=merge_phonemes))

                # ── DML / WebGPU 模式：多进程并行推理 ──
                if device in ('dml', 'webgpu'):
                    import multiprocessing
                    from pathlib import Path as _Path

                    # 获取用户选择的并行工作线程数
                    worker_sel = self.worker_choice.GetSelection()
                    num_workers = self.worker_choice.GetClientData(worker_sel)

                    # 收集所有 wav 文件
                    all_wavs = sorted([str(p) for p in _Path(wav_folder).rglob("*.wav") if p.with_suffix(".lab").exists()])
                    total_files = len(all_wavs)

                    if num_workers <= 1 or total_files < num_workers * 2:
                        # 线程数<=1或文件太少，退化为单进程推理
                        wx.CallAfter(self.infer_result_text.AppendText,
                            _('log.parallel_dml_skip').format(count=total_files, workers=num_workers, device=device.upper()))
                        _run_single_inference(model_path, wav_folder, language, dict_path,
                                            device, pad_times, pad_length, merge_phonemes)
                    else:
                        # 将文件列表平分给各工作进程
                        chunk_size = (total_files + num_workers - 1) // num_workers
                        wav_chunks = [all_wavs[i:i + chunk_size] for i in range(0, total_files, chunk_size)]

                        wx.CallAfter(self.infer_result_text.AppendText,
                            _('log.parallel_dml_start').format(total=total_files, workers=num_workers,
                                                               sizes=", ".join(str(len(c)) for c in wav_chunks), device=device.upper()))

                        _parallel_start = time.time()

                        # 创建消息队列，让子进程回传进度信息
                        msg_queue = multiprocessing.Queue()

                        processes = []
                        for i, chunk in enumerate(wav_chunks):
                            p = multiprocessing.Process(
                                target=onnx_infer.dml_worker,
                                args=(i + 1, str(model_path), wav_folder, language, str(dict_path),
                                      device, pad_times, pad_length, merge_phonemes, "AP", chunk, msg_queue)
                            )
                            processes.append(p)
                            p.start()
                        self._infer_processes = processes  # 供停止按钮终止进程

                        # 监控队列：将子进程消息实时显示到 GUI
                        def monitor_queue():
                            alive = [True] * len(processes)
                            while any(alive):
                                # 从队列读取所有可用消息
                                while not msg_queue.empty():
                                    try:
                                        msg = msg_queue.get_nowait()
                                        wx.CallAfter(self.infer_result_text.AppendText, msg + "\n")
                                        wx.CallAfter(self.infer_result_text.ShowPosition,
                                                     self.infer_result_text.GetLastPosition())
                                    except Exception:
                                        break
                                # 检查进程状态
                                for i, p in enumerate(processes):
                                    if alive[i] and not p.is_alive():
                                        alive[i] = False
                                time.sleep(0.2)
                            # 最后再清一次队列
                            while not msg_queue.empty():
                                try:
                                    msg = msg_queue.get_nowait()
                                    wx.CallAfter(self.infer_result_text.AppendText, msg + "\n")
                                except Exception:
                                    break

                        monitor_queue()
                        # 确保所有进程真正结束
                        for p in processes:
                            p.join()

                        if self._infer_stopped:
                            wx.CallAfter(self.infer_result_text.AppendText, _('log.infer_stopped'))
                        else:
                            _parallel_elapsed = time.time() - _parallel_start
                            total_msg = f"并行推理总耗时: {_parallel_elapsed:.1f}s (共 {total_files} 个文件，{num_workers} 进程，平均 {_parallel_elapsed / max(total_files, 1):.2f}s/个)"
                            wx.CallAfter(self.infer_result_text.AppendText, total_msg + "\n")
                            wx.CallAfter(self.infer_result_text.AppendText, _('log.infer_complete'))
                            wx.CallAfter(wx.MessageBox, _('msg.ok.infer_complete'), _('msg.success'),
                                         wx.OK | wx.ICON_INFORMATION)
                else:
                    # ── CPU 模式：单进程推理 ──
                    _run_single_inference(model_path, wav_folder, language, dict_path,
                                         device, pad_times, pad_length, merge_phonemes, self._infer_stop)

                wx.CallAfter(self.stop_infer_btn.Disable)
            except Exception:
                tb = traceback.format_exc()
                wx.CallAfter(self.infer_result_text.AppendText, _('log.infer_failed').format(error=tb))
                wx.CallAfter(self.stop_infer_btn.Disable)
                wx.CallAfter(wx.MessageBox, _('log.infer_failed').format(error=tb), _('msg.error'), wx.OK | wx.ICON_ERROR)

        def _run_single_inference(model_path, wav_folder, language, dict_path,
                                  device, pad_times, pad_length, merge_phonemes, stop_event=None):
            """单进程推理（CPU 模式或 DML 文件太少时使用）"""
            inference = onnx_infer.InferenceOnnx(model_path)
            wx.CallAfter(self.infer_result_text.AppendText, _('log.loading_config'))
            inference.load_config()
            wx.CallAfter(self.infer_result_text.AppendText, _('log.loading_model_weights'))
            inference.load_model(device=device)
            wx.CallAfter(self.infer_result_text.AppendText, _('log.init_decoder'))
            inference.init_decoder()

            def progress_callback(msg):
                wx.CallAfter(self.infer_result_text.AppendText, msg + "\n")
                wx.CallAfter(self.infer_result_text.ShowPosition, self.infer_result_text.GetLastPosition())

            inference.set_progress_callback(progress_callback)

            wx.CallAfter(self.infer_result_text.AppendText, _('log.loading_dataset'))
            inference.get_dataset(wav_folder, language=language, g2p="dictionary",
                                  dictionary_path=str(dict_path), in_format="lab")

            try:
                inference.infer(non_lexical_phonemes="AP", pad_times=pad_times,
                               pad_length=pad_length, merge_phonemes=merge_phonemes,
                               stop_event=stop_event)
            except onnx_infer.StopInference:
                wx.CallAfter(self.infer_result_text.AppendText, _('log.infer_stopped'))
                return

            wx.CallAfter(self.infer_result_text.AppendText, _('log.exporting'))
            inference.export(wav_folder)

            wx.CallAfter(self.infer_result_text.AppendText, _('log.infer_complete'))
            wx.CallAfter(wx.MessageBox, _('msg.ok.infer_complete'), _('msg.success'),
                         wx.OK | wx.ICON_INFORMATION)

        thread = threading.Thread(target=infer_thread)
        thread.start()

    def on_stop_infer(self, event):
        """强制停止当前 TextGrid 推理"""
        self._infer_stop.set()
        self._infer_stopped = True
        # 终止多进程推理子进程
        for p in self._infer_processes:
            if p.is_alive():
                try:
                    p.terminate()
                except Exception:
                    pass
        self.stop_infer_btn.Disable()
        self.infer_result_text.AppendText(_('log.infer_stop_requested'))

    def on_generate_json(self, event):
        wav_folder = self.json_path_text.GetValue().strip()
        dict_file = self.json_dict_choice.GetStringSelection()
        ignore = self.json_ignore_text.GetValue().strip()
        recording_list_path = self.json_recording_text.GetValue().strip()

        if not wav_folder:
            wx.MessageBox(_('msg.err.select_wav_folder'), _('msg.error'), wx.OK | wx.ICON_ERROR)
            return

        if not os.path.exists(wav_folder):
            wx.MessageBox(_('msg.err.wav_folder_not_exist'), _('msg.error'), wx.OK | wx.ICON_ERROR)
            return

        if not dict_file:
            wx.MessageBox(_('msg.err.select_dict'), _('msg.error'), wx.OK | wx.ICON_ERROR)
            return

        if recording_list_path and not os.path.exists(recording_list_path):
            wx.MessageBox(_('msg.err.recording_not_exist'), _('msg.error'), wx.OK | wx.ICON_ERROR)
            return
        
        def generate_json_thread():
            try:
                wx.CallAfter(self.json_result_text.Clear)
                wx.CallAfter(self.json_result_text.AppendText, _('log.start_generate_json'))

                # 获取字典路径
                selected_folder = self.json_folder_choice.GetStringSelection()
                dict_full_path = str(ROOT / 'HubertFA_model' / selected_folder / dict_file)

                if not os.path.exists(dict_full_path):
                    wx.CallAfter(wx.MessageBox, _('msg.err.dict_not_exist'), _('msg.error'), wx.OK | wx.ICON_ERROR)
                    return

                def process_folder(folder_path, folder_name=""):
                    textgrid_files = [f for f in os.listdir(folder_path) if f.endswith('.TextGrid')]

                    if textgrid_files:
                        display_name = folder_name if folder_name else folder_path
                        wx.CallAfter(self.json_result_text.AppendText, _('log.processing').format(folder=display_name))

                        # 生成ds_phone.json
                        rec_preset = recording_list_path if recording_list_path else None
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
                        
                        wx.CallAfter(self.json_result_text.AppendText, _('log.completed').format(folder=display_name))
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
                
                wx.CallAfter(self.json_result_text.AppendText, _('log.json_complete').format(count=processed_count))
                wx.CallAfter(wx.MessageBox, _('msg.ok.generate_json_complete').format(count=processed_count), _('msg.success'), wx.OK | wx.ICON_INFORMATION)

            except Exception:
                tb = traceback.format_exc()
                wx.CallAfter(self.json_result_text.AppendText, _('log.failed').format(error=tb))
                wx.CallAfter(wx.MessageBox, _('log.failed').format(error=tb), _('msg.error'), wx.OK | wx.ICON_ERROR)

        thread = threading.Thread(target=generate_json_thread)
        thread.start()

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    try:
        app = wx.App()
        frame = MainFrame()
        frame.Show()
        app.MainLoop()
    except Exception:
        error_msg = f"程序启动失败！\n{traceback.format_exc()}"
        print(error_msg, file=sys.stderr)  # 输出到cmd窗口
        input("按回车键退出...")  # 防止窗口立即关闭