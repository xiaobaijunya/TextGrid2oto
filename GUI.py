import wx
import os
import sys
import json
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
from inference import onnx_infer
from inference import tifa_infer
from inference import game_infer
from textgrid2json import del_SP, TextGrid2ds_json, ds_json2filter, ds_json2word
from textgrid2json import detect_short_phonemes
from textgrid2json import json2tg
from build_diffdataset import words_from_phones
from build_diffdataset import add_ph_num as phnum_builder
from build_diffdataset import csv2tg
from build_diffdataset.build_dataset import build_dataset as build_diff_dataset
from json2oto import json2CV_oto, json2oto, json2VCV_oto, json2test,json2arpasing_oto
sys.path.append(str(ROOT / 'tg2svdb'))
from tg2svdb import tg2sv_change
from i18n import _, register
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

        # 自动生成SP开关
        auto_sp_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.auto_sp_checkbox = wx.CheckBox(lab_wav_panel, label=_('lab.wavname.auto_sp'))
        self.auto_sp_checkbox.SetToolTip(_('lab.wavname.auto_sp.tooltip'))
        register(self.auto_sp_checkbox, 'lab.wavname.auto_sp.tooltip', 'tooltip')
        register(self.auto_sp_checkbox, 'lab.wavname.auto_sp', 'checkbox')
        self.auto_sp_checkbox.SetValue(True)
        auto_sp_sizer.Add(self.auto_sp_checkbox, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        wav_box_sizer.Add(auto_sp_sizer, 0, wx.ALL, 3)

        # 不生成R开关
        no_r_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.no_r_checkbox = wx.CheckBox(lab_wav_panel, label=_('lab.no_r'))
        self.no_r_checkbox.SetToolTip(_('lab.no_r.tooltip'))
        register(self.no_r_checkbox, 'lab.no_r.tooltip', 'tooltip')
        register(self.no_r_checkbox, 'lab.no_r', 'checkbox')
        self.no_r_checkbox.SetValue(False)
        no_r_sizer.Add(self.no_r_checkbox, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        wav_box_sizer.Add(no_r_sizer, 0, wx.ALL, 3)

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

        # 不生成R开关
        lab_index_no_r_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.lab_index_no_r_checkbox = wx.CheckBox(lab_index_panel, label=_('lab.no_r'))
        self.lab_index_no_r_checkbox.SetToolTip(_('lab.no_r.tooltip'))
        register(self.lab_index_no_r_checkbox, 'lab.no_r.tooltip', 'tooltip')
        register(self.lab_index_no_r_checkbox, 'lab.no_r', 'checkbox')
        self.lab_index_no_r_checkbox.SetValue(False)
        lab_index_no_r_sizer.Add(self.lab_index_no_r_checkbox, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        idx_box_sizer.Add(lab_index_no_r_sizer, 0, wx.ALL, 3)

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
        
        textgrid_outer_panel = wx.Panel(notebook)
        textgrid_notebook = wx.Notebook(textgrid_outer_panel)

        # 第一个标签页：TIFA 推理
        tifa_panel = wx.Panel(textgrid_notebook)
        self._build_tifa_panel(tifa_panel)

        # 第二个标签页：HubertFA 推理（原逻辑保持不变）
        textgrid_panel = wx.Panel(textgrid_notebook)
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

        textgrid_notebook.AddPage(tifa_panel, _('textgrid.tab.tifa'))
        textgrid_notebook.AddPage(textgrid_panel, _('textgrid.tab.hfa'))
        register(textgrid_notebook, 'textgrid.tab.tifa', 'notebook_tab', 0)
        register(textgrid_notebook, 'textgrid.tab.hfa', 'notebook_tab', 1)
        textgrid_outer_sizer = wx.BoxSizer(wx.VERTICAL)
        textgrid_outer_sizer.Add(textgrid_notebook, 1, wx.EXPAND | wx.ALL, 5)
        textgrid_outer_panel.SetSizer(textgrid_outer_sizer)

        # 音素检测面板
        detect_panel = wx.Panel(notebook)
        detect_sizer = wx.BoxSizer(wx.VERTICAL)

        detect_title = wx.StaticText(detect_panel, label=_('detect.title'))
        register(detect_title, 'detect.title')
        detect_sizer.Add(detect_title, 0, wx.ALL | wx.CENTER, 5)

        # ── 检测设置 ──
        detect_box = wx.StaticBox(detect_panel, label=_('detect.settings'))
        detect_box_sizer = wx.StaticBoxSizer(detect_box, wx.VERTICAL)
        register(detect_box, 'detect.settings', 'label')

        detect_path_sizer = wx.BoxSizer(wx.HORIZONTAL)
        detect_path_label = wx.StaticText(detect_panel, label=_('detect.folder'))
        register(detect_path_label, 'detect.folder')
        self.detect_path_text = wx.TextCtrl(detect_panel, size=(400, -1))
        detect_browse_btn = wx.Button(detect_panel, label=_('detect.browse_folder'))
        register(detect_browse_btn, 'detect.browse_folder', 'button')
        detect_browse_btn.Bind(wx.EVT_BUTTON, lambda event: self.on_browse_folder(event, self.detect_path_text))
        detect_path_sizer.Add(detect_path_label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        detect_path_sizer.Add(self.detect_path_text, 1, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        detect_path_sizer.Add(detect_browse_btn, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        detect_box_sizer.Add(detect_path_sizer, 0, wx.EXPAND | wx.ALL, 3)

        detect_threshold_sizer = wx.BoxSizer(wx.HORIZONTAL)
        detect_threshold_label = wx.StaticText(detect_panel, label=_('detect.threshold'))
        register(detect_threshold_label, 'detect.threshold')
        self.detect_threshold_text = wx.TextCtrl(detect_panel, value="30", size=(80, -1))
        self.detect_threshold_text.SetToolTip(_('detect.threshold.tooltip'))
        register(self.detect_threshold_text, 'detect.threshold.tooltip', 'tooltip')
        detect_threshold_unit = wx.StaticText(detect_panel, label=_('detect.threshold.unit'))
        register(detect_threshold_unit, 'detect.threshold.unit')
        detect_threshold_sizer.Add(detect_threshold_label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        detect_threshold_sizer.Add(self.detect_threshold_text, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        detect_threshold_sizer.Add(detect_threshold_unit, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        detect_box_sizer.Add(detect_threshold_sizer, 0, wx.ALL, 3)

        detect_btn = wx.Button(detect_panel, label=_('detect.run'))
        register(detect_btn, 'detect.run', 'button')
        detect_btn.Bind(wx.EVT_BUTTON, self.on_detect_short_phonemes)
        detect_box_sizer.Add(detect_btn, 0, wx.ALL | wx.CENTER, 5)

        detect_sizer.Add(detect_box_sizer, 0, wx.EXPAND | wx.ALL, 5)

        # 结果显示文本框
        detect_result_label = wx.StaticText(detect_panel, label=_('detect.result'))
        register(detect_result_label, 'detect.result')
        detect_sizer.Add(detect_result_label, 0, wx.ALL | wx.LEFT, 5)

        self.detect_result_text = wx.TextCtrl(detect_panel, style=wx.TE_MULTILINE | wx.TE_READONLY, size=(-1, 400))
        detect_sizer.Add(self.detect_result_text, 1, wx.EXPAND | wx.ALL, 5)

        detect_panel.SetSizer(detect_sizer)

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
        self.json_dict_choice = wx.Choice(json_panel, size=(170, -1))
        self.json_dict_choice.Bind(wx.EVT_CHOICE, self.on_json_dict_selected)
        self.json_dict_text = wx.TextCtrl(json_panel)
        self.json_dict_text.SetToolTip(_('json.dict.tooltip'))
        register(self.json_dict_text, 'json.dict.tooltip', 'tooltip')
        json_dict_browse_btn = wx.Button(json_panel, label=_('json.browse'), size=(60, -1))
        register(json_dict_browse_btn, 'json.browse', 'button')
        json_dict_browse_btn.Bind(wx.EVT_BUTTON, lambda event: self.on_browse_file(event, self.json_dict_text))
        json_dict_sizer.Add(json_dict_label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        json_dict_sizer.Add(self.json_dict_choice, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        json_dict_sizer.Add(self.json_dict_text, 1, wx.EXPAND | wx.ALL, 3)
        json_dict_sizer.Add(json_dict_browse_btn, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
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

        svdb_btn_row = wx.BoxSizer(wx.HORIZONTAL)
        svdb_generate_btn = wx.Button(svdb_panel, label=_('mark.svdb.generate'))
        register(svdb_generate_btn, 'mark.svdb.generate', 'button')
        svdb_generate_btn.Bind(wx.EVT_BUTTON, self.on_generate_svdb)
        svdb_tg_btn = wx.Button(svdb_panel, label=_('mark.svdb.generate_tg'))
        register(svdb_tg_btn, 'mark.svdb.generate_tg', 'button')
        svdb_tg_btn.SetToolTip(_('mark.svdb.generate_tg.tooltip'))
        register(svdb_tg_btn, 'mark.svdb.generate_tg.tooltip', 'tooltip')
        svdb_tg_btn.Bind(wx.EVT_BUTTON, self.on_generate_textgrid_from_json)
        svdb_btn_row.Add(svdb_generate_btn, 0, wx.ALL, 5)
        svdb_btn_row.Add(svdb_tg_btn, 0, wx.ALL, 5)
        svdb_box_sizer.Add(svdb_btn_row, 0, wx.CENTER)

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

        # ── transcriptions.csv 制作 ──
        trans_panel = wx.Panel(mark_notebook)
        self._build_transcriptions_panel(trans_panel)
        mark_notebook.AddPage(trans_panel, _('mark.transcriptions.title'))
        register(mark_notebook, 'mark.transcriptions.title', 'notebook_tab', 3)

        mark_sizer.Add(mark_notebook, 1, wx.EXPAND | wx.ALL, 5)
        mark_panel.SetSizer(mark_sizer)

        notebook.AddPage(lab_panel, _('notebook.lab'))
        register(notebook, 'notebook.lab', 'notebook_tab', 0)
        notebook.AddPage(textgrid_outer_panel, _('notebook.textgrid'))
        register(notebook, 'notebook.textgrid', 'notebook_tab', 1)
        notebook.AddPage(detect_panel, _('notebook.detect'))
        register(notebook, 'notebook.detect', 'notebook_tab', 2)
        notebook.AddPage(json_panel, _('notebook.json'))
        register(notebook, 'notebook.json', 'notebook_tab', 3)
        notebook.AddPage(mark_panel, _('notebook.mark'))
        register(notebook, 'notebook.mark', 'notebook_tab', 4)

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
        self._tifa_stop = threading.Event()
        self._words_stop = threading.Event()
        self._csv_stop = threading.Event()
        self._game_stop = threading.Event()
        self.load_models()
        self.load_tifa_models()
        self.load_json_model_folders()
        self.load_svdb_dicts()
        self.load_phnum_presets()
        self.load_game_models()
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

    def load_json_model_folders(self):
        """JSON 生成面板：HubertFA_model 与 TIFA_model 的模型目录都可选。"""
        self.json_folder_choice.Clear()
        hfa = ROOT / 'HubertFA_model'
        if os.path.exists(hfa):
            for d in sorted([x for x in os.listdir(hfa) if os.path.isdir(os.path.join(hfa, x))]):
                self.json_folder_choice.Append(d, ('HubertFA_model', d))
        tifa = ROOT / 'TIFA_model'
        if os.path.exists(tifa):
            for label, data in self._tifa_model_dirs():
                self.json_folder_choice.Append(f"[TIFA] {label}", ('TIFA_model', data))
        if self.json_folder_choice.GetCount():
            self.json_folder_choice.SetSelection(0)
            self.on_json_folder_selected(None)

    def _json_model_dir(self):
        """返回 JSON 生成面板当前选中的模型目录。"""
        data = self.json_folder_choice.GetClientData(self.json_folder_choice.GetSelection())
        if data:
            base, folder = data
            return (ROOT / base / folder) if folder else (ROOT / base)
        name = self.json_folder_choice.GetStringSelection()
        return (ROOT / 'HubertFA_model' / name) if name else None

    def on_json_folder_selected(self, event):
        """选模型文件夹时，把其中的字典列进下拉框，并默认写入第一个字典的完整路径。"""
        model_dir = self._json_model_dir()
        self.json_dict_choice.Clear()
        if model_dir and os.path.exists(model_dir):
            dict_files = sorted([f for f in os.listdir(model_dir) if f.endswith('.txt')])
            if dict_files:
                self.json_dict_choice.AppendItems(dict_files)
                self.json_dict_choice.SetSelection(0)
                self.on_json_dict_selected(None)

    def on_json_dict_selected(self, event):
        """从下拉框选字典时，把完整路径写入输入框（也可直接在输入框里改）。"""
        name = self.json_dict_choice.GetStringSelection()
        model_dir = self._json_model_dir()
        if name and model_dir:
            self.json_dict_text.SetValue(str(model_dir / name))
    
    def load_svdb_dicts(self):
        dict_dir = str(ROOT / 'tg2svdb' / 'dicts')
        if os.path.exists(dict_dir):
            dict_files = [f for f in os.listdir(dict_dir) if f.endswith('.txt')]
            self.svdb_dict_choice.Clear()
            self.svdb_dict_choice.AppendItems(dict_files)
            if dict_files:
                self.svdb_dict_choice.SetSelection(0)

    def load_phnum_presets(self):
        """把 tg2svdb/cv_dicts 下带 vowels/consonants 的语言目录列进下拉框。"""
        self.phnum_preset_choice.Clear()
        self.phnum_preset_choice.Append(_('mark.phnum.preset.custom'), '')
        base = ROOT / 'tg2svdb' / 'cv_dicts'
        if base.is_dir():
            for folder in sorted(p for p in base.iterdir() if p.is_dir()):
                if (folder / 'vowels.txt').is_file() and (folder / 'consonants.txt').is_file():
                    self.phnum_preset_choice.Append(folder.name, str(folder))
        self.phnum_preset_choice.SetSelection(0)

    def on_phnum_preset_selected(self, event):
        """选择预设后自动切到「元音 + 辅音文件」并填好路径。"""
        folder = self.phnum_preset_choice.GetClientData(self.phnum_preset_choice.GetSelection())
        if not folder:
            return
        self.phnum_mode_choice.SetSelection(1)  # 元音 + 辅音文件
        self.phnum_vowels_text.SetValue(os.path.join(folder, 'vowels.txt'))
        self.phnum_consonants_text.SetValue(os.path.join(folder, 'consonants.txt'))
        self.on_phnum_mode_changed(None)
    
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
                
                dict_path = str(ROOT / 'tg2svdb' / 'dicts' / dict_file)
                
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

    def on_generate_textgrid_from_json(self, event):
        """由 json 生成 TextGrid（与「生成 json」相反的流程）。"""
        folder = self.svdb_path_text.GetValue().strip()
        dict_file = self.svdb_dict_choice.GetStringSelection()
        tail_ratio_str = self.svdb_tail_ratio_text.GetValue().strip()

        if not folder:
            wx.MessageBox(_('msg.err.select_folder'), _('msg.error'), wx.OK | wx.ICON_ERROR)
            return
        if not os.path.isdir(folder):
            wx.MessageBox(_('msg.err.folder_not_exist'), _('msg.error'), wx.OK | wx.ICON_ERROR)
            return

        dict_path = None
        tail_ratio = 50.0
        if dict_file:
            dict_path = os.path.join(str(ROOT), 'tg2svdb', 'dicts', dict_file)
            if not os.path.isfile(dict_path):
                wx.MessageBox(_('msg.err.dict_not_exist'), _('msg.error'), wx.OK | wx.ICON_ERROR)
                return
            try:
                tail_ratio = float(tail_ratio_str)
                if tail_ratio <= 0 or tail_ratio >= 100:
                    wx.MessageBox(_('msg.err.tail_ratio_range'), _('msg.error'), wx.OK | wx.ICON_ERROR)
                    return
            except ValueError:
                wx.MessageBox(_('msg.err.tail_ratio_invalid'), _('msg.error'), wx.OK | wx.ICON_ERROR)
                return

        def worker():
            try:
                wx.CallAfter(self.svdb_result_text.Clear)
                wx.CallAfter(self.svdb_result_text.AppendText, _('log.json2tg.start'))

                def progress(message):
                    wx.CallAfter(self.svdb_result_text.AppendText, message + "\n")
                    wx.CallAfter(self.svdb_result_text.ShowPosition,
                                 self.svdb_result_text.GetLastPosition())

                stats = json2tg.run(folder, dictionary=dict_path, tail_ratio=tail_ratio,
                                    progress_callback=progress)

                wx.CallAfter(self.svdb_result_text.AppendText,
                             _('log.json2tg.complete').format(
                                 written=stats['written'], failed=stats['failed']))
                wx.CallAfter(wx.MessageBox,
                             _('msg.ok.json2tg_complete').format(count=stats['written']),
                             _('msg.success'), wx.OK | wx.ICON_INFORMATION)
            except Exception:
                tb = traceback.format_exc()
                wx.CallAfter(self.svdb_result_text.AppendText, _('log.failed').format(error=tb))
                wx.CallAfter(wx.MessageBox, _('log.failed').format(error=tb),
                             _('msg.error'), wx.OK | wx.ICON_ERROR)

        threading.Thread(target=worker, daemon=True).start()

    
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
                    # auto_oto.ini 是中间文件，始终覆盖写入，否则“不覆盖”时会另存为 auto_oto_01.ini，
                    # 下面读回的仍是不含本次结果的旧 auto_oto.ini
                    oto_rw.oto_write(os.path.join(wav_path, 'auto_oto.ini'), cv + vc, '', 'y', oto_encoding)
                    
                    oto_data = oto_rw.oto_read(os.path.join(wav_path, 'auto_oto.ini'))
                    if os.path.exists(oto_preset):
                        wx.CallAfter(self.oto_result_text.AppendText, _('log.apply_template').format(path=oto_preset))
                        oto_data = oto_rw.oto_apply_template(oto_data, oto_preset,pitch)
                    # “不覆盖”时实际写入的可能是 oto_01.ini，后续检测必须用实际路径
                    oto_path_written = oto_rw.oto_write(os.path.join(wav_path, 'oto.ini'), oto_data, pitch, cover, oto_encoding)
                    if not oto_path_written:
                        oto_path_written = os.path.join(wav_path, 'oto.ini')
                    
                    
                    
                    wx.CallAfter(self.oto_result_text.AppendText, _('log.check_missing'))
                    oto_check.run(oto_path_written, presamp_path, pitch, vcv_mode)
                
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

                        # auto_oto.ini 是中间文件，始终覆盖写入（原因同单音阶生成）
                        oto_rw.oto_write(os.path.join(sub_path, 'auto_oto.ini'), cv + vc, '', 'y', oto_encoding)

                        oto_data = oto_rw.oto_read(os.path.join(sub_path, 'auto_oto.ini'))
                        if os.path.exists(oto_preset):
                            oto_data = oto_rw.oto_apply_template(oto_data, oto_preset)
                        # “不覆盖”时实际写入的可能是 oto_01.ini，后续检测必须用实际路径
                        oto_path_written = oto_rw.oto_write(os.path.join(sub_path, 'oto.ini'), oto_data, pitch, cover, oto_encoding)
                        if not oto_path_written:
                            oto_path_written = os.path.join(sub_path, 'oto.ini')

                        oto_check.run(oto_path_written, presamp_path, pitch, vcv_mode)

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

    # ══════════════════════════════════════════════════════════════
    # transcriptions.csv 制作（标记生成 → 新标签页）
    # ══════════════════════════════════════════════════════════════

    def _build_transcriptions_panel(self, parent):
        """构建「transcriptions.csv 制作」标签页（内含五个子标签页）。"""
        outer = wx.BoxSizer(wx.VERTICAL)
        nb = wx.Notebook(parent)

        words_panel = wx.Panel(nb)
        self._build_words_page(words_panel)

        csv_panel = wx.Panel(nb)
        self._build_transcriptions_csv_page(csv_panel)

        phnum_panel = wx.Panel(nb)
        self._build_phnum_page(phnum_panel)

        game_panel = wx.ScrolledWindow(nb, style=wx.VSCROLL)
        game_panel.SetScrollRate(0, 10)
        self._build_game_page(game_panel)

        csv2tg_panel = wx.Panel(nb)
        self._build_csv2tg_page(csv2tg_panel)

        nb.AddPage(words_panel, _('mark.words.title'))
        register(nb, 'mark.words.title', 'notebook_tab', 0)
        nb.AddPage(csv_panel, _('mark.csv.title'))
        register(nb, 'mark.csv.title', 'notebook_tab', 1)
        nb.AddPage(phnum_panel, _('mark.phnum.title'))
        register(nb, 'mark.phnum.title', 'notebook_tab', 2)
        nb.AddPage(game_panel, _('mark.game.title'))
        register(nb, 'mark.game.title', 'notebook_tab', 3)
        nb.AddPage(csv2tg_panel, _('mark.csv2tg.title'))
        register(nb, 'mark.csv2tg.title', 'notebook_tab', 4)

        outer.Add(nb, 1, wx.EXPAND | wx.ALL, 5)
        parent.SetSizer(outer)

    def _build_words_page(self, parent):
        """子标签页一：从 phones 层复原 words 层。"""
        sizer = wx.BoxSizer(wx.VERTICAL)

        box = wx.StaticBox(parent, label=_('mark.words.settings'))
        register(box, 'mark.words.settings', 'label')
        box_sizer = wx.StaticBoxSizer(box, wx.VERTICAL)

        # TextGrid 文件夹
        row = wx.BoxSizer(wx.HORIZONTAL)
        label = wx.StaticText(parent, label=_('mark.words.tg_folder'))
        register(label, 'mark.words.tg_folder')
        self.words_tg_text = wx.TextCtrl(parent, size=(420, -1))
        browse_btn = wx.Button(parent, label=_('textgrid.browse_folder'))
        register(browse_btn, 'textgrid.browse_folder', 'button')
        browse_btn.Bind(wx.EVT_BUTTON, lambda event: self.on_browse_folder(event, self.words_tg_text))
        row.Add(label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        row.Add(self.words_tg_text, 1, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        row.Add(browse_btn, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        box_sizer.Add(row, 0, wx.EXPAND | wx.ALL, 3)

        # 字典文件
        row = wx.BoxSizer(wx.HORIZONTAL)
        label = wx.StaticText(parent, label=_('mark.words.dict'))
        register(label, 'mark.words.dict')
        self.words_dict_text = wx.TextCtrl(parent, size=(420, -1))
        dict_btn = wx.Button(parent, label=_('json.browse'), size=(60, -1))
        register(dict_btn, 'json.browse', 'button')
        dict_btn.Bind(wx.EVT_BUTTON, lambda event: self.on_browse_file(event, self.words_dict_text))
        row.Add(label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        row.Add(self.words_dict_text, 1, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        row.Add(dict_btn, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        box_sizer.Add(row, 0, wx.EXPAND | wx.ALL, 3)

        # 输出文件夹（可选）
        row = wx.BoxSizer(wx.HORIZONTAL)
        label = wx.StaticText(parent, label=_('mark.words.out_folder'))
        register(label, 'mark.words.out_folder')
        self.words_out_text = wx.TextCtrl(parent, size=(420, -1))
        out_btn = wx.Button(parent, label=_('textgrid.browse_folder'))
        register(out_btn, 'textgrid.browse_folder', 'button')
        out_btn.Bind(wx.EVT_BUTTON, lambda event: self.on_browse_folder(event, self.words_out_text))
        row.Add(label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        row.Add(self.words_out_text, 1, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        row.Add(out_btn, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        box_sizer.Add(row, 0, wx.EXPAND | wx.ALL, 3)

        hint = wx.StaticText(parent, label=_('mark.words.out_hint'))
        register(hint, 'mark.words.out_hint')
        box_sizer.Add(hint, 0, wx.LEFT | wx.BOTTOM, 8)

        # 覆盖选项
        self.words_overwrite_check = wx.CheckBox(parent, label=_('mark.words.overwrite'))
        register(self.words_overwrite_check, 'mark.words.overwrite', 'checkbox')
        self.words_overwrite_check.SetValue(True)
        box_sizer.Add(self.words_overwrite_check, 0, wx.ALL, 3)

        # 按钮
        btn_row = wx.BoxSizer(wx.HORIZONTAL)
        self.words_run_btn = wx.Button(parent, label=_('mark.words.run'))
        register(self.words_run_btn, 'mark.words.run', 'button')
        self.words_run_btn.Bind(wx.EVT_BUTTON, self.on_words_from_phones)
        self.words_stop_btn = wx.Button(parent, label=_('textgrid.stop_infer'))
        register(self.words_stop_btn, 'textgrid.stop_infer', 'button')
        self.words_stop_btn.Bind(wx.EVT_BUTTON, self.on_words_stop)
        self.words_stop_btn.Disable()
        btn_row.Add(self.words_run_btn, 0, wx.ALL, 5)
        btn_row.Add(self.words_stop_btn, 0, wx.ALL, 5)
        box_sizer.Add(btn_row, 0, wx.CENTER)

        sizer.Add(box_sizer, 0, wx.EXPAND | wx.ALL, 5)

        result_label = wx.StaticText(parent, label=_('mark.words.result'))
        register(result_label, 'mark.words.result')
        sizer.Add(result_label, 0, wx.ALL | wx.LEFT, 5)

        self.words_result_text = wx.TextCtrl(parent, style=wx.TE_MULTILINE | wx.TE_READONLY, size=(-1, 400))
        sizer.Add(self.words_result_text, 1, wx.EXPAND | wx.ALL, 5)

        parent.SetSizer(sizer)

    def _build_transcriptions_csv_page(self, parent):
        """子标签页二：由 wav + TextGrid 生成 transcriptions.csv。"""
        sizer = wx.BoxSizer(wx.VERTICAL)

        box = wx.StaticBox(parent, label=_('mark.csv.settings'))
        register(box, 'mark.csv.settings', 'label')
        box_sizer = wx.StaticBoxSizer(box, wx.VERTICAL)

        # wav 文件夹
        row = wx.BoxSizer(wx.HORIZONTAL)
        label = wx.StaticText(parent, label=_('mark.csv.wav_folder'))
        register(label, 'mark.csv.wav_folder')
        self.csv_wav_text = wx.TextCtrl(parent, size=(420, -1))
        wav_btn = wx.Button(parent, label=_('textgrid.browse_folder'))
        register(wav_btn, 'textgrid.browse_folder', 'button')
        wav_btn.Bind(wx.EVT_BUTTON, lambda event: self.on_browse_folder(event, self.csv_wav_text))
        row.Add(label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        row.Add(self.csv_wav_text, 1, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        row.Add(wav_btn, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        box_sizer.Add(row, 0, wx.EXPAND | wx.ALL, 3)

        # TextGrid 文件夹
        row = wx.BoxSizer(wx.HORIZONTAL)
        label = wx.StaticText(parent, label=_('mark.csv.tg_folder'))
        register(label, 'mark.csv.tg_folder')
        self.csv_tg_text = wx.TextCtrl(parent, size=(420, -1))
        tg_btn = wx.Button(parent, label=_('textgrid.browse_folder'))
        register(tg_btn, 'textgrid.browse_folder', 'button')
        tg_btn.Bind(wx.EVT_BUTTON, lambda event: self.on_browse_folder(event, self.csv_tg_text))
        row.Add(label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        row.Add(self.csv_tg_text, 1, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        row.Add(tg_btn, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        box_sizer.Add(row, 0, wx.EXPAND | wx.ALL, 3)

        # 数据集输出文件夹
        row = wx.BoxSizer(wx.HORIZONTAL)
        label = wx.StaticText(parent, label=_('mark.csv.dataset_folder'))
        register(label, 'mark.csv.dataset_folder')
        self.csv_dataset_text = wx.TextCtrl(parent, size=(420, -1))
        dataset_btn = wx.Button(parent, label=_('textgrid.browse_folder'))
        register(dataset_btn, 'textgrid.browse_folder', 'button')
        dataset_btn.Bind(wx.EVT_BUTTON, lambda event: self.on_browse_folder(event, self.csv_dataset_text))
        row.Add(label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        row.Add(self.csv_dataset_text, 1, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        row.Add(dataset_btn, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        box_sizer.Add(row, 0, wx.EXPAND | wx.ALL, 3)

        # 音频处理方式：保持原始 / 量化到 16bit 44.1kHz
        row = wx.BoxSizer(wx.HORIZONTAL)
        label = wx.StaticText(parent, label=_('mark.csv.audio_mode'))
        register(label, 'mark.csv.audio_mode')
        self.csv_audio_mode_choice = wx.Choice(parent, size=(420, -1))
        self.csv_audio_mode_choice.Append(_('mark.csv.audio.keep'), 'keep')
        self.csv_audio_mode_choice.Append(_('mark.csv.audio.convert16'), 'convert16')
        self.csv_audio_mode_choice.SetSelection(1)  # 默认量化
        register(self.csv_audio_mode_choice, '', 'choice',
                 [('mark.csv.audio.keep', 'keep'),
                  ('mark.csv.audio.convert16', 'convert16')])
        row.Add(label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        row.Add(self.csv_audio_mode_choice, 1, wx.EXPAND | wx.ALL, 3)
        box_sizer.Add(row, 0, wx.EXPAND | wx.ALL, 3)

        # 读取 TextGrid 的 words / notes / note_slur 层
        self.csv_read_tiers_check = wx.CheckBox(parent, label=_('mark.csv.read_tiers'))
        self.csv_read_tiers_check.SetToolTip(_('mark.csv.read_tiers.tooltip'))
        register(self.csv_read_tiers_check, 'mark.csv.read_tiers.tooltip', 'tooltip')
        register(self.csv_read_tiers_check, 'mark.csv.read_tiers', 'checkbox')
        self.csv_read_tiers_check.SetValue(True)
        box_sizer.Add(self.csv_read_tiers_check, 0, wx.ALL, 3)

        # 递归子文件夹
        self.csv_recursive_check = wx.CheckBox(parent, label=_('mark.csv.recursive'))
        self.csv_recursive_check.SetToolTip(_('mark.csv.recursive.tooltip'))
        register(self.csv_recursive_check, 'mark.csv.recursive.tooltip', 'tooltip')
        register(self.csv_recursive_check, 'mark.csv.recursive', 'checkbox')
        self.csv_recursive_check.SetValue(False)
        box_sizer.Add(self.csv_recursive_check, 0, wx.ALL, 3)

        # 按钮
        btn_row = wx.BoxSizer(wx.HORIZONTAL)
        self.csv_run_btn = wx.Button(parent, label=_('mark.csv.run'))
        register(self.csv_run_btn, 'mark.csv.run', 'button')
        self.csv_run_btn.Bind(wx.EVT_BUTTON, self.on_build_transcriptions_csv)
        self.csv_stop_btn = wx.Button(parent, label=_('textgrid.stop_infer'))
        register(self.csv_stop_btn, 'textgrid.stop_infer', 'button')
        self.csv_stop_btn.Bind(wx.EVT_BUTTON, self.on_csv_stop)
        self.csv_stop_btn.Disable()
        btn_row.Add(self.csv_run_btn, 0, wx.ALL, 5)
        btn_row.Add(self.csv_stop_btn, 0, wx.ALL, 5)
        box_sizer.Add(btn_row, 0, wx.CENTER)

        sizer.Add(box_sizer, 0, wx.EXPAND | wx.ALL, 5)

        result_label = wx.StaticText(parent, label=_('mark.csv.result'))
        register(result_label, 'mark.csv.result')
        sizer.Add(result_label, 0, wx.ALL | wx.LEFT, 5)

        self.csv_result_text = wx.TextCtrl(parent, style=wx.TE_MULTILINE | wx.TE_READONLY, size=(-1, 400))
        sizer.Add(self.csv_result_text, 1, wx.EXPAND | wx.ALL, 5)

        parent.SetSizer(sizer)

    def _build_phnum_page(self, parent):
        """子标签页三：为 transcriptions.csv 添加 ph_num。"""
        sizer = wx.BoxSizer(wx.VERTICAL)

        box = wx.StaticBox(parent, label=_('mark.phnum.settings'))
        register(box, 'mark.phnum.settings', 'label')
        box_sizer = wx.StaticBoxSizer(box, wx.VERTICAL)

        # transcriptions.csv
        row = wx.BoxSizer(wx.HORIZONTAL)
        label = wx.StaticText(parent, label=_('mark.phnum.csv'))
        register(label, 'mark.phnum.csv')
        self.phnum_csv_text = wx.TextCtrl(parent, size=(420, -1))
        csv_btn = wx.Button(parent, label=_('json.browse'), size=(60, -1))
        register(csv_btn, 'json.browse', 'button')
        csv_btn.Bind(wx.EVT_BUTTON, lambda event: self.on_browse_file(event, self.phnum_csv_text))
        row.Add(label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        row.Add(self.phnum_csv_text, 1, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        row.Add(csv_btn, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        box_sizer.Add(row, 0, wx.EXPAND | wx.ALL, 3)

        # 预设 C+V 字典（tg2svdb/cv_dicts）
        row = wx.BoxSizer(wx.HORIZONTAL)
        label = wx.StaticText(parent, label=_('mark.phnum.preset'))
        register(label, 'mark.phnum.preset')
        self.phnum_preset_choice = wx.Choice(parent, size=(420, -1))
        self.phnum_preset_choice.SetToolTip(_('mark.phnum.preset.tooltip'))
        register(self.phnum_preset_choice, 'mark.phnum.preset.tooltip', 'tooltip')
        self.phnum_preset_choice.Bind(wx.EVT_CHOICE, self.on_phnum_preset_selected)
        row.Add(label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        row.Add(self.phnum_preset_choice, 1, wx.EXPAND | wx.ALL, 3)
        box_sizer.Add(row, 0, wx.EXPAND | wx.ALL, 3)

        # 规则来源
        row = wx.BoxSizer(wx.HORIZONTAL)
        label = wx.StaticText(parent, label=_('mark.phnum.mode'))
        register(label, 'mark.phnum.mode')
        self.phnum_mode_choice = wx.Choice(parent, size=(420, -1))
        self.phnum_mode_choice.Append(_('mark.phnum.mode.dict'), 'dict')
        self.phnum_mode_choice.Append(_('mark.phnum.mode.files'), 'files')
        self.phnum_mode_choice.SetSelection(0)
        register(self.phnum_mode_choice, '', 'choice',
                 [('mark.phnum.mode.dict', 'dict'), ('mark.phnum.mode.files', 'files')])
        self.phnum_mode_choice.Bind(wx.EVT_CHOICE, self.on_phnum_mode_changed)
        row.Add(label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        row.Add(self.phnum_mode_choice, 1, wx.EXPAND | wx.ALL, 3)
        box_sizer.Add(row, 0, wx.EXPAND | wx.ALL, 3)

        # 字典文件
        row = wx.BoxSizer(wx.HORIZONTAL)
        label = wx.StaticText(parent, label=_('mark.phnum.dict'))
        register(label, 'mark.phnum.dict')
        self.phnum_dict_text = wx.TextCtrl(parent, size=(420, -1))
        self.phnum_dict_browse = wx.Button(parent, label=_('json.browse'), size=(60, -1))
        register(self.phnum_dict_browse, 'json.browse', 'button')
        self.phnum_dict_browse.Bind(wx.EVT_BUTTON,
                                    lambda event: self.on_browse_file(event, self.phnum_dict_text))
        row.Add(label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        row.Add(self.phnum_dict_text, 1, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        row.Add(self.phnum_dict_browse, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        box_sizer.Add(row, 0, wx.EXPAND | wx.ALL, 3)

        # 元音文件
        row = wx.BoxSizer(wx.HORIZONTAL)
        label = wx.StaticText(parent, label=_('mark.phnum.vowels'))
        register(label, 'mark.phnum.vowels')
        self.phnum_vowels_text = wx.TextCtrl(parent, size=(420, -1))
        self.phnum_vowels_browse = wx.Button(parent, label=_('json.browse'), size=(60, -1))
        register(self.phnum_vowels_browse, 'json.browse', 'button')
        self.phnum_vowels_browse.Bind(wx.EVT_BUTTON,
                                      lambda event: self.on_browse_file(event, self.phnum_vowels_text))
        row.Add(label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        row.Add(self.phnum_vowels_text, 1, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        row.Add(self.phnum_vowels_browse, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        box_sizer.Add(row, 0, wx.EXPAND | wx.ALL, 3)

        # 辅音文件
        row = wx.BoxSizer(wx.HORIZONTAL)
        label = wx.StaticText(parent, label=_('mark.phnum.consonants'))
        register(label, 'mark.phnum.consonants')
        self.phnum_consonants_text = wx.TextCtrl(parent, size=(420, -1))
        self.phnum_consonants_browse = wx.Button(parent, label=_('json.browse'), size=(60, -1))
        register(self.phnum_consonants_browse, 'json.browse', 'button')
        self.phnum_consonants_browse.Bind(wx.EVT_BUTTON,
                                          lambda event: self.on_browse_file(event, self.phnum_consonants_text))
        row.Add(label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        row.Add(self.phnum_consonants_text, 1, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        row.Add(self.phnum_consonants_browse, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        box_sizer.Add(row, 0, wx.EXPAND | wx.ALL, 3)

        hint = wx.StaticText(parent, label=_('mark.phnum.hint'))
        register(hint, 'mark.phnum.hint')
        box_sizer.Add(hint, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 8)

        phnum_btn = wx.Button(parent, label=_('mark.phnum.run'))
        register(phnum_btn, 'mark.phnum.run', 'button')
        phnum_btn.Bind(wx.EVT_BUTTON, self.on_add_ph_num)
        self.phnum_run_btn = phnum_btn
        box_sizer.Add(phnum_btn, 0, wx.ALL | wx.CENTER, 5)

        sizer.Add(box_sizer, 0, wx.EXPAND | wx.ALL, 5)

        result_label = wx.StaticText(parent, label=_('mark.phnum.result'))
        register(result_label, 'mark.phnum.result')
        sizer.Add(result_label, 0, wx.ALL | wx.LEFT, 5)

        self.phnum_result_text = wx.TextCtrl(parent, style=wx.TE_MULTILINE | wx.TE_READONLY, size=(-1, 400))
        sizer.Add(self.phnum_result_text, 1, wx.EXPAND | wx.ALL, 5)

        parent.SetSizer(sizer)
        self.on_phnum_mode_changed(None)

    def on_phnum_mode_changed(self, event):
        """切换规则来源时启用/禁用对应输入框。"""
        mode = self.phnum_mode_choice.GetClientData(self.phnum_mode_choice.GetSelection())
        use_dict = mode != 'files'

        self.phnum_dict_text.Enable(use_dict)
        self.phnum_dict_browse.Enable(use_dict)
        self.phnum_vowels_text.Enable(not use_dict)
        self.phnum_vowels_browse.Enable(not use_dict)
        self.phnum_consonants_text.Enable(not use_dict)
        self.phnum_consonants_browse.Enable(not use_dict)

    def on_add_ph_num(self, event):
        """为 transcriptions.csv 添加/更新 ph_num 列。"""
        csv_path = self.phnum_csv_text.GetValue().strip()
        mode = self.phnum_mode_choice.GetClientData(self.phnum_mode_choice.GetSelection())
        use_dict = mode != 'files'
        dict_path = self.phnum_dict_text.GetValue().strip()
        vowels_path = self.phnum_vowels_text.GetValue().strip()
        consonants_path = self.phnum_consonants_text.GetValue().strip()

        if not csv_path:
            wx.MessageBox(_('msg.err.select_csv'), _('msg.error'), wx.OK | wx.ICON_ERROR)
            return
        if not os.path.isfile(csv_path):
            wx.MessageBox(_('msg.err.csv_not_exist'), _('msg.error'), wx.OK | wx.ICON_ERROR)
            return

        if use_dict:
            if not dict_path:
                wx.MessageBox(_('msg.err.select_dictionary'), _('msg.error'), wx.OK | wx.ICON_ERROR)
                return
            if not os.path.isfile(dict_path):
                wx.MessageBox(_('msg.err.dict_not_exist'), _('msg.error'), wx.OK | wx.ICON_ERROR)
                return
            vowels_path = consonants_path = None
        else:
            if not vowels_path or not consonants_path:
                wx.MessageBox(_('msg.err.phnum_need_source'), _('msg.error'), wx.OK | wx.ICON_ERROR)
                return
            if not os.path.isfile(vowels_path) or not os.path.isfile(consonants_path):
                wx.MessageBox(_('msg.err.file_not_exist'), _('msg.error'), wx.OK | wx.ICON_ERROR)
                return
            dict_path = None

        def worker():
            try:
                wx.CallAfter(self.phnum_result_text.Clear)
                wx.CallAfter(self.phnum_run_btn.Disable)
                wx.CallAfter(self.phnum_result_text.AppendText, _('log.phnum.start'))

                def progress(message):
                    wx.CallAfter(self.phnum_result_text.AppendText, message + "\n")
                    wx.CallAfter(self.phnum_result_text.ShowPosition,
                                 self.phnum_result_text.GetLastPosition())

                stats = phnum_builder.run(
                    csv_path,
                    dictionary=dict_path,
                    vowels=vowels_path,
                    consonants=consonants_path,
                    progress_callback=progress,
                )

                wx.CallAfter(self.phnum_result_text.AppendText,
                             _('log.phnum.complete').format(count=stats['total']))
                wx.CallAfter(wx.MessageBox,
                             _('msg.ok.phnum_complete').format(count=stats['total']),
                             _('msg.success'), wx.OK | wx.ICON_INFORMATION)
            except Exception:
                tb = traceback.format_exc()
                wx.CallAfter(self.phnum_result_text.AppendText, _('log.failed').format(error=tb))
                wx.CallAfter(wx.MessageBox, _('log.failed').format(error=tb),
                             _('msg.error'), wx.OK | wx.ICON_ERROR)
            finally:
                wx.CallAfter(self.phnum_run_btn.Enable)

        threading.Thread(target=worker, daemon=True).start()

    def _build_game_page(self, parent):
        """子标签页四：GAME ONNX 推理，为 transcriptions.csv 添加 MIDI 标注。"""
        sizer = wx.BoxSizer(wx.VERTICAL)

        box = wx.StaticBox(parent, label=_('mark.game.settings'))
        register(box, 'mark.game.settings', 'label')
        box_sizer = wx.StaticBoxSizer(box, wx.VERTICAL)

        # GAME 模型目录
        row = wx.BoxSizer(wx.HORIZONTAL)
        label = wx.StaticText(parent, label=_('mark.game.model_dir'))
        register(label, 'mark.game.model_dir')
        self.game_model_choice = wx.Choice(parent, size=(360, -1))
        self.game_model_choice.Bind(wx.EVT_CHOICE, self.on_game_model_selected)
        model_btn = wx.Button(parent, label=_('textgrid.browse_folder'))
        register(model_btn, 'textgrid.browse_folder', 'button')
        model_btn.Bind(wx.EVT_BUTTON, self.on_game_browse_model_dir)
        row.Add(label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        row.Add(self.game_model_choice, 1, wx.EXPAND | wx.ALL, 3)
        row.Add(model_btn, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        box_sizer.Add(row, 0, wx.EXPAND | wx.ALL, 3)

        # transcriptions.csv
        row = wx.BoxSizer(wx.HORIZONTAL)
        label = wx.StaticText(parent, label=_('mark.game.csv'))
        register(label, 'mark.game.csv')
        self.game_csv_text = wx.TextCtrl(parent, size=(360, -1))
        csv_btn = wx.Button(parent, label=_('json.browse'), size=(60, -1))
        register(csv_btn, 'json.browse', 'button')
        csv_btn.Bind(wx.EVT_BUTTON, lambda event: self.on_browse_file(event, self.game_csv_text))
        row.Add(label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        row.Add(self.game_csv_text, 1, wx.EXPAND | wx.ALL, 3)
        row.Add(csv_btn, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        box_sizer.Add(row, 0, wx.EXPAND | wx.ALL, 3)

        # 语言 + 设备
        row = wx.BoxSizer(wx.HORIZONTAL)
        lang_label = wx.StaticText(parent, label=_('mark.game.language'))
        register(lang_label, 'mark.game.language')
        self.game_language_choice = wx.Choice(parent, size=(150, -1))
        device_label = wx.StaticText(parent, label=_('mark.game.device'))
        register(device_label, 'mark.game.device')
        self.game_device_choice = wx.Choice(parent, size=(180, -1))
        self.game_device_choice.Append(_('textgrid.device.cpu'), 'cpu')
        self.game_device_choice.Append(_('textgrid.device.dml'), 'dml')
        self.game_device_choice.SetSelection(0)
        register(self.game_device_choice, '', 'choice',
                 [('textgrid.device.cpu', 'cpu'), ('textgrid.device.dml', 'dml')])
        row.Add(lang_label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        row.Add(self.game_language_choice, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        row.Add(device_label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        row.Add(self.game_device_choice, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        box_sizer.Add(row, 0, wx.ALL, 3)

        # 输出文件名 + 覆盖
        row = wx.BoxSizer(wx.HORIZONTAL)
        label = wx.StaticText(parent, label=_('mark.game.output_name'))
        register(label, 'mark.game.output_name')
        self.game_output_name_text = wx.TextCtrl(parent, value="transcriptions-midi.csv", size=(220, -1))
        self.game_overwrite_check = wx.CheckBox(parent, label=_('mark.game.overwrite'))
        register(self.game_overwrite_check, 'mark.game.overwrite', 'checkbox')
        self.game_overwrite_check.SetValue(False)
        self.game_overwrite_check.Bind(wx.EVT_CHECKBOX, self.on_game_overwrite_changed)
        row.Add(label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        row.Add(self.game_output_name_text, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        row.Add(self.game_overwrite_check, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        box_sizer.Add(row, 0, wx.ALL, 3)

        # 无声音素
        row = wx.BoxSizer(wx.HORIZONTAL)
        label = wx.StaticText(parent, label=_('mark.game.uv_vocab'))
        register(label, 'mark.game.uv_vocab')
        self.game_uv_text = wx.TextCtrl(parent, value="EP,AP,SP,br,sil", size=(360, -1))
        row.Add(label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        row.Add(self.game_uv_text, 1, wx.EXPAND | wx.ALL, 3)
        box_sizer.Add(row, 0, wx.EXPAND | wx.ALL, 3)

        # UV 判定规则
        row = wx.BoxSizer(wx.HORIZONTAL)
        word_label = wx.StaticText(parent, label=_('mark.game.uv_word_cond'))
        register(word_label, 'mark.game.uv_word_cond')
        self.game_uv_word_choice = wx.Choice(parent, size=(160, -1))
        self.game_uv_word_choice.Append(_('mark.game.uv.word.lead'), 'lead')
        self.game_uv_word_choice.Append(_('mark.game.uv.word.all'), 'all')
        self.game_uv_word_choice.SetSelection(0)
        register(self.game_uv_word_choice, '', 'choice',
                 [('mark.game.uv.word.lead', 'lead'), ('mark.game.uv.word.all', 'all')])
        note_label = wx.StaticText(parent, label=_('mark.game.uv_note_cond'))
        register(note_label, 'mark.game.uv_note_cond')
        self.game_uv_note_choice = wx.Choice(parent, size=(160, -1))
        self.game_uv_note_choice.Append(_('mark.game.uv.note.predict'), 'predict')
        self.game_uv_note_choice.Append(_('mark.game.uv.note.follow'), 'follow')
        self.game_uv_note_choice.SetSelection(0)
        register(self.game_uv_note_choice, '', 'choice',
                 [('mark.game.uv.note.predict', 'predict'), ('mark.game.uv.note.follow', 'follow')])
        row.Add(word_label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        row.Add(self.game_uv_word_choice, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        row.Add(note_label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        row.Add(self.game_uv_note_choice, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        box_sizer.Add(row, 0, wx.ALL, 3)

        self.game_use_wb_check = wx.CheckBox(parent, label=_('mark.game.use_wb'))
        register(self.game_use_wb_check, 'mark.game.use_wb', 'checkbox')
        self.game_use_wb_check.SetValue(True)
        box_sizer.Add(self.game_use_wb_check, 0, wx.ALL, 3)

        # 推理参数
        row = wx.BoxSizer(wx.HORIZONTAL)
        for key, attr, value in (
            ('mark.game.seg_threshold', 'game_seg_threshold_text', '0.2'),
            ('mark.game.seg_radius', 'game_seg_radius_text', '0.02'),
            ('mark.game.t0', 'game_t0_text', '0.0'),
        ):
            label = wx.StaticText(parent, label=_(key))
            register(label, key)
            text = wx.TextCtrl(parent, value=value, size=(55, -1))
            setattr(self, attr, text)
            row.Add(label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
            row.Add(text, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        box_sizer.Add(row, 0, wx.ALL, 3)

        row = wx.BoxSizer(wx.HORIZONTAL)
        for key, attr, value in (
            ('mark.game.nsteps', 'game_nsteps_text', '8'),
            ('mark.game.est_threshold', 'game_est_threshold_text', '0.2'),
            ('mark.game.max_chunk', 'game_max_chunk_text', '30'),
        ):
            label = wx.StaticText(parent, label=_(key))
            register(label, key)
            text = wx.TextCtrl(parent, value=value, size=(55, -1))
            setattr(self, attr, text)
            row.Add(label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
            row.Add(text, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        self.game_max_chunk_text.SetToolTip(_('mark.game.max_chunk.tooltip'))
        register(self.game_max_chunk_text, 'mark.game.max_chunk.tooltip', 'tooltip')
        box_sizer.Add(row, 0, wx.ALL, 3)

        # 按钮
        btn_row = wx.BoxSizer(wx.HORIZONTAL)
        self.game_run_btn = wx.Button(parent, label=_('mark.game.run'))
        register(self.game_run_btn, 'mark.game.run', 'button')
        self.game_run_btn.Bind(wx.EVT_BUTTON, self.on_game_align)
        self.game_stop_btn = wx.Button(parent, label=_('textgrid.stop_infer'))
        register(self.game_stop_btn, 'textgrid.stop_infer', 'button')
        self.game_stop_btn.Bind(wx.EVT_BUTTON, self.on_game_stop)
        self.game_stop_btn.Disable()
        btn_row.Add(self.game_run_btn, 0, wx.ALL, 5)
        btn_row.Add(self.game_stop_btn, 0, wx.ALL, 5)
        box_sizer.Add(btn_row, 0, wx.CENTER)

        sizer.Add(box_sizer, 0, wx.EXPAND | wx.ALL, 5)

        result_label = wx.StaticText(parent, label=_('mark.game.result'))
        register(result_label, 'mark.game.result')
        sizer.Add(result_label, 0, wx.ALL | wx.LEFT, 5)

        self.game_result_text = wx.TextCtrl(parent, style=wx.TE_MULTILINE | wx.TE_READONLY, size=(-1, 260))
        sizer.Add(self.game_result_text, 1, wx.EXPAND | wx.ALL, 5)

        parent.SetSizer(sizer)
        parent.FitInside()

    def load_game_models(self):
        """扫描 ROOT/GAME_model 下带 config.json 的 ONNX 模型目录。"""
        self.game_model_choice.Clear()
        base = ROOT / 'GAME_model'
        if base.exists():
            for folder in sorted(base.iterdir()):
                if folder.is_dir() and (folder / 'config.json').is_file():
                    self.game_model_choice.Append(folder.name, str(folder))
        if self.game_model_choice.GetCount():
            self.game_model_choice.SetSelection(0)
            self.on_game_model_selected(None)

    def _game_model_dir(self) -> str:
        if self.game_model_choice.GetSelection() == wx.NOT_FOUND:
            return ''
        return self.game_model_choice.GetClientData(self.game_model_choice.GetSelection()) or ''

    def on_game_browse_model_dir(self, event):
        dialog = wx.DirDialog(None, _('msg.select_folder'), style=wx.DD_DEFAULT_STYLE)
        if dialog.ShowModal() == wx.ID_OK:
            path = dialog.GetPath()
            label = os.path.basename(path.rstrip('\\/')) or path
            index = self.game_model_choice.FindString(label)
            if index == wx.NOT_FOUND:
                index = self.game_model_choice.Append(label, path)
            self.game_model_choice.SetSelection(index)
            self.on_game_model_selected(None)
        dialog.Destroy()

    def on_game_model_selected(self, event):
        """根据 config.json 动态填充语言列表。"""
        model_dir = self._game_model_dir()
        self.game_language_choice.Clear()
        self.game_language_choice.Append(_('mark.game.language.auto'), '')
        if model_dir:
            try:
                with open(os.path.join(model_dir, 'config.json'), 'r', encoding='utf-8') as f:
                    config = json.load(f)
                for lang in (config.get('languages') or {}).keys():
                    self.game_language_choice.Append(lang, lang)
            except Exception:
                pass
        self.game_language_choice.SetSelection(0)

    def on_game_overwrite_changed(self, event):
        self.game_output_name_text.Enable(not self.game_overwrite_check.GetValue())

    def on_game_stop(self, event):
        self._game_stop.set()
        self.game_stop_btn.Disable()
        self.game_result_text.AppendText(_('log.infer_stop_requested'))

    def on_game_align(self, event):
        """运行 GAME ONNX 推理，为 CSV 添加 note_seq / note_dur / note_slur。"""
        model_dir = self._game_model_dir()
        csv_path = self.game_csv_text.GetValue().strip()
        language = self.game_language_choice.GetClientData(self.game_language_choice.GetSelection()) or None
        device = self.game_device_choice.GetClientData(self.game_device_choice.GetSelection()) or 'cpu'
        overwrite = self.game_overwrite_check.GetValue()
        output_name = self.game_output_name_text.GetValue().strip()
        uv_vocab = self.game_uv_text.GetValue().strip()
        uv_word_cond = self.game_uv_word_choice.GetClientData(self.game_uv_word_choice.GetSelection()) or 'lead'
        uv_note_cond = self.game_uv_note_choice.GetClientData(self.game_uv_note_choice.GetSelection()) or 'predict'
        use_wb = self.game_use_wb_check.GetValue()

        if not model_dir:
            wx.MessageBox(_('msg.err.select_game_model'), _('msg.error'), wx.OK | wx.ICON_ERROR)
            return
        required_files = ('encoder.onnx', 'segmenter.onnx', 'estimator.onnx', 'dur2bd.onnx', 'bd2dur.onnx', 'config.json')
        missing = [name for name in required_files if not os.path.isfile(os.path.join(model_dir, name))]
        if missing:
            wx.MessageBox(_('msg.err.game_model_invalid').format(files=", ".join(missing)),
                          _('msg.error'), wx.OK | wx.ICON_ERROR)
            return
        if not csv_path:
            wx.MessageBox(_('msg.err.select_csv'), _('msg.error'), wx.OK | wx.ICON_ERROR)
            return
        if not os.path.isfile(csv_path):
            wx.MessageBox(_('msg.err.csv_not_exist'), _('msg.error'), wx.OK | wx.ICON_ERROR)
            return
        if not overwrite and not output_name:
            wx.MessageBox(_('msg.err.game_output_name'), _('msg.error'), wx.OK | wx.ICON_ERROR)
            return

        try:
            seg_threshold = float(self.game_seg_threshold_text.GetValue().strip())
            seg_radius = float(self.game_seg_radius_text.GetValue().strip())
            t0 = float(self.game_t0_text.GetValue().strip())
            nsteps = int(self.game_nsteps_text.GetValue().strip())
            est_threshold = float(self.game_est_threshold_text.GetValue().strip())
            max_chunk_sec = float(self.game_max_chunk_text.GetValue().strip())
            if not (0 < seg_threshold < 1 and 0 < est_threshold < 1 and 0 <= t0 < 1
                    and nsteps > 0 and seg_radius > 0 and max_chunk_sec >= 0):
                raise ValueError
        except ValueError:
            wx.MessageBox(_('msg.err.game_params'), _('msg.error'), wx.OK | wx.ICON_ERROR)
            return

        output_path = None
        if not overwrite:
            output_path = os.path.join(os.path.dirname(os.path.abspath(csv_path)), output_name)
        output_display = csv_path if overwrite else output_path

        def worker():
            try:
                wx.CallAfter(self.game_result_text.Clear)
                self._game_stop.clear()
                wx.CallAfter(self.game_run_btn.Disable)
                wx.CallAfter(self.game_stop_btn.Enable)
                wx.CallAfter(self.game_result_text.AppendText, _('log.game.start'))
                wx.CallAfter(self.game_result_text.AppendText,
                             _('log.game.options').format(
                                 lang=language or _('mark.game.language.auto'),
                                 device=str(device).upper(),
                                 chunk=f"{max_chunk_sec:g}",
                                 output=output_display))

                def progress(message):
                    wx.CallAfter(self.game_result_text.AppendText, message + "\n")
                    wx.CallAfter(self.game_result_text.ShowPosition,
                                 self.game_result_text.GetLastPosition())

                stats = game_infer.align_transcriptions(
                    csv_path, model_dir,
                    language=language,
                    device=device,
                    output_path=output_path,
                    overwrite=overwrite,
                    use_wb=use_wb,
                    uv_vocab=uv_vocab,
                    uv_word_cond=uv_word_cond,
                    uv_note_cond=uv_note_cond,
                    seg_threshold=seg_threshold,
                    seg_radius=seg_radius,
                    t0=t0,
                    nsteps=nsteps,
                    est_threshold=est_threshold,
                    max_chunk_sec=max_chunk_sec,
                    progress_callback=progress,
                    stop_event=self._game_stop,
                )

                if stats['stopped']:
                    wx.CallAfter(self.game_result_text.AppendText, _('log.game.stopped'))
                else:
                    wx.CallAfter(self.game_result_text.AppendText,
                                 _('log.game.complete').format(count=stats['updated']))
                    wx.CallAfter(wx.MessageBox,
                                 _('msg.ok.game_complete').format(count=stats['updated']),
                                 _('msg.success'), wx.OK | wx.ICON_INFORMATION)
            except Exception:
                tb = traceback.format_exc()
                wx.CallAfter(self.game_result_text.AppendText, _('log.failed').format(error=tb))
                wx.CallAfter(wx.MessageBox, _('log.failed').format(error=tb),
                             _('msg.error'), wx.OK | wx.ICON_ERROR)
            finally:
                wx.CallAfter(self.game_stop_btn.Disable)
                wx.CallAfter(self.game_run_btn.Enable)

        threading.Thread(target=worker, daemon=True).start()

    def _build_csv2tg_page(self, parent):
        """子标签页五：把 transcriptions.csv 反推为 TextGrid。"""
        sizer = wx.BoxSizer(wx.VERTICAL)

        box = wx.StaticBox(parent, label=_('mark.csv2tg.settings'))
        register(box, 'mark.csv2tg.settings', 'label')
        box_sizer = wx.StaticBoxSizer(box, wx.VERTICAL)

        # transcriptions.csv
        row = wx.BoxSizer(wx.HORIZONTAL)
        label = wx.StaticText(parent, label=_('mark.csv2tg.csv'))
        register(label, 'mark.csv2tg.csv')
        self.csv2tg_csv_text = wx.TextCtrl(parent, size=(420, -1))
        csv_btn = wx.Button(parent, label=_('json.browse'), size=(60, -1))
        register(csv_btn, 'json.browse', 'button')
        csv_btn.Bind(wx.EVT_BUTTON, lambda event: self.on_browse_file(event, self.csv2tg_csv_text))
        row.Add(label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        row.Add(self.csv2tg_csv_text, 1, wx.EXPAND | wx.ALL, 3)
        row.Add(csv_btn, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        box_sizer.Add(row, 0, wx.EXPAND | wx.ALL, 3)

        # 输出目录
        row = wx.BoxSizer(wx.HORIZONTAL)
        label = wx.StaticText(parent, label=_('mark.csv2tg.out_dir'))
        register(label, 'mark.csv2tg.out_dir')
        self.csv2tg_out_text = wx.TextCtrl(parent, size=(420, -1))
        out_btn = wx.Button(parent, label=_('textgrid.browse_folder'))
        register(out_btn, 'textgrid.browse_folder', 'button')
        out_btn.Bind(wx.EVT_BUTTON, lambda event: self.on_browse_folder(event, self.csv2tg_out_text))
        row.Add(label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        row.Add(self.csv2tg_out_text, 1, wx.EXPAND | wx.ALL, 3)
        row.Add(out_btn, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        box_sizer.Add(row, 0, wx.EXPAND | wx.ALL, 3)

        hint = wx.StaticText(parent, label=_('mark.csv2tg.out_hint'))
        register(hint, 'mark.csv2tg.out_hint')
        box_sizer.Add(hint, 0, wx.LEFT | wx.BOTTOM, 8)

        # 字典（可选）
        row = wx.BoxSizer(wx.HORIZONTAL)
        label = wx.StaticText(parent, label=_('mark.csv2tg.dictionary'))
        register(label, 'mark.csv2tg.dictionary')
        self.csv2tg_dict_text = wx.TextCtrl(parent, size=(420, -1))
        dict_btn = wx.Button(parent, label=_('json.browse'), size=(60, -1))
        register(dict_btn, 'json.browse', 'button')
        dict_btn.Bind(wx.EVT_BUTTON, lambda event: self.on_browse_file(event, self.csv2tg_dict_text))
        row.Add(label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        row.Add(self.csv2tg_dict_text, 1, wx.EXPAND | wx.ALL, 3)
        row.Add(dict_btn, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        box_sizer.Add(row, 0, wx.EXPAND | wx.ALL, 3)

        self.csv2tg_overwrite_check = wx.CheckBox(parent, label=_('mark.csv2tg.overwrite'))
        register(self.csv2tg_overwrite_check, 'mark.csv2tg.overwrite', 'checkbox')
        self.csv2tg_overwrite_check.SetValue(True)
        box_sizer.Add(self.csv2tg_overwrite_check, 0, wx.ALL, 3)

        hint2 = wx.StaticText(parent, label=_('mark.csv2tg.hint'))
        register(hint2, 'mark.csv2tg.hint')
        box_sizer.Add(hint2, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 8)

        self.csv2tg_run_btn = wx.Button(parent, label=_('mark.csv2tg.run'))
        register(self.csv2tg_run_btn, 'mark.csv2tg.run', 'button')
        self.csv2tg_run_btn.Bind(wx.EVT_BUTTON, self.on_csv_to_tg)
        box_sizer.Add(self.csv2tg_run_btn, 0, wx.ALL | wx.CENTER, 5)

        sizer.Add(box_sizer, 0, wx.EXPAND | wx.ALL, 5)

        result_label = wx.StaticText(parent, label=_('mark.csv2tg.result'))
        register(result_label, 'mark.csv2tg.result')
        sizer.Add(result_label, 0, wx.ALL | wx.LEFT, 5)

        self.csv2tg_result_text = wx.TextCtrl(parent, style=wx.TE_MULTILINE | wx.TE_READONLY, size=(-1, 400))
        sizer.Add(self.csv2tg_result_text, 1, wx.EXPAND | wx.ALL, 5)

        parent.SetSizer(sizer)

    def on_csv_to_tg(self, event):
        """把 transcriptions.csv 反推为 TextGrid。"""
        csv_path = self.csv2tg_csv_text.GetValue().strip()
        out_dir = self.csv2tg_out_text.GetValue().strip()
        dictionary = self.csv2tg_dict_text.GetValue().strip()
        overwrite = self.csv2tg_overwrite_check.GetValue()

        if not csv_path:
            wx.MessageBox(_('msg.err.select_csv'), _('msg.error'), wx.OK | wx.ICON_ERROR)
            return
        if not os.path.isfile(csv_path):
            wx.MessageBox(_('msg.err.csv_not_exist'), _('msg.error'), wx.OK | wx.ICON_ERROR)
            return
        if out_dir and not os.path.isdir(out_dir):
            wx.MessageBox(_('msg.err.folder_not_exist'), _('msg.error'), wx.OK | wx.ICON_ERROR)
            return
        if dictionary and not os.path.isfile(dictionary):
            wx.MessageBox(_('msg.err.dict_not_exist'), _('msg.error'), wx.OK | wx.ICON_ERROR)
            return

        def worker():
            try:
                wx.CallAfter(self.csv2tg_result_text.Clear)
                wx.CallAfter(self.csv2tg_run_btn.Disable)
                wx.CallAfter(self.csv2tg_result_text.AppendText, _('log.csv2tg.start'))

                def progress(message):
                    wx.CallAfter(self.csv2tg_result_text.AppendText, message + "\n")
                    wx.CallAfter(self.csv2tg_result_text.ShowPosition,
                                 self.csv2tg_result_text.GetLastPosition())

                stats = csv2tg.run(
                    csv_path,
                    out_dir=out_dir or None,
                    dictionary=dictionary or None,
                    overwrite=overwrite,
                    progress_callback=progress,
                )

                wx.CallAfter(self.csv2tg_result_text.AppendText,
                             _('log.csv2tg.complete').format(count=stats['written']))
                wx.CallAfter(wx.MessageBox,
                             _('msg.ok.csv2tg_complete').format(count=stats['written']),
                             _('msg.success'), wx.OK | wx.ICON_INFORMATION)
            except Exception:
                tb = traceback.format_exc()
                wx.CallAfter(self.csv2tg_result_text.AppendText, _('log.failed').format(error=tb))
                wx.CallAfter(wx.MessageBox, _('log.failed').format(error=tb),
                             _('msg.error'), wx.OK | wx.ICON_ERROR)
            finally:
                wx.CallAfter(self.csv2tg_run_btn.Enable)

        threading.Thread(target=worker, daemon=True).start()

    def on_words_stop(self, event):
        """停止 words 层复原。"""
        self._words_stop.set()
        self.words_stop_btn.Disable()
        self.words_result_text.AppendText(_('log.infer_stop_requested'))

    def on_csv_stop(self, event):
        """停止 transcriptions.csv 生成。"""
        self._csv_stop.set()
        self.csv_stop_btn.Disable()
        self.csv_result_text.AppendText(_('log.infer_stop_requested'))

    def on_words_from_phones(self, event):
        """从 phones 层复原 words 层。"""
        tg_dir = self.words_tg_text.GetValue().strip()
        dict_path = self.words_dict_text.GetValue().strip()
        out_dir = self.words_out_text.GetValue().strip()
        overwrite = self.words_overwrite_check.GetValue()

        if not tg_dir:
            wx.MessageBox(_('msg.err.select_tg_folder'), _('msg.error'), wx.OK | wx.ICON_ERROR)
            return
        if not os.path.isdir(tg_dir):
            wx.MessageBox(_('msg.err.textgrid_not_exist'), _('msg.error'), wx.OK | wx.ICON_ERROR)
            return
        if not dict_path:
            wx.MessageBox(_('msg.err.select_dictionary'), _('msg.error'), wx.OK | wx.ICON_ERROR)
            return
        if not os.path.isfile(dict_path):
            wx.MessageBox(_('msg.err.dict_not_exist'), _('msg.error'), wx.OK | wx.ICON_ERROR)
            return

        def worker():
            try:
                wx.CallAfter(self.words_result_text.Clear)
                self._words_stop.clear()
                wx.CallAfter(self.words_run_btn.Disable)
                wx.CallAfter(self.words_stop_btn.Enable)
                wx.CallAfter(self.words_result_text.AppendText, _('log.words.start'))
                wx.CallAfter(self.words_result_text.AppendText,
                             _('log.words.dict_info').format(path=dict_path))

                def progress(message):
                    wx.CallAfter(self.words_result_text.AppendText, message + "\n")
                    wx.CallAfter(self.words_result_text.ShowPosition,
                                 self.words_result_text.GetLastPosition())

                stats = words_from_phones.run(
                    tg_dir, dict_path,
                    out_dir=out_dir or None,
                    overwrite=overwrite,
                    recursive=True,
                    progress_callback=progress,
                    stop_event=self._words_stop,
                )

                if stats["total"] == 0:
                    wx.CallAfter(self.words_result_text.AppendText, _('log.words.no_files'))
                    return
                if stats["stopped"]:
                    wx.CallAfter(self.words_result_text.AppendText, _('log.words.stopped'))
                    return
                wx.CallAfter(self.words_result_text.AppendText, _('log.words.complete'))
                wx.CallAfter(wx.MessageBox,
                             _('msg.ok.words_complete').format(count=stats["processed"]),
                             _('msg.success'), wx.OK | wx.ICON_INFORMATION)
            except Exception:
                tb = traceback.format_exc()
                wx.CallAfter(self.words_result_text.AppendText, _('log.failed').format(error=tb))
                wx.CallAfter(wx.MessageBox, _('log.failed').format(error=tb),
                             _('msg.error'), wx.OK | wx.ICON_ERROR)
            finally:
                wx.CallAfter(self.words_stop_btn.Disable)
                wx.CallAfter(self.words_run_btn.Enable)

        threading.Thread(target=worker, daemon=True).start()

    def on_build_transcriptions_csv(self, event):
        """由 wav + TextGrid 生成 transcriptions.csv。"""
        wav_folder = self.csv_wav_text.GetValue().strip()
        tg_folder = self.csv_tg_text.GetValue().strip()
        dataset_folder = self.csv_dataset_text.GetValue().strip()
        recursive = self.csv_recursive_check.GetValue()
        read_extra_tiers = self.csv_read_tiers_check.GetValue()
        audio_mode = self.csv_audio_mode_choice.GetClientData(self.csv_audio_mode_choice.GetSelection()) or 'convert16'
        convert_audio = audio_mode == 'convert16'

        if not wav_folder:
            wx.MessageBox(_('msg.err.select_wav_folder'), _('msg.error'), wx.OK | wx.ICON_ERROR)
            return
        if not os.path.isdir(wav_folder):
            wx.MessageBox(_('msg.err.wav_folder_not_exist'), _('msg.error'), wx.OK | wx.ICON_ERROR)
            return
        if not tg_folder:
            wx.MessageBox(_('msg.err.select_tg_folder'), _('msg.error'), wx.OK | wx.ICON_ERROR)
            return
        if not os.path.isdir(tg_folder):
            wx.MessageBox(_('msg.err.textgrid_not_exist'), _('msg.error'), wx.OK | wx.ICON_ERROR)
            return
        if not dataset_folder:
            wx.MessageBox(_('msg.err.select_dataset_folder'), _('msg.error'), wx.OK | wx.ICON_ERROR)
            return

        def worker():
            try:
                wx.CallAfter(self.csv_result_text.Clear)
                self._csv_stop.clear()
                wx.CallAfter(self.csv_run_btn.Disable)
                wx.CallAfter(self.csv_stop_btn.Enable)
                wx.CallAfter(self.csv_result_text.AppendText, _('log.csv.start'))
                wx.CallAfter(self.csv_result_text.AppendText,
                             _('log.csv.options').format(
                                 recursive=_('msg.on') if recursive else _('msg.off'),
                                 audio=self.csv_audio_mode_choice.GetStringSelection()))

                def progress(message):
                    wx.CallAfter(self.csv_result_text.AppendText, message + "\n")
                    wx.CallAfter(self.csv_result_text.ShowPosition,
                                 self.csv_result_text.GetLastPosition())

                stats = build_diff_dataset(
                    wav_folder, tg_folder, dataset_folder,
                    wav_subtype="PCM_16",
                    progress_callback=progress,
                    stop_event=self._csv_stop,
                    recursive=recursive,
                    convert_audio=convert_audio,
                    read_extra_tiers=read_extra_tiers,
                )

                if stats["total"] == 0:
                    wx.CallAfter(self.csv_result_text.AppendText, _('log.csv.no_wav'))
                    return
                if stats["stopped"]:
                    wx.CallAfter(self.csv_result_text.AppendText, _('log.csv.stopped'))
                    return
                wx.CallAfter(self.csv_result_text.AppendText, _('log.csv.complete'))
                wx.CallAfter(wx.MessageBox,
                             _('msg.ok.csv_complete').format(count=stats["processed"]),
                             _('msg.success'), wx.OK | wx.ICON_INFORMATION)
            except Exception:
                tb = traceback.format_exc()
                wx.CallAfter(self.csv_result_text.AppendText, _('log.failed').format(error=tb))
                wx.CallAfter(wx.MessageBox, _('log.failed').format(error=tb),
                             _('msg.error'), wx.OK | wx.ICON_ERROR)
            finally:
                wx.CallAfter(self.csv_stop_btn.Disable)
                wx.CallAfter(self.csv_run_btn.Enable)

        threading.Thread(target=worker, daemon=True).start()

    # ══════════════════════════════════════════════════════════════
    # TIFA 推理（TextGrid 面板第一个标签页）
    # ══════════════════════════════════════════════════════════════

    def _build_tifa_panel(self, parent):
        """构建 TIFA ONNX 推理标签页。"""
        sizer = wx.BoxSizer(wx.VERTICAL)

        title = wx.StaticText(parent, label=_('tifa.title'))
        register(title, 'tifa.title')
        sizer.Add(title, 0, wx.ALL | wx.CENTER, 5)

        # ── 音源文件夹 ──
        folder_box = wx.StaticBox(parent, label=_('textgrid.folder'))
        register(folder_box, 'textgrid.folder', 'label')
        folder_box_sizer = wx.StaticBoxSizer(folder_box, wx.HORIZONTAL)
        self.tifa_folder_text = wx.TextCtrl(parent, size=(500, -1))
        tifa_browse_btn = wx.Button(parent, label=_('textgrid.browse_folder'))
        register(tifa_browse_btn, 'textgrid.browse_folder', 'button')
        tifa_browse_btn.Bind(wx.EVT_BUTTON, lambda event: self.on_browse_folder(event, self.tifa_folder_text))
        folder_box_sizer.Add(self.tifa_folder_text, 1, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        folder_box_sizer.Add(tifa_browse_btn, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        sizer.Add(folder_box_sizer, 0, wx.EXPAND | wx.ALL, 5)

        content_sizer = wx.BoxSizer(wx.HORIZONTAL)

        # ── 左：模型选择 ──
        left_box = wx.StaticBox(parent, label=_('textgrid.model_section'))
        register(left_box, 'textgrid.model_section', 'label')
        left_sizer = wx.StaticBoxSizer(left_box, wx.VERTICAL)

        row = wx.BoxSizer(wx.HORIZONTAL)
        model_folder_label = wx.StaticText(parent, label=_('textgrid.model_folder'))
        register(model_folder_label, 'textgrid.model_folder')
        self.tifa_model_folder_choice = wx.Choice(parent, size=(250, -1))
        self.tifa_model_folder_choice.Bind(wx.EVT_CHOICE, self.on_tifa_model_folder_selected)
        tifa_model_browse_btn = wx.Button(parent, label=_('json.browse'), size=(60, -1))
        register(tifa_model_browse_btn, 'json.browse', 'button')
        tifa_model_browse_btn.Bind(wx.EVT_BUTTON, self.on_tifa_browse_model_dir)
        row.Add(model_folder_label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        row.Add(self.tifa_model_folder_choice, 1, wx.EXPAND | wx.ALL, 3)
        row.Add(tifa_model_browse_btn, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        left_sizer.Add(row, 0, wx.EXPAND | wx.ALL, 3)

        row = wx.BoxSizer(wx.HORIZONTAL)
        dict_label = wx.StaticText(parent, label=_('textgrid.dict'))
        register(dict_label, 'textgrid.dict')
        self.tifa_dict_choice = wx.Choice(parent, size=(250, -1))
        row.Add(dict_label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        row.Add(self.tifa_dict_choice, 1, wx.EXPAND | wx.ALL, 3)
        left_sizer.Add(row, 0, wx.EXPAND | wx.ALL, 3)

        content_sizer.Add(left_sizer, 1, wx.EXPAND | wx.ALL, 5)
        content_sizer.Add(wx.StaticLine(parent, style=wx.LI_VERTICAL), 0,
                          wx.EXPAND | wx.LEFT | wx.RIGHT, 5)

        # ── 右：推理设置 ──
        right_box = wx.StaticBox(parent, label=_('textgrid.infer_section'))
        register(right_box, 'textgrid.infer_section', 'label')
        right_sizer = wx.StaticBoxSizer(right_box, wx.VERTICAL)

        row = wx.BoxSizer(wx.HORIZONTAL)
        device_label = wx.StaticText(parent, label=_('textgrid.device'))
        register(device_label, 'textgrid.device')
        self.tifa_device_choice = wx.Choice(parent, size=(400, -1))
        self.tifa_device_choice.Append(_('textgrid.device.cpu'), "cpu")
        self.tifa_device_choice.Append(_('textgrid.device.dml'), "dml")
        self.tifa_device_choice.Append(_('textgrid.device.webgpu'), "webgpu")
        self.tifa_device_choice.SetSelection(0)
        register(self.tifa_device_choice, '', 'choice',
                 [('textgrid.device.cpu', 'cpu'), ('textgrid.device.dml', 'dml'),
                  ('textgrid.device.webgpu', 'webgpu')])
        row.Add(device_label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        row.Add(self.tifa_device_choice, 1, wx.EXPAND | wx.ALL, 3)
        right_sizer.Add(row, 0, wx.EXPAND | wx.ALL, 3)

        row = wx.BoxSizer(wx.HORIZONTAL)
        batch_label = wx.StaticText(parent, label=_('textgrid.batch'))
        register(batch_label, 'textgrid.batch')
        self.tifa_batch_choice = wx.Choice(parent, size=(400, -1))
        self.tifa_batch_choice.SetToolTip(_('textgrid.batch.tooltip'))
        register(self.tifa_batch_choice, 'textgrid.batch.tooltip', 'tooltip')
        for value in (1, 2, 4, 8, 16, 32):
            self.tifa_batch_choice.Append(str(value), value)
        self.tifa_batch_choice.SetSelection(3)  # 8
        row.Add(batch_label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        row.Add(self.tifa_batch_choice, 1, wx.EXPAND | wx.ALL, 3)
        right_sizer.Add(row, 0, wx.EXPAND | wx.ALL, 3)

        row = wx.BoxSizer(wx.HORIZONTAL)
        skip_label = wx.StaticText(parent, label=_('textgrid.skip_penalty'))
        register(skip_label, 'textgrid.skip_penalty')
        self.tifa_skip_penalty_text = wx.TextCtrl(parent, value="0.5", size=(80, -1))
        self.tifa_skip_penalty_text.SetToolTip(_('textgrid.skip_penalty.tooltip'))
        register(self.tifa_skip_penalty_text, 'textgrid.skip_penalty.tooltip', 'tooltip')
        row.Add(skip_label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        row.Add(self.tifa_skip_penalty_text, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        right_sizer.Add(row, 0, wx.ALL, 3)

        # 推理后删除多余SP
        row = wx.BoxSizer(wx.HORIZONTAL)
        self.tifa_del_sp_checkbox = wx.CheckBox(parent, label=_('tifa.del_sp'))
        self.tifa_del_sp_checkbox.SetToolTip(_('tifa.del_sp.tooltip'))
        register(self.tifa_del_sp_checkbox, 'tifa.del_sp.tooltip', 'tooltip')
        register(self.tifa_del_sp_checkbox, 'tifa.del_sp', 'checkbox')
        self.tifa_del_sp_checkbox.SetValue(True)
        row.Add(self.tifa_del_sp_checkbox, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        right_sizer.Add(row, 0, wx.ALL, 3)

        content_sizer.Add(right_sizer, 1, wx.EXPAND | wx.ALL, 5)
        sizer.Add(content_sizer, 0, wx.EXPAND | wx.ALL, 10)

        # ── 按钮 ──
        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        infer_btn = wx.Button(parent, label=_('textgrid.infer'))
        register(infer_btn, 'textgrid.infer', 'button')
        infer_btn.Bind(wx.EVT_BUTTON, self.on_tifa_infer)
        btn_sizer.Add(infer_btn, 0, wx.ALL | wx.CENTER, 5)
        self.tifa_stop_btn = wx.Button(parent, label=_('textgrid.stop_infer'))
        register(self.tifa_stop_btn, 'textgrid.stop_infer', 'button')
        self.tifa_stop_btn.Bind(wx.EVT_BUTTON, self.on_tifa_stop)
        self.tifa_stop_btn.Disable()
        btn_sizer.Add(self.tifa_stop_btn, 0, wx.ALL | wx.CENTER, 5)
        sizer.Add(btn_sizer, 0, wx.ALL | wx.CENTER, 5)

        result_label = wx.StaticText(parent, label=_('textgrid.result'))
        register(result_label, 'textgrid.result')
        sizer.Add(result_label, 0, wx.ALL | wx.LEFT, 5)

        self.tifa_result_text = wx.TextCtrl(parent, style=wx.TE_MULTILINE | wx.TE_READONLY, size=(-1, 400))
        sizer.Add(self.tifa_result_text, 1, wx.EXPAND | wx.ALL, 5)

        parent.SetSizer(sizer)

    def _tifa_model_dirs(self):
        """返回 [(显示标签, 客户端数据)]：TIFA_model 下的每个子目录。
        若一个子目录都没有，则退回 TIFA_model 根目录本身，保证下拉里总有可用项。"""
        base = ROOT / 'TIFA_model'
        if not os.path.exists(base):
            return []
        subdirs = sorted([x for x in os.listdir(base) if os.path.isdir(os.path.join(base, x))])
        if subdirs:
            return [(d, d) for d in subdirs]
        return [(_('tifa.model_default'), '')]

    def load_tifa_models(self):
        """扫描 ROOT/TIFA_model 下的模型目录与字典文件。"""
        self.tifa_model_folder_choice.Clear()
        for label, data in self._tifa_model_dirs():
            self.tifa_model_folder_choice.Append(label, data)
        if self.tifa_model_folder_choice.GetCount():
            self.tifa_model_folder_choice.SetSelection(0)
            self.on_tifa_model_folder_selected(None)

    def _tifa_model_dir(self):
        data = self.tifa_model_folder_choice.GetClientData(self.tifa_model_folder_choice.GetSelection())
        if data:
            p = Path(data)
            return p if p.is_absolute() else (ROOT / 'TIFA_model' / p)
        return ROOT / 'TIFA_model'

    def on_tifa_browse_model_dir(self, event):
        """手动选择一个模型目录（不限于 TIFA_model 之下）。"""
        dialog = wx.DirDialog(None, _('msg.select_folder'), style=wx.DD_DEFAULT_STYLE)
        if dialog.ShowModal() == wx.ID_OK:
            path = dialog.GetPath()
            label = os.path.basename(path.rstrip('\\/')) or path
            idx = self.tifa_model_folder_choice.FindString(label)
            if idx == wx.NOT_FOUND:
                idx = self.tifa_model_folder_choice.Append(label, path)
            self.tifa_model_folder_choice.SetSelection(idx)
            self.on_tifa_model_folder_selected(None)
        dialog.Destroy()

    def on_tifa_model_folder_selected(self, event):
        model_dir = self._tifa_model_dir()
        self.tifa_dict_choice.Clear()
        if os.path.exists(model_dir):
            dict_files = sorted([f for f in os.listdir(model_dir) if f.endswith('.txt')])
            self.tifa_dict_choice.AppendItems(dict_files)
            if dict_files:
                self.tifa_dict_choice.SetSelection(0)

    def on_tifa_stop(self, event):
        """强制停止 TIFA 推理。"""
        self._tifa_stop.set()
        self.tifa_stop_btn.Disable()
        self.tifa_result_text.AppendText(_('log.infer_stop_requested'))

    def on_tifa_infer(self, event):
        wav_folder = self.tifa_folder_text.GetValue().strip()
        dict_file = self.tifa_dict_choice.GetStringSelection()

        if not wav_folder:
            wx.MessageBox(_('msg.err.select_folder'), _('msg.error'), wx.OK | wx.ICON_ERROR)
            return
        if not os.path.exists(wav_folder):
            wx.MessageBox(_('msg.err.folder_not_exist'), _('msg.error'), wx.OK | wx.ICON_ERROR)
            return
        if not dict_file:
            wx.MessageBox(_('msg.err.select_dict'), _('msg.error'), wx.OK | wx.ICON_ERROR)
            return

        model_dir = self._tifa_model_dir()
        dict_path = model_dir / dict_file
        if not os.path.exists(dict_path):
            wx.MessageBox(_('msg.err.dict_not_exist'), _('msg.error'), wx.OK | wx.ICON_ERROR)
            return
        if not (model_dir / 'model.onnx').exists():
            wx.MessageBox(_('msg.err.tifa_model_missing').format(path=str(model_dir)),
                          _('msg.error'), wx.OK | wx.ICON_ERROR)
            return

        device = self.tifa_device_choice.GetClientData(self.tifa_device_choice.GetSelection()) or 'cpu'
        batch_size = self.tifa_batch_choice.GetClientData(self.tifa_batch_choice.GetSelection()) or 8
        del_sp_enabled = self.tifa_del_sp_checkbox.GetValue()
        try:
            skip_penalty = float(self.tifa_skip_penalty_text.GetValue().strip())
        except ValueError:
            wx.MessageBox(_('msg.err.params_format'), _('msg.error'), wx.OK | wx.ICON_ERROR)
            return

        # 语言前缀：TIFA 词表自带语言前缀，解析时会自动回退，这里传 None 即可
        language = None

        def tifa_infer_thread():
            try:
                wx.CallAfter(self.tifa_result_text.Clear)
                self._tifa_stop.clear()
                self._tifa_stopped = False
                wx.CallAfter(self.tifa_stop_btn.Enable)

                def progress_callback(msg):
                    wx.CallAfter(self.tifa_result_text.AppendText, msg + "\n")
                    wx.CallAfter(self.tifa_result_text.ShowPosition, self.tifa_result_text.GetLastPosition())

                wx.CallAfter(self.tifa_result_text.AppendText, _('log.model_info').format(model=model_dir.name))
                wx.CallAfter(self.tifa_result_text.AppendText, _('log.dict_info').format(dict=dict_file))
                wx.CallAfter(self.tifa_result_text.AppendText,
                             _('log.tifa_batch').format(batch=batch_size, penalty=skip_penalty))

                inference = tifa_infer.TifaInference(model_dir)
                wx.CallAfter(self.tifa_result_text.AppendText, _('log.loading_config'))
                inference.load_config()
                wx.CallAfter(self.tifa_result_text.AppendText, _('log.loading_model_weights'))
                inference.load_model(device=device)
                inference.set_progress_callback(progress_callback)

                wx.CallAfter(self.tifa_result_text.AppendText, _('log.loading_dataset'))
                inference.get_dataset(wav_folder, language=language, g2p="dictionary",
                                      dictionary_path=str(dict_path), in_format="lab")

                wx.CallAfter(self.tifa_result_text.AppendText, _('log.start_infer'))
                try:
                    inference.infer(batch_size=batch_size, skip_penalty=skip_penalty,
                                    stop_event=self._tifa_stop)
                except tifa_infer.StopInference:
                    wx.CallAfter(self.tifa_result_text.AppendText, _('log.infer_stopped'))
                    wx.CallAfter(self.tifa_stop_btn.Disable)
                    return

                wx.CallAfter(self.tifa_result_text.AppendText, _('log.exporting'))
                inference.export(wav_folder)

                # ── 推理完成后可自动删除多余的 SP 音素 ──
                if del_sp_enabled:
                    wx.CallAfter(self.tifa_result_text.AppendText, _('log.del_sp_start'))
                    try:
                        with TextRedirector(self.tifa_result_text):
                            del_SP.process_all_textgrid_files(wav_folder, "AP,SP,EP", True)
                        wx.CallAfter(self.tifa_result_text.AppendText, _('log.del_sp_done'))
                    except Exception:
                        wx.CallAfter(self.tifa_result_text.AppendText,
                                     _('log.del_sp_failed').format(error=traceback.format_exc()))

                wx.CallAfter(self.tifa_result_text.AppendText, _('log.infer_complete'))
                wx.CallAfter(self.tifa_stop_btn.Disable)
                wx.CallAfter(wx.MessageBox, _('msg.ok.infer_complete'), _('msg.success'),
                             wx.OK | wx.ICON_INFORMATION)
            except Exception:
                tb = traceback.format_exc()
                wx.CallAfter(self.tifa_result_text.AppendText, _('log.infer_failed').format(error=tb))
                wx.CallAfter(self.tifa_stop_btn.Disable)
                wx.CallAfter(wx.MessageBox, _('log.infer_failed').format(error=tb),
                             _('msg.error'), wx.OK | wx.ICON_ERROR)

        thread = threading.Thread(target=tifa_infer_thread)
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

                auto_sp = self.auto_sp_checkbox.GetValue()
                drop_r = self.no_r_checkbox.GetValue()
                wx.CallAfter(self.lab_result_text.AppendText,
                             _('log.auto_sp_state').format(state=_('msg.on') if auto_sp else _('msg.off')))
                wx.CallAfter(self.lab_result_text.AppendText,
                             _('log.no_r_state').format(state=_('msg.on') if drop_r else _('msg.off')))

                with TextRedirector(self.lab_result_text):
                    wavname2lab.run(path, cuts, auto_sp, drop_r)

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

                drop_r = self.lab_index_no_r_checkbox.GetValue()
                wx.CallAfter(self.lab_index_result_text.AppendText,
                             _('log.no_r_state').format(state=_('msg.on') if drop_r else _('msg.off')))

                with TextRedirector(self.lab_index_result_text):
                    index2lab.run(wav_path, index_path, cuts, drop_r)

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

    def on_detect_short_phonemes(self, event):
        """检测 TextGrid 中时长过短的音素"""
        wav_folder = self.detect_path_text.GetValue().strip()
        threshold_str = self.detect_threshold_text.GetValue().strip()

        if not wav_folder:
            wx.MessageBox(_('msg.err.select_folder'), _('msg.error'), wx.OK | wx.ICON_ERROR)
            return

        if not os.path.exists(wav_folder):
            wx.MessageBox(_('msg.err.folder_not_exist'), _('msg.error'), wx.OK | wx.ICON_ERROR)
            return

        try:
            threshold_ms = float(threshold_str)
            if threshold_ms <= 0:
                raise ValueError
        except ValueError:
            wx.MessageBox(_('msg.err.detect_threshold'), _('msg.error'), wx.OK | wx.ICON_ERROR)
            return

        # 提前检查是否有 TextGrid
        has_textgrid = any(True for _ in Path(wav_folder).rglob('*.TextGrid'))
        if not has_textgrid:
            wx.MessageBox(_('msg.err.textgrid_not_exist'), _('msg.error'), wx.OK | wx.ICON_ERROR)
            return

        def detect_thread():
            try:
                wx.CallAfter(self.detect_result_text.Clear)
                wx.CallAfter(self.detect_result_text.AppendText,
                             _('log.detect.start').format(threshold=threshold_ms))

                # TextRedirector 会同时把 print 内容写入结果框和原命令行
                with TextRedirector(self.detect_result_text):
                    results, file_count = detect_short_phonemes.run(wav_folder, threshold_ms)

                wx.CallAfter(self.detect_result_text.AppendText,
                             _('log.detect.summary').format(count=len(results), files=file_count))
                wx.CallAfter(self.detect_result_text.ShowPosition,
                             self.detect_result_text.GetLastPosition())
                wx.CallAfter(wx.MessageBox,
                             _('msg.ok.detect_complete').format(count=len(results), files=file_count),
                             _('msg.success'), wx.OK | wx.ICON_INFORMATION)
            except Exception:
                tb = traceback.format_exc()
                wx.CallAfter(self.detect_result_text.AppendText, _('log.failed').format(error=tb))
                wx.CallAfter(wx.MessageBox, _('log.failed').format(error=tb), _('msg.error'), wx.OK | wx.ICON_ERROR)

        thread = threading.Thread(target=detect_thread)
        thread.start()

    def on_generate_json(self, event):
        wav_folder = self.json_path_text.GetValue().strip()
        dict_full_path = self.json_dict_text.GetValue().strip()
        ignore = self.json_ignore_text.GetValue().strip()
        recording_list_path = self.json_recording_text.GetValue().strip()

        if not wav_folder:
            wx.MessageBox(_('msg.err.select_wav_folder'), _('msg.error'), wx.OK | wx.ICON_ERROR)
            return

        if not os.path.exists(wav_folder):
            wx.MessageBox(_('msg.err.wav_folder_not_exist'), _('msg.error'), wx.OK | wx.ICON_ERROR)
            return

        if not dict_full_path:
            wx.MessageBox(_('msg.err.select_dict'), _('msg.error'), wx.OK | wx.ICON_ERROR)
            return

        if not os.path.isfile(dict_full_path):
            wx.MessageBox(_('msg.err.dict_not_exist'), _('msg.error'), wx.OK | wx.ICON_ERROR)
            return

        if recording_list_path and not os.path.exists(recording_list_path):
            wx.MessageBox(_('msg.err.recording_not_exist'), _('msg.error'), wx.OK | wx.ICON_ERROR)
            return
        
        def generate_json_thread():
            try:
                wx.CallAfter(self.json_result_text.Clear)
                wx.CallAfter(self.json_result_text.AppendText, _('log.start_generate_json'))

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