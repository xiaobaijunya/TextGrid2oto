import wx
import os
import sys
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
        super().__init__(None, title=_('window.title'), size=(960, 720))
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
        
        lab_wav_title = wx.StaticText(lab_wav_panel, label=_('lab.tab.wavname'))
        register(lab_wav_title, 'lab.tab.wavname')
        lab_wav_sizer.Add(lab_wav_title, 0, wx.ALL | wx.CENTER, 10)
        
        path_sizer = wx.BoxSizer(wx.HORIZONTAL)
        path_label = wx.StaticText(lab_wav_panel, label=_('lab.wavname.folder'))
        register(path_label, 'lab.wavname.folder')
        self.path_text = wx.TextCtrl(lab_wav_panel, size=(400, -1))
        browse_btn = wx.Button(lab_wav_panel, label=_('lab.wavname.browse_folder'))
        register(browse_btn, 'lab.wavname.browse_folder', 'button')
        browse_btn.Bind(wx.EVT_BUTTON, lambda event: self.on_browse_folder(event, self.path_text))
        path_sizer.Add(path_label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        path_sizer.Add(self.path_text, 1, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        path_sizer.Add(browse_btn, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        lab_wav_sizer.Add(path_sizer, 0, wx.EXPAND | wx.ALL, 10)
        
        separator_sizer = wx.BoxSizer(wx.HORIZONTAL)
        separator_label = wx.StaticText(lab_wav_panel, label=_('lab.wavname.separator'))
        register(separator_label, 'lab.wavname.separator')
        self.separator_text = wx.TextCtrl(lab_wav_panel, value="_,-", size=(200, -1))
        separator_sizer.Add(separator_label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        separator_sizer.Add(self.separator_text, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        lab_wav_sizer.Add(separator_sizer, 0, wx.ALL, 10)
        
        generate_btn = wx.Button(lab_wav_panel, label=_('lab.wavname.generate'))
        register(generate_btn, 'lab.wavname.generate', 'button')
        generate_btn.Bind(wx.EVT_BUTTON, self.on_generate_lab)
        lab_wav_sizer.Add(generate_btn, 0, wx.ALL | wx.CENTER, 10)
        
        # 结果显示文本框
        lab_result_label = wx.StaticText(lab_wav_panel, label=_('lab.wavname.result'))
        register(lab_result_label, 'lab.wavname.result')
        lab_wav_sizer.Add(lab_result_label, 0, wx.ALL | wx.LEFT, 10)
        
        self.lab_result_text = wx.TextCtrl(lab_wav_panel, style=wx.TE_MULTILINE | wx.TE_READONLY, size=(-1, 400))
        lab_wav_sizer.Add(self.lab_result_text, 1, wx.EXPAND | wx.ALL, 10)
        
        lab_wav_panel.SetSizer(lab_wav_sizer)
        
        # 第二个标签页：根据index生成lab
        lab_index_panel = wx.Panel(lab_notebook)
        lab_index_sizer = wx.BoxSizer(wx.VERTICAL)
        
        lab_index_title = wx.StaticText(lab_index_panel, label=_('lab.tab.index'))
        register(lab_index_title, 'lab.tab.index')
        lab_index_sizer.Add(lab_index_title, 0, wx.ALL | wx.CENTER, 10)
        
        # WAV路径框
        lab_wav_path_sizer = wx.BoxSizer(wx.HORIZONTAL)
        lab_wav_path_label = wx.StaticText(lab_index_panel, label=_('lab.index.wav_path'))
        register(lab_wav_path_label, 'lab.index.wav_path')
        self.lab_wav_path_text = wx.TextCtrl(lab_index_panel, size=(400, -1))
        lab_wav_browse_btn = wx.Button(lab_index_panel, label=_('lab.index.browse_wav'))
        register(lab_wav_browse_btn, 'lab.index.browse_wav', 'button')
        lab_wav_browse_btn.Bind(wx.EVT_BUTTON, lambda event: self.on_browse_folder(event, self.lab_wav_path_text))
        lab_wav_path_sizer.Add(lab_wav_path_label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        lab_wav_path_sizer.Add(self.lab_wav_path_text, 1, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        lab_wav_path_sizer.Add(lab_wav_browse_btn, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        lab_index_sizer.Add(lab_wav_path_sizer, 0, wx.EXPAND | wx.ALL, 10)
        
        # Index路径框
        lab_index_path_sizer = wx.BoxSizer(wx.HORIZONTAL)
        lab_index_path_label = wx.StaticText(lab_index_panel, label=_('lab.index.index_path'))
        register(lab_index_path_label, 'lab.index.index_path')
        self.lab_index_path_text = wx.TextCtrl(lab_index_panel, size=(400, -1))
        lab_index_browse_btn = wx.Button(lab_index_panel, label=_('lab.index.browse_file'))
        register(lab_index_browse_btn, 'lab.index.browse_file', 'button')
        lab_index_browse_btn.Bind(wx.EVT_BUTTON, lambda event: self.on_browse_file(event, self.lab_index_path_text))
        lab_index_path_sizer.Add(lab_index_path_label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        lab_index_path_sizer.Add(self.lab_index_path_text, 1, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        lab_index_path_sizer.Add(lab_index_browse_btn, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        lab_index_sizer.Add(lab_index_path_sizer, 0, wx.EXPAND | wx.ALL, 10)
        
        # 分隔符输入框
        lab_separator_sizer = wx.BoxSizer(wx.HORIZONTAL)
        lab_separator_label = wx.StaticText(lab_index_panel, label=_('lab.index.separator'))
        register(lab_separator_label, 'lab.index.separator')
        self.lab_separator_text = wx.TextCtrl(lab_index_panel, value="_,-", size=(200, -1))
        lab_separator_sizer.Add(lab_separator_label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        lab_separator_sizer.Add(self.lab_separator_text, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        lab_index_sizer.Add(lab_separator_sizer, 0, wx.ALL, 10)

        lab_index_generate_btn = wx.Button(lab_index_panel, label=_('lab.index.generate'))
        register(lab_index_generate_btn, 'lab.index.generate', 'button')
        lab_index_generate_btn.Bind(wx.EVT_BUTTON, self.on_generate_lab_from_index)
        lab_index_sizer.Add(lab_index_generate_btn, 0, wx.ALL | wx.CENTER, 10)
        
        # 结果显示文本框
        lab_index_result_label = wx.StaticText(lab_index_panel, label=_('lab.index.result'))
        register(lab_index_result_label, 'lab.index.result')
        lab_index_sizer.Add(lab_index_result_label, 0, wx.ALL | wx.LEFT, 10)
        
        self.lab_index_result_text = wx.TextCtrl(lab_index_panel, style=wx.TE_MULTILINE | wx.TE_READONLY, size=(-1, 400))
        lab_index_sizer.Add(self.lab_index_result_text, 1, wx.EXPAND | wx.ALL, 10)
        
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
        textgrid_sizer.Add(textgrid_title, 0, wx.ALL | wx.CENTER, 10)

        # 顶部：音源文件夹（占据整个宽度）
        folder_sizer = wx.BoxSizer(wx.HORIZONTAL)
        folder_label = wx.StaticText(textgrid_panel, label=_('textgrid.folder'))
        register(folder_label, 'textgrid.folder')
        self.textgrid_folder_text = wx.TextCtrl(textgrid_panel, size=(500, -1))
        browse_folder_btn = wx.Button(textgrid_panel, label=_('textgrid.browse_folder'))
        register(browse_folder_btn, 'textgrid.browse_folder', 'button')
        browse_folder_btn.Bind(wx.EVT_BUTTON, lambda event: self.on_browse_folder(event, self.textgrid_folder_text))
        folder_sizer.Add(folder_label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        folder_sizer.Add(self.textgrid_folder_text, 1, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        folder_sizer.Add(browse_folder_btn, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        textgrid_sizer.Add(folder_sizer, 0, wx.EXPAND | wx.ALL, 10)

        # 主内容区域：左右两栏
        main_content_sizer = wx.BoxSizer(wx.HORIZONTAL)

        # 左侧栏
        left_panel = wx.Panel(textgrid_panel)
        left_sizer = wx.BoxSizer(wx.VERTICAL)

        model_folder_sizer = wx.BoxSizer(wx.HORIZONTAL)
        model_folder_label = wx.StaticText(left_panel, label=_('textgrid.model_folder'))
        register(model_folder_label, 'textgrid.model_folder')
        self.model_folder_choice = wx.Choice(left_panel, size=(250, -1))
        self.model_folder_choice.Bind(wx.EVT_CHOICE, self.on_model_folder_selected)
        model_folder_sizer.Add(model_folder_label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        model_folder_sizer.Add(self.model_folder_choice, 1, wx.EXPAND | wx.ALL, 5)
        left_sizer.Add(model_folder_sizer, 0, wx.ALL, 10)

        model_file_sizer = wx.BoxSizer(wx.HORIZONTAL)
        model_file_label = wx.StaticText(left_panel, label=_('textgrid.model_file'))
        register(model_file_label, 'textgrid.model_file')
        self.model_file_choice = wx.Choice(left_panel, size=(250, -1))
        model_file_sizer.Add(model_file_label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        model_file_sizer.Add(self.model_file_choice, 1, wx.EXPAND | wx.ALL, 5)
        left_sizer.Add(model_file_sizer, 0, wx.ALL, 10)

        dict_sizer = wx.BoxSizer(wx.HORIZONTAL)
        dict_label = wx.StaticText(left_panel, label=_('textgrid.dict'))
        register(dict_label, 'textgrid.dict')
        self.dict_choice = wx.Choice(left_panel, size=(250, -1))
        dict_sizer.Add(dict_label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        dict_sizer.Add(self.dict_choice, 1, wx.EXPAND | wx.ALL, 5)
        left_sizer.Add(dict_sizer, 0, wx.ALL, 10)

        left_panel.SetSizer(left_sizer)
        main_content_sizer.Add(left_panel, 1, wx.EXPAND | wx.ALL, 5)

        # 中间分隔线
        separator = wx.StaticLine(textgrid_panel, style=wx.LI_VERTICAL)
        main_content_sizer.Add(separator, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 5)

        # 右侧栏：推理设置
        right_panel = wx.Panel(textgrid_panel)
        right_sizer = wx.BoxSizer(wx.VERTICAL)

        # 设备选择
        device_sizer = wx.BoxSizer(wx.HORIZONTAL)
        device_label = wx.StaticText(right_panel, label=_('textgrid.device'))
        register(device_label, 'textgrid.device')
        self.device_choice = wx.Choice(right_panel, size=(400, -1))
        self.device_choice.Append(_('textgrid.device.cpu'), "cpu")
        self.device_choice.Append(_('textgrid.device.dml'), "dml")
        self.device_choice.SetSelection(0)
        register(self.device_choice, '', 'choice', [('textgrid.device.cpu', 'cpu'), ('textgrid.device.dml', 'dml')])
        device_sizer.Add(device_label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        device_sizer.Add(self.device_choice, 1, wx.EXPAND | wx.ALL, 5)
        right_sizer.Add(device_sizer, 0, wx.ALL, 10)

        # pad_times选择
        pad_times_sizer = wx.BoxSizer(wx.HORIZONTAL)
        pad_times_label = wx.StaticText(right_panel, label=_('textgrid.pad_times'))
        register(pad_times_label, 'textgrid.pad_times')
        self.pad_times_choice = wx.Choice(right_panel, size=(400, -1))
        self.pad_times_choice.SetToolTip(_('textgrid.pad_times.tooltip'))
        register(self.pad_times_choice, 'textgrid.pad_times.tooltip', 'tooltip')
        self.pad_times_choice.Append(_('textgrid.pad_times.1'), 1)
        self.pad_times_choice.Append(_('textgrid.pad_times.3'), 3)
        self.pad_times_choice.Append(_('textgrid.pad_times.5'), 5)
        self.pad_times_choice.SetSelection(1)
        register(self.pad_times_choice, '', 'choice', [('textgrid.pad_times.1', 1), ('textgrid.pad_times.3', 3), ('textgrid.pad_times.5', 5)])
        pad_times_sizer.Add(pad_times_label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        pad_times_sizer.Add(self.pad_times_choice, 1, wx.EXPAND | wx.ALL, 5)
        right_sizer.Add(pad_times_sizer, 0, wx.ALL, 10)

        # pad_length选择
        pad_length_sizer = wx.BoxSizer(wx.HORIZONTAL)
        pad_length_label = wx.StaticText(right_panel, label=_('textgrid.pad_length'))
        register(pad_length_label, 'textgrid.pad_length')
        self.pad_length_choice = wx.Choice(right_panel, size=(400, -1))
        self.pad_length_choice.SetToolTip(_('textgrid.pad_length.tooltip'))
        register(self.pad_length_choice, 'textgrid.pad_length.tooltip', 'tooltip')
        self.pad_length_choice.Append(_('textgrid.pad_length.3'), 3)
        self.pad_length_choice.Append(_('textgrid.pad_length.5'), 5)
        self.pad_length_choice.Append(_('textgrid.pad_length.7'), 7)
        self.pad_length_choice.Append(_('textgrid.pad_length.10'), 10)
        self.pad_length_choice.SetSelection(1)
        register(self.pad_length_choice, '', 'choice', [('textgrid.pad_length.3', 3), ('textgrid.pad_length.5', 5), ('textgrid.pad_length.7', 7), ('textgrid.pad_length.10', 10)])
        pad_length_sizer.Add(pad_length_label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        pad_length_sizer.Add(self.pad_length_choice, 1, wx.EXPAND | wx.ALL, 5)
        right_sizer.Add(pad_length_sizer, 0, wx.ALL, 10)

        # 合并重复音素选项
        merge_phonemes_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.merge_phonemes_checkbox = wx.CheckBox(right_panel, label=_('textgrid.merge_phonemes'))
        self.merge_phonemes_checkbox.SetToolTip(_('textgrid.merge_phonemes.tooltip'))
        register(self.merge_phonemes_checkbox, 'textgrid.merge_phonemes.tooltip', 'tooltip')
        register(self.merge_phonemes_checkbox, 'textgrid.merge_phonemes', 'checkbox')
        self.merge_phonemes_checkbox.SetValue(True)
        merge_phonemes_sizer.Add(self.merge_phonemes_checkbox, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        right_sizer.Add(merge_phonemes_sizer, 0, wx.ALL, 10)


        right_panel.SetSizer(right_sizer)
        main_content_sizer.Add(right_panel, 1, wx.EXPAND | wx.ALL, 5)

        textgrid_sizer.Add(main_content_sizer, 0, wx.EXPAND | wx.ALL, 10)

        infer_btn = wx.Button(textgrid_panel, label=_('textgrid.infer'))
        register(infer_btn, 'textgrid.infer', 'button')
        infer_btn.Bind(wx.EVT_BUTTON, self.on_infer)
        textgrid_sizer.Add(infer_btn, 0, wx.ALL | wx.CENTER, 10)

        # 结果显示文本框
        infer_result_label = wx.StaticText(textgrid_panel, label=_('textgrid.result'))
        register(infer_result_label, 'textgrid.result')
        textgrid_sizer.Add(infer_result_label, 0, wx.ALL | wx.LEFT, 10)

        self.infer_result_text = wx.TextCtrl(textgrid_panel, style=wx.TE_MULTILINE | wx.TE_READONLY, size=(-1, 400))
        textgrid_sizer.Add(self.infer_result_text, 1, wx.EXPAND | wx.ALL, 10)

        textgrid_panel.SetSizer(textgrid_sizer)

        # JSON生成面板
        json_panel = wx.Panel(notebook)
        json_sizer = wx.BoxSizer(wx.VERTICAL)
        
        json_title = wx.StaticText(json_panel, label=_('json.title'))
        register(json_title, 'json.title')
        json_sizer.Add(json_title, 0, wx.ALL | wx.CENTER, 10)
        
        # WAV路径框
        json_path_sizer = wx.BoxSizer(wx.HORIZONTAL)
        json_path_label = wx.StaticText(json_panel, label=_('json.wav_folder'))
        register(json_path_label, 'json.wav_folder')
        self.json_path_text = wx.TextCtrl(json_panel, size=(400, -1))
        json_browse_btn = wx.Button(json_panel, label=_('json.browse_folder'))
        register(json_browse_btn, 'json.browse_folder', 'button')
        json_browse_btn.Bind(wx.EVT_BUTTON, lambda event: self.on_browse_folder(event, self.json_path_text))
        json_path_sizer.Add(json_path_label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        json_path_sizer.Add(self.json_path_text, 1, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        json_path_sizer.Add(json_browse_btn, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        json_sizer.Add(json_path_sizer, 0, wx.EXPAND | wx.ALL, 10)

        json_folder_sizer = wx.BoxSizer(wx.HORIZONTAL)
        json_folder_label = wx.StaticText(json_panel, label=_('json.model_folder'))
        register(json_folder_label, 'json.model_folder')
        self.json_folder_choice = wx.Choice(json_panel, size=(300, -1))
        self.json_folder_choice.Bind(wx.EVT_CHOICE, self.on_json_folder_selected)
        json_folder_sizer.Add(json_folder_label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        json_folder_sizer.Add(self.json_folder_choice, 1, wx.EXPAND | wx.ALL, 5)
        json_sizer.Add(json_folder_sizer, 0, wx.ALL, 10)

        # 模型字典选择
        json_dict_sizer = wx.BoxSizer(wx.HORIZONTAL)
        json_dict_label = wx.StaticText(json_panel, label=_('json.model_dict'))
        register(json_dict_label, 'json.model_dict')
        self.json_dict_choice = wx.Choice(json_panel, size=(300, -1))
        self.json_dict_choice.Bind(wx.EVT_CHOICE, self.on_json_dict_selected)
        json_dict_sizer.Add(json_dict_label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        json_dict_sizer.Add(self.json_dict_choice, 1, wx.EXPAND | wx.ALL, 5)
        json_sizer.Add(json_dict_sizer, 0, wx.ALL, 10)
        
        # 忽略音素
        json_ignore_sizer = wx.BoxSizer(wx.HORIZONTAL)
        json_ignore_label = wx.StaticText(json_panel, label=_('json.ignore'))
        register(json_ignore_label, 'json.ignore')
        self.json_ignore_text = wx.TextCtrl(json_panel, value="AP,SP,EP,R", size=(200, -1))
        json_ignore_sizer.Add(json_ignore_label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        json_ignore_sizer.Add(self.json_ignore_text, 1, wx.EXPAND | wx.ALL, 5)
        json_sizer.Add(json_ignore_sizer, 0, wx.ALL, 10)

        # 录音表路径
        json_recording_sizer = wx.BoxSizer(wx.HORIZONTAL)
        json_recording_label = wx.StaticText(json_panel, label=_('json.recording'))
        register(json_recording_label, 'json.recording')
        self.json_recording_text = wx.TextCtrl(json_panel, value="", size=(300, -1))
        json_recording_browse_btn = wx.Button(json_panel, label=_('json.browse'), size=(60, -1))
        register(json_recording_browse_btn, 'json.browse', 'button')
        json_recording_browse_btn.Bind(wx.EVT_BUTTON, self.on_browse_json_recording)
        json_recording_sizer.Add(json_recording_label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        json_recording_sizer.Add(self.json_recording_text, 1, wx.EXPAND | wx.ALL, 5)
        json_recording_sizer.Add(json_recording_browse_btn, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        json_sizer.Add(json_recording_sizer, 0, wx.ALL, 10)

        # 生成JSON按钮
        generate_json_btn = wx.Button(json_panel, label=_('json.generate'))
        register(generate_json_btn, 'json.generate', 'button')
        generate_json_btn.Bind(wx.EVT_BUTTON, self.on_generate_json)
        json_sizer.Add(generate_json_btn, 0, wx.ALL | wx.CENTER, 10)
        
        # 结果显示文本框
        json_result_label = wx.StaticText(json_panel, label=_('json.result'))
        register(json_result_label, 'json.result')
        json_sizer.Add(json_result_label, 0, wx.ALL | wx.LEFT, 10)
        
        self.json_result_text = wx.TextCtrl(json_panel, style=wx.TE_MULTILINE | wx.TE_READONLY, size=(-1, 400))
        json_sizer.Add(self.json_result_text, 1, wx.EXPAND | wx.ALL, 10)
        
        json_panel.SetSizer(json_sizer)

        mark_panel = wx.Panel(notebook)
        mark_sizer = wx.BoxSizer(wx.VERTICAL)

        mark_notebook = wx.Notebook(mark_panel)

        oto_panel = wx.Panel(mark_notebook)
        oto_sizer = wx.BoxSizer(wx.VERTICAL)
        
        oto_title = wx.StaticText(oto_panel, label=_('mark.oto.title'))
        register(oto_title, 'mark.oto.title')
        oto_sizer.Add(oto_title, 0, wx.ALL | wx.CENTER, 10)
        
        # WAV路径框
        oto_path_sizer = wx.BoxSizer(wx.HORIZONTAL)
        oto_path_label = wx.StaticText(oto_panel, label=_('mark.oto.folder'))
        register(oto_path_label, 'mark.oto.folder')
        self.oto_path_text = wx.TextCtrl(oto_panel, size=(400, -1))
        oto_browse_btn = wx.Button(oto_panel, label=_('mark.oto.browse_folder'))
        register(oto_browse_btn, 'mark.oto.browse_folder', 'button')
        oto_browse_btn.Bind(wx.EVT_BUTTON, lambda event: self.on_browse_folder(event, self.oto_path_text))
        oto_path_sizer.Add(oto_path_label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        oto_path_sizer.Add(self.oto_path_text, 1, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        oto_path_sizer.Add(oto_browse_btn, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        oto_sizer.Add(oto_path_sizer, 0, wx.EXPAND | wx.ALL, 10)
        
        # Presamp路径和预设
        presamp_sizer = wx.BoxSizer(wx.HORIZONTAL)
        presamp_path_label = wx.StaticText(oto_panel, label=_('mark.oto.presamp_path'))
        register(presamp_path_label, 'mark.oto.presamp_path')
        self.oto_presamp_path_text = wx.TextCtrl(oto_panel, size=(300, -1))
        presamp_browse_btn = wx.Button(oto_panel, label=_('mark.oto.browse'))
        register(presamp_browse_btn, 'mark.oto.browse', 'button')
        presamp_browse_btn.Bind(wx.EVT_BUTTON, lambda event: self.on_browse_file(event, self.oto_presamp_path_text))
        presamp_preset_label = wx.StaticText(oto_panel, label=_('mark.oto.presamp_preset'))
        register(presamp_preset_label, 'mark.oto.presamp_preset')
        self.oto_presamp_choice = wx.Choice(oto_panel, size=(150, -1))
        self.oto_presamp_choice.Bind(wx.EVT_CHOICE, self.on_oto_presamp_selected)
        presamp_sizer.Add(presamp_path_label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        presamp_sizer.Add(self.oto_presamp_path_text, 1, wx.EXPAND | wx.ALL, 5)
        presamp_sizer.Add(presamp_browse_btn, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        presamp_sizer.Add(presamp_preset_label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        presamp_sizer.Add(self.oto_presamp_choice, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        oto_sizer.Add(presamp_sizer, 0, wx.EXPAND | wx.ALL, 10)
        
        # oto模板路径
        oto_preset_sizer = wx.BoxSizer(wx.HORIZONTAL)
        oto_preset_label = wx.StaticText(oto_panel, label=_('mark.oto.template'))
        register(oto_preset_label, 'mark.oto.template')
        self.oto_preset_text = wx.TextCtrl(oto_panel, value="", size=(300, -1))
        oto_preset_browse_btn = wx.Button(oto_panel, label=_('mark.oto.browse'))
        register(oto_preset_browse_btn, 'mark.oto.browse', 'button')
        oto_preset_browse_btn.Bind(wx.EVT_BUTTON, lambda event: self.on_browse_file(event, self.oto_preset_text))
        oto_preset_sizer.Add(oto_preset_label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        oto_preset_sizer.Add(self.oto_preset_text, 1, wx.EXPAND | wx.ALL, 5)
        oto_preset_sizer.Add(oto_preset_browse_btn, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        oto_sizer.Add(oto_preset_sizer, 0, wx.EXPAND | wx.ALL, 10)
        
        # 生成模式和编码
        mode_encoding_sizer = wx.BoxSizer(wx.HORIZONTAL)
        oto_mode_label = wx.StaticText(oto_panel, label=_('mark.oto.mode'))
        register(oto_mode_label, 'mark.oto.mode')
        self.oto_mode_choice = wx.Choice(oto_panel, choices=["CVVC", "VCV", "CVV","ARPAsing", "Test"], size=(150, -1))
        self.oto_mode_choice.SetSelection(0)
        self.oto_mode_choice.Bind(wx.EVT_CHOICE, self.on_oto_mode_changed)
        oto_encoding_label = wx.StaticText(oto_panel, label=_('mark.oto.encoding'))
        register(oto_encoding_label, 'mark.oto.encoding')
        self.oto_encoding_choice = wx.Choice(oto_panel, choices=["utf-8", "shift-jis", "gbk"], size=(150, -1))
        self.oto_encoding_choice.SetSelection(0)
        self.oto_cover_checkbox = wx.CheckBox(oto_panel, label=_('mark.oto.cover'))
        register(self.oto_cover_checkbox, 'mark.oto.cover', 'checkbox')
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
        oto_cv_sum_label = wx.StaticText(oto_panel, label=_('mark.oto.cv_params'))
        register(oto_cv_sum_label, 'mark.oto.cv_params')
        self.oto_cv_sum_text = wx.TextCtrl(oto_panel, value="1,3,1.5,1,4", size=(120, -1))
        self.oto_cv_sum_text.SetToolTip(_('mark.oto.params_tooltip'))
        register(self.oto_cv_sum_text, 'mark.oto.params_tooltip', 'tooltip')
        oto_cv_sum_sizer.Add(oto_cv_sum_label, 0, wx.ALL, 2)
        oto_cv_sum_sizer.Add(self.oto_cv_sum_text, 0, wx.ALL, 2)
        
        oto_vc_sum_sizer = wx.BoxSizer(wx.VERTICAL)
        oto_vc_sum_label = wx.StaticText(oto_panel, label=_('mark.oto.vc_params'))
        register(oto_vc_sum_label, 'mark.oto.vc_params')
        self.oto_vc_sum_text = wx.TextCtrl(oto_panel, value="3,0,2,1,3", size=(120, -1))
        self.oto_vc_sum_text.SetToolTip(_('mark.oto.params_tooltip'))
        register(self.oto_vc_sum_text, 'mark.oto.params_tooltip', 'tooltip')
        oto_vc_sum_sizer.Add(oto_vc_sum_label, 0, wx.ALL, 2)
        oto_vc_sum_sizer.Add(self.oto_vc_sum_text, 0, wx.ALL, 2)
        
        oto_vv_sum_sizer = wx.BoxSizer(wx.VERTICAL)
        oto_vv_sum_label = wx.StaticText(oto_panel, label=_('mark.oto.vv_params'))
        register(oto_vv_sum_label, 'mark.oto.vv_params')
        self.oto_vv_sum_text = wx.TextCtrl(oto_panel, value="3,3,1.5,1,3", size=(120, -1))
        self.oto_vv_sum_text.SetToolTip(_('mark.oto.params_tooltip'))
        register(self.oto_vv_sum_text, 'mark.oto.params_tooltip', 'tooltip')
        oto_vv_sum_sizer.Add(oto_vv_sum_label, 0, wx.ALL, 2)
        oto_vv_sum_sizer.Add(self.oto_vv_sum_text, 0, wx.ALL, 2)
        
        params1_sizer.Add(oto_cv_sum_sizer, 0, wx.ALL, 5)
        params1_sizer.Add(oto_vc_sum_sizer, 0, wx.ALL, 5)
        params1_sizer.Add(oto_vv_sum_sizer, 0, wx.ALL, 5)
        oto_sizer.Add(params1_sizer, 0, wx.ALL, 0)
        
        # 参数组2：CV偏移、VC偏移、音阶后缀

        oto_cv_offset_sizer = wx.BoxSizer(wx.VERTICAL)
        oto_cv_offset_label = wx.StaticText(oto_panel, label=_('mark.oto.cv_offset'))
        register(oto_cv_offset_label, 'mark.oto.cv_offset')
        self.oto_cv_offset_text = wx.TextCtrl(oto_panel, value="0,0,0,0,0", size=(120, -1))
        oto_cv_offset_sizer.Add(oto_cv_offset_label, 0, wx.ALL, 2)
        oto_cv_offset_sizer.Add(self.oto_cv_offset_text, 0, wx.ALL, 2)
        
        oto_vc_offset_sizer = wx.BoxSizer(wx.VERTICAL)
        oto_vc_offset_label = wx.StaticText(oto_panel, label=_('mark.oto.vc_offset'))
        register(oto_vc_offset_label, 'mark.oto.vc_offset')
        self.oto_vc_offset_text = wx.TextCtrl(oto_panel, value="0,0,0,0,0", size=(120, -1))
        oto_vc_offset_sizer.Add(oto_vc_offset_label, 0, wx.ALL, 2)
        oto_vc_offset_sizer.Add(self.oto_vc_offset_text, 0, wx.ALL, 2)

        params1_sizer.Add(oto_cv_offset_sizer, 0, wx.ALL, 5)
        params1_sizer.Add(oto_vc_offset_sizer, 0, wx.ALL, 5)

        oto_cv_repeat_sizer = wx.BoxSizer(wx.VERTICAL)
        oto_cv_repeat_label = wx.StaticText(oto_panel, label=_('mark.oto.cv_repeat'))
        register(oto_cv_repeat_label, 'mark.oto.cv_repeat')
        self.oto_cv_repeat_text = wx.TextCtrl(oto_panel, value="1", size=(80, -1))
        oto_cv_repeat_sizer.Add(oto_cv_repeat_label, 0, wx.ALL, 2)
        oto_cv_repeat_sizer.Add(self.oto_cv_repeat_text, 0, wx.ALL, 2)
        
        oto_vc_repeat_sizer = wx.BoxSizer(wx.VERTICAL)
        oto_vc_repeat_label = wx.StaticText(oto_panel, label=_('mark.oto.vc_repeat'))
        register(oto_vc_repeat_label, 'mark.oto.vc_repeat')
        self.oto_vc_repeat_text = wx.TextCtrl(oto_panel, value="1", size=(80, -1))
        oto_vc_repeat_sizer.Add(oto_vc_repeat_label, 0, wx.ALL, 2)
        oto_vc_repeat_sizer.Add(self.oto_vc_repeat_text, 0, wx.ALL, 2)
        
        oto_ignore_sizer = wx.BoxSizer(wx.VERTICAL)
        oto_ignore_label = wx.StaticText(oto_panel, label=_('mark.oto.ignore'))
        register(oto_ignore_label, 'mark.oto.ignore')
        self.oto_ignore_text = wx.TextCtrl(oto_panel, value="AP,SP,EP,R", size=(130, -1))
        oto_ignore_sizer.Add(oto_ignore_label, 0, wx.ALL, 2)
        oto_ignore_sizer.Add(self.oto_ignore_text, 0, wx.ALL, 2)

        params2_sizer = wx.BoxSizer(wx.HORIZONTAL)
        oto_pitch_sizer = wx.BoxSizer(wx.VERTICAL)
        oto_pitch_label = wx.StaticText(oto_panel, label=_('mark.oto.pitch'))
        register(oto_pitch_label, 'mark.oto.pitch')
        self.oto_pitch_text = wx.TextCtrl(oto_panel, value="", size=(100, -1))
        oto_pitch_sizer.Add(oto_pitch_label, 0, wx.ALL, 2)
        oto_pitch_sizer.Add(self.oto_pitch_text, 0, wx.ALL, 2)

        params2_sizer.Add(oto_cv_repeat_sizer, 0, wx.ALL, 5)
        params2_sizer.Add(oto_vc_repeat_sizer, 0, wx.ALL, 5)
        params2_sizer.Add(oto_ignore_sizer, 0, wx.ALL, 5)
        params2_sizer.Add(oto_pitch_sizer, 0, wx.ALL, 5)
        oto_sizer.Add(params2_sizer, 0, wx.ALL, 10)


        

        
        # 生成OTO按钮
        oto_generate_btn = wx.Button(oto_panel, label=_('mark.oto.generate'))
        register(oto_generate_btn, 'mark.oto.generate', 'button')
        oto_generate_btn.Bind(wx.EVT_BUTTON, self.on_generate_oto)
        oto_sizer.Add(oto_generate_btn, 0, wx.ALL | wx.CENTER, 10)
        
        # 结果显示文本框
        oto_result_label = wx.StaticText(oto_panel, label=_('mark.oto.result'))
        register(oto_result_label, 'mark.oto.result')
        oto_sizer.Add(oto_result_label, 0, wx.ALL | wx.LEFT, 0)
        
        self.oto_result_text = wx.TextCtrl(oto_panel, style=wx.TE_MULTILINE | wx.TE_READONLY, size=(-1, 400))
        oto_sizer.Add(self.oto_result_text, 1, wx.EXPAND | wx.ALL, 10)
        
        oto_panel.SetSizer(oto_sizer)

        svdb_panel = wx.Panel(mark_notebook)
        svdb_sizer = wx.BoxSizer(wx.VERTICAL)
        
        svdb_title = wx.StaticText(svdb_panel, label=_('mark.svdb.title'))
        register(svdb_title, 'mark.svdb.title')
        svdb_sizer.Add(svdb_title, 0, wx.ALL | wx.CENTER, 10)
        
        svdb_path_sizer = wx.BoxSizer(wx.HORIZONTAL)
        svdb_path_label = wx.StaticText(svdb_panel, label=_('mark.svdb.folder'))
        register(svdb_path_label, 'mark.svdb.folder')
        self.svdb_path_text = wx.TextCtrl(svdb_panel, size=(400, -1))
        svdb_browse_btn = wx.Button(svdb_panel, label=_('mark.svdb.browse_folder'))
        register(svdb_browse_btn, 'mark.svdb.browse_folder', 'button')
        svdb_browse_btn.Bind(wx.EVT_BUTTON, lambda event: self.on_browse_folder(event, self.svdb_path_text))
        svdb_path_sizer.Add(svdb_path_label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        svdb_path_sizer.Add(self.svdb_path_text, 1, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        svdb_path_sizer.Add(svdb_browse_btn, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        svdb_sizer.Add(svdb_path_sizer, 0, wx.EXPAND | wx.ALL, 10)
        
        svdb_dict_sizer = wx.BoxSizer(wx.HORIZONTAL)
        svdb_dict_label = wx.StaticText(svdb_panel, label=_('mark.svdb.dict'))
        register(svdb_dict_label, 'mark.svdb.dict')
        self.svdb_dict_choice = wx.Choice(svdb_panel, size=(300, -1))
        svdb_dict_sizer.Add(svdb_dict_label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        svdb_dict_sizer.Add(self.svdb_dict_choice, 1, wx.EXPAND | wx.ALL, 5)
        svdb_sizer.Add(svdb_dict_sizer, 0, wx.ALL, 10)

        svdb_tail_ratio_sizer = wx.BoxSizer(wx.HORIZONTAL)
        svdb_tail_ratio_label = wx.StaticText(svdb_panel, label=_('mark.svdb.tail_ratio'))
        register(svdb_tail_ratio_label, 'mark.svdb.tail_ratio')
        self.svdb_tail_ratio_text = wx.TextCtrl(svdb_panel, value="50", size=(40, -1))
        svdb_tail_ratio_percent = wx.StaticText(svdb_panel, label=_('mark.svdb.tail_percent'))
        svdb_tail_ratio_sizer.Add(svdb_tail_ratio_label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        svdb_tail_ratio_sizer.Add(self.svdb_tail_ratio_text, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        svdb_tail_ratio_sizer.Add(svdb_tail_ratio_percent, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        svdb_sizer.Add(svdb_tail_ratio_sizer, 0, wx.ALL, 10)
        
        svdb_generate_btn = wx.Button(svdb_panel, label=_('mark.svdb.generate'))
        register(svdb_generate_btn, 'mark.svdb.generate', 'button')
        svdb_generate_btn.Bind(wx.EVT_BUTTON, self.on_generate_svdb)
        svdb_sizer.Add(svdb_generate_btn, 0, wx.ALL | wx.CENTER, 10)
        
        svdb_result_label = wx.StaticText(svdb_panel, label=_('mark.svdb.result'))
        register(svdb_result_label, 'mark.svdb.result')
        svdb_sizer.Add(svdb_result_label, 0, wx.ALL | wx.LEFT, 10)
        
        self.svdb_result_text = wx.TextCtrl(svdb_panel, style=wx.TE_MULTILINE | wx.TE_READONLY, size=(-1, 400))
        svdb_sizer.Add(self.svdb_result_text, 1, wx.EXPAND | wx.ALL, 10)
        
        svdb_panel.SetSizer(svdb_sizer)

        # v3db_panel = wx.Panel(mark_notebook)
        # v3db_sizer = wx.BoxSizer(wx.VERTICAL)
        # v3db_text = wx.StaticText(v3db_panel, label=_('mark.v3db.title'))
        # register(v3db_text, 'mark.v3db.title')
        # v3db_sizer.Add(v3db_text, 0, wx.ALL | wx.CENTER, 20)
        # v3db_panel.SetSizer(v3db_sizer)

        mark_notebook.AddPage(oto_panel, _('mark.oto.title'))
        register(mark_notebook, 'mark.oto.title', 'notebook_tab', 0)
        mark_notebook.AddPage(svdb_panel, _('mark.svdb.title'))
        register(mark_notebook, 'mark.svdb.title', 'notebook_tab', 1)
        # mark_notebook.AddPage(v3db_panel, _('mark.v3db.title'))
        # register(mark_notebook, 'mark.v3db.title', 'notebook_tab', 2)

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
        self.load_models()
        self.load_svdb_dicts()
        self.load_presamp()
        self.load_presamp()

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
            except Exception as e:
                wx.CallAfter(self.svdb_result_text.AppendText, _('log.failed').format(error=str(e)))
                wx.CallAfter(wx.MessageBox, _('log.failed').format(error=str(e)), _('msg.error'), wx.OK | wx.ICON_ERROR)
        
        thread = threading.Thread(target=generate_svdb_thread)
        thread.start()

    
    def load_presamp(self):
        presamp_dir = str(ROOT / 'presamp')
        if os.path.exists(presamp_dir):
            presamp_files = [f for f in os.listdir(presamp_dir) if os.path.isfile(os.path.join(presamp_dir, f))]
            self.oto_presamp_choice.Clear()
            self.oto_presamp_choice.AppendItems(presamp_files)
            if presamp_files:
                self.oto_presamp_choice.SetSelection(0)
    
    def on_language_switch(self, event):
        """切换语言（实时生效）"""
        lang_code = self.lang_choice.GetClientData(self.lang_choice.GetSelection())
        from i18n import switch_lang, update_all
        switch_lang(lang_code)
        update_all()

    def on_oto_presamp_selected(self, event):
        presamp_file = self.oto_presamp_choice.GetStringSelection()
        if presamp_file:
            presamp_dir = str(ROOT / 'presamp')
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
        elif mode == 3:  # VCV
            self.oto_cv_sum_text.SetValue("1,3,1.5,1,2")
            self.oto_vc_sum_text.SetValue("2.5,3,1.5,1,3")
            self.oto_vv_sum_text.SetValue("0,0,0,0,0")
            self.oto_cv_offset_text.SetValue("0,0,0,0,0")
            self.oto_vc_offset_text.SetValue("0,0,0,0,0")
        elif mode == 4:  # Test
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
            wx.MessageBox(_('msg.err.select_folder'), _('msg.error'), wx.OK | wx.ICON_ERROR)
            return
        
        if not os.path.exists(wav_path):
            wx.MessageBox(_('msg.err.folder_not_exist'), _('msg.error'), wx.OK | wx.ICON_ERROR)
            return
        
        if not presamp_path:
            wx.MessageBox(_('msg.err.select_presamp'), _('msg.error'), wx.OK | wx.ICON_ERROR)
            return
        
        if not os.path.exists(presamp_path):
            wx.MessageBox(_('msg.err.presamp_not_exist'), _('msg.error'), wx.OK | wx.ICON_ERROR)
            return
        
        try:
            cv_sum = [float(x) for x in self.oto_cv_sum_text.GetValue().split(',')]
            vc_sum = [float(x) for x in self.oto_vc_sum_text.GetValue().split(',')]
            vv_sum = [float(x) for x in self.oto_vv_sum_text.GetValue().split(',')]
            cv_offset = [float(x) for x in self.oto_cv_offset_text.GetValue().split(',')]
            vc_offset = [float(x) for x in self.oto_vc_offset_text.GetValue().split(',')]
        except ValueError:
            wx.MessageBox(_('msg.err.params_format'), _('msg.error'), wx.OK | wx.ICON_ERROR)
            return
        
        pitch = self.oto_pitch_text.GetValue()
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
            wx.MessageBox(_('msg.err.textgrid_not_exist'), _('msg.error'), wx.OK | wx.ICON_ERROR)
            return

        word_phone_path = os.path.join(textgrid_path, 'json', 'word_phone.json')
        
        if not os.path.exists(word_phone_path):
            wx.MessageBox(_('msg.err.json_not_exist'), _('msg.error'), wx.OK | wx.ICON_ERROR)
            return
        
        if not os.path.exists(oto_preset) and oto_preset != "":
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
                    oto_rw.oto_write(os.path.join(wav_path, 'auto_oto.ini'), cv + vc, pitch, cover, oto_encoding)
                    
                    oto_data = oto_rw.oto_read(os.path.join(wav_path, 'auto_oto.ini'))
                    if os.path.exists(oto_preset):
                        wx.CallAfter(self.oto_result_text.AppendText, _('log.apply_template').format(path=oto_preset))
                        oto_data = oto_rw.oto_apply_template(oto_data, oto_preset)
                    oto_rw.oto_write(os.path.join(wav_path, 'oto.ini'), oto_data, pitch, cover, oto_encoding)
                    
                    
                    
                    wx.CallAfter(self.oto_result_text.AppendText, _('log.check_missing'))
                    oto_check.run(os.path.join(wav_path, 'oto.ini'), presamp_path, pitch, vcv_mode)
                
                wx.CallAfter(self.oto_result_text.AppendText, _('log.oto_complete'))
                wx.CallAfter(wx.MessageBox, _('msg.ok.oto_complete'), _('msg.success'), wx.OK | wx.ICON_INFORMATION)
            except Exception as e:
                wx.CallAfter(self.oto_result_text.AppendText, _('log.failed').format(error=str(e)))
                wx.CallAfter(wx.MessageBox, _('log.failed').format(error=str(e)), _('msg.error'), wx.OK | wx.ICON_ERROR)
        
        thread = threading.Thread(target=generate_oto_thread)
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
    
    def on_browse_file(self, event, text_ctrl):
        dialog = wx.FileDialog(None, _('msg.select_file'), style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)
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
            except Exception as e:
                wx.CallAfter(self.lab_result_text.AppendText, _('log.failed').format(error=str(e)))
                wx.CallAfter(wx.MessageBox, _('log.failed').format(error=str(e)), _('msg.error'), wx.OK | wx.ICON_ERROR)

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
            except Exception as e:
                wx.CallAfter(self.lab_index_result_text.AppendText, _('log.failed').format(error=str(e)))
                wx.CallAfter(wx.MessageBox, _('log.failed').format(error=str(e)), _('msg.error'), wx.OK | wx.ICON_ERROR)

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
                wx.CallAfter(self.infer_result_text.AppendText, _('log.loading_model'))

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

                inference = onnx_infer.InferenceOnnx(model_path)
                wx.CallAfter(self.infer_result_text.AppendText, _('log.loading_config'))
                inference.load_config()
                wx.CallAfter(self.infer_result_text.AppendText, _('log.loading_model_weights'))
                inference.load_model(device=device)  # 传递设备选择
                wx.CallAfter(self.infer_result_text.AppendText, _('log.init_decoder'))
                inference.init_decoder()

                def progress_callback(msg):
                    wx.CallAfter(self.infer_result_text.AppendText, msg + "\n")
                    wx.CallAfter(self.infer_result_text.ShowPosition, self.infer_result_text.GetLastPosition())

                inference.set_progress_callback(progress_callback)

                wx.CallAfter(self.infer_result_text.AppendText, _('log.loading_dataset'))
                inference.get_dataset(wav_folder, language=language, g2p="dictionary", dictionary_path=str(dict_path), in_format="lab")

                # # 获取用户选择的推理参数
                pad_times_selection = self.pad_times_choice.GetSelection()
                pad_times = self.pad_times_choice.GetClientData(pad_times_selection)

                pad_length_selection = self.pad_length_choice.GetSelection()
                pad_length = self.pad_length_choice.GetClientData(pad_length_selection)

                # pad_times = 2
                # pad_length = 5

                wx.CallAfter(self.infer_result_text.AppendText, _('log.start_infer'))
                wx.CallAfter(self.infer_result_text.AppendText, _('log.infer_times').format(times=pad_times))
                wx.CallAfter(self.infer_result_text.AppendText, _('log.pad_length').format(len=pad_length))
                merge_phonemes = self.merge_phonemes_checkbox.GetValue()
                wx.CallAfter(self.infer_result_text.AppendText, _('log.merge_phonemes').format(merge=merge_phonemes))
                inference.infer(non_lexical_phonemes="AP", pad_times=pad_times, pad_length=pad_length, merge_phonemes=merge_phonemes)
                wx.CallAfter(self.infer_result_text.AppendText, _('log.exporting'))
                inference.export(wav_folder)

                wx.CallAfter(self.infer_result_text.AppendText, _('log.infer_complete'))
                wx.CallAfter(wx.MessageBox, _('msg.ok.infer_complete'), _('msg.success'), wx.OK | wx.ICON_INFORMATION)
            except Exception as e:
                wx.CallAfter(self.infer_result_text.AppendText, _('log.infer_failed').format(error=str(e)))
                wx.CallAfter(wx.MessageBox, _('log.infer_failed').format(error=str(e)), _('msg.error'), wx.OK | wx.ICON_ERROR)

        thread = threading.Thread(target=infer_thread)
        thread.start()



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

            except Exception as e:
                wx.CallAfter(self.json_result_text.AppendText, _('log.failed').format(error=str(e)))
                wx.CallAfter(wx.MessageBox, _('log.failed').format(error=str(e)), _('msg.error'), wx.OK | wx.ICON_ERROR)

        thread = threading.Thread(target=generate_json_thread)
        thread.start()

if __name__ == "__main__":
    try:
        app = wx.App()
        frame = MainFrame()
        frame.Show()
        app.MainLoop()
    except Exception as e:
        error_msg = f"程序启动失败！\n错误: {str(e)}"
        print(error_msg)  # 输出到cmd窗口
        input("按回车键退出...")  # 防止窗口立即关闭