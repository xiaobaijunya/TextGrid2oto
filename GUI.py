import wx
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import wavname2lab
import onnx_infer
from pathlib import Path
from textgrid2json import del_SP, TextGrid2ds_json, ds_json2filter, ds_json2word
from json2oto import json2CV_oto, json2oto, json2VCV_oto, json2test

class MainFrame(wx.Frame):
    def __init__(self):
        super().__init__(None, title="UTAU自动标注工具", size=(800, 600))
        
        panel = wx.Panel(self)
        
        notebook = wx.Notebook(panel)
        
        lab_panel = wx.Panel(notebook)
        lab_sizer = wx.BoxSizer(wx.VERTICAL)
        
        lab_title = wx.StaticText(lab_panel, label="1.LAB生成")
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
        
        self.lab_result_text = wx.TextCtrl(lab_panel, style=wx.TE_MULTILINE | wx.TE_READONLY, size=(-1, 150))
        lab_sizer.Add(self.lab_result_text, 0, wx.EXPAND | wx.ALL, 10)
        
        lab_panel.SetSizer(lab_sizer)
        
        textgrid_panel = wx.Panel(notebook)
        textgrid_sizer = wx.BoxSizer(wx.VERTICAL)

        textgrid_title = wx.StaticText(textgrid_panel, label="2.TextGrid推理")
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

        self.infer_result_text = wx.TextCtrl(textgrid_panel, style=wx.TE_MULTILINE | wx.TE_READONLY, size=(-1, 150))
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
        self.json_ignore_text = wx.TextCtrl(json_panel, value="AP,SP,EP", size=(200, -1))
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
        
        self.json_result_text = wx.TextCtrl(json_panel, style=wx.TE_MULTILINE | wx.TE_READONLY, size=(-1, 150))
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

        self.clean_result_text = wx.TextCtrl(clean_panel, style=wx.TE_MULTILINE | wx.TE_READONLY, size=(-1, 150))
        clean_sizer.Add(self.clean_result_text, 0, wx.EXPAND | wx.ALL, 10)

        clean_panel.SetSizer(clean_sizer)

        mark_panel = wx.Panel(notebook)
        mark_sizer = wx.BoxSizer(wx.VERTICAL)

        mark_notebook = wx.Notebook(mark_panel)

        oto_panel = wx.Panel(mark_notebook)
        oto_sizer = wx.BoxSizer(wx.VERTICAL)
        oto_text = wx.StaticText(oto_panel, label="OTO生成")
        oto_sizer.Add(oto_text, 0, wx.ALL | wx.CENTER, 20)
        oto_panel.SetSizer(oto_sizer)

        svdb_panel = wx.Panel(mark_notebook)
        svdb_sizer = wx.BoxSizer(wx.VERTICAL)
        svdb_text = wx.StaticText(svdb_panel, label="SVDB")
        svdb_sizer.Add(svdb_text, 0, wx.ALL | wx.CENTER, 20)
        svdb_panel.SetSizer(svdb_sizer)

        v3db_panel = wx.Panel(mark_notebook)
        v3db_sizer = wx.BoxSizer(wx.VERTICAL)
        v3db_text = wx.StaticText(v3db_panel, label="V3DB")
        v3db_sizer.Add(v3db_text, 0, wx.ALL | wx.CENTER, 20)
        v3db_panel.SetSizer(v3db_sizer)

        mark_notebook.AddPage(oto_panel, "OTO生成")
        mark_notebook.AddPage(svdb_panel, "SVDB")
        mark_notebook.AddPage(v3db_panel, "V3DB")

        mark_sizer.Add(mark_notebook, 1, wx.EXPAND | wx.ALL, 5)
        mark_panel.SetSizer(mark_sizer)

        notebook.AddPage(lab_panel, "LAB生成")
        notebook.AddPage(textgrid_panel, "TextGrid推理")
        notebook.AddPage(clean_panel, "TextGrid清洗")
        notebook.AddPage(json_panel, "JSON生成")
        notebook.AddPage(mark_panel, "标记生成")

        main_sizer = wx.BoxSizer(wx.VERTICAL)
        main_sizer.Add(notebook, 1, wx.EXPAND)
        panel.SetSizer(main_sizer)
        
        # 在所有控件创建完成后加载模型
        self.load_models()

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

    def on_browse_folder(self, event, text_ctrl):
        dialog = wx.DirDialog(None, "选择文件夹", style=wx.DD_DEFAULT_STYLE)
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
        
        try:
            self.lab_result_text.Clear()
            self.lab_result_text.AppendText("正在生成LAB文件...\n")

            wavname2lab.run(path, cuts)

            self.lab_result_text.AppendText("LAB文件生成完成！\n")
            wx.MessageBox("LAB文件生成完成", "成功", wx.OK | wx.ICON_INFORMATION)
        except Exception as e:
            error_msg = f"生成失败：{str(e)}"
            self.lab_result_text.AppendText(error_msg + "\n")
            wx.MessageBox(error_msg, "错误", wx.OK | wx.ICON_ERROR)

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

        try:
            self.infer_result_text.Clear()
            self.infer_result_text.AppendText("正在加载模型...\n")

            model_path = Path(os.path.dirname(os.path.abspath(__file__))) / 'HubertFA_model' / model_folder / model_file
            dict_path = Path(os.path.dirname(os.path.abspath(__file__))) / 'HubertFA_model' / model_folder / dict_file

            language = dict_file.split('.')[0].split('-')
            language = language[0] if len(language) == 1 else language

            self.infer_result_text.AppendText(f"模型: {model_file}\n")
            self.infer_result_text.AppendText(f"字典: {dict_file}\n")
            self.infer_result_text.AppendText(f"语言: {language}\n")

            inference = onnx_infer.InferenceOnnx(model_path)
            self.infer_result_text.AppendText("加载配置...\n")
            inference.load_config()
            self.infer_result_text.AppendText("加载模型...\n")
            inference.load_model()
            self.infer_result_text.AppendText("初始化解码器...\n")
            inference.init_decoder()
            self.infer_result_text.AppendText("加载数据集...\n")
            inference.get_dataset(wav_folder, language=language, g2p="dictionary", dictionary_path=str(dict_path), in_format="lab")
            self.infer_result_text.AppendText("开始推理...\n")
            inference.infer(non_lexical_phonemes="AP", pad_times=1, pad_length=5)
            self.infer_result_text.AppendText("导出结果...\n")
            inference.export(wav_folder)

            self.infer_result_text.AppendText("TextGrid推理完成！\n")
            wx.MessageBox("TextGrid推理完成", "成功", wx.OK | wx.ICON_INFORMATION)
        except Exception as e:
            error_msg = f"推理失败：{str(e)}"
            self.infer_result_text.AppendText(error_msg + "\n")
            wx.MessageBox(error_msg, "错误", wx.OK | wx.ICON_ERROR)

    def on_clean_sp(self, event):
        wav_folder = self.clean_path_text.GetValue().strip()
        
        if not wav_folder:
            wx.MessageBox("请选择WAV文件夹", "错误", wx.OK | wx.ICON_ERROR)
            return
        
        if not os.path.exists(wav_folder):
            wx.MessageBox("WAV文件夹不存在", "错误", wx.OK | wx.ICON_ERROR)
            return
        
        try:
            self.clean_result_text.Clear()
            self.clean_result_text.AppendText("正在清理SP标记...\n")
            
            ignore = 'AP,SP,EP'
            delete_sp = True
            
            deleted_files = del_SP.process_all_textgrid_files(wav_folder, ignore, delete_sp)
            
            if deleted_files and deleted_files[0] != '没有文件中有SP被删除':
                result_msg = "以下文件中有SP被删除（建议复核标记）：\n"
                result_msg += ', '.join(deleted_files)
                self.clean_result_text.AppendText(result_msg + "\n")
                wx.MessageBox("SP清理完成，请查看结果", "成功", wx.OK | wx.ICON_INFORMATION)
            else:
                self.clean_result_text.AppendText("没有文件中有SP被删除。\n")
                wx.MessageBox("没有文件中有SP被删除", "提示", wx.OK | wx.ICON_INFORMATION)
                
        except Exception as e:
            error_msg = f"清理SP失败：{str(e)}"
            self.clean_result_text.AppendText(error_msg + "\n")
            wx.MessageBox(error_msg, "错误", wx.OK | wx.ICON_ERROR)

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
        
        try:
            self.json_result_text.Clear()
            self.json_result_text.AppendText("正在生成JSON文件...\n")
            
            # 获取字典路径
            selected_folder = self.model_folder_choice.GetStringSelection()
            dict_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'HubertFA_model', selected_folder, dict_file)
            
            if not os.path.exists(dict_path):
                wx.MessageBox("字典文件不存在", "错误", wx.OK | wx.ICON_ERROR)
                return
            
            # 生成ds_phone.json
            rec_preset = None
            TextGrid2ds_json.run(wav_folder, rec_preset)
            self.json_result_text.AppendText("1. 生成ds_phone.json完成\n")
            
            # 过滤音素
            json_path = os.path.join(wav_folder, 'json', 'ds_phone.json')
            dict_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'HubertFA_model', selected_folder, dict_file)
            ds_json2filter.run(dict_file, json_path, ignore)
            self.json_result_text.AppendText("2. 过滤音素完成\n")
            
            # 生成word.json
            filter_json_path = os.path.join(wav_folder, 'json', 'ds_phone_filter.json')
            ds_json2word.run(dict_file, filter_json_path)
            self.json_result_text.AppendText("3. 生成word_phone.json完成\n")
            
            self.json_result_text.AppendText("JSON文件生成完成！\n")
            wx.MessageBox("JSON文件生成完成", "成功", wx.OK | wx.ICON_INFORMATION)
                
        except Exception as e:
            error_msg = f"生成JSON失败：{str(e)}"
            self.json_result_text.AppendText(error_msg + "\n")
            wx.MessageBox(error_msg, "错误", wx.OK | wx.ICON_ERROR)

if __name__ == "__main__":
    app = wx.App()
    frame = MainFrame()
    frame.Show()
    app.MainLoop()