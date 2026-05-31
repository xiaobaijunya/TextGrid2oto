"""共享 OTO 参数面板 — 用于普通 OTO 生成和多音阶 OTO 生成"""

import wx
import os
import json
from pathlib import Path

from i18n import _, register

ROOT = Path(__file__).resolve().parent.parent
PRESETS_FILE = ROOT / "config" / "oto_presets.json"
PRESAMP_DIR = ROOT / "presamp"

# ── 模式名列表（与 json 中的 key 一致） ────────
MODE_NAMES = ["CVVC", "VCV", "CVV", "ARPAsing", "Test"]

# ── 加载预设 ─────────────────────────────────────
def _load_all_presets() -> dict:
    """从 oto_presets.json 加载所有预设"""
    if PRESETS_FILE.exists():
        try:
            with open(PRESETS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                # 去掉 _comment, _format_version 等元数据
                return {k: v for k, v in data.items() if not k.startswith("_")}
        except Exception:
            pass
    return {}

_ALL_PRESETS = _load_all_presets()


class OtoParamPanel(wx.Panel):
    """共享 OTO 参数面板

    包含参数预设、Presamp、模板、模式、编码、覆盖、CV/VC/VV参数、偏移、重复、忽略等。
    不包含音源文件夹（由父级各自处理）。
    """

    def __init__(self, parent, browse_folder_cb=None, show_pitch=True):
        super().__init__(parent)
        self.browse_folder_cb = browse_folder_cb
        self._show_pitch = show_pitch
        self._init_ui()

    def _init_ui(self):
        sizer = wx.BoxSizer(wx.VERTICAL)

        # ── Presamp ────────────────────────────────
        presamp_box = wx.StaticBox(self, label=_("mark.oto.presamp_path"))
        presamp_sizer = wx.StaticBoxSizer(presamp_box, wx.VERTICAL)
        self._presamp_label_ctrl = presamp_box  # 用于翻译
        register(presamp_box, "mark.oto.presamp_path", "label")

        # Presamp 路径 + 浏览 + 预设
        presamp_row = wx.BoxSizer(wx.HORIZONTAL)
        self.presamp_path_text = wx.TextCtrl(self, size=(300, -1))
        self.presamp_browse_btn = wx.Button(self, label=_("mark.oto.browse"))
        register(self.presamp_browse_btn, "mark.oto.browse", "button")
        self.presamp_browse_btn.Bind(wx.EVT_BUTTON, self._on_browse_presamp)
        self.presamp_preset_label = wx.StaticText(self, label=_("mark.oto.presamp_preset"))
        register(self.presamp_preset_label, "mark.oto.presamp_preset")
        self.presamp_preset_choice = wx.Choice(self, size=(150, -1))
        self.presamp_preset_choice.Bind(wx.EVT_CHOICE, self._on_presamp_preset_selected)
        presamp_row.Add(self.presamp_path_text, 1, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 2)
        presamp_row.Add(self.presamp_browse_btn, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 2)
        presamp_row.Add(self.presamp_preset_label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 2)
        presamp_row.Add(self.presamp_preset_choice, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 2)
        presamp_sizer.Add(presamp_row, 0, wx.EXPAND, 0)
        sizer.Add(presamp_sizer, 0, wx.EXPAND | wx.ALL, 5)

        # ── OTO 模板 ───────────────────────────────
        template_box = wx.StaticBox(self, label=_("mark.oto.template"))
        template_sizer = wx.StaticBoxSizer(template_box, wx.VERTICAL)
        self._template_label_ctrl = template_box
        register(template_box, "mark.oto.template", "label")

        template_row = wx.BoxSizer(wx.HORIZONTAL)
        self.template_path_text = wx.TextCtrl(self, value="", size=(300, -1))
        self.template_browse_btn = wx.Button(self, label=_("mark.oto.browse"))
        register(self.template_browse_btn, "mark.oto.browse", "button")
        self.template_browse_btn.Bind(wx.EVT_BUTTON, self._on_browse_template)
        template_row.Add(self.template_path_text, 1, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 2)
        template_row.Add(self.template_browse_btn, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 2)
        template_sizer.Add(template_row, 0, wx.EXPAND, 0)
        sizer.Add(template_sizer, 0, wx.EXPAND | wx.ALL, 5)

        # ── 模式 / 编码 / 覆盖 ──────────────────────
        mode_box = wx.StaticBox(self, label=_("mark.oto.mode"))
        mode_sizer = wx.StaticBoxSizer(mode_box, wx.HORIZONTAL)
        self._mode_label_ctrl = mode_box
        register(mode_box, "mark.oto.mode", "label")

        self.mode_choice = wx.Choice(self, choices=MODE_NAMES, size=(120, -1))
        self.mode_choice.SetSelection(0)
        self.mode_choice.Bind(wx.EVT_CHOICE, self._on_mode_changed)
        self.encoding_label = wx.StaticText(self, label=_("mark.oto.encoding"))
        register(self.encoding_label, "mark.oto.encoding")
        self.encoding_choice = wx.Choice(self, choices=["utf-8", "shift-jis", "gbk"], size=(100, -1))
        self.encoding_choice.SetSelection(0)
        self.cover_checkbox = wx.CheckBox(self, label=_("mark.oto.cover"))
        register(self.cover_checkbox, "mark.oto.cover", "checkbox")
        self.cover_checkbox.SetValue(True)
        mode_sizer.Add(self.mode_choice, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 2)
        mode_sizer.Add(self.encoding_label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 2)
        mode_sizer.Add(self.encoding_choice, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 2)
        mode_sizer.Add(self.cover_checkbox, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 2)
        mode_sizer.AddStretchSpacer()
        sizer.Add(mode_sizer, 0, wx.EXPAND | wx.ALL, 5)

        # ── 参数预设 ────────────────────────────────
        preset_box = wx.StaticBox(self, label=_("mark.oto.preset_group"))
        preset_box_sizer = wx.StaticBoxSizer(preset_box, wx.HORIZONTAL)
        self._preset_box_label = preset_box
        register(preset_box, "mark.oto.preset_group", "label")

        self.preset_label = wx.StaticText(self, label=_("mark.oto.preset"))
        register(self.preset_label, "mark.oto.preset")
        self.preset_choice = wx.Choice(self, size=(180, -1))
        self.preset_choice.Bind(wx.EVT_CHOICE, self._on_preset_selected)

        # 音阶后缀
        self.pitch_label = wx.StaticText(self, label=_("mark.oto.pitch"))
        register(self.pitch_label, "mark.oto.pitch")
        self.pitch_text = wx.TextCtrl(self, value="", size=(80, -1))
        if not self._show_pitch:
            self.pitch_label.Hide()
            self.pitch_text.Hide()

        # 忽略音素
        self.ignore_label = wx.StaticText(self, label=_("mark.oto.ignore"))
        register(self.ignore_label, "mark.oto.ignore")
        self.ignore_text = wx.TextCtrl(self, value="AP,SP,EP,R", size=(140, -1))

        preset_box_sizer.Add(self.preset_label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 2)
        preset_box_sizer.Add(self.preset_choice, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 2)
        preset_box_sizer.Add(self.pitch_label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 2)
        preset_box_sizer.Add(self.pitch_text, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 2)
        preset_box_sizer.Add(self.ignore_label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 2)
        preset_box_sizer.Add(self.ignore_text, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 2)
        preset_box_sizer.AddStretchSpacer()
        sizer.Add(preset_box_sizer, 0, wx.EXPAND | wx.ALL, 5)

        # ── 参数数值 ────────────────────────────────
        params_box = wx.StaticBox(self, label=_("mark.oto.params_group"))
        params_sizer = wx.StaticBoxSizer(params_box, wx.VERTICAL)
        self._params_box_label = params_box
        register(params_box, "mark.oto.params_group", "label")

        # 第一行：CV / VC / VV 参数
        row1 = wx.BoxSizer(wx.HORIZONTAL)
        row1.Add(self._make_param_group(params_box, "mark.oto.cv_params", "cv_sum_text"), 1, wx.EXPAND | wx.ALL, 2)
        row1.Add(self._make_param_group(params_box, "mark.oto.vc_params", "vc_sum_text"), 1, wx.EXPAND | wx.ALL, 2)
        row1.Add(self._make_param_group(params_box, "mark.oto.vv_params", "vv_sum_text"), 1, wx.EXPAND | wx.ALL, 2)
        params_sizer.Add(row1, 0, wx.EXPAND, 0)

        # 第二行：CV偏移 / VC偏移
        row2 = wx.BoxSizer(wx.HORIZONTAL)
        row2.Add(self._make_param_group(params_box, "mark.oto.cv_offset", "cv_offset_text"), 1, wx.EXPAND | wx.ALL, 2)
        row2.Add(self._make_param_group(params_box, "mark.oto.vc_offset", "vc_offset_text"), 1, wx.EXPAND | wx.ALL, 2)
        row2.AddStretchSpacer()
        params_sizer.Add(row2, 0, wx.EXPAND, 0)

        # 第三行：CV重复 / VC重复
        row3 = wx.BoxSizer(wx.HORIZONTAL)
        row3.Add(self._make_param_group(params_box, "mark.oto.cv_repeat", "cv_repeat_text"), 0, wx.ALL, 2)
        row3.Add(self._make_param_group(params_box, "mark.oto.vc_repeat", "vc_repeat_text"), 0, wx.ALL, 2)
        row3.AddStretchSpacer()
        params_sizer.Add(row3, 0, wx.EXPAND, 0)

        sizer.Add(params_sizer, 0, wx.EXPAND | wx.ALL, 5)

        self.SetSizer(sizer)
        self._load_presamp_presets()
        self._refresh_preset_choices()
        self._apply_preset()  # 加载默认预设

    # ── 辅助：创建带标签的参数字段 ──────────────────
    def _make_param_group(self, parent_box, label_key, attr_name):
        """创建一个带标签的 TextCtrl，标签为 StaticBox 的子 StaticText"""
        vsizer = wx.BoxSizer(wx.VERTICAL)
        label = wx.StaticText(self, label=_(label_key))
        register(label, label_key)
        text = wx.TextCtrl(self, value="", size=(110, -1))
        text.SetToolTip(_("mark.oto.params_tooltip"))
        register(text, "mark.oto.params_tooltip", "tooltip")
        vsizer.Add(label, 0, wx.ALL, 1)
        vsizer.Add(text, 0, wx.ALL, 1)
        setattr(self, attr_name, text)
        return vsizer

    # ── Presamp 预设 ──────────────────────────────
    def _load_presamp_presets(self):
        self.presamp_preset_choice.Clear()
        if PRESAMP_DIR.exists():
            files = sorted([f for f in os.listdir(str(PRESAMP_DIR))
                           if os.path.isfile(os.path.join(str(PRESAMP_DIR), f))])
            self.presamp_preset_choice.AppendItems(files)
            if files:
                self.presamp_preset_choice.SetSelection(0)
                # 自动填入路径
                self._on_presamp_preset_selected(None)

    def _on_presamp_preset_selected(self, event):
        name = self.presamp_preset_choice.GetStringSelection()
        if name:
            self.presamp_path_text.SetValue(str(PRESAMP_DIR / name))

    def _on_browse_presamp(self, event):
        dialog = wx.FileDialog(self, "", style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)
        if dialog.ShowModal() == wx.ID_CANCEL:
            return
        self.presamp_path_text.SetValue(dialog.GetPath())
        dialog.Destroy()

    def _on_browse_template(self, event):
        dialog = wx.FileDialog(self, "", style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)
        if dialog.ShowModal() == wx.ID_CANCEL:
            return
        self.template_path_text.SetValue(dialog.GetPath())
        dialog.Destroy()

    # ── 模式切换 → 刷新预设下拉列表 ──────────────
    def _on_mode_changed(self, event):
        self._refresh_preset_choices()
        self._apply_preset()

    def _refresh_preset_choices(self):
        """根据当前模式刷新参数预设下拉列表"""
        mode_name = self.get_mode_name()
        presets_dict = _ALL_PRESETS.get(mode_name, {})
        preset_list = presets_dict.get("presets", {})
        self.preset_choice.Clear()
        names = list(preset_list.keys())
        if names:
            self.preset_choice.AppendItems(names)
            # 选择默认预设
            default = presets_dict.get("default_preset", names[0])
            if default in names:
                self.preset_choice.SetSelection(names.index(default))
            else:
                self.preset_choice.SetSelection(0)

    def _on_preset_selected(self, event):
        self._apply_preset()

    def _apply_preset(self):
        """将当前选中的预设的值填入各文本框"""
        mode_name = self.get_mode_name()
        presets_dict = _ALL_PRESETS.get(mode_name, {})
        preset_list = presets_dict.get("presets", {})
        name = self.preset_choice.GetStringSelection()
        if not name or name not in preset_list:
            # 回退：使用硬编码的默认值
            self._apply_hardcoded_defaults()
            return
        values = preset_list[name]
        self.cv_sum_text.SetValue(values.get("cv_params", ""))
        self.vc_sum_text.SetValue(values.get("vc_params", ""))
        self.vv_sum_text.SetValue(values.get("vv_params", ""))
        self.cv_offset_text.SetValue(values.get("cv_offset", "0,0,0,0,0"))
        self.vc_offset_text.SetValue(values.get("vc_offset", "0,0,0,0,0"))
        self.cv_repeat_text.SetValue(values.get("cv_repeat", "1"))
        self.vc_repeat_text.SetValue(values.get("vc_repeat", "1"))

    def _apply_hardcoded_defaults(self):
        """预设文件加载失败时的硬编码后备"""
        self.cv_sum_text.SetValue("1,3,1.5,1,2")
        self.vc_sum_text.SetValue("3,0,2,1,3")
        self.vv_sum_text.SetValue("3,3,1.5,1,3")
        self.cv_offset_text.SetValue("0,0,0,0,0")
        self.vc_offset_text.SetValue("0,0,0,0,0")
        self.cv_repeat_text.SetValue("1")
        self.vc_repeat_text.SetValue("1")

    # ── 公共读取方法 ──────────────────────────────
    def get_mode_name(self) -> str:
        idx = self.mode_choice.GetSelection()
        return MODE_NAMES[idx] if 0 <= idx < len(MODE_NAMES) else "CVVC"

    def get_mode_index(self) -> int:
        return self.mode_choice.GetSelection()

    def get_params(self) -> dict:
        """返回当前所有参数 (含验证)"""
        try:
            cv_sum = [float(x) for x in self.cv_sum_text.GetValue().split(",")]
            vc_sum = [float(x) for x in self.vc_sum_text.GetValue().split(",")]
            vv_sum = [float(x) for x in self.vv_sum_text.GetValue().split(",")]
            cv_offset = [float(x) for x in self.cv_offset_text.GetValue().split(",")]
            vc_offset = [float(x) for x in self.vc_offset_text.GetValue().split(",")]
        except ValueError:
            raise ValueError("参数格式错误")

        return {
            "presamp_path": self.presamp_path_text.GetValue().strip(),
            "template_path": self.template_path_text.GetValue().strip(),
            "mode_index": self.get_mode_index(),
            "mode_name": self.get_mode_name(),
            "encoding": self.encoding_choice.GetStringSelection(),
            "cover": "y" if self.cover_checkbox.GetValue() else "n",
            "pitch": self.pitch_text.GetValue().strip(),
            "ignore": self.ignore_text.GetValue().strip(),
            "cv_sum": cv_sum,
            "vc_sum": vc_sum,
            "vv_sum": vv_sum,
            "cv_offset": cv_offset,
            "vc_offset": vc_offset,
            "cv_repeat": self.cv_repeat_text.GetValue().strip() or "1",
            "vc_repeat": self.vc_repeat_text.GetValue().strip() or "1",
        }

    def get_oto_params(self) -> dict:
        """get_params 的别名，对外一致"""
        return self.get_params()
