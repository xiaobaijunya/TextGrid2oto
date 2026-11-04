# coding=utf-8
"""GUI 的「ASR 识别（txt / lab / json）」页面。

以 mixin 形式挂到 ``MainFrame``，避免 GUI.py 继续膨胀。
页面只保留基础功能：

    音频（文件夹 / 多选 / 批量 CSV）
      -> ASR（系统提示词 + 参考歌词）
      -> Qwen3-ForcedAligner 逐字时间戳
      -> .txt / .lab / _ASR.json

批量 CSV 使用英文表头 ``path,title,lyrics``：第 1 列是音频或文件夹路径
（相对路径按 CSV 所在目录解析），第 2 列是标题（可空），第 3 列是歌词。
一行指向文件夹时，该文件夹内所有音频都会使用这段歌词。
"""

from __future__ import annotations

import os
import threading
import traceback
from pathlib import Path

import wx

from i18n import _, register

ROOT = Path(__file__).resolve().parent.parent


def _append_output(text_ctrl, text: str) -> None:
    """在 GUI 线程内追加日志并滚到底部。"""
    if not text_ctrl:
        return
    if text:
        text_ctrl.AppendText(text)
        lines = text_ctrl.GetNumberOfLines()
        if lines > 8000:
            cut_pos = text_ctrl.XYToPosition(0, lines - 4000)
            if cut_pos != wx.NOT_FOUND and cut_pos > 0:
                text_ctrl.Remove(0, cut_pos)
    text_ctrl.ShowPosition(text_ctrl.GetLastPosition())


class AsrPanelMixin:
    """MainFrame 的 ASR 页面扩展。"""

    # ── 构建界面 ────────────────────────────────────────────
    def _build_asr_panel(self, parent):
        from asr_lab import asr_suite

        self._asr_stop = threading.Event()
        self._asr_selected_files: list[str] = []
        self._asr_lyrics_map: dict[str, str] = {}
        # 每个音频「上次写入草稿文件的内容」，用于只保存真正发生变化的歌词
        self._asr_saved_lyrics: dict[str, str] = {}
        self._asr_current_audio: str | None = None
        self._asr_csv_notes: list[str] = []
        self._asr_audio_paths: list[Path] = []

        # 歌词保存：每 3 秒巡检一次，只有内容变化才写盘；
        # 切換音频、开始识别、关闭窗口、手动点「保存歌词」时也会立即保存。
        self._asr_lyrics_timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self._asr_lyrics_autosave_tick, self._asr_lyrics_timer)
        self._asr_lyrics_timer.Start(3000)
        self.Bind(wx.EVT_CLOSE, self._asr_on_close)

        sizer = wx.BoxSizer(wx.VERTICAL)

        # ── 音源 ──
        src_box = wx.StaticBox(parent, label=_('lab.asr.source'))
        register(src_box, 'lab.asr.source', 'label')
        src_sizer = wx.StaticBoxSizer(src_box, wx.VERTICAL)

        row = wx.BoxSizer(wx.HORIZONTAL)
        label = wx.StaticText(parent, label=_('lab.asr.folder'))
        register(label, 'lab.asr.folder')
        self.asr_folder_text = wx.TextCtrl(parent, size=(320, -1))
        browse_btn = wx.Button(parent, label=_('textgrid.browse_folder'))
        register(browse_btn, 'textgrid.browse_folder', 'button')
        browse_btn.Bind(wx.EVT_BUTTON, lambda event: self._asr_browse_folder(event))
        files_btn = wx.Button(parent, label=_('textgrid.browse_files'))
        register(files_btn, 'textgrid.browse_files', 'button')
        files_btn.Bind(wx.EVT_BUTTON, lambda event: self._asr_browse_files(event))
        row.Add(label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        row.Add(self.asr_folder_text, 1, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        row.Add(browse_btn, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        row.Add(files_btn, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        src_sizer.Add(row, 0, wx.EXPAND | wx.ALL, 3)

        # 批量 CSV
        row = wx.BoxSizer(wx.HORIZONTAL)
        csv_label = wx.StaticText(parent, label=_('lab.asr.csv'))
        register(csv_label, 'lab.asr.csv')
        self.asr_csv_text = wx.TextCtrl(parent, size=(300, -1))
        self.asr_csv_text.SetToolTip(_('lab.asr.csv.tooltip'))
        register(self.asr_csv_text, 'lab.asr.csv.tooltip', 'tooltip')
        csv_btn = wx.Button(parent, label=_('json.browse'), size=(60, -1))
        csv_btn.Bind(wx.EVT_BUTTON, self._asr_browse_csv)
        csv_tpl_btn = wx.Button(parent, label=_('lab.asr.csv.template'))
        register(csv_tpl_btn, 'lab.asr.csv.template', 'button')
        csv_tpl_btn.SetToolTip(_('lab.asr.csv.template.tooltip'))
        register(csv_tpl_btn, 'lab.asr.csv.template.tooltip', 'tooltip')
        csv_tpl_btn.Bind(wx.EVT_BUTTON, self._asr_save_csv_template)
        row.Add(csv_label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        row.Add(self.asr_csv_text, 1, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        row.Add(csv_btn, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        row.Add(csv_tpl_btn, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        src_sizer.Add(row, 0, wx.EXPAND | wx.ALL, 3)

        self.asr_csv_status = wx.StaticText(parent, label=_('lab.asr.csv.hint'))
        register(self.asr_csv_status, 'lab.asr.csv.hint')
        src_sizer.Add(self.asr_csv_status, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 8)

        row = wx.BoxSizer(wx.HORIZONTAL)
        self.asr_recursive_check = wx.CheckBox(parent, label=_('lab.asr.recursive'))
        register(self.asr_recursive_check, 'lab.asr.recursive', 'checkbox')
        self.asr_recursive_check.SetValue(True)
        refresh_btn = wx.Button(parent, label=_('lab.asr.refresh'))
        register(refresh_btn, 'lab.asr.refresh', 'button')
        refresh_btn.Bind(wx.EVT_BUTTON, self._asr_refresh_audio_list)
        row.Add(self.asr_recursive_check, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        row.Add(refresh_btn, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        src_sizer.Add(row, 0, wx.ALL, 3)
        sizer.Add(src_sizer, 0, wx.EXPAND | wx.ALL, 5)

        # ── 模型 ──
        model_box = wx.StaticBox(parent, label=_('lab.asr.settings'))
        register(model_box, 'lab.asr.settings', 'label')
        model_sizer = wx.StaticBoxSizer(model_box, wx.VERTICAL)

        row = wx.BoxSizer(wx.HORIZONTAL)
        run_label = wx.StaticText(parent, label=_('lab.asr.run_mode'))
        register(run_label, 'lab.asr.run_mode')
        self.asr_run_mode_choice = wx.Choice(parent, size=(190, -1))
        self.asr_run_mode_choice.Append(_('lab.asr.run_mode.full'), asr_suite.RUN_MODE_FULL)
        self.asr_run_mode_choice.Append(_('lab.asr.run_mode.asr'), asr_suite.RUN_MODE_ASR_ONLY)
        self.asr_run_mode_choice.Append(_('lab.asr.run_mode.align'), asr_suite.RUN_MODE_ALIGN_ONLY)
        self.asr_run_mode_choice.SetSelection(0)
        self.asr_run_mode_choice.Bind(wx.EVT_CHOICE, self.on_asr_run_mode_changed)
        register(self.asr_run_mode_choice, '', 'choice', [
            ('lab.asr.run_mode.full', asr_suite.RUN_MODE_FULL),
            ('lab.asr.run_mode.asr', asr_suite.RUN_MODE_ASR_ONLY),
            ('lab.asr.run_mode.align', asr_suite.RUN_MODE_ALIGN_ONLY),
        ])
        self.asr_run_mode_choice.SetToolTip(_('lab.asr.run_mode.tooltip'))
        register(self.asr_run_mode_choice, 'lab.asr.run_mode.tooltip', 'tooltip')
        row.Add(run_label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        row.Add(self.asr_run_mode_choice, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        model_sizer.Add(row, 0, wx.ALL, 3)

        row = wx.BoxSizer(wx.HORIZONTAL)
        label = wx.StaticText(parent, label=_('lab.asr.model'))
        register(label, 'lab.asr.model')
        self.asr_model_text = wx.TextCtrl(parent, size=(420, -1))
        model_btn = wx.Button(parent, label=_('textgrid.browse_folder'))
        register(model_btn, 'textgrid.browse_folder', 'button')
        model_btn.Bind(wx.EVT_BUTTON, lambda event: self.on_browse_folder(event, self.asr_model_text))
        row.Add(label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        row.Add(self.asr_model_text, 1, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        row.Add(model_btn, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        model_sizer.Add(row, 0, wx.EXPAND | wx.ALL, 3)

        row = wx.BoxSizer(wx.HORIZONTAL)
        label = wx.StaticText(parent, label=_('lab.asr.aligner_model'))
        register(label, 'lab.asr.aligner_model')
        self.asr_aligner_text = wx.TextCtrl(parent, size=(420, -1))
        aligner_btn = wx.Button(parent, label=_('textgrid.browse_folder'))
        register(aligner_btn, 'textgrid.browse_folder', 'button')
        aligner_btn.Bind(wx.EVT_BUTTON, lambda event: self.on_browse_folder(event, self.asr_aligner_text))
        row.Add(label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        row.Add(self.asr_aligner_text, 1, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        row.Add(aligner_btn, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        model_sizer.Add(row, 0, wx.EXPAND | wx.ALL, 3)

        model_hint = wx.StaticText(parent, label=_('lab.asr.model_hint'))
        register(model_hint, 'lab.asr.model_hint')
        model_sizer.Add(model_hint, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 8)

        row = wx.BoxSizer(wx.HORIZONTAL)
        lang_label = wx.StaticText(parent, label=_('lab.asr.language'))
        register(lang_label, 'lab.asr.language')
        self.asr_language_choice = wx.Choice(parent, size=(90, -1))
        self.asr_language_choice.Append(_('lab.asr.language.zh'), 'zh')
        self.asr_language_choice.Append(_('lab.asr.language.ja'), 'ja')
        self.asr_language_choice.SetSelection(0)
        self.asr_language_choice.Bind(wx.EVT_CHOICE, self.on_asr_language_changed)
        register(self.asr_language_choice, '', 'choice', [
            ('lab.asr.language.zh', 'zh'),
            ('lab.asr.language.ja', 'ja'),
        ])

        mode_label = wx.StaticText(parent, label=_('lab.asr.output_mode'))
        register(mode_label, 'lab.asr.output_mode')
        self.asr_mode_choice = wx.Choice(parent, size=(90, -1))

        device_label = wx.StaticText(parent, label=_('lab.asr.device'))
        register(device_label, 'lab.asr.device')
        self.asr_device_choice = wx.Choice(parent, size=(110, -1))
        self.asr_device_choice.Append(_('lab.asr.device.auto'), asr_suite.DEVICE_AUTO)
        self.asr_device_choice.Append(_('lab.asr.device.cuda'), asr_suite.DEVICE_CUDA)
        self.asr_device_choice.Append(_('lab.asr.device.cpu'), asr_suite.DEVICE_CPU)
        self.asr_device_choice.SetSelection(0)
        self.asr_device_choice.SetToolTip(_('lab.asr.device.tooltip'))
        register(self.asr_device_choice, 'lab.asr.device.tooltip', 'tooltip')
        register(self.asr_device_choice, '', 'choice', [
            ('lab.asr.device.auto', asr_suite.DEVICE_AUTO),
            ('lab.asr.device.cuda', asr_suite.DEVICE_CUDA),
            ('lab.asr.device.cpu', asr_suite.DEVICE_CPU),
        ])

        dtype_label = wx.StaticText(parent, label=_('lab.asr.dtype'))
        register(dtype_label, 'lab.asr.dtype')
        self.asr_dtype_choice = wx.Choice(parent, size=(110, -1))
        self.asr_dtype_choice.Append(_('lab.asr.dtype.auto'), asr_suite.DTYPE_AUTO)
        self.asr_dtype_choice.Append(_('lab.asr.dtype.bf16'), asr_suite.DTYPE_BF16)
        self.asr_dtype_choice.Append(_('lab.asr.dtype.fp16'), asr_suite.DTYPE_FP16)
        self.asr_dtype_choice.Append(_('lab.asr.dtype.fp32'), asr_suite.DTYPE_FP32)
        self.asr_dtype_choice.SetSelection(0)
        self.asr_dtype_choice.SetToolTip(_('lab.asr.dtype.tooltip'))
        register(self.asr_dtype_choice, 'lab.asr.dtype.tooltip', 'tooltip')
        register(self.asr_dtype_choice, '', 'choice', [
            ('lab.asr.dtype.auto', asr_suite.DTYPE_AUTO),
            ('lab.asr.dtype.bf16', asr_suite.DTYPE_BF16),
            ('lab.asr.dtype.fp16', asr_suite.DTYPE_FP16),
            ('lab.asr.dtype.fp32', asr_suite.DTYPE_FP32),
        ])

        batch_label = wx.StaticText(parent, label=_('lab.asr.batch'))
        register(batch_label, 'lab.asr.batch')
        self.asr_batch_choice = wx.Choice(parent, size=(60, -1))
        for value in (1, 2, 4, 8):
            self.asr_batch_choice.Append(str(value), value)
        self.asr_batch_choice.SetSelection(0)
        self.asr_batch_choice.SetToolTip(_('lab.asr.batch.tooltip'))
        register(self.asr_batch_choice, 'lab.asr.batch.tooltip', 'tooltip')

        row.Add(lang_label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        row.Add(self.asr_language_choice, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        row.Add(mode_label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        row.Add(self.asr_mode_choice, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        row.Add(device_label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        row.Add(self.asr_device_choice, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        row.Add(dtype_label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        row.Add(self.asr_dtype_choice, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        row.Add(batch_label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        row.Add(self.asr_batch_choice, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        model_sizer.Add(row, 0, wx.ALL, 3)

        # ── 分段执行（降低峰值显存） / Aligner 省显存 ──
        row = wx.BoxSizer(wx.HORIZONTAL)
        chunk_label = wx.StaticText(parent, label=_('lab.asr.chunk'))
        register(chunk_label, 'lab.asr.chunk')
        self.asr_chunk_spin = wx.SpinCtrlDouble(
            parent,
            min=float(asr_suite.MIN_CHUNK_SEC),
            max=float(asr_suite.MAX_CHUNK_SEC),
            initial=float(asr_suite.DEFAULT_CHUNK_SEC),
            inc=10.0, size=(100, -1), style=wx.SP_ARROW_KEYS)
        self.asr_chunk_spin.SetDigits(0)
        self.asr_chunk_spin.SetToolTip(_('lab.asr.chunk.tooltip'))
        register(self.asr_chunk_spin, 'lab.asr.chunk.tooltip', 'tooltip')
        self.asr_aligner_cpu_check = wx.CheckBox(parent, label=_('lab.asr.aligner_cpu'))
        self.asr_aligner_cpu_check.SetValue(False)
        self.asr_aligner_cpu_check.SetToolTip(_('lab.asr.aligner_cpu.tooltip'))
        register(self.asr_aligner_cpu_check, 'lab.asr.aligner_cpu.tooltip', 'tooltip')
        register(self.asr_aligner_cpu_check, 'lab.asr.aligner_cpu', 'checkbox')
        self.asr_encoder_cpu_check = wx.CheckBox(parent, label=_('lab.asr.encoder_cpu'))
        self.asr_encoder_cpu_check.SetValue(False)
        self.asr_encoder_cpu_check.SetToolTip(_('lab.asr.encoder_cpu.tooltip'))
        register(self.asr_encoder_cpu_check, 'lab.asr.encoder_cpu.tooltip', 'tooltip')
        register(self.asr_encoder_cpu_check, 'lab.asr.encoder_cpu', 'checkbox')
        self.asr_skip_silence_check = wx.CheckBox(parent, label=_('lab.asr.skip_silence'))
        self.asr_skip_silence_check.SetValue(True)
        self.asr_skip_silence_check.SetToolTip(_('lab.asr.skip_silence.tooltip'))
        register(self.asr_skip_silence_check, 'lab.asr.skip_silence.tooltip', 'tooltip')
        register(self.asr_skip_silence_check, 'lab.asr.skip_silence', 'checkbox')
        row.Add(chunk_label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        row.Add(self.asr_chunk_spin, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        row.Add(self.asr_aligner_cpu_check, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 8)
        row.Add(self.asr_encoder_cpu_check, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 8)
        row.Add(self.asr_skip_silence_check, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 8)
        model_sizer.Add(row, 0, wx.ALL, 3)

        engine_hint = wx.StaticText(parent, label=_('lab.asr.engine_hint'))
        register(engine_hint, 'lab.asr.engine_hint')
        model_sizer.Add(engine_hint, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 8)
        sizer.Add(model_sizer, 0, wx.EXPAND | wx.ALL, 5)

        # ── 提示词 ──
        prompt_box = wx.StaticBox(parent, label=_('lab.asr.prompt'))
        register(prompt_box, 'lab.asr.prompt', 'label')
        prompt_sizer = wx.StaticBoxSizer(prompt_box, wx.VERTICAL)

        system_label = wx.StaticText(parent, label=_('lab.asr.system_prompt'))
        register(system_label, 'lab.asr.system_prompt')
        prompt_sizer.Add(system_label, 0, wx.ALL, 3)
        self.asr_system_prompt_text = wx.TextCtrl(
            parent, value=asr_suite.DEFAULT_SYSTEM_PROMPT,
            style=wx.TE_MULTILINE, size=(-1, 70))
        prompt_sizer.Add(self.asr_system_prompt_text, 0, wx.EXPAND | wx.ALL, 3)

        row = wx.BoxSizer(wx.HORIZONTAL)
        lyrics_label = wx.StaticText(parent, label=_('lab.asr.lyrics'))
        register(lyrics_label, 'lab.asr.lyrics')
        import_btn = wx.Button(parent, label=_('lab.asr.lyrics.import'))
        register(import_btn, 'lab.asr.lyrics.import', 'button')
        import_btn.Bind(wx.EVT_BUTTON, self.on_asr_import_lyrics)
        clear_btn = wx.Button(parent, label=_('lab.asr.lyrics.clear'))
        register(clear_btn, 'lab.asr.lyrics.clear', 'button')
        clear_btn.Bind(wx.EVT_BUTTON, self.on_asr_clear_lyrics)
        save_btn = wx.Button(parent, label=_('lab.asr.lyrics.save'))
        save_btn.SetToolTip(_('lab.asr.lyrics.save.tooltip'))
        register(save_btn, 'lab.asr.lyrics.save.tooltip', 'tooltip')
        register(save_btn, 'lab.asr.lyrics.save', 'button')
        save_btn.Bind(wx.EVT_BUTTON, self.on_asr_save_lyrics)
        row.Add(lyrics_label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        row.AddStretchSpacer()
        self.asr_lyrics_status = wx.StaticText(parent, label="")
        row.Add(self.asr_lyrics_status, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        row.Add(save_btn, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        row.Add(import_btn, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        row.Add(clear_btn, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        prompt_sizer.Add(row, 0, wx.EXPAND | wx.ALL, 3)

        self.asr_lyrics_text = wx.TextCtrl(parent, style=wx.TE_MULTILINE, size=(-1, 90))
        prompt_sizer.Add(self.asr_lyrics_text, 0, wx.EXPAND | wx.ALL, 3)

        hint = wx.StaticText(parent, label=_('lab.asr.prompt.hint'))
        register(hint, 'lab.asr.prompt.hint')
        prompt_sizer.Add(hint, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 8)
        sizer.Add(prompt_sizer, 0, wx.EXPAND | wx.ALL, 5)

        # ── 音频列表（用于逐文件查看/编辑歌词）──
        list_box = wx.StaticBox(parent, label=_('lab.asr.audio_list'))
        register(list_box, 'lab.asr.audio_list', 'label')
        list_sizer = wx.StaticBoxSizer(list_box, wx.VERTICAL)
        self.asr_audio_list = wx.ListBox(parent, style=wx.LB_SINGLE, size=(-1, 110))
        self.asr_audio_list.Bind(wx.EVT_LISTBOX, self._asr_on_list_select)
        list_sizer.Add(self.asr_audio_list, 1, wx.EXPAND | wx.ALL, 3)
        sizer.Add(list_sizer, 0, wx.EXPAND | wx.ALL, 5)

        # ── 输出 ──
        out_box = wx.StaticBox(parent, label=_('lab.asr.output'))
        register(out_box, 'lab.asr.output', 'label')
        out_sizer = wx.StaticBoxSizer(out_box, wx.VERTICAL)

        row = wx.BoxSizer(wx.HORIZONTAL)
        label = wx.StaticText(parent, label=_('lab.asr.out_dir'))
        register(label, 'lab.asr.out_dir')
        self.asr_out_text = wx.TextCtrl(parent, size=(380, -1))
        out_btn = wx.Button(parent, label=_('textgrid.browse_folder'))
        register(out_btn, 'textgrid.browse_folder', 'button')
        out_btn.Bind(wx.EVT_BUTTON, lambda event: self.on_browse_folder(event, self.asr_out_text))
        row.Add(label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        row.Add(self.asr_out_text, 1, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        row.Add(out_btn, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        out_sizer.Add(row, 0, wx.EXPAND | wx.ALL, 3)

        row = wx.BoxSizer(wx.HORIZONTAL)
        self.asr_save_txt_check = wx.CheckBox(parent, label=_('lab.asr.save_txt'))
        register(self.asr_save_txt_check, 'lab.asr.save_txt', 'checkbox')
        self.asr_save_txt_check.SetValue(True)
        self.asr_save_lab_check = wx.CheckBox(parent, label=_('lab.asr.save_lab'))
        register(self.asr_save_lab_check, 'lab.asr.save_lab', 'checkbox')
        self.asr_save_lab_check.SetValue(True)
        self.asr_save_json_check = wx.CheckBox(parent, label=_('lab.asr.save_json'))
        register(self.asr_save_json_check, 'lab.asr.save_json', 'checkbox')
        self.asr_save_json_check.SetValue(True)
        self.asr_overwrite_check = wx.CheckBox(parent, label=_('lab.asr.overwrite'))
        register(self.asr_overwrite_check, 'lab.asr.overwrite', 'checkbox')
        self.asr_overwrite_check.SetValue(True)
        row.Add(self.asr_save_txt_check, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        row.Add(self.asr_save_lab_check, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        row.Add(self.asr_save_json_check, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        row.Add(self.asr_overwrite_check, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        out_sizer.Add(row, 0, wx.ALL, 3)

        hint = wx.StaticText(parent, label=_('lab.asr.out_hint'))
        register(hint, 'lab.asr.out_hint')
        out_sizer.Add(hint, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 8)
        sizer.Add(out_sizer, 0, wx.EXPAND | wx.ALL, 5)

        # ── 运行 ──
        btn_row = wx.BoxSizer(wx.HORIZONTAL)
        self.asr_run_btn = wx.Button(parent, label=_('lab.asr.run'))
        register(self.asr_run_btn, 'lab.asr.run', 'button')
        self.asr_run_btn.Bind(wx.EVT_BUTTON, self.on_asr_generate)
        self.asr_stop_btn = wx.Button(parent, label=_('lab.asr.stop'))
        register(self.asr_stop_btn, 'lab.asr.stop', 'button')
        self.asr_stop_btn.Bind(wx.EVT_BUTTON, self.on_asr_stop)
        self.asr_stop_btn.Disable()
        btn_row.Add(self.asr_run_btn, 0, wx.ALL, 5)
        btn_row.Add(self.asr_stop_btn, 0, wx.ALL, 5)
        sizer.Add(btn_row, 0, wx.CENTER)

        result_label = wx.StaticText(parent, label=_('lab.asr.result'))
        register(result_label, 'lab.asr.result')
        sizer.Add(result_label, 0, wx.ALL | wx.LEFT, 5)
        self.asr_result_text = wx.TextCtrl(parent, style=wx.TE_MULTILINE | wx.TE_READONLY, size=(-1, 220))
        sizer.Add(self.asr_result_text, 1, wx.EXPAND | wx.ALL, 5)

        parent.SetSizer(sizer)
        parent.FitInside()

        self.on_asr_language_changed(None)
        self._asr_auto_fill_models()
        self._asr_update_controls()
        self._asr_refresh_audio_list(None)

    # ── 路径探测 ────────────────────────────────────────────
    @staticmethod
    def _asr_candidate_dirs():
        """ASR / Aligner 权重候选目录（官方 transformers 权重优先）。"""
        roots = [ROOT / 'ASR_model', ROOT / 'models']

        def _scan(keyword):
            found = []
            for root in roots:
                if not root.is_dir():
                    continue
                for child in sorted(root.iterdir()):
                    if child.is_dir() and keyword in child.name.lower():
                        found.append(child)
            # 旧的量化/ONNX 目录排到最后（它们不含 transformers 权重）
            found.sort(key=lambda p: any(tag in p.name.lower() for tag in ('dml', 'gguf', 'onnx')))
            return found

        return {"asr": _scan('asr'), "aligner": _scan('aligner')}

    def _asr_auto_fill_models(self):
        from asr_lab import asr_suite

        candidates = self._asr_candidate_dirs()
        if not self.asr_model_text.GetValue().strip():
            for candidate in candidates["asr"]:
                if asr_suite.is_transformers_model_dir(str(candidate)):
                    self.asr_model_text.SetValue(str(candidate))
                    break
        if not self.asr_aligner_text.GetValue().strip():
            for candidate in candidates["aligner"]:
                if asr_suite.is_transformers_model_dir(str(candidate)):
                    self.asr_aligner_text.SetValue(str(candidate))
                    break

    # ── 事件：音源 / CSV ────────────────────────────────────
    def _asr_browse_folder(self, event):
        if self.on_browse_folder(event, self.asr_folder_text):
            self._asr_selected_files = []
            self.asr_csv_text.SetValue("")
            self._asr_refresh_audio_list(None)

    def _asr_browse_files(self, event):
        dialog = wx.FileDialog(
            None, _('msg.select_files'),
            wildcard="Audio (*.wav;*.flac;*.mp3;*.ogg;*.m4a)|*.wav;*.flac;*.mp3;*.ogg;*.m4a"
                     "|WAV (*.wav)|*.wav|FLAC (*.flac)|*.flac|All files (*.*)|*.*",
            style=wx.FD_OPEN | wx.FD_MULTIPLE | wx.FD_FILE_MUST_EXIST)
        if dialog.ShowModal() != wx.ID_OK:
            dialog.Destroy()
            return
        paths = list(dialog.GetPaths())
        dialog.Destroy()
        if not paths:
            return
        self._asr_selected_files = paths
        self.asr_csv_text.SetValue("")
        names = ", ".join(os.path.basename(path) for path in paths[:3])
        if len(paths) > 3:
            names += f" ... (+{len(paths) - 3})"
        self.asr_folder_text.SetValue(_('msg.files_selected').format(count=len(paths), names=names))
        self._asr_refresh_audio_list(None)

    def _asr_browse_csv(self, event):
        dialog = wx.FileDialog(
            None, _('msg.select_file'),
            wildcard="CSV (*.csv)|*.csv|All files (*.*)|*.*",
            style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)
        if dialog.ShowModal() != wx.ID_OK:
            dialog.Destroy()
            return
        path = dialog.GetPath()
        dialog.Destroy()
        self.asr_csv_text.SetValue(path)
        self._asr_selected_files = []
        self._asr_refresh_audio_list(None)

    def _asr_save_csv_template(self, event):
        from asr_lab import asr_suite

        dialog = wx.FileDialog(
            None, _('msg.select_save_path'),
            defaultFile="asr_batch_template.csv",
            wildcard="CSV (*.csv)|*.csv",
            style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT)
        if dialog.ShowModal() != wx.ID_OK:
            dialog.Destroy()
            return
        path = dialog.GetPath()
        dialog.Destroy()
        try:
            saved = asr_suite.save_csv_template(path)
            wx.MessageBox(_('msg.ok.csv_template_saved').format(path=saved),
                          _('msg.success'), wx.OK | wx.ICON_INFORMATION)
        except Exception:
            wx.MessageBox(traceback.format_exc(), _('msg.error'), wx.OK | wx.ICON_ERROR)

    def _asr_refresh_audio_list(self, event):
        from asr_lab import asr_suite

        # 刷新前先把当前编辑写盘，避免丢改动
        self._asr_flush_lyrics(None)

        csv_path = self.asr_csv_text.GetValue().strip()
        self._asr_lyrics_map = {}
        self._asr_saved_lyrics = {}
        self._asr_csv_notes = []

        if csv_path:
            if not os.path.isfile(csv_path):
                paths = []
                self._asr_csv_notes = [f"CSV 文件不存在：{csv_path}"]
            else:
                try:
                    paths, lyrics_map, notes = asr_suite.collect_csv_tasks(
                        csv_path,
                        recursive=self.asr_recursive_check.GetValue(),
                    )
                    self._asr_lyrics_map = dict(lyrics_map)
                    self._asr_csv_notes = list(notes)
                except Exception as error:
                    paths = []
                    self._asr_csv_notes = [f"CSV 读取失败：{error}"]
        else:
            folder = self.asr_folder_text.GetValue().strip()
            selected = [path for path in self._asr_selected_files if os.path.isfile(path)]
            if selected:
                paths = [Path(path) for path in selected]
            elif folder and os.path.isdir(folder):
                paths = asr_suite.collect_audio_paths(
                    folder, recursive=self.asr_recursive_check.GetValue())
            else:
                paths = []

        self._asr_audio_paths = paths
        self.asr_audio_list.Clear()
        for path in paths:
            self.asr_audio_list.Append(f"{path.name}    ({path.parent})", str(path))

        if csv_path and paths:
            self.asr_csv_status.SetLabel(
                _('lab.asr.csv.status').format(
                    count=len(paths), extra=len(self._asr_csv_notes)))
        else:
            self.asr_csv_status.SetLabel(_('lab.asr.csv.hint'))

        if paths:
            self.asr_audio_list.SetSelection(0)
            self._asr_load_lyrics_for(paths[0])
        else:
            self._asr_current_audio = None
            self.asr_lyrics_text.ChangeValue("")
            self.asr_lyrics_status.SetLabel("")

    def _asr_on_list_select(self, event):
        index = self.asr_audio_list.GetSelection()
        if index == wx.NOT_FOUND:
            return
        path = Path(self.asr_audio_list.GetClientData(index))
        self._asr_load_lyrics_for(path)

    def _asr_load_lyrics_for(self, audio_path: Path):
        from asr_lab import asr_suite

        key = str(audio_path)
        if self._asr_current_audio and self._asr_current_audio != key:
            self._asr_flush_lyrics(None)
        self._asr_current_audio = key
        self.asr_lyrics_status.SetLabel("")
        # 优先级：页面编辑保存的 _lyrics.txt > CSV/列表里的歌词 > 同名 .txt/.lrc
        edited = asr_suite.load_edited_lyrics(audio_path)
        lyrics = edited
        if not lyrics.strip():
            lyrics = self._asr_lyrics_map.get(key, "")
        if not lyrics.strip():
            lyrics = asr_suite.load_sidecar_lyrics(audio_path)
        self._asr_lyrics_map[key] = lyrics
        self._asr_saved_lyrics[key] = lyrics
        self.asr_lyrics_text.ChangeValue(lyrics)
        if edited.strip():
            self.asr_lyrics_status.SetLabel(
                _('lab.asr.lyrics.saved').format(file=f"{audio_path.stem}_lyrics.txt"))

    def _asr_current_lyrics_text(self) -> str:
        """当前输入框里的歌词（控件未创建时返回空字符串）。"""
        try:
            return self.asr_lyrics_text.GetValue()
        except Exception:
            return ""

    def _asr_lyrics_autosave_tick(self, event=None):
        """定时巡检：输入框内容与上次写盘不同才保存（不影响输入框快捷键）。"""
        if not self._asr_current_audio:
            return
        if self._asr_current_lyrics_text() == self._asr_saved_lyrics.get(
                self._asr_current_audio, ""):
            return
        self._asr_flush_lyrics()

    def _asr_flush_lyrics(self, event=None, force: bool = False):
        """把当前音频的歌词写入 ``{音频名}_lyrics.txt``。

        Args:
            force: 为 True 时即使内容与上次相同也写一次盘（手动保存用）。
        """
        from asr_lab import asr_suite

        key = self._asr_current_audio
        if not key:
            return
        text = self._asr_current_lyrics_text()
        self._asr_lyrics_map[key] = text
        if not force and text == self._asr_saved_lyrics.get(key, ""):
            return

        saved = asr_suite.save_lyrics_sidecar(Path(key), text)
        self._asr_saved_lyrics[key] = text
        if saved is None:
            self.asr_lyrics_status.SetLabel("")
        else:
            self.asr_lyrics_status.SetLabel(
                _('lab.asr.lyrics.saved').format(file=saved.name))

    def on_asr_save_lyrics(self, event=None):
        """手动保存当前音频的歌词。"""
        if not self._asr_current_audio:
            self.asr_lyrics_status.SetLabel(_('lab.asr.lyrics.no_audio'))
            return
        self._asr_flush_lyrics(force=True)
        if not self._asr_current_lyrics_text().strip():
            self.asr_lyrics_status.SetLabel(_('lab.asr.lyrics.cleared'))

    def _asr_on_close(self, event):
        try:
            self._asr_lyrics_timer.Stop()
        except Exception:
            pass
        self._asr_flush_lyrics()
        event.Skip()

    def on_asr_import_lyrics(self, event):
        dialog = wx.FileDialog(
            None, _('msg.select_file'),
            wildcard="Lyrics (*.txt;*.lrc)|*.txt;*.lrc|Text (*.txt)|*.txt|LRC (*.lrc)|*.lrc",
            style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)
        if dialog.ShowModal() != wx.ID_OK:
            dialog.Destroy()
            return
        path = dialog.GetPath()
        dialog.Destroy()
        try:
            from asr_lab import asr_suite

            text = asr_suite.read_text_auto(Path(path))
        except Exception:
            wx.MessageBox(traceback.format_exc(), _('msg.error'), wx.OK | wx.ICON_ERROR)
            return
        self.asr_lyrics_text.ChangeValue(text)
        if self._asr_current_audio:
            self._asr_lyrics_map[self._asr_current_audio] = text
            self._asr_flush_lyrics(force=True)

    def on_asr_clear_lyrics(self, event):
        self.asr_lyrics_text.ChangeValue("")
        if self._asr_current_audio:
            self._asr_lyrics_map[self._asr_current_audio] = ""
            self._asr_flush_lyrics(force=True)
            self.asr_lyrics_status.SetLabel(_('lab.asr.lyrics.cleared'))

    # ── 事件：选项 ──────────────────────────────────────────
    def _asr_update_controls(self):
        """按运行模式启用/禁用相关控件。"""
        from asr_lab import asr_suite

        mode = self.asr_run_mode_choice.GetClientData(self.asr_run_mode_choice.GetSelection())
        align_only = mode == asr_suite.RUN_MODE_ALIGN_ONLY
        needs_aligner = mode in (asr_suite.RUN_MODE_FULL, asr_suite.RUN_MODE_ALIGN_ONLY)

        self.asr_model_text.Enable(not align_only)
        self.asr_aligner_text.Enable(needs_aligner)
        # Aligner 用 CPU 只在「两个模型都加载」时才有意义
        self.asr_aligner_cpu_check.Enable(mode == asr_suite.RUN_MODE_FULL)
        # 只对齐模式不重新识别，提示词/歌词输入框用不上
        self.asr_system_prompt_text.Enable(not align_only)
        self.asr_lyrics_text.Enable(not align_only)
        self.asr_save_txt_check.Enable(not align_only)
        self.asr_run_btn.SetLabel(
            _('lab.asr.run.align') if align_only else _('lab.asr.run'))

    def on_asr_run_mode_changed(self, event=None):
        self._asr_update_controls()

    def on_asr_language_changed(self, event):
        from asr_lab import asr_suite

        language = self.asr_language_choice.GetClientData(self.asr_language_choice.GetSelection()) or 'zh'
        self.asr_mode_choice.Clear()
        for value, label in asr_suite.OUTPUT_MODES.get(language, asr_suite.OUTPUT_MODES['zh']):
            self.asr_mode_choice.Append(label, value)
        if self.asr_mode_choice.GetCount():
            self.asr_mode_choice.SetSelection(0)

    # ── 运行 ────────────────────────────────────────────────
    def on_asr_stop(self, event):
        self._asr_stop.set()
        self.asr_stop_btn.Disable()
        self.asr_result_text.AppendText(_('log.infer_stop_requested'))

    def on_asr_generate(self, event):
        from asr_lab import asr_suite

        folder = self.asr_folder_text.GetValue().strip()
        csv_path = self.asr_csv_text.GetValue().strip()
        selected = [path for path in self._asr_selected_files if os.path.isfile(path)]

        if csv_path:
            if not os.path.isfile(csv_path):
                wx.MessageBox(f"CSV 文件不存在：{csv_path}", _('msg.error'), wx.OK | wx.ICON_ERROR)
                return
        elif not selected and (not folder or not os.path.isdir(folder)):
            wx.MessageBox(_('lab.asr.need_audio'), _('msg.error'), wx.OK | wx.ICON_ERROR)
            return

        settings = asr_suite.AsrSettings(
            asr_model_dir=self.asr_model_text.GetValue().strip(),
            aligner_model_dir=self.asr_aligner_text.GetValue().strip(),
            device=self.asr_device_choice.GetClientData(self.asr_device_choice.GetSelection())
            or asr_suite.DEVICE_AUTO,
            dtype=self.asr_dtype_choice.GetClientData(self.asr_dtype_choice.GetSelection())
            or asr_suite.DTYPE_AUTO,
            language=self.asr_language_choice.GetClientData(self.asr_language_choice.GetSelection()) or 'zh',
            output_mode=self.asr_mode_choice.GetClientData(self.asr_mode_choice.GetSelection()) or 'pinyin',
            system_prompt=self.asr_system_prompt_text.GetValue(),
            out_dir=self.asr_out_text.GetValue().strip(),
            recursive=self.asr_recursive_check.GetValue(),
            overwrite=self.asr_overwrite_check.GetValue(),
            batch_size=self.asr_batch_choice.GetClientData(self.asr_batch_choice.GetSelection()) or 1,
            save_txt=self.asr_save_txt_check.GetValue(),
            save_lab=self.asr_save_lab_check.GetValue(),
            save_json=self.asr_save_json_check.GetValue(),
            chunk_size_sec=float(self.asr_chunk_spin.GetValue()),
            aligner_on_cpu=self.asr_aligner_cpu_check.GetValue(),
            encoder_on_cpu=self.asr_encoder_cpu_check.GetValue(),
            skip_silence=self.asr_skip_silence_check.GetValue(),
            run_mode=self.asr_run_mode_choice.GetClientData(
                self.asr_run_mode_choice.GetSelection()) or asr_suite.RUN_MODE_FULL,
        )

        try:
            asr_suite.validate_settings(settings)
        except Exception as error:
            wx.MessageBox(str(error), _('msg.error'), wx.OK | wx.ICON_ERROR)
            return

        if csv_path:
            try:
                audio_paths, csv_map, notes = asr_suite.collect_csv_tasks(
                    csv_path, recursive=settings.recursive)
            except Exception as error:
                wx.MessageBox(f"CSV 读取失败：{error}", _('msg.error'), wx.OK | wx.ICON_ERROR)
                return
            lyrics_map = dict(csv_map)
        else:
            audio_paths = asr_suite.collect_audio_paths(
                folder, selected, recursive=settings.recursive)
            notes = []
            lyrics_map = {}

        # 页面上手动编辑过的歌词优先（点「开始识别」前先把改动落盘）
        if self._asr_current_audio:
            self._asr_flush_lyrics(force=True)
        for key, value in self._asr_lyrics_map.items():
            if str(value).strip():
                lyrics_map[key] = value

        if not audio_paths:
            wx.MessageBox(_('lab.asr.need_audio'), _('msg.error'), wx.OK | wx.ICON_ERROR)
            return

        def worker():
            try:
                wx.CallAfter(self.asr_result_text.Clear)
                self._asr_stop.clear()
                wx.CallAfter(self.asr_run_btn.Disable)
                wx.CallAfter(self.asr_stop_btn.Enable)

                def progress(message):
                    wx.CallAfter(_append_output, self.asr_result_text, message + "\n")

                def stream(text):
                    # 流式片段：不加换行，直接续在 "> " 后面
                    wx.CallAfter(_append_output, self.asr_result_text, text)

                stats = asr_suite.run_batch(
                    audio_paths,
                    settings,
                    lyrics_map=lyrics_map,
                    notes=notes,
                    progress_callback=progress,
                    stream_callback=stream,
                    cancel_checker=self._asr_stop.is_set,
                )

                if stats["stopped"]:
                    progress(_('lab.asr.stopped'))
                else:
                    progress(_('lab.asr.done').format(
                        written=stats['written'], failed=stats['failed']))
                    if stats["errors"]:
                        progress("\n".join(f"✗ {item}" for item in stats["errors"]))
            except Exception:
                tb = traceback.format_exc()
                wx.CallAfter(_append_output, self.asr_result_text, _('log.failed').format(error=tb))
                wx.CallAfter(wx.MessageBox, _('log.failed').format(error=tb),
                             _('msg.error'), wx.OK | wx.ICON_ERROR)
            finally:
                wx.CallAfter(self.asr_stop_btn.Disable)
                wx.CallAfter(self.asr_run_btn.Enable)

        threading.Thread(target=worker, daemon=True).start()
