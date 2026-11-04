"""ASR 相关模块。

- ``asr_suite``：Qwen3-ASR + Qwen3-ForcedAligner 批量推理
  （官方 ``qwen_asr`` 包，torch + transformers 后端；输出 .txt / .lab / _ASR.json）
- ``asr_panel_ui``：GUI 的 ASR 页面（mixin）
- ``LyricFA``：中/日文 G2P 词典（生成 .lab 用）

注意：这里不在包初始化时导入具体模块，避免启动 GUI 时被
librosa / torch 等重依赖拖慢。
"""
