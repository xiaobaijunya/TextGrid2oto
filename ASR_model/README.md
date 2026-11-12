# ASR_model（模型权重不入库）

ASR（语音识别 + 强制对齐）使用的 HuggingFace / ModelScope 权重目录，
对应 GUI「ASR 识别」面板里的 **ASR 模型** 与 **对齐模型**。

> 本目录**只保留这份说明**，权重文件（`*.safetensors` 等，单个 1~4 GB）
> 被 `.gitignore` 排除，也不随安装包分发，需自行下载后放入。
> 请勿删除本文件，它保证克隆仓库 / 解压安装包后目录仍然存在。

## 目录结构

```
ASR_model/
├── Qwen--Qwen3-ASR-1.7B/                 ← 识别模型（约 3.5 GB）
│   └── snapshots/master/                 # 含 config.json + *.safetensors + tokenizer
└── Qwen--Qwen3-ForcedAligner-0.6B/       ← 对齐模型（约 1.2 GB）
    └── snapshots/master/                 # 含 config.json + *.safetensors + tokenizer
```

也支持直接放成「权重目录」形态（`config.json` 与 `*.safetensors` 在同一个文件夹里）：

```
ASR_model/Qwen3-ASR-1.7B/{config.json, *.safetensors, tokenizer*}
ASR_model/Qwen3-ForcedAligner-0.6B/{config.json, *.safetensors, tokenizer*}
```

程序会自动在 `ASR_model/` 与 `models/` 下扫描名字含 `asr` / `aligner`
且能解析出权重的目录，并在面板里自动填入；也可以手动点「浏览」选择任意目录。

## 获取模型

- 识别模型：<https://huggingface.co/Qwen/Qwen3-ASR-1.7B>
- 对齐模型：<https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B>

下载方式二选一：

```bash
# 方式 1：huggingface-cli（推荐）
pip install -U huggingface_hub
huggingface-cli download Qwen/Qwen3-ASR-1.7B \
    --local-dir ASR_model/Qwen3-ASR-1.7B
huggingface-cli download Qwen/Qwen3-ForcedAligner-0.6B \
    --local-dir ASR_model/Qwen3-ForcedAligner-0.6B
```

```bash
# 方式 2：ModelScope（国内网络更快）
pip install -U modelscope
modelscope download --model Qwen/Qwen3-ASR-1.7B \
    --local_dir ASR_model/Qwen--Qwen3-ASR-1.7B
modelscope download --model Qwen/Qwen3-ForcedAligner-0.6B \
    --local_dir ASR_model/Qwen--Qwen3-ForcedAligner-0.6B
```

## 注意

1. 只用 HubertFA / TIFA / GAME 推理时，**本目录可以整空**，不影响其他功能。
2. ASR 需要 `torch + transformers`（`requirements-asr.txt`），显存不足时可在面板
   勾选「对齐模型跑在 CPU」「encoder 卸载到内存」「分段执行」。
3. 缺少权重时报错信息里会列出当前目录，按上面结构补齐即可。
