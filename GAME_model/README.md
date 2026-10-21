# GAME_model（模型权重不入库）

GAME ONNX 模型目录，用于给 `transcriptions.csv` 生成 `note_seq` / `note_dur` / `note_slur`（MIDI 标注）。

> 本仓库**只保留目录结构与 `config.json`**，`*.onnx` 权重被 `.gitignore` 排除，需自行下载后放入。
> 请勿删除本文件，它保证克隆仓库后目录仍然存在。

## 目录结构

```
GAME_model/
├── GAME-1.0.3-medium-onnx/
│   ├── encoder.onnx                ← 需自行下载（本仓库不含）
│   ├── segmenter.onnx              ← 需自行下载（本仓库不含）
│   ├── estimator.onnx              ← 需自行下载（本仓库不含）
│   ├── dur2bd.onnx                 ← 需自行下载（本仓库不含）
│   ├── bd2dur.onnx                 ← 需自行下载（本仓库不含）
│   └── config.json                 # 已包含
└── GAME-1.0.3-small-onnx/
    ├── encoder.onnx                ← 需自行下载（本仓库不含）
    ├── segmenter.onnx              ← 需自行下载（本仓库不含）
    ├── estimator.onnx              ← 需自行下载（本仓库不含）
    ├── dur2bd.onnx                 ← 需自行下载（本仓库不含）
    ├── bd2dur.onnx                 ← 需自行下载（本仓库不含）
    └── config.json                 # 已包含
```

## 必需文件

每个模型文件夹（`GAME_model/<模型名>/`）必须同时包含下面 6 个文件，**缺一不可**，GUI 会逐个检查：

```
encoder.onnx
segmenter.onnx
estimator.onnx
dur2bd.onnx
bd2dur.onnx
config.json
```

`config.json` 决定可用语言与采样率等参数（当前支持 `en` / `ja` / `yue` / `zh`），已随仓库提供，无需下载。

## 获取模型

模型来源：[openvpi/GAME](https://github.com/openvpi/GAME/releases) 下载 ONNX 后按上述文件名放入。
本项目不随仓库分发权重文件。

## 说明

- `medium` 显存占用更高；`small` 更轻量。
- 推理面板中的「最大分块」默认 30 秒，显存不足时可降到 8~16 秒。
