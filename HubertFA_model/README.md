# HubertFA_model（模型权重不入库）

HubertFA 强制对齐（forced alignment）的 ONNX 模型目录。

> 本仓库**只保留目录结构、配置和字典文件**，`*.onnx` 权重被 `.gitignore` 排除，需自行下载后放入。
> 请勿删除本文件，它保证克隆仓库后目录仍然存在。

## 目录结构

```
HubertFA_model/
├──  1218_hfa_model/
    ├── model.onnx                  ← 需自行下载（本仓库不含）
    ├── config.json
    ├── vocab.json
    ├── VERSION                     # 内容必须为 5
    ├── zh.txt
    ├── ds-zh-pinyin.txt
    ├── en-arpasing.txt
    ├── ja-hira-假名.txt
    └── ja-romaji-罗马音.txt
```

## 下载与放置

- 📦 **HubertFA_model 下载（Release v0.1.0）**：
  <https://github.com/xiaobaijunya/TextGrid2oto/releases/tag/v0.1.0>
  资源文件：`HubertFA_model.zip`（约 244 MB）
  SHA256：`fde3cd59795754e7e1b2fa5495e40b43fe1efa28e3588765022542abe6ec8eb7`
- 模型源自 [HubertFA v0.0.7](https://github.com/wolfgitpr/HubertFA/releases/tag/v0.0.7)，为适配本项目架构做了部分修改。

解压后，把每个模型文件夹里的 `model.onnx` 放到本目录对应位置即可（其余文件本仓库已包含）：

```
HubertFA_model/1218_hfa_model/model.onnx
```

## 注意事项

1. 层级必须是 `HubertFA_model/<模型名>/model.onnx`，**不要多套一层同名目录**
   （出现 `HubertFA_model/HubertFA_model/...` 就是错的）。
2. `model.onnx` 必须与 `vocab.json`、`config.json`、`VERSION` 位于同一个模型文件夹内；
   推理时缺少 `VERSION` 或 `VERSION` 不是 `5` 会直接报错。
3. 在 GUI 中选择模型时，要选到**具体的模型文件夹**（如 `1218_hfa_model`），而不是 `HubertFA_model` 这一层。
