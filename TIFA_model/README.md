# TIFA_model（模型权重不入库）

TIFA 强制对齐的 ONNX 模型目录。

> 本仓库**只保留目录结构、配置和字典文件**，`*.onnx` 权重被 `.gitignore` 排除，需自行下载后放入。
> 请勿删除本文件，它保证克隆仓库后目录仍然存在。

## 目录结构

```
TIFA_model/
└── TIFA-1.0-ST/
    ├── model.onnx                
    ├── spectrogram.onnx         
    ├── prepare.onnx               
    ├── score.onnx                 
    ├── select.onnx                 
    ├── config.json
    ├── vocabulary.json
    ├── ds-zh-pinyin-lite.txt
    ├── en-arpasing.txt
    ├── ja-hira-假名.txt
    ├── ja-romaji-罗马音.txt
    └── jyutping_dict.txt
```


## 放置方式

模型来源为 TIFA 官方权重导出的 ONNX 版本（请在 TIFA 官方发布页获取）。
下载后按原文件名放入 `TIFA_model/TIFA-1.0-ST/` 即可，例如：

```
TIFA_model/TIFA-1.0-ST/model.onnx
```

> GUI 会列出 `TIFA_model` 下的所有子目录供选择，因此也可以新建子目录存放其他版本；
> 若 `TIFA_model` 下没有任何子目录，程序会退回使用 `TIFA_model` 根目录作为模型目录。
