# asr_lab —— Qwen3-ASR + Qwen3-ForcedAligner

统一推理入口：`asr_suite.py`；GUI 页面：`asr_panel_ui.py`。

```
音频文件夹 / 选中的音频
  └─ 每条音频 → ASR（系统提示词 + 参考歌词）
       → Qwen3-ForcedAligner 逐字时间戳
       → {名}.txt / {名}.lab / {名}_ASR.json
```

## 推理后端

只保留官方实现：项目根目录内置的 `qwen_asr` 包（与 `QwenLM/Qwen3-ASR` 同源），
后端为 **torch + transformers**。

| 模块 | 作用 |
| --- | --- |
| `qwen_asr.Qwen3ASRModel` | ASR 识别（开启时间戳时连同 ForceAligner 一起加载） |
| `qwen_asr.Qwen3ForcedAligner` | 逐字时间戳；「只对齐」模式单独加载它 |

依赖见 `requirements-asr.txt`（`torch` / `transformers` / `accelerate` / `nagisa` / `soynlp`），
**不需要** `onnxruntime` 或 llama.cpp。

工作方式与官方 `app.py` 完全一致：

* 不做用户级切片；开启时间戳时按官方 180 秒上限内部分段，
  切点会自动吸附到边界附近的静音处，不会把一个字切成两半；
* 不启用歌词匹配，ASR 输出什么就用什么；
* 运行模式三选一：`识别+对齐` / `只识别`（不加载 Aligner）/ `只对齐`（读已有 `.txt`，只加载 Aligner）；
* 设备（自动 / CUDA / CPU）与精度（自动 / bfloat16 / float16 / float32）可选，
  模型目录支持 ModelScope 缓存布局（`<模型>/snapshots/<版本>` 会自动定位）。

## 输出

| 文件 | 内容 |
| --- | --- |
| `{名}.txt` | ASR 原始文本 |
| `{名}.lab` | 拼音（中文）或罗马音（日文），空格分隔；也可选汉字 / 假名 |
| `{名}_ASR.json` | 合并结果：`text`、`lab`、`items`（逐字 `text/pinyin/start/end`）、提示词、耗时等 |

`_ASR.json` 可直接被 `textgrid2json/aligner_json2tg.py` 转成 Praat TextGrid。

## 提示词

页面有两个输入框：

1. **系统提示词**（默认已填好，可改）
2. **歌词 / 补充提示**（可为空；每个音频可单独编辑）

最终提示词 = `系统提示词 + "\n\n" + 参考歌词块`。

编辑歌词后每 **3 秒巡检一次**（内容有变化才写盘），也可点「保存歌词」立即落盘；
切换音频、刷新列表、开始识别、关闭窗口时也会立即保存。读取优先级：

1. `{名}_lyrics.txt`（页面编辑保存的草稿）
2. CSV / 列表中提供的歌词
3. 同目录同名 `.txt` / `.lrc`（会跳过本程序上次生成的识别结果）

点「清空」会删除 `{名}_lyrics.txt`，恢复使用 CSV / 同名文本。

## 批量 CSV

页面支持一个 CSV 统一给多个音频/文件夹套歌词，表头固定为英文：

```csv
path,title,lyrics
10,,如果那两个字没有颤抖
11,歌名,第一句歌词
segments/12.wav,,这是单条音频的歌词
```

- `path`：音频文件或文件夹；**相对路径按 CSV 所在目录解析**
- `title`：标题，仅用于日志，可空
- `lyrics`：歌词；该行指向文件夹时，文件夹内（可勾选递归）所有音频都用这段歌词
- CSV 填写后**优先于**页面上的文件夹 / 手动选择
- 「生成模板」按钮会写出带 `path,title,lyrics` 表头的 UTF-8 BOM 模板（Excel 直接打开不乱码）
- 兼容旧模板：`lyrics` 列为空、而第 2 列是多行长文本时，会回退把它当歌词，并在日志里提示

