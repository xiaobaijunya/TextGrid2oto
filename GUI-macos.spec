# -*- mode: python ; coding: utf-8 -*-
# 打包命令：pyinstaller GUI-macos.spec
#
# 布局与 GUI.spec 一致：用户可见的资源放在可执行文件同级，
# Python 依赖与 DLL 收在 _internal/ 里。
import os
import shutil

# 需要暴露到可执行文件同级的资源目录（相对项目根目录）
VISIBLE_DIRS = [
    'i18n',
    'config',
    'img',
    'presamp',
    'tg2svdb/dicts',
]

a = Analysis(
    ['GUI.py'],
    pathex=[],
    binaries=[],
    datas=[
        # ASR：LyricFA 词典（G2P 用）与官方 qwen_asr 的分词词表，
        # 运行时通过 Path(__file__) 相对定位，必须保持在各自包内。
        ('asr_lab/LyricFA/Dicts', 'asr_lab/LyricFA/Dicts'),
        ('qwen_asr/inference/assets', 'qwen_asr/inference/assets'),
    ],
    hiddenimports=[
        # 官方 Qwen3-ASR（torch + transformers 后端，内置在项目根目录）
        'qwen_asr',
        'qwen_asr.inference.qwen3_asr',
        'qwen_asr.inference.qwen3_forced_aligner',
        'qwen_asr.core.transformers_backend',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    noarchive=False,
    # 等价于 python -OO：级别 2 会去掉 assert 语句和文档字符串，
    # 减小体积、略微加快启动。可用环境变量 PYI_OPTIMIZE 覆盖。
    optimize=int(os.environ.get('PYI_OPTIMIZE', '2')),
)
pyz = PYZ(a.pure)
# macOS: 创建 .app 包
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='TextGrid2oto',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='TextGrid2oto.app',
)

# ── 把「用户可见」的资源复制到可执行文件同级 ─────────────────
# macOS 下 sys.executable 位于 TextGrid2oto.app/ 内，故写入 coll.name 即可。


def _ignore_junk(_dir, names):
    """复制时跳过 __pycache__ 等构建垃圾。"""
    return [n for n in names if n == '__pycache__' or n.endswith(('.pyc', '.pyo'))]


for _rel in VISIBLE_DIRS:
    _src = os.path.join(SPECPATH, _rel)
    _dst = os.path.join(coll.name, _rel)
    if not os.path.isdir(_src):
        print(f'[WARN] 资源目录不存在，已跳过: {_src}')
        continue
    shutil.rmtree(_dst, ignore_errors=True)
    shutil.copytree(_src, _dst, ignore=_ignore_junk)
    print(f'[OK] {_rel}  ->  {_dst}')

# ── 模型目录：只带「目录结构 + README + 配置/字典」，不带权重 ──
# 权重（*.onnx / *.safetensors / …）体积过大，由用户从 Release 单独下载
# 后放进对应目录；包内保留目录结构和 README 指引，避免用户不知道放哪里。

MODEL_DIRS = [
    'HubertFA_model',
    'TIFA_model',
    'GAME_model',
]

MODEL_WEIGHT_SUFFIXES = (
    '.onnx', '.onnx_data', '.ort', '.gguf',
    '.safetensors', '.bin', '.pt', '.pth', '.ckpt',
)


def _ignore_weights(_dir, names):
    ignored = _ignore_junk(_dir, names)
    ignored += [n for n in names if n.lower().endswith(MODEL_WEIGHT_SUFFIXES)]
    return ignored


for _rel in MODEL_DIRS:
    _src = os.path.join(SPECPATH, _rel)
    _dst = os.path.join(coll.name, _rel)
    if not os.path.isdir(_src):
        print(f'[WARN] 模型目录不存在，已跳过: {_src}')
        continue
    shutil.rmtree(_dst, ignore_errors=True)
    shutil.copytree(_src, _dst, ignore=_ignore_weights)
    print(f'[OK] {_rel}（不含权重）  ->  {_dst}')

# 根目录说明文档 + ASR 模型指引（ASR 权重太大，只带 README）
for _rel in ('README.MD', 'ASR_model/README.md'):
    _src = os.path.join(SPECPATH, _rel)
    _dst = os.path.join(coll.name, _rel)
    if not os.path.isfile(_src):
        print(f'[WARN] 说明文档不存在，已跳过: {_src}')
        continue
    os.makedirs(os.path.dirname(_dst), exist_ok=True)
    shutil.copy2(_src, _dst)
    print(f'[OK] {_rel}  ->  {_dst}')