# -*- mode: python ; coding: utf-8 -*-
# 打包命令：pyinstaller GUI.spec
#
# ── 最终目录布局 ────────────────────────────────────────────
#   dist/TextGrid2oto/
#   ├── TextGrid2oto.exe
#   ├── i18n/  config/  img/  presamp/  tg2svdb/字典/   ← 用户可见，与 exe 同级
#   ├── HubertFA_model/                                  ← 用户自行下载放入
#   └── _internal/                                       ← Python 依赖与 DLL，用户无需关心
#
# 程序通过 get_app_root()（sys.executable 的父目录）查找上述资源，
# 因此它们必须与 exe 同级，不能待在 _internal 里。
# 但 PyInstaller 6 会把 datas 统一放进 _internal，且禁止用 ../ 逃逸，
# 所以这里改为「COLLECT 之后再手动复制」的方式（见文件末尾）。

import os
import shutil

# 需要暴露到 exe 同级的资源目录（相对项目根目录）
VISIBLE_DIRS = [
    'i18n',
    'config',
    'img',
    'presamp',
    'tg2svdb/字典',
]

a = Analysis(
    ['GUI.py'],
    pathex=[],
    binaries=[],
    datas=[],          # 可见资源不在这里声明，见文件末尾的复制步骤
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    exclude_binaries=True,
    name='TextGrid2oto',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='./img/TextGrid2oto.ico'
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='TextGrid2oto'
)

# ── 把「用户可见」的资源复制到 exe 同级 ──────────────────────
# 执行到此处时 COLLECT 已完成 dist 目录构建（Target.__postinit__ 会立即
# 调用 assemble()），所以可以直接写入 coll.name（即 dist/TextGrid2oto）。


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