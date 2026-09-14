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
    noarchive=False,
    optimize=0,
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