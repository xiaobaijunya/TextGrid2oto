# -*- mode: python ; coding: utf-8 -*-
#pyinstaller GUI.spec

# 需要随程序一起分发的资源（源路径, 打包后目录）
# 注意：ONNX 模型（HubertFA_model，约 244MB）体积过大，不打包，
#       由用户自行到 Release 下载后放到程序根目录。
datas = [
    ('img/TextGrid2oto.ico', 'img'),
    ('i18n', 'i18n'),
    ('config', 'config'),
    ('presamp', 'presamp'),
    ('tg2svdb/字典', 'tg2svdb/字典'),
]

a = Analysis(
    ['GUI.py'],
    pathex=[],
    binaries=[],
    datas=datas,
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