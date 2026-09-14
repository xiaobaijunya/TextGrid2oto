# -*- mode: python ; coding: utf-8 -*-
#pyinstaller GUI.spec
a = Analysis(
    ['GUI.py'],
    pathex=[],
    binaries=[],
    # 需要随程序一起分发的资源（源路径, 打包后目录）
    # 注意：ONNX 模型（HubertFA_model）体积过大，不打包，由用户自行下载。
    datas=[
        ("img/TextGrid2oto.ico", "img"),
        ("i18n", "i18n"),
        ("config", "config"),
        ("presamp", "presamp"),
        ("tg2svdb/字典", "tg2svdb/字典")
    ],
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