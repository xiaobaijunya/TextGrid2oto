# -*- mode: python ; coding: utf-8 -*-
#pyinstaller GUI.spec

a = Analysis(
    ['GUI.py'],
    pathex=[],
    binaries=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

#    datas=[
#    ("presamp", "presamp"),
#    ("HubertFA_model/1218_hfa_model","HubertFA_model/1218_hfa_model"),
#    ("tg2svdb/字典","tg2svdb/字典"),
#    ("img/TextGrid2oto.ico","img")
#],


pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='tg转换多种标记',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='./img/TextGrid2oto.ico'
)
