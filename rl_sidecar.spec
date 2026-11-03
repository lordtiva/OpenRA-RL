# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['C:/Users/lordc/Desktop/OpenRA-RL/ra_rl/play_skirmish.py'],
    pathex=['C:/Users/lordc/Desktop/OpenRA-RL'],
    binaries=[],
    datas=[],
    hiddenimports=['torch', 'numpy', 'grpc', 'ra_env', 'ra_rl'],
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
    a.binaries,
    a.datas,
    [],
    name='rl_sidecar',
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
)
