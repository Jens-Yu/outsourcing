# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['bending_reinforcement_analysis_2.0.py'],
    pathex=[],
    binaries=[],
    datas=[('D:\\Programfiles\\outsourcing\\python_software\\Figure_1.png', '.'), ('D:\\Programfiles\\outsourcing\\python_software\\Figure_2.png', '.'), ('D:\\Programfiles\\outsourcing\\python_software\\Figure_3.png', '.')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=True,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [('v', None, 'OPTION')],
    name='bending_reinforcement_analysis_2.0',
    debug=True,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
