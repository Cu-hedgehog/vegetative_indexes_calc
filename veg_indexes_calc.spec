# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_submodules, collect_data_files
rast_hidden_imports = collect_submodules('rasterio')
rast_datas = collect_data_files('rasterio', subdir=None, include_py_files=True)

a = Analysis(
    ['veg_indexes_calc.py'],
    pathex=['path\\to\\.venv\\Lib\\site-packages'],
    binaries=[],
    datas=rast_datas+[('./help/ARVI_help.txt','help'), ('./help/GNDVI_help.txt','help'), ('./help/MSAVI_help.txt','help'), ('./help/NDVI_help.txt','help'), ('./help/NDWI_help.txt','help'), ('./help/SAVI_help.txt','help'), ('./models/model.learner.pth','models')],
    hiddenimports=rast_hidden_imports+[],
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
    [],
    exclude_binaries=True,
    name='Vegetative Indexes Calculator',
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
    name='Vegetative Indexes Calculator',
)
