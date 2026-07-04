# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller build specification for the Student Risk Assessment app.

Produces a single Windows executable (dist/RiskSolvingApp.exe) that bundles:
  * the PySide6 GUI,
  * the ML stack (xgboost + shap + scikit-learn + imbalanced-learn),
  * the optional Anthropic SDK,
  * the pre-trained model artifact (app/artifacts/*.json).

Build locally with:  pyinstaller RiskSolvingApp.spec
(Run `python train.py` first so the model artifact exists to bundle.)
"""

import os
from PyInstaller.utils.hooks import (
    collect_all, collect_data_files, collect_dynamic_libs,
)

block_cipher = None

# Bundle the trained model so the app never has to train on first launch.
datas = []
for fname in ("risk_model.json", "risk_model.meta.json"):
    path = os.path.join("app", "artifacts", fname)
    if os.path.exists(path):
        datas.append((path, os.path.join("app", "artifacts")))

binaries = []
hiddenimports = []

# xgboost ships a native DLL (and a VERSION file) but its `xgboost.testing`
# submodule hard-imports pytest/hypothesis — so we grab the data + shared
# library directly instead of walking its submodules, and let PyInstaller's
# import graph pick up the code paths actually used by the app.
datas += collect_data_files("xgboost")
binaries += collect_dynamic_libs("xgboost")

# These packages ship data files / lazy imports the default analysis can miss.
for pkg in ("shap", "imblearn", "sklearn", "anthropic"):
    pkg_datas, pkg_binaries, pkg_hidden = collect_all(pkg)
    datas += pkg_datas
    binaries += pkg_binaries
    hiddenimports += pkg_hidden

a = Analysis(
    ["run.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter", "pytest", "PyInstaller"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="RiskSolvingApp",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,        # GUI app — no console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
