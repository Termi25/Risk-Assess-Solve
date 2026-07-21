# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller build specification for the Student Risk Assessment app.

Produces a single Windows executable (dist/RiskSolvingApp.exe) that bundles:
  * the PySide6 GUI,
  * the ML stack (xgboost + shap + lime + scikit-learn + imbalanced-learn),
  * the optional Anthropic SDK,
  * the pre-trained model artifact (app/artifacts/*.json).

Build locally with:  pyinstaller RiskSolvingApp.spec
(Run `python train.py` first so the model artifact exists to bundle.)
"""

import os
from PyInstaller.utils.hooks import (
    collect_all, collect_data_files, collect_dynamic_libs, copy_metadata,
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
# The cloud SDKs (anthropic, google.genai) are imported lazily inside
# llm_client, so collect_all is what actually pulls them into the bundle.
for pkg in ("shap", "imblearn", "sklearn", "anthropic", "keyring", "google.genai"):
    pkg_datas, pkg_binaries, pkg_hidden = collect_all(pkg)
    datas += pkg_datas
    binaries += pkg_binaries
    hiddenimports += pkg_hidden

# keyring discovers its OS backend via importlib.metadata entry points, which
# PyInstaller can miss — bundle its package metadata and pin the Windows backend
# (plus the pywin32-ctypes module it relies on) explicitly.
datas += copy_metadata("keyring")
hiddenimports += [
    "keyring.backends.Windows",
    "keyring.backends.chainer",
    "keyring.backends.fail",
    "win32ctypes.core",
    "win32ctypes.core.ctypes",
]

# lime is imported lazily inside lime_explainer, so the import graph never sees
# it — name the tabular module explicitly. Only `lime.lime_image` needs
# scikit-image (and its imageio/networkx/tifffile chain); this app uses the
# tabular explainer only, so those are excluded below to keep the exe small.
hiddenimports += ["lime", "lime.lime_tabular", "lime.discretize"]

a = Analysis(
    ["run.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "tkinter", "pytest", "PyInstaller",
        # Pulled in transitively by lime for its image explainer, which this
        # app never imports. Verified: lime/__init__.py is empty and
        # lime.lime_tabular does not touch skimage.
        "skimage", "imageio", "networkx", "tifffile", "lime.lime_image",
    ],
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
