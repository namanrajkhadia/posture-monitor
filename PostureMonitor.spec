# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for Posture Monitor.

Build with:
    pyinstaller --noconfirm PostureMonitor.spec

Produces dist/PostureMonitor/ — a self-contained folder that runs without
Python installed. Zip the folder for distribution.
"""
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# Mediapipe ships .binarypb / .tflite assets next to the python modules; without
# this PyInstaller ships the .py files but not the model graphs. Same for cv2.
hidden_imports = []
hidden_imports += collect_submodules("mediapipe")
hidden_imports += ["pystray._win32"]

datas = []
datas += collect_data_files("mediapipe", include_py_files=False)
datas += collect_data_files("cv2")
# Bundle the pose model and the icon next to the launcher.
datas += [
    ("pose_landmarker_lite.task", "."),
    ("icon.ico", "."),
]

a = Analysis(
    ["launch.pyw"],
    pathex=["."],
    binaries=[],
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Big optional deps mediapipe pulls in transitively that we don't use.
        # Note: matplotlib must stay — mediapipe.tasks.python.vision imports it
        # via drawing_utils even when we don't use the drawing helpers.
        "torch", "tensorflow", "jax", "jaxlib",
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
    [],
    exclude_binaries=True,
    name="PostureMonitor",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,           # GUI app — no console window
    icon="icon.ico",
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="PostureMonitor",
)
