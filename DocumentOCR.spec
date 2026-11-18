# -*- mode: python ; coding: utf-8 -*-
"""
Сборка:  pyinstaller --noconfirm --clean DocumentOCR.spec

Переменные окружения:
  DOCOCR_DEBUG=1   -> консольная версия (видны traceback при падении)
  DOCOCR_ONEDIR=1  -> папка вместо одного exe (старт 2-3 с вместо 15-25 с)
"""
import os
from pathlib import Path
from PyInstaller.utils.hooks import (collect_all, collect_data_files,
                                     collect_dynamic_libs, collect_submodules)

ROOT = Path(SPECPATH)
APP = "DocumentOCR"
DEBUG = os.environ.get("DOCOCR_DEBUG", "0") == "1"
ONEDIR = os.environ.get("DOCOCR_ONEDIR", "0") == "1"

datas, binaries, hidden = [], [], []


def safe_all(pkg):
    try:
        return collect_all(pkg)
    except Exception as e:
        print("[WARN] collect_all(%s): %s" % (pkg, e))
        return [], [], []


# --- 1. Модели OCR рядом с проектом (положит prepare_models.py) ---
if (ROOT / "ocr_models").is_dir():
    datas.append((str(ROOT / "ocr_models"), "ocr_models"))
    print("[INFO] bundled ocr_models from project")

# --- 2. rapidocr целиком: onnx + config.yaml + словари символов ---
d, b, h = safe_all("rapidocr_onnxruntime")
datas += d
binaries += b
hidden += h

# --- 3. onnxruntime: нативные DLL + провайдеры ---
try:
    binaries += collect_dynamic_libs("onnxruntime")
    datas += collect_data_files("onnxruntime")
    hidden += collect_submodules("onnxruntime")
except Exception as e:
    print("[WARN] onnxruntime collect: %s" % e)
hidden += ["onnxruntime.capi._pybind_state"]

# --- 4. PyMuPDF (pymupdf + fitz) ---
d, b, h = safe_all("pymupdf")
datas += d
binaries += b
hidden += h
hidden += ["fitz"]

# --- 5. Модули проекта и мелкие зависимости ---
hidden += ["pipeline", "gpr_core", "pdf_io", "exporters", "ocr_engines",
           "openpyxl", "openpyxl.cell._writer", "PIL._imaging",
           "cv2", "numpy", "yaml"]

# --- 6. Что НЕ тащим в сборку ---
excludes = [
    "torch", "torchvision", "torchaudio", "easyocr",
    "paddle", "paddleocr", "paddlex", "tensorflow",
    "pandas", "scipy", "matplotlib", "tkinter",
    "IPython", "jupyter", "notebook", "pytest",
    "PyQt6.QtWebEngineCore", "PyQt6.QtWebEngineWidgets", "PyQt6.QtQuick",
    "PyQt6.QtQuick3D", "PyQt6.QtQml", "PyQt6.Qt3DCore", "PyQt6.Qt3DRender",
    "PyQt6.QtBluetooth", "PyQt6.QtMultimedia", "PyQt6.QtMultimediaWidgets",
    "PyQt6.QtSensors", "PyQt6.QtSerialPort", "PyQt6.QtPositioning",
    "PyQt6.QtRemoteObjects", "PyQt6.QtDesigner", "PyQt6.QtHelp",
    "PyQt6.QtTest", "PyQt6.QtNfc", "PyQt6.QtSpatialAudio",
    "PyQt6.QtTextToSpeech", "PyQt6.QtWebSockets", "PyQt6.QtCharts",
    "PyQt6.QtDataVisualization", "PyQt6.QtNetworkAuth",
]

print("[INFO] datas=%d binaries=%d hidden=%d excludes=%d"
      % (len(datas), len(binaries), len(hidden), len(excludes)))

a = Analysis(
    [str(ROOT / "main.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)

pyz = PYZ(a.pure)

icon = str(ROOT / "icon.ico") if (ROOT / "icon.ico").exists() else None

if ONEDIR:
    exe = EXE(
        pyz, a.scripts, [],
        exclude_binaries=True,
        name=APP, debug=False, strip=False, upx=False,
        console=DEBUG, disable_windowed_traceback=False, icon=icon,
    )
    coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name=APP)
else:
    exe = EXE(
        pyz, a.scripts, a.binaries, a.datas, [],
        name=APP, debug=False, strip=False, upx=False,
        console=DEBUG, disable_windowed_traceback=False, icon=icon,
        runtime_tmpdir=None,
    )
