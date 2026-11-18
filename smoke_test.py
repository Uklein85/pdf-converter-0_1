"""Предсборочная проверка. ASCII-вывод (cp1252-safe)."""
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("DOCOCR_DISABLE_OPTIONAL", "1")
ROOT = Path(__file__).resolve().parent


def fail(msg):
    print("[FAIL] %s" % msg)
    sys.exit(1)


def main():
    try:
        import cv2
        import numpy as np
        import ocr_engines
        from ocr_engines import OCREngineManager
    except Exception as e:
        fail("import: %s: %s" % (type(e).__name__, e))
    print("[OK] imports")

    # Путь к моделям определяем сами, если функции в модуле нет
    getter = getattr(ocr_engines, "get_models_dir", None)
    md = Path(getter()) if callable(getter) else ROOT / "ocr_models"
    print("[INFO] models dir: %s (exists=%s)" % (md, md.exists()))

    rapid = md / "rapidocr"
    if not rapid.is_dir() or not list(rapid.glob("*.onnx")):
        pkg = getattr(ocr_engines, "_package_models", None)
        alt = pkg() if callable(pkg) else None
        if alt is None:
            fail("no .onnx models - run: python prepare_models.py")
        rapid = Path(alt)
        print("[INFO] fallback to package models: %s" % rapid)
    for f in sorted(rapid.glob("*.onnx")):
        print("       %s (%.1f MB)" % (f.name, f.stat().st_size / 1048576))
    print("[OK] models present")

    try:
        m = OCREngineManager(["ru", "en"])
        OCREngineManager(languages=["ru", "en"])
    except TypeError as e:
        fail("OCREngineManager signature: %s" % e)
    engines = m.get_available_engines()
    print("[OK] manager calls, engines=%s" % engines)
    if not engines:
        fail("no OCR engine available - see log above")

    img = np.full((120, 640, 3), 255, dtype=np.uint8)
    cv2.putText(img, "TEST 12345", (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 4)
    try:
        res = m.recognize(img, with_details=False)
    except TypeError:
        res = m.recognize(img)
    print("[OK] OCR: engine=%s text=%r conf=%.2f"
          % (res.engine, res.text, res.confidence))
    if "12345" not in res.text.replace(" ", ""):
        print("[WARN] digits not recognized - check rec model")

    # Новые модули ГПР - только если они есть в репозитории
    for name in ("gpr_core", "pdf_io", "exporters", "pipeline"):
        if (ROOT / (name + ".py")).exists():
            try:
                __import__(name)
                print("[OK] import %s" % name)
            except Exception as e:
                fail("import %s: %s: %s" % (name, type(e).__name__, e))
        else:
            print("[SKIP] %s.py not in repo" % name)

    print("")
    print("[SUCCESS] required checks passed - ready to package")
    return 0


if __name__ == "__main__":
    sys.exit(main())
