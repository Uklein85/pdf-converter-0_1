"""Предсборочная проверка. ASCII-вывод (cp1252-safe)."""
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("PYTHONUTF8", "1")
ROOT = Path(__file__).resolve().parent


def fail(msg):
    print("[FAIL] %s" % msg)
    sys.exit(1)


def local_models_dir():
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent / "ocr_models"
    return ROOT / "ocr_models"


def main():
    # ---------- 1. Обязательный модуль ----------
    try:
        import cv2
        import numpy as np
        import ocr_engines
        from ocr_engines import OCREngineManager
    except Exception as e:
        fail("import ocr_engines/cv2/numpy: %s: %s" % (type(e).__name__, e))
    print("[OK] core imports")

    # ---------- 2. Папка моделей ----------
    getter = getattr(ocr_engines, "get_models_dir", None)
    md = Path(getter()) if callable(getter) else local_models_dir()
    print("[INFO] models dir: %s (exists=%s)" % (md, md.exists()))
    rapid = md / "rapidocr"
    if not rapid.is_dir() or not list(rapid.glob("*.onnx")):
        fail("no .onnx in %s - run: python prepare_models.py" % rapid)
    for f in sorted(rapid.glob("*.onnx")):
        print("       %s (%.1f MB)" % (f.name, f.stat().st_size / 1048576))
    print("[OK] models present")

    # ---------- 3. Оба стиля вызова менеджера ----------
    try:
        m = OCREngineManager(["ru", "en"])
        OCREngineManager(languages=["ru", "en"])
    except TypeError as e:
        fail("OCREngineManager signature (unhashable list?): %s" % e)
    engines = m.get_available_engines()
    print("[OK] manager calls, engines=%s" % engines)
    if not engines:
        fail("no OCR engine available - see log above")

    # ---------- 4. Реальный OCR ----------
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
        print("[WARN] digits not recognized - check rec model/dict")

    # ---------- 5. Новые модули ГПР (если добавлены) ----------
    have = {}
    for name in ("gpr_core", "pdf_io", "exporters", "pipeline"):
        if (ROOT / (name + ".py")).exists():
            try:
                have[name] = __import__(name)
                print("[OK] import %s" % name)
            except Exception as e:
                fail("import %s: %s: %s" % (name, type(e).__name__, e))
        else:
            print("[SKIP] %s.py not in repo" % name)

    if "gpr_core" in have:
        gpr = have["gpr_core"]
        matrix = [
            ["Наименование людских ресурсов", "ИТОГО кол-во штатных единиц",
             2026, 2026, 2026, 2026, "Кол-во раб часов в месяц", "Итого чел.-час"],
            ["Наименование людских ресурсов", "ИТОГО кол-во штатных единиц",
             "Январь", "Февраль", "Март", "Апрель",
             "Кол-во раб часов в месяц", "Итого чел.-час"],
            ["Рабочие профессии", "10", "2", "3", "2", "3", "304", "3 040"],
            ["ИТР", "4", "1", "1", "1", "1", "304", "1 216"],
            ["ИТОГО ПК", "14", "3", "4", "3", "4", "304", "4 256"],
        ]
        t = gpr.build_table(matrix, page=1, meta={"object": "TEST"}, kind="people")
        if t is None:
            fail("gpr_core.build_table returned None")
        v = gpr.validate(t)
        print("[OK] GPR table: rows=%d failed=%d" % (v["rows"], v["failed"]))
        if v["failed"]:
            for r in t.rows:
                if r.checks:
                    print("       %s: %s" % (r.name, r.checks))
            fail("checksum validation failed")
        if gpr.norm_num("1 604 208") != 1604208:
            fail("norm_num broken")
        print("[OK] number parsing")

        if "exporters" in have:
            d = t.as_dict()
            d["sheet_name"] = t.sheet_name
            result = {"source": "synthetic.pdf", "engine": "smoke",
                      "meta": {"object": "TEST"}, "tables": [d], "pages": [],
                      "text": "",
                      "stats": {"pages": 1, "tables": 1, "rows": v["rows"],
                                "errors": 0, "text_pages": 1, "ocr_pages": 0}}
            with tempfile.TemporaryDirectory() as td:
                h = have["exporters"].to_html(result, str(Path(td) / "t.html"))
                x = have["exporters"].to_xlsx(result, str(Path(td) / "t.xlsx"))
                if Path(h).stat().st_size < 500 or Path(x).stat().st_size < 2000:
                    fail("export produced empty files")
            print("[OK] HTML + XLSX export")
    else:
        print("[SKIP] GPR checks - add gpr_core.py to enable")

    print("")
    print("[SUCCESS] required checks passed - ready to package")
    return 0


if __name__ == "__main__":
    sys.exit(main())
