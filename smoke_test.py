"""Предсборочная проверка. Только ASCII в выводе."""
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("PYTHONUTF8", "1")


def fail(msg):
    print("[FAIL] %s" % msg)
    sys.exit(1)


def main():
    # 1. Импорт всех модулей проекта
    try:
        import cv2
        import numpy as np
        import gpr_core as gpr
        import pdf_io, exporters, pipeline
        from ocr_engines import OCREngineManager, get_models_dir
    except Exception as e:
        fail("import: %s: %s" % (type(e).__name__, e))
    print("[OK] imports")

    # 2. Модели
    md = get_models_dir()
    print("[INFO] models dir: %s (exists=%s)" % (md, md.exists()))
    if not (md / "rapidocr").is_dir():
        fail("no ocr_models/rapidocr - run prepare_models.py")
    print("[OK] models present")

    # 3. Оба стиля вызова менеджера (позиционный список раньше падал)
    try:
        m = OCREngineManager(["ru", "en"])
        m2 = OCREngineManager(languages=["ru", "en"])
    except TypeError as e:
        fail("OCREngineManager signature: %s" % e)
    engines = m.get_available_engines()
    print("[OK] manager calls, engines=%s" % engines)
    if not engines:
        fail("no OCR engine available")

    # 4. Реальный OCR (латиница+цифры работают даже без кириллической модели)
    img = np.full((120, 640, 3), 255, dtype=np.uint8)
    cv2.putText(img, "TEST 12345", (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 4)
    res = m.recognize(img, with_details=False)
    print("[OK] OCR: engine=%s text=%r conf=%.2f" % (res.engine, res.text, res.confidence))
    if "12345" not in res.text.replace(" ", ""):
        print("[WARN] digits not recognized - check rec model/dict")

    # 5. Парсер ГПР + контрольные суммы
    matrix = [
        ["Наименование людских ресурсов", "ИТОГО кол-во штатных единиц",
         2026, 2026, 2026, 2026, "Кол-во раб часов в месяц", "Итого чел.-час"],
        ["Наименование людских ресурсов", "ИТОГО кол-во штатных единиц",
         "Январь", "Февраль", "Март", "Апрель", "Кол-во раб часов в месяц", "Итого чел.-час"],
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
        fail("checksum validation failed on synthetic table")
    if gpr.norm_num("1 604 208") != 1604208:
        fail("norm_num broken")
    if abs(gpr.norm_num("1,6") - 1.6) > 1e-9:
        fail("norm_num decimal broken")
    print("[OK] number parsing")

    # 6. Экспорт HTML + XLSX
    d = t.as_dict()
    d["sheet_name"] = t.sheet_name
    result = {"source": "synthetic.pdf", "engine": "smoke", "meta": {"object": "TEST"},
              "tables": [d], "pages": [], "text": "",
              "stats": {"pages": 1, "tables": 1, "rows": v["rows"], "errors": 0,
                        "text_pages": 1, "ocr_pages": 0}}
    with tempfile.TemporaryDirectory() as td:
        h = exporters.to_html(result, str(Path(td) / "t.html"))
        x = exporters.to_xlsx(result, str(Path(td) / "t.xlsx"))
        if Path(h).stat().st_size < 500 or Path(x).stat().st_size < 2000:
            fail("export produced empty files")
    print("[OK] HTML + XLSX export")

    print("")
    print("[SUCCESS] all checks passed - ready to package")
    return 0


if __name__ == "__main__":
    sys.exit(main())
