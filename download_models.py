"""Подготовка моделей ПЕРЕД сборкой exe. Копирует модели из pip-пакета, ничего не качает."""
import shutil
import sys
from pathlib import Path


def main() -> int:
    dst = Path(__file__).parent / "ocr_models" / "rapidocr"
    dst.mkdir(parents=True, exist_ok=True)
    try:
        import rapidocr_onnxruntime as r
    except ImportError:
        print("[ERROR] pip install rapidocr-onnxruntime")
        return 1
    pkg = Path(r.__file__).parent
    onnx = sorted(pkg.rglob("*.onnx"))
    if not onnx:
        print("[ERROR] В пакете нет .onnx моделей")
        return 1
    want = {"det": "ch_PP-OCRv4_det_infer.onnx",
            "rec": "ch_PP-OCRv4_rec_infer.onnx",
            "cls": "ch_ppocr_mobile_v2.0_cls_infer.onnx"}
    for kind, name in want.items():
        cand = [p for p in onnx if kind in p.name.lower() and "server" not in p.name.lower()]
        if cand:
            shutil.copy2(cand[0], dst / name)
            print(f"[OK] {kind}: {cand[0].name}")
    for extra in list(pkg.rglob("*.txt")) + list(pkg.rglob("*.yaml")):
        if extra.stat().st_size < 5_000_000:
            shutil.copy2(extra, dst / extra.name)

    cyr = Path(__file__).parent / "ocr_models" / "cyrillic"
    cyr.mkdir(parents=True, exist_ok=True)
    if not list(cyr.glob("*.onnx")):
        print("\n[ВНИМАНИЕ] Кириллическая rec-модель не найдена.")
        print("Без неё OCR читает русский как латиницу ('BoANTen' вместо 'Водители').")
        print("Положите в ocr_models/cyrillic/ файлы:")
        print("  cyrillic_rec.onnx  и  cyrillic_dict.txt")
        print("Для PDF с текстовым слоем OCR не нужен — таблицы берутся точно.\n")
    print(f"[SUCCESS] Модели: {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
