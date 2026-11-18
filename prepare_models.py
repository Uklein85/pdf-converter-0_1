"""Подготовка OCR-моделей перед сборкой. Только ASCII в выводе (cp1252-safe)."""
import shutil
import sys
from pathlib import Path

WANT = {
    "det": "ch_PP-OCRv4_det_infer.onnx",
    "rec": "ch_PP-OCRv4_rec_infer.onnx",
    "cls": "ch_ppocr_mobile_v2.0_cls_infer.onnx",
}


def main() -> int:
    root = Path(__file__).resolve().parent
    dst = root / "ocr_models" / "rapidocr"
    dst.mkdir(parents=True, exist_ok=True)

    try:
        import rapidocr_onnxruntime as r
    except ImportError:
        print("[ERROR] pip install rapidocr-onnxruntime")
        return 1

    pkg = Path(r.__file__).parent
    onnx = sorted(pkg.rglob("*.onnx"))
    print("[INFO] package: %s" % pkg)
    for p in onnx:
        print("       %s (%.1f MB)" % (p.name, p.stat().st_size / 1048576))
    if not onnx:
        print("[ERROR] no .onnx inside package")
        return 1

    for kind, name in WANT.items():
        cand = [p for p in onnx
                if kind in p.name.lower() and "server" not in p.name.lower()]
        if cand:
            shutil.copy2(cand[0], dst / name)
            print("[OK] %s -> %s" % (kind, name))
        elif kind != "cls":
            print("[ERROR] missing required model: %s" % kind)
            return 1

    for extra in list(pkg.rglob("*.txt")) + list(pkg.rglob("*.yaml")):
        if extra.stat().st_size < 5_000_000:
            shutil.copy2(extra, dst / extra.name)

    cyr = root / "ocr_models" / "cyrillic"
    cyr.mkdir(parents=True, exist_ok=True)
    if not list(cyr.glob("*.onnx")):
        print("")
        print("[NOTE] Cyrillic rec-model not found.")
        print("       PDF with text layer works without OCR (exact tables).")
        print("       For scanned Russian pages put into ocr_models/cyrillic/:")
        print("         cyrillic_rec.onnx + cyrillic_dict.txt")
        print("")

    total = sum(f.stat().st_size for f in dst.rglob("*") if f.is_file())
    print("[SUCCESS] models ready: %s (%.1f MB)" % (dst, total / 1048576))
    return 0


if __name__ == "__main__":
    sys.exit(main())
