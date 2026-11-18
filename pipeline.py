"""Конвейер: текстовый слой PDF -> векторные таблицы -> OCR -> таблицы ГПР -> валидация."""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import cv2
import numpy as np

import gpr_core as gpr
import pdf_io as pio

log = logging.getLogger("pipeline")


def _jpg(image: np.ndarray, max_w: int = 1600, quality: int = 80) -> bytes:
    h, w = image.shape[:2]
    if w > max_w:
        k = max_w / float(w)
        image = cv2.resize(image, (max_w, int(h * k)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return buf.tobytes() if ok else b""


def process(path: str, ocr=None, mode: str = "auto", dpi: int = 300,
            progress: Optional[Callable[[int, str], None]] = None,
            keep_page_images: bool = True,
            max_pages: Optional[int] = None) -> Dict[str, Any]:
    """
    mode: auto | text | ocr
      auto — текстовый слой, если он есть; иначе OCR
      text — только текстовый слой (быстро, точно)
      ocr  — принудительно OCR
    """
    def tick(p, m):
        if progress:
            progress(int(max(0, min(100, p))), m)

    src = str(Path(path).resolve())
    ext = Path(src).suffix.lower()
    result: Dict[str, Any] = {
        "source": src, "generated": datetime.now().isoformat(timespec="seconds"),
        "engine": "", "meta": {}, "tables": [], "pages": [], "text": "",
        "stats": {"pages": 0, "tables": 0, "rows": 0, "errors": 0,
                  "text_pages": 0, "ocr_pages": 0},
    }
    page_images: List[bytes] = []

    if ext in (".xlsx", ".xls"):
        return _process_excel(src, result, tick)

    if ext in (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"):
        tick(10, "Загрузка изображения")
        img = cv2.imdecode(np.fromfile(src, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Не удалось прочитать изображение")
        if ocr is None or not getattr(ocr, "available", False):
            raise RuntimeError("Для изображений нужен OCR-движок")
        img = pio.prepare_for_ocr(pio.deskew(img))
        tick(35, "Распознавание")
        res = ocr.recognize(img)
        matrices = _ocr_matrices(img, res, ocr, tick, 40, 80)
        text = res.text
        meta = gpr.parse_meta(text)
        result["meta"] = meta
        kind = gpr.detect_kind(text)
        _add_tables(result, matrices, 1, meta, kind, "ocr")
        result["text"] = text
        result["pages"] = [{"number": 1, "mode": "ocr", "text": text,
                            "confidence": res.confidence,
                            "tables": len(result["tables"])}]
        result["engine"] = "OCR (RapidOCR)"
        result["stats"].update(pages=1, ocr_pages=1)
        if keep_page_images:
            page_images.append(_jpg(img))
        _finalize(result, tick)
        result["page_images"] = page_images
        return result

    if ext != ".pdf":
        raise ValueError(f"Формат не поддерживается: {ext}")

    doc = pio.open_doc(src)
    total = len(doc) if not max_pages else min(len(doc), max_pages)
    ocr_needed = False
    texts: List[str] = []

    for i in range(total):
        page = doc[i]
        base = 5 + 85 * i / max(total, 1)
        span = 85 / max(total, 1)
        tick(base, f"Страница {i + 1}/{total}")

        use_text = (mode == "text") or (mode == "auto" and pio.has_text_layer(page))
        if mode == "ocr":
            use_text = False

        matrices: List[List[List[str]]] = []
        if use_text:
            page_text = page.get_text("text") or ""
            words = pio.page_words(page)
            matrices = pio.vector_tables(page)
            src_kind = "text"
            if not matrices:
                m = pio.words_to_matrix(words)
                if m:
                    matrices = [m]
            conf = 1.0
            result["stats"]["text_pages"] += 1
            tick(base + span * 0.6, f"Страница {i + 1}: текстовый слой")
        else:
            if ocr is None or not getattr(ocr, "available", False):
                raise RuntimeError(f"Страница {i + 1}: нет текстового слоя, а OCR недоступен")
            ocr_needed = True
            img = pio.render_page(page, dpi=dpi)
            img = pio.prepare_for_ocr(pio.deskew(img))
            res = ocr.recognize(img)
            page_text = res.text
            conf = res.confidence
            matrices = _ocr_matrices(img, res, ocr, tick, base, base + span)
            src_kind = "ocr"
            result["stats"]["ocr_pages"] += 1
            if keep_page_images:
                page_images.append(_jpg(img))

        meta = gpr.parse_meta(page_text)
        if not result["meta"]:
            result["meta"] = meta
        kind = gpr.detect_kind(page_text)
        _add_tables(result, matrices, i + 1, meta or result["meta"], kind, src_kind)

        texts.append(f"\n--- Страница {i + 1} ---\n{page_text}\n")
        result["pages"].append({"number": i + 1, "mode": src_kind, "text": page_text,
                                "confidence": conf, "tables": len(matrices)})

    doc.close()
    result["text"] = "".join(texts)
    result["engine"] = ("Текстовый слой PDF" if not ocr_needed
                        else ("OCR (RapidOCR)" if result["stats"]["text_pages"] == 0
                              else "Смешанный: текст + OCR"))
    result["stats"]["pages"] = total
    _finalize(result, tick)
    result["page_images"] = page_images
    return result


def _ocr_matrices(img, res, ocr, tick, p0, p1) -> List[List[List[str]]]:
    tick(p0 + (p1 - p0) * 0.3, "Поиск сетки таблицы")
    ys, xs = pio.detect_grid(img)
    if len(ys) >= 5 and len(xs) >= 5:
        tick(p0 + (p1 - p0) * 0.5, f"OCR ячеек: {len(ys) - 1}x{len(xs) - 1}")
        m = pio.grid_matrix(img, ys, xs, ocr)
        if m and len(m) >= 5:
            return [m]
    tick(p0 + (p1 - p0) * 0.8, "Сборка таблицы из блоков OCR")
    m = pio.blocks_to_matrix(res.blocks)
    return [m] if m else []


def _add_tables(result, matrices, page, meta, kind, source):
    for k, mat in enumerate(matrices, 1):
        t = gpr.build_table(mat, page=page, meta=meta, kind=kind, source=source, idx=k)
        if t is not None:
            result["tables"].append(t)


def _finalize(result, tick):
    tick(93, "Проверка контрольных сумм")
    hours = gpr.guess_hours(result["tables"])
    rows = errors = 0
    dicts = []
    for t in result["tables"]:
        v = gpr.validate(t, hours_default=hours)
        rows += v["rows"]
        errors += v["failed"]
        d = t.as_dict()
        d["sheet_name"] = t.sheet_name
        dicts.append(d)
    result["tables"] = dicts
    result["stats"].update(tables=len(dicts), rows=rows, errors=errors)
    result["hours_per_month"] = hours
    tick(100, f"Готово: таблиц {len(dicts)}, строк {rows}, расхождений {errors}")


def _process_excel(src, result, tick):
    from openpyxl import load_workbook
    tick(20, "Чтение Excel")
    wb = load_workbook(src, data_only=True, read_only=True)
    texts = []
    for si, name in enumerate(wb.sheetnames, 1):
        ws = wb[name]
        mat = [["" if c is None else str(c) for c in row]
               for row in ws.iter_rows(values_only=True)]
        text = "\n".join(" ".join(r) for r in mat)
        texts.append(f"\n--- Лист: {name} ---\n{text}\n")
        meta = gpr.parse_meta(text) or result["meta"]
        if not result["meta"]:
            result["meta"] = meta
        _add_tables(result, [mat], si, meta, gpr.detect_kind(text), "text")
        result["pages"].append({"number": si, "mode": "text", "text": text,
                                "confidence": 1.0, "tables": 1})
        tick(20 + 70 * si / max(len(wb.sheetnames), 1), f"Лист {name}")
    wb.close()
    result["text"] = "".join(texts)
    result["engine"] = "Excel (нативные данные)"
    result["stats"]["pages"] = len(result["pages"])
    result["stats"]["text_pages"] = len(result["pages"])
    _finalize(result, tick)
    result["page_images"] = []
    return result


def build_report(result: Dict[str, Any], compress: Optional[Dict[str, Any]] = None) -> str:
    st = result["stats"]
    lines = ["DocumentOCR — отчет об обработке",
             "=" * 60,
             f"Исходный файл : {result['source']}",
             f"Режим         : {result['engine']}",
             f"Сформировано  : {result['generated']}",
             f"Страниц       : {st['pages']} (текст: {st['text_pages']}, OCR: {st['ocr_pages']})",
             f"Таблиц ГПР    : {st['tables']}",
             f"Строк         : {st['rows']}",
             f"Расхождений   : {st['errors']}", ""]
    for t in result["tables"]:
        lines.append(f"[{t['id']}] {t['title']}")
        lines.append(f"    страница {t['page']}, источник {t['source']}, "
                     f"строк {t['validation']['rows']}, расхождений {t['validation']['failed']}")
        for r in t["rows"]:
            if r["checks"]:
                lines.append(f"    ! {r['name']}: {'; '.join(r['checks'])}")
        lines.append("")
    if compress:
        lines += ["Сжатие PDF", "-" * 60,
                  f"Было   : {compress['before'] / 1048576:.2f} МБ",
                  f"Стало  : {compress['after'] / 1048576:.2f} МБ",
                  f"Выигрыш: {compress['ratio']:.1f}%",
                  f"Метод  : {compress['method']}", ""]
    lines.append("Разработчик: Уклеин И.В. | АО «СТНГ» | г. Новосибирск | 2026")
    return "\n".join(lines)
