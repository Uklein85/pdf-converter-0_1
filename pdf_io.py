"""Работа с PDF: текстовый слой, векторные таблицы, рендер, сжатие, вложения."""
from __future__ import annotations

import io
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

log = logging.getLogger("pdf")

try:
    import pymupdf as fitz
except ImportError:  # pragma: no cover
    import fitz  # type: ignore


def open_doc(path: str):
    return fitz.open(path)


def page_words(page) -> List[tuple]:
    try:
        return page.get_text("words") or []
    except Exception:
        return []


def has_text_layer(page, min_words: int = 40, min_chars: int = 200) -> bool:
    w = page_words(page)
    chars = sum(len(x[4]) for x in w)
    return len(w) >= min_words and chars >= min_chars


def render_page(page, dpi: int = 300, max_side: int = 6500) -> np.ndarray:
    r = page.rect
    zoom = dpi / 72.0
    long_side = max(r.width, r.height) * zoom
    if long_side > max_side:
        zoom = max_side / max(r.width, r.height)
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 3:
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    if pix.n == 4:
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
    return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)


def vector_tables(page) -> List[List[List[str]]]:
    """Таблицы из векторной разметки + текстового слоя."""
    try:
        found = page.find_tables()
    except Exception as e:
        log.debug("find_tables: %s", e)
        return []
    out = []
    for t in getattr(found, "tables", []) or []:
        try:
            data = t.extract()
        except Exception:
            continue
        if data and len(data) >= 4:
            out.append([[("" if c is None else str(c)) for c in row] for row in data])
    return out


# ------------------------------------------------------------------ геометрия
def words_to_matrix(words: List[tuple], min_gap: float = 5.0) -> List[List[str]]:
    """Кластеризация слов в строки/колонки по геометрии (без разметки таблицы)."""
    if not words:
        return []
    heights = [w[3] - w[1] for w in words]
    tol = max(2.0, float(np.median(heights)) * 0.65)

    items = sorted(words, key=lambda w: ((w[1] + w[3]) / 2.0, w[0]))
    lines: List[Dict[str, Any]] = []
    for w in items:
        cy = (w[1] + w[3]) / 2.0
        if lines and abs(cy - lines[-1]["cy"]) <= tol:
            lines[-1]["ws"].append(w)
            lines[-1]["cy"] = float(np.mean([(x[1] + x[3]) / 2.0 for x in lines[-1]["ws"]]))
        else:
            lines.append({"cy": cy, "ws": [w]})

    x0 = min(w[0] for w in words)
    x1 = max(w[2] for w in words)
    width = int(x1 - x0) + 2
    cover = np.zeros(width, dtype=np.int32)
    for w in words:
        a, b = int(w[0] - x0), int(w[2] - x0) + 1
        cover[a:min(b, width)] += 1

    bounds = [0]
    run = 0
    for i, v in enumerate(cover):
        if v == 0:
            run += 1
        else:
            if run >= min_gap and i > 2:
                bounds.append(i - run // 2)
            run = 0
    bounds.append(width)
    if len(bounds) < 3:
        bounds = [0, width]

    matrix: List[List[str]] = []
    for ln in lines:
        cells = [""] * (len(bounds) - 1)
        for w in sorted(ln["ws"], key=lambda w: w[0]):
            cx = (w[0] + w[2]) / 2.0 - x0
            k = 0
            for j in range(len(bounds) - 1):
                if bounds[j] <= cx < bounds[j + 1]:
                    k = j
                    break
            cells[k] = (cells[k] + " " + w[4]).strip()
        if any(cells):
            matrix.append(cells)
    return matrix


def detect_grid(image: np.ndarray) -> Tuple[List[int], List[int]]:
    """Координаты горизонтальных и вертикальных линий таблицы."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    bw = cv2.adaptiveThreshold(255 - gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY, 25, 12)
    h, w = bw.shape
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(30, w // 25), 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(20, h // 25)))
    hor = cv2.morphologyEx(bw, cv2.MORPH_OPEN, hk)
    ver = cv2.morphologyEx(bw, cv2.MORPH_OPEN, vk)

    def positions(mask, axis):
        proj = (mask > 0).sum(axis=axis)
        thr = max(3, int(0.35 * (mask.shape[1] if axis == 1 else mask.shape[0])))
        idx = np.where(proj > thr)[0]
        groups: List[int] = []
        for i in idx:
            if groups and i - groups[-1] <= 3:
                continue
            groups.append(int(i))
        return groups

    ys = positions(hor, 1)
    xs = positions(ver, 0)
    return ys, xs


def grid_matrix(image: np.ndarray, ocr, ys: List[int], xs: List[int],
                max_cells: int = 6000) -> List[List[str]]:
    """OCR по ячейкам сетки."""
    if len(ys) < 4 or len(xs) < 4:
        return []
    if (len(ys) - 1) * (len(xs) - 1) > max_cells:
        return []
    h, w = image.shape[:2]
    matrix: List[List[str]] = []
    for i in range(len(ys) - 1):
        y0, y1 = ys[i] + 2, ys[i + 1] - 2
        if y1 - y0 < 6:
            continue
        row: List[str] = []
        for j in range(len(xs) - 1):
            x0, x1 = xs[j] + 2, xs[j + 1] - 2
            if x1 - x0 < 5:
                row.append("")
                continue
            crop = image[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]
            txt, _ = ocr.read_cell(crop)
            row.append(txt.strip())
        if any(row):
            matrix.append(row)
    return matrix


def blocks_to_matrix(blocks, min_gap_ratio: float = 0.6) -> List[List[str]]:
    """Кластеризация OCR-блоков в строки/колонки (таблица без разметки)."""
    if not blocks:
        return []
    hs = [b.h for b in blocks if b.h > 0]
    tol = max(3.0, float(np.median(hs)) * 0.7) if hs else 8.0
    lines: List[Dict[str, Any]] = []
    for b in sorted(blocks, key=lambda b: b.cy):
        if lines and abs(b.cy - lines[-1]["cy"]) <= tol:
            lines[-1]["bs"].append(b)
            lines[-1]["cy"] = float(np.mean([x.cy for x in lines[-1]["bs"]]))
        else:
            lines.append({"cy": b.cy, "bs": [b]})
    matrix = [[b.text for b in sorted(ln["bs"], key=lambda b: b.cx)] for ln in lines]
    return [r for r in matrix if any(c.strip() for c in r)]


def deskew(image: np.ndarray, limit: float = 4.0) -> np.ndarray:
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        pts = cv2.findNonZero(bw)
        if pts is None:
            return image
        angle = cv2.minAreaRect(pts)[-1]
        if angle > 45:
            angle -= 90
        if abs(angle) < 0.25 or abs(angle) > limit:
            return image
        h, w = image.shape[:2]
        m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        return cv2.warpAffine(image, m, (w, h), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)
    except Exception:
        return image


def prepare_for_ocr(image: np.ndarray) -> np.ndarray:
    """Мягкая предобработка: НЕ уничтожает тонкие линии и мелкие цифры."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    gray = cv2.bilateralFilter(gray, 5, 25, 25)
    blur = cv2.GaussianBlur(gray, (0, 0), 1.2)
    sharp = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)
    return cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)


# ------------------------------------------------------------------ сжатие
def compress_pdf(src: str, dst: str, target_dpi: int = 200, quality: int = 85,
                 progress=None) -> Dict[str, Any]:
    """
    Сжатие PDF без потери текстового слоя и без видимой потери качества.
    - изображения выше target_dpi уменьшаются и пережимаются в JPEG q=quality
    - мусор/дубликаты объектов удаляются, потоки deflate
    """
    before = os.path.getsize(src)
    doc = open_doc(src)
    replaced = 0
    method = "none"

    def tick(p, m):
        if progress:
            progress(int(p), m)

    tick(5, "Анализ изображений PDF...")

    if hasattr(doc, "rewrite_images"):
        try:
            doc.rewrite_images(dpi_threshold=target_dpi + 40, dpi_target=target_dpi,
                               quality=quality, lossy=True, lossless=True,
                               bitonal=False, gray=False, set_to_gray=False)
            method = "rewrite_images"
            replaced = -1
        except Exception as e:
            log.warning("rewrite_images: %s", e)

    if method == "none":
        from PIL import Image
        can_replace = hasattr(fitz.Page, "replace_image")
        total = len(doc)
        for pno in range(total):
            tick(5 + 70 * (pno + 1) / max(total, 1), f"Сжатие страницы {pno + 1}/{total}")
            page = doc[pno]
            for info in page.get_images(full=True):
                xref = info[0]
                try:
                    base = doc.extract_image(xref)
                except Exception:
                    continue
                raw = base.get("image") or b""
                if len(raw) < 20000:
                    continue
                try:
                    im = Image.open(io.BytesIO(raw))
                    im.load()
                except Exception:
                    continue
                if im.mode in ("P", "RGBA", "LA"):
                    im = im.convert("RGB")
                elif im.mode == "CMYK":
                    im = im.convert("RGB")

                rects = []
                try:
                    rects = page.get_image_rects(xref)
                except Exception:
                    pass
                max_pts = max([max(r.width, r.height) for r in rects] or [0])
                need = int(max_pts / 72.0 * target_dpi) if max_pts else 0
                if need and max(im.size) > need * 1.1:
                    k = need / float(max(im.size))
                    im = im.resize((max(1, int(im.width * k)), max(1, int(im.height * k))),
                                   Image.LANCZOS)
                buf = io.BytesIO()
                im.save(buf, format="JPEG", quality=quality, optimize=True, progressive=True)
                data = buf.getvalue()
                if len(data) >= len(raw) * 0.95:
                    continue
                if can_replace:
                    try:
                        page.replace_image(xref, stream=data)
                        replaced += 1
                        method = "replace_image"
                    except Exception as e:
                        log.debug("replace_image: %s", e)

    if method == "none":
        tick(40, "Пересборка страниц с сохранением текстового слоя...")
        doc.close()
        return _raster_rebuild(src, dst, target_dpi, quality, before, progress)

    tick(80, "Оптимизация структуры PDF...")
    try:
        doc.subset_fonts()
    except Exception:
        pass
    try:
        doc.save(dst, garbage=4, deflate=True, deflate_images=True,
                 deflate_fonts=True, clean=True)
    except Exception:
        doc.save(dst, garbage=4, deflate=True)
    doc.close()

    after = os.path.getsize(dst)
    return {"before": before, "after": after,
            "ratio": (1 - after / before) * 100 if before else 0.0,
            "method": method, "images": replaced}


def _raster_rebuild(src, dst, dpi, quality, before, progress=None) -> Dict[str, Any]:
    """Крайний случай: страницы как JPEG + невидимый текстовый слой (поиск сохраняется)."""
    from PIL import Image
    srcdoc = open_doc(src)
    out = fitz.open()
    total = len(srcdoc)
    for i in range(total):
        if progress:
            progress(5 + 80 * (i + 1) / max(total, 1), f"Пересборка {i + 1}/{total}")
        page = srcdoc[i]
        text = page.get_text("text") or ""
        img = render_page(page, dpi=dpi)
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=quality, optimize=True, progressive=True)
        new = out.new_page(width=page.rect.width, height=page.rect.height)
        new.insert_image(new.rect, stream=buf.getvalue())
        if text.strip():
            try:
                new.insert_textbox(new.rect, text, fontsize=6, render_mode=3,
                                   fontname="helv", overlay=True)
            except Exception:
                pass
    out.save(dst, garbage=4, deflate=True)
    out.close()
    srcdoc.close()
    after = os.path.getsize(dst)
    return {"before": before, "after": after,
            "ratio": (1 - after / before) * 100 if before else 0.0,
            "method": "raster+text", "images": total}


def attach_files(pdf_path: str, items: Dict[str, bytes]) -> None:
    """Вложить результаты (HTML/XLSX/JSON) внутрь PDF как вложения."""
    doc = open_doc(pdf_path)
    existing = set(doc.embfile_names() or [])
    for name, data in items.items():
        try:
            if name in existing:
                doc.embfile_del(name)
            doc.embfile_add(name, data, filename=name, ufilename=name,
                            desc="DocumentOCR: результат распознавания")
        except Exception as e:
            log.error("Вложение %s: %s", name, e)
    tmp = pdf_path + ".tmp"
    doc.save(tmp, garbage=3, deflate=True)
    doc.close()
    os.replace(tmp, pdf_path)
