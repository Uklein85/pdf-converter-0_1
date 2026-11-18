"""
OCR-движок: автономный RapidOCR (ONNX).
Модели берутся из сборки, ничего не скачивается при запуске.
Поддерживает кириллическую rec-модель, если она положена в ocr_models/cyrillic/.
"""
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ocr")

DET_NAME = "ch_PP-OCRv4_det_infer.onnx"
REC_NAME = "ch_PP-OCRv4_rec_infer.onnx"
CLS_NAME = "ch_ppocr_mobile_v2.0_cls_infer.onnx"


def app_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def models_root() -> Path:
    """_MEIPASS (onefile) -> рядом с exe -> _internal -> папка проекта."""
    cands: List[Path] = []
    if getattr(sys, "frozen", False):
        mp = getattr(sys, "_MEIPASS", None)
        if mp:
            cands.append(Path(mp) / "ocr_models")
        cands.append(app_dir() / "ocr_models")
        cands.append(app_dir() / "_internal" / "ocr_models")
    cands.append(Path(__file__).resolve().parent / "ocr_models")
    for c in cands:
        if c.is_dir():
            return c
    return cands[-1]


def _package_models() -> Optional[Path]:
    try:
        import rapidocr_onnxruntime as r
        p = Path(r.__file__).parent
        for sub in ("models", "model", ""):
            d = (p / sub) if sub else p
            if any(d.glob("*.onnx")):
                return d
    except Exception:
        pass
    return None


def resolve_models() -> Dict[str, Optional[Path]]:
    """Ищем det/rec/cls + кириллический словарь."""
    roots = [models_root() / "rapidocr", models_root()]
    pkg = _package_models()
    if pkg:
        roots.append(pkg)

    found: Dict[str, Optional[Path]] = {"det": None, "rec": None, "cls": None,
                                       "keys": None, "cyr_rec": None, "cyr_keys": None}

    for root in roots:
        if not root or not root.is_dir():
            continue
        for f in root.rglob("*.onnx"):
            n = f.name.lower()
            if "det" in n and found["det"] is None and "server" not in n:
                found["det"] = f
            elif "rec" in n and "server" not in n:
                if "cyril" in n or "ru_" in n or "latin" in n:
                    if found["cyr_rec"] is None:
                        found["cyr_rec"] = f
                elif found["rec"] is None:
                    found["rec"] = f
            elif "cls" in n and found["cls"] is None:
                found["cls"] = f
        for f in root.rglob("*.txt"):
            n = f.name.lower()
            if "cyril" in n and found["cyr_keys"] is None:
                found["cyr_keys"] = f
            elif ("keys" in n or "dict" in n) and found["keys"] is None:
                found["keys"] = f

    cyr_dir = models_root() / "cyrillic"
    if cyr_dir.is_dir():
        for f in cyr_dir.glob("*.onnx"):
            found["cyr_rec"] = f
        for f in cyr_dir.glob("*.txt"):
            found["cyr_keys"] = f
    return found


@dataclass
class Block:
    text: str
    score: float
    bbox: List[List[float]]

    @property
    def cx(self) -> float:
        return float(np.mean([p[0] for p in self.bbox]))

    @property
    def cy(self) -> float:
        return float(np.mean([p[1] for p in self.bbox]))

    @property
    def h(self) -> float:
        ys = [p[1] for p in self.bbox]
        return float(max(ys) - min(ys))


@dataclass
class OCRResult:
    text: str
    blocks: List[Block] = field(default_factory=list)
    confidence: float = 0.0
    engine: str = "RapidOCR"


class RapidOCREngine:
    name = "RapidOCR"

    def __init__(self, languages: Optional[List[str]] = None, prefer_cyrillic: bool = True):
        self.languages = languages or ["ru", "en"]
        self._eng = None
        self._cyr = None
        self.available = False
        self.dict_kind = "ch/latin"

        m = resolve_models()
        if not m["det"] or not (m["rec"] or m["cyr_rec"]):
            log.error("Модели OCR не найдены. Папка: %s", models_root())
            log.error("Запустите download_models.py перед сборкой.")
            return

        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError as e:
            log.error("rapidocr-onnxruntime не установлен: %s", e)
            return

        rec = m["cyr_rec"] if (prefer_cyrillic and m["cyr_rec"]) else m["rec"]
        keys = m["cyr_keys"] if (prefer_cyrillic and m["cyr_rec"] and m["cyr_keys"]) else m["keys"]

        self._eng = self._make(RapidOCR, m["det"], rec, m["cls"], keys)
        if self._eng is None:
            return
        self.available = True
        self.dict_kind = "cyrillic" if (rec is m["cyr_rec"] and m["cyr_rec"]) else "ch/latin"
        log.info("RapidOCR готов. det=%s rec=%s dict=%s",
                 m["det"].name, Path(rec).name, self.dict_kind)

        if self.dict_kind != "cyrillic":
            log.warning("ВНИМАНИЕ: rec-модель без кириллицы. Русский текст будет искажен.")
            log.warning("Положите cyrillic_rec.onnx + cyrillic_dict.txt в ocr_models/cyrillic/")

        if m["cyr_rec"] and rec is not m["cyr_rec"]:
            self._cyr = self._make(RapidOCR, m["det"], m["cyr_rec"], m["cls"], m["cyr_keys"])

    @staticmethod
    def _make(cls, det, rec, cls_m, keys):
        base = {"det_model_path": str(det), "rec_model_path": str(rec)}
        if cls_m:
            base["cls_model_path"] = str(cls_m)
        attempts = []
        if keys:
            attempts.append({**base, "rec_keys_path": str(keys)})
            attempts.append({**base, "keys_path": str(keys)})
        attempts.append(base)
        for kw in attempts:
            try:
                return cls(**kw)
            except TypeError:
                continue
            except Exception as e:
                log.error("Инициализация RapidOCR: %s: %s", type(e).__name__, e)
                return None
        return None

    def _call(self, img: np.ndarray, engine=None, **kw):
        eng = engine or self._eng
        try:
            return eng(img, **kw)
        except TypeError:
            return eng(img)

    def recognize(self, image: np.ndarray) -> OCRResult:
        if not self.available:
            raise RuntimeError("OCR недоступен")
        out, _ = self._call(image)
        blocks: List[Block] = []
        if out:
            for row in out:
                try:
                    box, txt, score = row[0], row[1], float(row[2])
                except Exception:
                    continue
                bbox = [[float(p[0]), float(p[1])] for p in box]
                blocks.append(Block(str(txt), score, bbox))
        text = "\n".join(b.text for b in blocks)
        conf = float(np.mean([b.score for b in blocks])) if blocks else 0.0
        return OCRResult(text=text, blocks=blocks, confidence=conf, engine=self.name)

    def read_cell(self, crop: np.ndarray) -> Tuple[str, float]:
        """OCR одной ячейки: без детектора, только распознавание строки."""
        if crop is None or crop.size == 0:
            return "", 0.0
        h, w = crop.shape[:2]
        if h < 48:
            k = 48.0 / max(h, 1)
            crop = cv2.resize(crop, (max(int(w * k), 16), 48), interpolation=cv2.INTER_CUBIC)
        crop = cv2.copyMakeBorder(crop, 6, 6, 6, 6, cv2.BORDER_REPLICATE)
        out, _ = self._call(crop, use_det=False, use_cls=False, use_rec=True)
        if not out:
            return "", 0.0
        row = out[0]
        if isinstance(row, (list, tuple)) and len(row) >= 2:
            return str(row[0]), float(row[1])
        return "", 0.0


class OCREngineManager:
    """Совместим со старыми вызовами: OCREngineManager(['ru','en']) и (languages=[...])."""

    def __init__(self, preferred_engine=None, languages=None):
        if isinstance(preferred_engine, (list, tuple, set)):
            if languages is None:
                languages = list(preferred_engine)
            preferred_engine = None
        if isinstance(languages, str):
            languages = [languages]
        self.languages = [str(x) for x in (languages or ["ru", "en"])]

        self.engines: Dict[str, Any] = {}
        eng = RapidOCREngine(self.languages)
        self.engines[eng.name] = eng
        self.current_engine = eng if eng.available else None
        if self.current_engine is None:
            log.error("Ни один OCR-движок не доступен (текстовый слой PDF все равно работает)")

    def get_available_engines(self) -> List[str]:
        return [n for n, e in self.engines.items() if getattr(e, "available", False)]

    @property
    def available(self) -> bool:
        return self.current_engine is not None

    def switch_engine(self, name) -> bool:
        if isinstance(name, str) and name in self.engines and self.engines[name].available:
            self.current_engine = self.engines[name]
            return True
        return False

    def recognize(self, image: np.ndarray, with_details: bool = True) -> OCRResult:
        if self.current_engine is None:
            raise RuntimeError("Нет доступных OCR движков")
        return self.current_engine.recognize(image)

    def read_cell(self, crop: np.ndarray) -> Tuple[str, float]:
        if self.current_engine is None:
            return "", 0.0
        return self.current_engine.read_cell(crop)
        # ============================================================
# СОВМЕСТИМОСТЬ: единый поиск моделей + запрет скачивания.
# Блок намеренно в конце файла - перекрывает старые определения.
# ============================================================
def get_models_dir():
    """_MEIPASS (onefile) -> рядом с exe -> _internal -> папка проекта."""
    import sys as _sys
    from pathlib import Path as _Path

    if getattr(_sys, "frozen", False):
        exe_dir = _Path(_sys.executable).resolve().parent
        fallback = exe_dir / "ocr_models"
        cands = []
        mp = getattr(_sys, "_MEIPASS", None)
        if mp:
            cands.append(_Path(mp) / "ocr_models")
        cands.append(exe_dir / "ocr_models")
        cands.append(exe_dir / "_internal" / "ocr_models")
    else:
        here = _Path(__file__).resolve().parent
        fallback = here / "ocr_models"
        cands = [here / "ocr_models"]

    for p in cands:
        r = p / "rapidocr"
        if r.is_dir() and any(r.glob("*.onnx")):
            return p
    for p in cands:
        if p.is_dir():
            return p
    return fallback


def ensure_rapidocr_models():
    """Только проверка наличия. НИЧЕГО не скачивает (убирает HTTP 404)."""
    from pathlib import Path as _Path

    models_dir = _Path(get_models_dir()) / "rapidocr"
    required = [
        "ch_PP-OCRv4_det_infer.onnx",
        "ch_PP-OCRv4_rec_infer.onnx",
    ]
    missing = [n for n in required
               if not (models_dir / n).exists()
               or (models_dir / n).stat().st_size < 500_000]
    if missing:
        raise FileNotFoundError(
            "Нет моделей OCR: " + ", ".join(missing)
            + " | папка: " + str(models_dir)
            + " | запустите prepare_models.py перед сборкой"
        )
    return models_dir


def models_root():
    """Алиас для совместимости с новыми модулями."""
    return get_models_dir()
