"""
OCR Engines - автономная версия для .exe сборки.
Модели НЕ скачиваются при запуске: они вшиты в сборку.
Логи только ASCII (cp1252-safe для GitHub Actions).
"""
from __future__ import annotations

import logging
import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ocr_engines")

DET_NAME = "ch_PP-OCRv4_det_infer.onnx"
REC_NAME = "ch_PP-OCRv4_rec_infer.onnx"
CLS_NAME = "ch_ppocr_mobile_v2.0_cls_infer.onnx"


# ============================================================
# ПУТИ К МОДЕЛЯМ
# ============================================================
def get_models_dir() -> Path:
    """_MEIPASS (onefile) -> рядом с exe -> _internal -> папка проекта."""
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        fallback = exe_dir / "ocr_models"
        cands: List[Path] = []
        mp = getattr(sys, "_MEIPASS", None)
        if mp:
            cands.append(Path(mp) / "ocr_models")
        cands.append(exe_dir / "ocr_models")
        cands.append(exe_dir / "_internal" / "ocr_models")
    else:
        here = Path(__file__).resolve().parent
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


def models_root() -> Path:
    """Алиас для новых модулей."""
    return get_models_dir()


def _package_models() -> Optional[Path]:
    """Резерв: модели прямо из установленного pip-пакета."""
    try:
        import rapidocr_onnxruntime as r
        p = Path(r.__file__).parent
        if any(p.rglob("*.onnx")):
            return p
    except Exception:
        pass
    return None


def ensure_rapidocr_models() -> Path:
    """Только проверка наличия. НИЧЕГО не скачивает (убирает HTTP 404)."""
    primary = get_models_dir() / "rapidocr"
    required = [DET_NAME, REC_NAME]

    def ok(d: Path) -> bool:
        return all((d / n).exists() and (d / n).stat().st_size > 500_000
                   for n in required)

    if ok(primary):
        return primary

    pkg = _package_models()
    if pkg is not None:
        logger.info("Модели взяты из pip-пакета: %s", pkg)
        return pkg

    missing = [n for n in required
               if not (primary / n).exists()
               or (primary / n).stat().st_size <= 500_000]
    raise FileNotFoundError(
        "Нет моделей OCR: " + ", ".join(missing)
        + " | папка: " + str(primary)
        + " | запустите prepare_models.py перед сборкой"
    )


def _pick(d: Path, kind: str, canonical: str) -> Optional[Path]:
    exact = d / canonical
    if exact.exists():
        return exact
    for f in sorted(d.rglob("*.onnx")):
        n = f.name.lower()
        if kind in n and "server" not in n:
            return f
    return None


# ============================================================
# СТРУКТУРЫ ДАННЫХ
# ============================================================
@dataclass
class TextBlock:
    text: str
    confidence: float
    bbox: List[List[float]] = field(default_factory=list)
    font_size: float = 0.0
    is_bold: bool = False
    language: str = ""

    @property
    def score(self) -> float:
        return self.confidence

    @property
    def cx(self) -> float:
        if not self.bbox:
            return 0.0
        return float(np.mean([p[0] for p in self.bbox]))

    @property
    def cy(self) -> float:
        if not self.bbox:
            return 0.0
        return float(np.mean([p[1] for p in self.bbox]))

    @property
    def h(self) -> float:
        if not self.bbox:
            return 0.0
        ys = [p[1] for p in self.bbox]
        return float(max(ys) - min(ys))


@dataclass
class ImageRegion:
    bbox: List[List[float]]
    image: np.ndarray
    confidence: float
    region_type: str = "image"


@dataclass
class OCRResult:
    text: str
    blocks: List[TextBlock] = field(default_factory=list)
    images: List[ImageRegion] = field(default_factory=list)
    confidence: float = 0.0
    engine: str = "RapidOCR"
    language: str = "ru+en"
    metadata: Dict = field(default_factory=dict)


# ============================================================
# БАЗОВЫЙ КЛАСС
# ============================================================
class OCREngine(ABC):
    def __init__(self, languages: Optional[List[str]] = None):
        self.languages = list(languages) if languages else ["ru", "en"]
        self.name = "BaseEngine"
        self._is_available = False

    @property
    def is_available(self) -> bool:
        return self._is_available

    @property
    def available(self) -> bool:
        return self._is_available

    @abstractmethod
    def recognize(self, image: np.ndarray) -> OCRResult:
        ...

    def recognize_with_details(self, image: np.ndarray) -> OCRResult:
        return self.recognize(image)

    def read_cell(self, crop: np.ndarray) -> Tuple[str, float]:
        return "", 0.0

    @staticmethod
    def _lang_of(text: str) -> str:
        cyr = sum(1 for c in text if "\u0400" <= c <= "\u04FF")
        lat = sum(1 for c in text if c.isascii() and c.isalpha())
        if cyr > lat:
            return "ru"
        if lat > cyr:
            return "en"
        return "unknown"

    def _empty(self) -> OCRResult:
        return OCRResult(text="", blocks=[], images=[], confidence=0.0,
                         engine=self.name, language="+".join(self.languages))


# ============================================================
# RAPIDOCR (ОСНОВНОЙ)
# ============================================================
class RapidOCREngine(OCREngine):
    def __init__(self, languages: Optional[List[str]] = None):
        super().__init__(languages)
        self.name = "RapidOCR"
        self._engine = None
        self.dict_kind = "ch/latin"

        try:
            models_dir = ensure_rapidocr_models()
        except Exception as e:
            logger.error("RapidOCR: %s", e)
            return

        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError as e:
            logger.warning("RapidOCR не установлен: %s", e)
            logger.warning("Решение: pip install rapidocr-onnxruntime")
            return

        det = _pick(models_dir, "det", DET_NAME)
        rec = _pick(models_dir, "rec", REC_NAME)
        cls_m = _pick(models_dir, "cls", CLS_NAME)
        if det is None or rec is None:
            logger.error("RapidOCR: не найдены det/rec в %s", models_dir)
            return

        keys = None
        cyr_dir = get_models_dir() / "cyrillic"
        if cyr_dir.is_dir():
            cyr_onnx = list(cyr_dir.glob("*.onnx"))
            cyr_txt = list(cyr_dir.glob("*.txt"))
            if cyr_onnx:
                rec = cyr_onnx[0]
                keys = cyr_txt[0] if cyr_txt else None
                self.dict_kind = "cyrillic"
        if keys is None:
            for f in sorted(models_dir.rglob("*.txt")):
                if "keys" in f.name.lower() or "dict" in f.name.lower():
                    keys = f
                    break

        base: Dict[str, Any] = {"det_model_path": str(det), "rec_model_path": str(rec)}
        if cls_m is not None:
            base["cls_model_path"] = str(cls_m)

        attempts: List[Dict[str, Any]] = []
        if keys is not None:
            attempts.append({**base, "rec_keys_path": str(keys)})
            attempts.append({**base, "keys_path": str(keys)})
        attempts.append(base)

        for kw in attempts:
            try:
                self._engine = RapidOCR(**kw)
                break
            except TypeError:
                continue
            except Exception as e:
                logger.error("RapidOCR init: %s: %s", type(e).__name__, e)
                return

        if self._engine is None:
            logger.error("RapidOCR: не удалось инициализировать")
            return

        self._is_available = True
        logger.info("RapidOCR готов. models=%s dict=%s", models_dir, self.dict_kind)
        if self.dict_kind != "cyrillic":
            logger.warning("rec-модель без кириллицы: русский будет искажен.")
            logger.warning("Положите cyrillic_rec.onnx + cyrillic_dict.txt в ocr_models/cyrillic/")

    def _call(self, img: np.ndarray, **kw):
        try:
            return self._engine(img, **kw)
        except TypeError:
            return self._engine(img)

    def recognize(self, image: np.ndarray) -> OCRResult:
        if not self._is_available or self._engine is None:
            raise RuntimeError("RapidOCR недоступен")
        out, _ = self._call(image)
        if not out:
            return self._empty()

        blocks: List[TextBlock] = []
        for row in out:
            try:
                box, txt, score = row[0], str(row[1]), float(row[2])
            except Exception:
                continue
            bbox = [[float(p[0]), float(p[1])] for p in box]
            blocks.append(TextBlock(text=txt, confidence=score, bbox=bbox,
                                    language=self._lang_of(txt)))
        if not blocks:
            return self._empty()
        return OCRResult(
            text="\n".join(b.text for b in blocks),
            blocks=blocks, images=[],
            confidence=float(np.mean([b.confidence for b in blocks])),
            engine=self.name, language="+".join(self.languages))

    def recognize_with_details(self, image: np.ndarray) -> OCRResult:
        res = self.recognize(image)
        for b in res.blocks:
            if b.bbox and len(b.bbox) >= 3:
                b.font_size = b.h
                b.is_bold = self._estimate_bold(image, b.bbox)
        res.images = self._detect_images(image)
        return res

    def read_cell(self, crop: np.ndarray) -> Tuple[str, float]:
        if not self._is_available or crop is None or crop.size == 0:
            return "", 0.0
        h, w = crop.shape[:2]
        if h < 48:
            k = 48.0 / max(h, 1)
            crop = cv2.resize(crop, (max(int(w * k), 16), 48),
                              interpolation=cv2.INTER_CUBIC)
        crop = cv2.copyMakeBorder(crop, 6, 6, 6, 6, cv2.BORDER_REPLICATE)
        out, _ = self._call(crop, use_det=False, use_cls=False, use_rec=True)
        if not out:
            return "", 0.0
        row = out[0]
        if isinstance(row, (list, tuple)) and len(row) >= 2:
            return str(row[0]), float(row[1])
        return "", 0.0

    @staticmethod
    def _estimate_bold(image: np.ndarray, bbox: List) -> bool:
        try:
            xs = [int(p[0]) for p in bbox]
            ys = [int(p[1]) for p in bbox]
            x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
            if x1 <= x0 or y1 <= y0:
                return False
            roi = image[y0:y1, x0:x1]
            if roi.size == 0:
                return False
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
            bw = cv2.threshold(gray, 0, 255,
                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            return float(np.sum(bw > 0)) / bw.size > 0.35
        except Exception:
            return False

    @staticmethod
    def _detect_images(image: np.ndarray) -> List[ImageRegion]:
        out: List[ImageRegion] = []
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
            bw = cv2.threshold(gray, 0, 255,
                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            total = image.shape[0] * image.shape[1]
            for c in cnts:
                x, y, w, h = cv2.boundingRect(c)
                area = w * h
                if 10000 < area < total * 0.5:
                    roi = bw[y:y + h, x:x + w]
                    if float(np.sum(roi > 0)) / roi.size < 0.3:
                        out.append(ImageRegion(
                            bbox=[[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
                            image=image[y:y + h, x:x + w],
                            confidence=0.8, region_type="image"))
        except Exception:
            pass
        return out


# ============================================================
# ОПЦИОНАЛЬНЫЕ ДВИЖКИ (в exe исключены)
# ============================================================
class EasyOCREngine(OCREngine):
    def __init__(self, languages: Optional[List[str]] = None, gpu: bool = False):
        super().__init__(languages)
        self.name = "EasyOCR"
        self._reader = None
        if os.environ.get("DOCOCR_DISABLE_OPTIONAL", "0") == "1":
            return
        try:
            import easyocr
            self._reader = easyocr.Reader(self.languages, gpu=gpu, verbose=False)
            self._is_available = True
            logger.info("EasyOCR готов")
        except ImportError as e:
            logger.info("EasyOCR пропущен: %s", e)
        except Exception as e:
            logger.warning("EasyOCR: %s: %s", type(e).__name__, e)

    def recognize(self, image: np.ndarray) -> OCRResult:
        if not self._is_available:
            raise RuntimeError("EasyOCR недоступен")
        blocks: List[TextBlock] = []
        for box, txt, score in self._reader.readtext(image):
            bbox = [[float(p[0]), float(p[1])] for p in box]
            blocks.append(TextBlock(text=str(txt), confidence=float(score),
                                    bbox=bbox, language=self._lang_of(str(txt))))
        if not blocks:
            return self._empty()
        return OCRResult(text="\n".join(b.text for b in blocks), blocks=blocks,
                         confidence=float(np.mean([b.confidence for b in blocks])),
                         engine=self.name, language="+".join(self.languages))


class PaddleOCREngine(OCREngine):
    def __init__(self, languages: Optional[List[str]] = None, use_gpu: bool = False):
        super().__init__(languages)
        self.name = "PaddleOCR"
        self._ocr = None
        if os.environ.get("DOCOCR_DISABLE_OPTIONAL", "0") == "1":
            return
        try:
            from paddleocr import PaddleOCR
            lang = self.languages[0] if self.languages else "ru"
            try:
                self._ocr = PaddleOCR(use_angle_cls=True, lang=lang, use_gpu=use_gpu)
            except TypeError:
                self._ocr = PaddleOCR(use_angle_cls=True, lang=lang)
            self._is_available = True
            logger.info("PaddleOCR готов (lang=%s)", lang)
        except ImportError as e:
            logger.info("PaddleOCR пропущен: %s", e)
        except Exception as e:
            logger.warning("PaddleOCR: %s: %s", type(e).__name__, e)

    def recognize(self, image: np.ndarray) -> OCRResult:
        if not self._is_available:
            raise RuntimeError("PaddleOCR недоступен")
        res = self._ocr.ocr(image, cls=True)
        if not res or not res[0]:
            return self._empty()
        blocks: List[TextBlock] = []
        for box, (txt, score) in res[0]:
            bbox = [[float(p[0]), float(p[1])] for p in box]
            blocks.append(TextBlock(text=str(txt), confidence=float(score),
                                    bbox=bbox, language=self._lang_of(str(txt))))
        if not blocks:
            return self._empty()
        return OCRResult(text="\n".join(b.text for b in blocks), blocks=blocks,
                         confidence=float(np.mean([b.confidence for b in blocks])),
                         engine=self.name, language="+".join(self.languages))


# ============================================================
# МЕНЕДЖЕР
# ============================================================
class OCREngineManager:
    """
    Допустимые вызовы:
        OCREngineManager()
        OCREngineManager(['ru', 'en'])
        OCREngineManager(languages=['ru', 'en'])
        OCREngineManager('RapidOCR', ['ru', 'en'])
    """

    PRIORITY = ["RapidOCR", "PaddleOCR", "EasyOCR"]

    def __init__(self, preferred_engine=None, languages=None):
        if isinstance(preferred_engine, (list, tuple, set)):
            if languages is None:
                languages = list(preferred_engine)
            preferred_engine = None
        elif preferred_engine is not None and not isinstance(preferred_engine, str):
            preferred_engine = str(preferred_engine)

        if languages is None:
            languages = ["ru", "en"]
        elif isinstance(languages, str):
            languages = [languages]
        else:
            languages = [str(x) for x in languages]
        if not languages:
            languages = ["ru", "en"]

        self.languages: List[str] = languages
        self.engines: Dict[str, OCREngine] = {}
        self.current_engine: Optional[OCREngine] = None
        self._init_all_engines(preferred_engine)

    def _init_all_engines(self, preferred: Optional[str] = None):
        logger.info("Инициализация OCR движков...")
        logger.info("Языки: %s", ", ".join(self.languages))
        logger.info("Папка моделей: %s", get_models_dir())

        factories = [
            ("RapidOCR", lambda: RapidOCREngine(list(self.languages))),
            ("PaddleOCR", lambda: PaddleOCREngine(list(self.languages))),
            ("EasyOCR", lambda: EasyOCREngine(list(self.languages))),
        ]
        for name, factory in factories:
            try:
                engine = factory()
            except Exception as e:
                logger.warning("  %s: ошибка инициализации (%s: %s)",
                               name, type(e).__name__, e)
                continue
            self.engines[str(name)] = engine
            logger.info("  %s: %s", name,
                        "доступен" if engine.is_available else "недоступен")

        chosen: Optional[OCREngine] = None
        if isinstance(preferred, str) and preferred:
            cand = self.engines.get(preferred)
            if cand is not None and cand.is_available:
                chosen = cand
        if chosen is None:
            for name in self.PRIORITY:
                cand = self.engines.get(name)
                if cand is not None and cand.is_available:
                    chosen = cand
                    break
        if chosen is None:
            for cand in self.engines.values():
                if cand.is_available:
                    chosen = cand
                    break

        self.current_engine = chosen
        if chosen is not None:
            logger.info("Выбран движок: %s", chosen.name)
        else:
            logger.error("Ни один OCR движок не доступен")

    @property
    def available(self) -> bool:
        return self.current_engine is not None

    def get_available_engines(self) -> List[str]:
        return [n for n, e in self.engines.items() if e.is_available]

    def switch_engine(self, engine_name) -> bool:
        if not isinstance(engine_name, str):
            return False
        e = self.engines.get(engine_name)
        if e is not None and e.is_available:
            self.current_engine = e
            logger.info("Переключено на: %s", engine_name)
            return True
        return False

    def recognize(self, image: np.ndarray, with_details: bool = True) -> OCRResult:
        if self.current_engine is None:
            raise RuntimeError("Нет доступных OCR движков")
        try:
            if with_details:
                return self.current_engine.recognize_with_details(image)
            return self.current_engine.recognize(image)
        except Exception as e:
            logger.error("Ошибка распознавания: %s", e)
            return self._try_fallback(image, with_details)

    def read_cell(self, crop: np.ndarray) -> Tuple[str, float]:
        if self.current_engine is None:
            return "", 0.0
        return self.current_engine.read_cell(crop)

    def _try_fallback(self, image: np.ndarray, with_details: bool) -> OCRResult:
        for name, e in self.engines.items():
            if not e.is_available or e is self.current_engine:
                continue
            logger.info("Попытка fallback: %s", name)
            try:
                res = e.recognize_with_details(image) if with_details else e.recognize(image)
                self.current_engine = e
                return res
            except Exception as ex:
                logger.warning("Fallback %s: %s", name, ex)
        raise RuntimeError("Все OCR движки не смогли обработать изображение")
