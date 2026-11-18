"""
Предметный парсер ГПР / ГПТЛР (графики движения людских и технических ресурсов).
Понимает шапку (год + месяц), итоги, часы, коэффициент сменности, пиковую потребность.
Проверяет контрольные суммы — это ловит ошибки OCR в цифрах.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

MONTHS = ["январь", "февраль", "март", "апрель", "май", "июнь",
          "июль", "август", "сентябрь", "октябрь", "ноябрь", "декабрь"]
MONTHS_SHORT = ["янв", "фев", "мар", "апр", "май", "июн",
                "июл", "авг", "сен", "окт", "ноя", "дек"]

# Латинские двойники кириллицы (для OCR-шапок)
_FOLD = str.maketrans({"a": "а", "e": "е", "o": "о", "p": "р", "c": "с",
                       "y": "у", "x": "х", "k": "к", "m": "м", "t": "т",
                       "h": "н", "b": "в", "r": "г", "i": "и"})
_NUM_LOOK = str.maketrans({"O": "0", "О": "0", "о": "0", "D": "0",
                           "З": "3", "з": "3", "б": "6", "I": "1",
                           "l": "1", "|": "1", "S": "5", "B": "8"})

SPACES = "\u00a0\u2009\u202f\u2007"


def clean(s: Any) -> str:
    if s is None:
        return ""
    t = str(s)
    for sp in SPACES:
        t = t.replace(sp, " ")
    return re.sub(r"\s+", " ", t).strip()


def fold(s: str) -> str:
    return clean(s).lower().translate(_FOLD)


def month_index(cell: Any) -> Optional[int]:
    c = fold(cell).replace(" ", "").replace(".", "")
    if not c or len(c) > 14:
        return None
    for i, m in enumerate(MONTHS):
        if c == m:
            return i
    for i, m in enumerate(MONTHS):
        if c.startswith(m[:4]) or c.startswith(MONTHS_SHORT[i]):
            return i
    return None


def norm_num(s: Any) -> Optional[float]:
    """'1 604 208' -> 1604208 ; '1,6' -> 1.6 ; '' -> None"""
    t = clean(s)
    if not t:
        return None
    if re.fullmatch(r"\d{1,3}(?: \d{3})+(?:[.,]\d+)?", t):
        t = t.replace(" ", "")
    digits = sum(ch.isdigit() for ch in t)
    if digits and digits >= len(t.replace(" ", "")) * 0.5:
        t = t.translate(_NUM_LOOK)
    t = t.replace(",", ".")
    t = re.sub(r"[^0-9.\-]", "", t)
    if t in ("", "-", ".", "-.", "--"):
        return None
    try:
        v = float(t)
    except ValueError:
        return None
    return int(v) if float(v).is_integer() else v


def fmt(v: Any) -> str:
    if v is None or v == "":
        return ""
    if isinstance(v, float) and not v.is_integer():
        return f"{v:.2f}".rstrip("0").rstrip(".").replace(".", ",")
    return f"{int(v):,}".replace(",", " ")


# ---------------------------------------------------------------- шапка документа
META_PATTERNS = [
    ("contractor", r"Подрядчик\s*:\s*(.+?)(?=\s*(?:СС/СП|Наименование|$))"),
    ("ss_sp", r"СС/СП\s*:\s*(\S+)"),
    ("construction", r"Наименование стройки\s*:\s*(.+?)(?=\s*Код стройки|$)"),
    ("construction_code", r"Код стройки\s*:\s*([\d\-]+)"),
    ("project", r"Наименование проекта\s*:\s*(\S+)"),
    ("project_code", r"Код проекта\s*:\s*(\d+)"),
    ("nomenclature", r"Номенклатурная группа\s*:\s*([\d.\-]+)"),
    ("object", r"Наименование объекта\s*:\s*(.+?)(?=\s*Тип ГП|$)"),
    ("gpr_type", r"Тип ГП\w*\s*:\s*(\S+)"),
    ("month", r"Месяц\s*:\s*([^\n]+)"),
]


def parse_meta(text: str) -> Dict[str, str]:
    flat = clean(text)
    out: Dict[str, str] = {}
    for key, pat in META_PATTERNS:
        m = re.search(pat, flat, re.IGNORECASE)
        if m:
            out[key] = clean(m.group(1))[:300]
    return out


def detect_kind(text: str) -> Optional[str]:
    t = fold(text)
    if "людских ресурс" in t or "чел.-час" in t or "чел.-час" in t:
        return "people"
    if "технических ресурс" in t or "машино-час" in t:
        return "tech"
    return None


KIND_TITLE = {"people": "График движения людских ресурсов",
              "tech": "График движения технических ресурсов"}


# ---------------------------------------------------------------- модель таблицы
@dataclass
class Column:
    index: int
    month: int
    year: Optional[int]

    @property
    def label(self) -> str:
        name = MONTHS[self.month].capitalize()
        return f"{self.year} {name}" if self.year else name


@dataclass
class Row:
    name: str
    code: str = ""
    total: Optional[float] = None
    values: List[Optional[float]] = field(default_factory=list)
    hours: Optional[float] = None
    sum_hours: Optional[float] = None
    peak: Optional[float] = None
    shift: Optional[float] = None
    level: int = 0
    checks: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.checks

    def as_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "code": self.code, "total": self.total,
                "values": self.values, "hours": self.hours, "sum_hours": self.sum_hours,
                "peak": self.peak, "shift": self.shift, "level": self.level,
                "checks": self.checks, "ok": self.ok}


@dataclass
class GPRTable:
    kind: str
    page: int
    meta: Dict[str, str]
    columns: List[Column]
    rows: List[Row]
    name_header: str = "Наименование"
    source: str = "text"
    idx: int = 1

    @property
    def title(self) -> str:
        obj = self.meta.get("object", "")
        base = KIND_TITLE.get(self.kind, "Таблица")
        return f"{base} — {obj}" if obj else base

    @property
    def sheet_name(self) -> str:
        obj = self.meta.get("object", "")
        tail = ""
        if "линейн" in obj.lower():
            tail = "_ЛЧ"
        elif "кранов" in obj.lower():
            tail = "_КУ"
        prefix = "ЛР" if self.kind == "people" else "ТР"
        return f"{prefix}{tail}_{self.page}"[:31]

    def as_dict(self) -> Dict[str, Any]:
        failed = sum(1 for r in self.rows if not r.ok)
        return {"id": f"{self.kind}_p{self.page}_{self.idx}", "kind": self.kind,
                "title": self.title, "page": self.page, "source": self.source,
                "meta": self.meta,
                "columns": [{"index": c.index, "month": c.month, "year": c.year,
                             "label": c.label} for c in self.columns],
                "name_header": self.name_header,
                "rows": [r.as_dict() for r in self.rows],
                "validation": {"rows": len(self.rows), "failed": failed}}


# ---------------------------------------------------------------- разбор матрицы
def _find_header(matrix: Sequence[Sequence[Any]]) -> Optional[int]:
    best, best_n = None, 0
    for i, row in enumerate(matrix[:20]):
        n = sum(1 for c in row if month_index(c) is not None)
        if n > best_n:
            best, best_n = i, n
    return best if best_n >= 4 else None


def _find_years(matrix, header_idx) -> Dict[int, int]:
    """Сопоставляем колонку месяца с годом из строки выше."""
    years: List[tuple] = []
    for i in range(max(0, header_idx - 3), header_idx):
        for j, c in enumerate(matrix[i]):
            t = clean(c)
            if re.fullmatch(r"20\d\d", t):
                years.append((j, int(t)))
    if not years:
        return {}
    years.sort()
    out: Dict[int, int] = {}
    month_cols = [j for j, c in enumerate(matrix[header_idx]) if month_index(c) is not None]
    for j in month_cols:
        cand = [y for x, y in years if x <= j]
        out[j] = cand[-1] if cand else years[0][1]
    return out


def _col_by_keyword(matrix, header_idx, keywords, exclude=()) -> Optional[int]:
    keys = [fold(k) for k in keywords]
    for i in range(max(0, header_idx - 3), min(len(matrix), header_idx + 2)):
        for j, c in enumerate(matrix[i]):
            f = fold(c)
            if not f or any(e in f for e in exclude):
                continue
            if any(k in f for k in keys):
                return j
    return None


def _level(name: str, code: str, kind: str) -> int:
    if kind == "tech" and code:
        return max(0, code.count("."))
    n = fold(name)
    if n.startswith("итого") or n.startswith("всего"):
        return 0
    groups = ["рабочие профессии", "основные рабочие", "водители и механизаторы",
              "итр", "руководители", "специалисты и служащие", "линейные итр",
              "вспомогательные работники", "обслуживающий персонал",
              "автотранспорт", "прицепная техника", "оборудование"]
    return 1 if any(n.startswith(g) for g in groups) else 2


def build_table(matrix: Sequence[Sequence[Any]], page: int, meta: Dict[str, str],
                kind: Optional[str] = None, source: str = "text",
                idx: int = 1) -> Optional[GPRTable]:
    if not matrix or len(matrix) < 4:
        return None
    width = max(len(r) for r in matrix)
    mat = [list(r) + [""] * (width - len(r)) for r in matrix]

    h = _find_header(mat)
    if h is None:
        return None

    kind = kind or detect_kind(" ".join(clean(c) for r in mat[:h + 2] for c in r)) or "people"
    years = _find_years(mat, h)
    columns = [Column(j, month_index(c), years.get(j))
               for j, c in enumerate(mat[h]) if month_index(c) is not None]
    if len(columns) < 4:
        return None

    month_idx = {c.index for c in columns}
    first_month = min(month_idx)
    last_month = max(month_idx)

    code_col = _col_by_keyword(mat, h, ["№ п/п", "n п/п", "классификатор"])
    name_col = _col_by_keyword(mat, h, ["наименование людских", "наименование технических",
                                        "наименование", "подкласс"])
    if name_col is None:
        name_col = 1 if (code_col == 0 and width > 2) else 0
    if code_col is None and name_col > 0:
        code_col = 0

    total_col = _col_by_keyword(mat, h, ["итого кол-во", "кол-во штатных", "итого кол"],
                                exclude=("час",))
    hours_col = _col_by_keyword(mat, h, ["кол-во раб часов", "раб часов", "количество машино",
                                         "машино-часов"])
    sum_col = _col_by_keyword(mat, h, ["итого чел", "чел.-час", "итого"],
                              exclude=("кол-во", "штатных"))
    peak_col = _col_by_keyword(mat, h, ["пиковая", "пиков"])
    shift_col = _col_by_keyword(mat, h, ["сменности", "коэф-т", "коэфф"])

    reserved = {name_col, code_col, total_col, hours_col, sum_col, peak_col, shift_col}
    reserved.discard(None)
    if sum_col is not None and sum_col in month_idx:
        sum_col = None
    if total_col is not None and total_col in month_idx:
        total_col = None

    # Итоговые колонки правее последнего месяца, если по ключевым словам не нашли
    tail = [j for j in range(last_month + 1, width) if j not in reserved]
    if kind == "people":
        if hours_col is None and tail:
            hours_col = tail[0]
        if sum_col is None and len(tail) > 1:
            sum_col = tail[-1]
    else:
        if peak_col is None and tail:
            peak_col = tail[0]
        if sum_col is None and len(tail) > 1:
            sum_col = tail[-1]

    name_header = clean(mat[h][name_col]) or KIND_TITLE[kind]
    rows: List[Row] = []
    pending_name = ""

    for raw in mat[h + 1:]:
        cells = [clean(c) for c in raw]
        if not any(cells):
            continue
        joined = fold(" ".join(cells))
        if sum(1 for c in cells if month_index(c) is not None) >= 4:
            continue
        if "согласовано" in joined or "директор" in joined or "2026 г" in joined:
            break

        name = clean(cells[name_col]) if name_col < len(cells) else ""
        code = clean(cells[code_col]) if code_col is not None and code_col < len(cells) else ""
        if code and not re.fullmatch(r"[\d.]+", code):
            code = ""
        vals = [norm_num(cells[c.index]) if c.index < len(cells) else None for c in columns]
        has_num = any(v is not None for v in vals)

        # перенос длинного наименования на следующую строку
        if name and not has_num and not code:
            nxt = raw
            if any(norm_num(c) is not None for j, c in enumerate(nxt) if j in month_idx):
                pending_name = (pending_name + " " + name).strip()
                continue
        if not name and not has_num:
            continue
        if pending_name:
            name = (pending_name + " " + name).strip()
            pending_name = ""
        if not name and code:
            name = f"Код {code}"
        if not name:
            continue

        rows.append(Row(
            name=name, code=code,
            total=norm_num(cells[total_col]) if total_col is not None and total_col < len(cells) else None,
            values=vals,
            hours=norm_num(cells[hours_col]) if hours_col is not None and hours_col < len(cells) else None,
            sum_hours=norm_num(cells[sum_col]) if sum_col is not None and sum_col < len(cells) else None,
            peak=norm_num(cells[peak_col]) if peak_col is not None and peak_col < len(cells) else None,
            shift=norm_num(cells[shift_col]) if shift_col is not None and shift_col < len(cells) else None,
            level=_level(name, code, kind),
        ))

    if len(rows) < 3:
        return None
    return GPRTable(kind=kind, page=page, meta=meta, columns=columns, rows=rows,
                    name_header=name_header, source=source, idx=idx)


# ---------------------------------------------------------------- валидация
def validate(table: GPRTable, hours_default: Optional[float] = None) -> Dict[str, int]:
    failed = 0
    for r in table.rows:
        r.checks = []
        vals = [v for v in r.values if v is not None]
        s = sum(vals) if vals else 0
        hours = r.hours or hours_default

        if table.kind == "people":
            if r.total is not None and vals and abs(s - r.total) > 0.5:
                r.checks.append(f"сумма месяцев {fmt(s)} != ИТОГО {fmt(r.total)}")
            if r.total is not None and hours and r.sum_hours is not None:
                expect = r.total * hours
                if abs(expect - r.sum_hours) > 1.5:
                    r.checks.append(f"{fmt(r.total)} x {fmt(hours)} = {fmt(expect)} != {fmt(r.sum_hours)}")
        else:
            if r.peak is not None and vals and abs(max(vals) - r.peak) > 0.5:
                r.checks.append(f"пик {fmt(r.peak)} != max месяцев {fmt(max(vals))}")
            if hours and r.sum_hours is not None and vals:
                k = r.shift or 1.0
                expect = s * hours * k
                if abs(expect - r.sum_hours) > max(2.0, r.sum_hours * 0.005):
                    r.checks.append(f"{fmt(s)} x {fmt(hours)} x {fmt(k)} = {fmt(expect)} != {fmt(r.sum_hours)}")
        if r.checks:
            failed += 1
    return {"rows": len(table.rows), "failed": failed}


def guess_hours(tables: List[GPRTable]) -> Optional[float]:
    from collections import Counter
    c = Counter(r.hours for t in tables for r in t.rows if r.hours)
    return c.most_common(1)[0][0] if c else None
