"""Экспорт результатов ГПР: HTML (самодостаточный), XLSX, JSON, ZIP-пакет."""
from __future__ import annotations

import base64
import html
import json
import os
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from gpr_core import MONTHS, fmt


# ------------------------------------------------------------------ HTML
CSS = """
:root{--line:#d8dee9;--head:#eef2f7;--ok:#1a7f37;--bad:#b42318;--accent:#0b62c4}
*{box-sizing:border-box}
body{font-family:'Segoe UI',Arial,sans-serif;margin:0;padding:24px;background:#f6f8fa;color:#1f2328}
h1{font-size:22px;margin:0 0 4px}
h2{font-size:17px;margin:28px 0 8px;padding-bottom:6px;border-bottom:2px solid var(--accent)}
.sub{color:#57606a;font-size:13px;margin-bottom:16px}
.meta{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:6px 18px;
 background:#fff;border:1px solid var(--line);border-radius:8px;padding:14px;margin-bottom:18px}
.meta div{font-size:13px}.meta b{color:#57606a;font-weight:600}
.wrap{overflow:auto;background:#fff;border:1px solid var(--line);border-radius:8px;max-height:78vh}
table{border-collapse:collapse;font-size:12px;width:100%}
th,td{border:1px solid var(--line);padding:4px 7px;white-space:nowrap}
thead th{background:var(--head);position:sticky;top:0;z-index:3;text-align:center;font-weight:600}
thead tr:nth-child(2) th{top:26px}
td.name,th.name{text-align:left;white-space:normal;min-width:300px;max-width:520px;
 position:sticky;left:0;background:#fff;z-index:2}
thead th.name{background:var(--head);z-index:4}
td.num{text-align:right;font-variant-numeric:tabular-nums}
tr.l0 td{font-weight:700;background:#eaf2fd}
tr.l1 td{font-weight:600;background:#f4f7fb}
tr.l0 td.name,tr.l1 td.name{background:inherit}
tbody tr:hover td{background:#fffbe6}
.bad{color:var(--bad);font-weight:600}
.ok{color:var(--ok)}
.badge{display:inline-block;padding:2px 8px;border-radius:10px;font-size:12px;margin-left:8px}
.badge.g{background:#dafbe1;color:var(--ok)}.badge.r{background:#ffebe9;color:var(--bad)}
details{margin:14px 0;background:#fff;border:1px solid var(--line);border-radius:8px;padding:10px}
summary{cursor:pointer;font-weight:600}
img.page{max-width:100%;border:1px solid var(--line);margin-top:10px}
footer{margin-top:26px;color:#57606a;font-size:12px}
"""


def _esc(s: Any) -> str:
    return html.escape("" if s is None else str(s))


def to_html(result: Dict[str, Any], out_path: str,
            page_images: Optional[List[bytes]] = None,
            embed_source: Optional[str] = None) -> str:
    tables = result.get("tables", [])
    meta = result.get("meta", {})
    st = result.get("stats", {})
    parts: List[str] = []
    parts.append("<!DOCTYPE html><html lang='ru'><head><meta charset='utf-8'>")
    parts.append("<meta name='viewport' content='width=device-width,initial-scale=1'>")
    parts.append(f"<title>{_esc(Path(result.get('source','')).name)} — ГПР</title>")
    parts.append(f"<style>{CSS}</style></head><body>")
    parts.append("<h1>Графики движения ресурсов (ГПР)</h1>")
    parts.append(f"<div class='sub'>Источник: <b>{_esc(Path(result.get('source','')).name)}</b>"
                 f" &nbsp;•&nbsp; режим: {_esc(result.get('engine',''))}"
                 f" &nbsp;•&nbsp; сформировано: {datetime.now():%d.%m.%Y %H:%M}</div>")

    labels = [("contractor", "Подрядчик"), ("construction", "Наименование стройки"),
              ("construction_code", "Код стройки"), ("project", "Наименование проекта"),
              ("project_code", "Код проекта"), ("nomenclature", "Номенклатурная группа"),
              ("object", "Наименование объекта"), ("gpr_type", "Тип ГПТЛР"), ("month", "Месяц")]
    if meta:
        parts.append("<div class='meta'>")
        for k, t in labels:
            if meta.get(k):
                parts.append(f"<div><b>{t}:</b> {_esc(meta[k])}</div>")
        parts.append("</div>")

    parts.append(f"<div class='sub'>Страниц: {st.get('pages',0)} &nbsp;•&nbsp; "
                 f"таблиц: {st.get('tables',0)} &nbsp;•&nbsp; строк: {st.get('rows',0)} "
                 f"&nbsp;•&nbsp; расхождений: {st.get('errors',0)}</div>")

    for t in tables:
        v = t.get("validation", {})
        cls, txt = ("g", "контрольные суммы сошлись") if not v.get("failed") else \
                   ("r", f"расхождений: {v['failed']}")
        parts.append(f"<h2>{_esc(t['title'])}<span class='badge {cls}'>{txt}</span></h2>")
        parts.append(f"<div class='sub'>Страница {t['page']} • источник данных: "
                     f"{'текстовый слой PDF' if t['source']=='text' else 'OCR'}</div>")
        parts.append("<div class='wrap'><table>")

        cols = t["columns"]
        years: List[Any] = [c.get("year") for c in cols]
        parts.append("<thead><tr>")
        parts.append(f"<th class='name' rowspan='2'>{_esc(t.get('name_header') or 'Наименование')}</th>")
        if t["kind"] == "tech":
            parts.append("<th rowspan='2'>Коэф-т<br>сменности</th>")
        parts.append("<th rowspan='2'>ИТОГО<br>кол-во</th>")
        i = 0
        while i < len(years):
            j = i
            while j + 1 < len(years) and years[j + 1] == years[i]:
                j += 1
            span = j - i + 1
            parts.append(f"<th colspan='{span}'>{_esc(years[i] or '')}</th>")
            i = j + 1
        if t["kind"] == "tech":
            parts.append("<th rowspan='2'>Пиковая<br>потребность</th>")
            parts.append("<th rowspan='2'>Кол-во<br>машино-часов</th>")
        else:
            parts.append("<th rowspan='2'>Кол-во раб.<br>часов в месяц</th>")
        parts.append("<th rowspan='2'>ИТОГО<br>чел.-час / маш.-час</th>")
        parts.append("<th rowspan='2'>Проверка</th></tr><tr>")
        for c in cols:
            parts.append(f"<th>{_esc(MONTHS[c['month']].capitalize())}</th>")
        parts.append("</tr></thead><tbody>")

        for r in t["rows"]:
            parts.append(f"<tr class='l{r.get('level',2)}'>")
            nm = (r.get("code", "") + " " + r["name"]).strip() if t["kind"] == "tech" else r["name"]
            parts.append(f"<td class='name'>{_esc(nm)}</td>")
            if t["kind"] == "tech":
                parts.append(f"<td class='num'>{_esc(fmt(r.get('shift')))}</td>")
            parts.append(f"<td class='num'>{_esc(fmt(r.get('total')))}</td>")
            for val in r["values"]:
                parts.append(f"<td class='num'>{_esc(fmt(val))}</td>")
            if t["kind"] == "tech":
                parts.append(f"<td class='num'>{_esc(fmt(r.get('peak')))}</td>")
            parts.append(f"<td class='num'>{_esc(fmt(r.get('hours')))}</td>")
            parts.append(f"<td class='num'>{_esc(fmt(r.get('sum_hours')))}</td>")
            if r.get("checks"):
                parts.append("<td class='bad'>" + _esc("; ".join(r["checks"])) + "</td>")
            else:
                parts.append("<td class='ok'>OK</td>")
            parts.append("</tr>")
        parts.append("</tbody></table></div>")

    if page_images:
        parts.append("<details><summary>Исходные страницы документа "
                     f"({len(page_images)} шт.)</summary>")
        for i, jpg in enumerate(page_images, 1):
            b64 = base64.b64encode(jpg).decode("ascii")
            parts.append(f"<div><b>Страница {i}</b></div>")
            parts.append(f"<img class='page' src='data:image/jpeg;base64,{b64}' alt='page {i}'>")
        parts.append("</details>")

    if embed_source and os.path.exists(embed_source):
        b64 = base64.b64encode(Path(embed_source).read_bytes()).decode("ascii")
        name = Path(embed_source).name
        parts.append(f"<p><a download='{_esc(name)}' "
                     f"href='data:application/pdf;base64,{b64}'>Скачать исходный файл</a></p>")

    parts.append("<footer>DocumentOCR • АО «СТНГ» • разработчик Уклеин И.В., "
                 "г. Новосибирск, 2026</footer></body></html>")
    Path(out_path).write_text("\n".join(parts), encoding="utf-8")
    return out_path


# ------------------------------------------------------------------ XLSX
def to_xlsx(result: Dict[str, Any], out_path: str) -> str:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter

    thin = Side(style="thin", color="BFBFBF")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)
    head_fill = PatternFill("solid", fgColor="EEF2F7")
    bad_fill = PatternFill("solid", fgColor="FFEBE9")
    bold = Font(bold=True)

    wb = Workbook()
    ws = wb.active
    ws.title = "Сводка"
    ws.append(["DocumentOCR — результат распознавания ГПР"])
    ws["A1"].font = Font(bold=True, size=14)
    ws.append([])
    ws.append(["Исходный файл", Path(result.get("source", "")).name])
    ws.append(["Режим", result.get("engine", "")])
    ws.append(["Сформировано", f"{datetime.now():%d.%m.%Y %H:%M}"])
    for k, t in [("contractor", "Подрядчик"), ("construction", "Наименование стройки"),
                 ("construction_code", "Код стройки"), ("project", "Проект"),
                 ("project_code", "Код проекта"), ("nomenclature", "Номенклатурная группа"),
                 ("object", "Объект"), ("gpr_type", "Тип ГПТЛР"), ("month", "Месяц")]:
        if result.get("meta", {}).get(k):
            ws.append([t, result["meta"][k]])
    ws.append([])
    ws.append(["Таблица", "Лист", "Страница", "Строк", "Расхождений"])
    for c in ws[ws.max_row]:
        c.font = bold
    ws.column_dimensions["A"].width = 62
    ws.column_dimensions["B"].width = 18
    for col in "CDE":
        ws.column_dimensions[col].width = 14

    used = set()
    for t in result.get("tables", []):
        base = t.get("sheet_name") or f"{t['kind']}_{t['page']}"
        name = base
        n = 2
        while name in used:
            name = f"{base[:28]}_{n}"
            n += 1
        used.add(name)
        sh = wb.create_sheet(name)
        ws.append([t["title"], name, t["page"], t["validation"]["rows"], t["validation"]["failed"]])

        cols = t["columns"]
        is_tech = t["kind"] == "tech"
        headers1 = [t.get("name_header") or "Наименование"]
        headers2 = [""]
        if is_tech:
            headers1 += ["Коэф-т сменности"]
            headers2 += [""]
        headers1 += ["ИТОГО кол-во"]
        headers2 += [""]
        for c in cols:
            headers1.append(c.get("year") or "")
            headers2.append(MONTHS[c["month"]].capitalize())
        if is_tech:
            headers1 += ["Пиковая потребность", "Кол-во машино-часов"]
            headers2 += ["", ""]
        else:
            headers1 += ["Кол-во раб. часов в месяц"]
            headers2 += [""]
        headers1 += ["ИТОГО чел.-час / маш.-час", "Проверка"]
        headers2 += ["", ""]

        sh.append(headers1)
        sh.append(headers2)
        for row in (sh[1], sh[2]):
            for cell in row:
                cell.font = bold
                cell.fill = head_fill
                cell.border = border
                cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

        # объединяем одинаковые года в шапке
        start = len(headers1) - len(cols) - (3 if is_tech else 2) - 1
        first_month_col = 3 if is_tech else 2
        i = 0
        while i < len(cols):
            j = i
            while j + 1 < len(cols) and cols[j + 1].get("year") == cols[i].get("year"):
                j += 1
            if j > i:
                sh.merge_cells(start_row=1, start_column=first_month_col + i + 1,
                               end_row=1, end_column=first_month_col + j + 1)
            i = j + 1

        for r in t["rows"]:
            nm = (r.get("code", "") + " " + r["name"]).strip() if is_tech else r["name"]
            line = [nm]
            if is_tech:
                line.append(r.get("shift"))
            line.append(r.get("total"))
            line.extend(r["values"])
            if is_tech:
                line.append(r.get("peak"))
            line.append(r.get("hours"))
            line.append(r.get("sum_hours"))
            line.append("; ".join(r.get("checks", [])) or "OK")
            sh.append(line)
            rr = sh.max_row
            for ci, cell in enumerate(sh[rr], start=1):
                cell.border = border
                if ci == 1:
                    cell.alignment = Alignment(wrap_text=True, vertical="center")
                else:
                    cell.alignment = Alignment(horizontal="right")
                    if isinstance(cell.value, (int, float)):
                        cell.number_format = "#,##0" if float(cell.value).is_integer() else "0.00"
                if r.get("level", 2) <= 1:
                    cell.font = bold
            if r.get("checks"):
                sh.cell(rr, len(line)).fill = bad_fill
                sh.cell(rr, len(line)).font = Font(color="B42318", bold=True)

        sh.column_dimensions["A"].width = 58
        for ci in range(2, len(headers1) + 1):
            sh.column_dimensions[get_column_letter(ci)].width = 12
        sh.column_dimensions[get_column_letter(len(headers1))].width = 46
        sh.freeze_panes = sh.cell(3, first_month_col + 1)
        sh.auto_filter.ref = f"A2:{get_column_letter(len(headers1))}{sh.max_row}"

    wb.save(out_path)
    return out_path


def to_json(result: Dict[str, Any], out_path: str) -> str:
    Path(out_path).write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str),
                              encoding="utf-8")
    return out_path


def make_zip(zip_path: str, files: List[str], source: Optional[str] = None,
             report: Optional[str] = None) -> str:
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for f in files:
            if f and os.path.exists(f):
                z.write(f, arcname=Path(f).name)
        if source and os.path.exists(source):
            z.write(source, arcname="source/" + Path(source).name)
        if report:
            z.writestr("REPORT.txt", report)
    return zip_path
