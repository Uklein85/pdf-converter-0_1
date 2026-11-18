"""DocumentOCR — ГПР: таблицы, сжатие PDF, экспорт HTML/XLSX/PDF с вложениями."""
from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (QApplication, QComboBox, QFileDialog, QFrame, QGroupBox,
                             QHBoxLayout, QHeaderView, QLabel, QMainWindow, QMessageBox,
                             QProgressBar, QPushButton, QSpinBox, QSplitter, QTabWidget,
                             QTableWidget, QTableWidgetItem, QTextEdit, QVBoxLayout, QWidget,
                             QCheckBox)

import exporters
import gpr_core as gpr
import pdf_io as pio
import pipeline
from ocr_engines import OCREngineManager

SUPPORTED = ("Документы (*.pdf *.xlsx *.xls *.jpg *.jpeg *.png *.tif *.tiff *.bmp *.webp);;"
             "Все файлы (*.*)")


class Worker(QThread):
    progress = pyqtSignal(int, str)
    done = pyqtSignal(dict)
    fail = pyqtSignal(str)

    def __init__(self, task, **kw):
        super().__init__()
        self.task, self.kw = task, kw

    def run(self):
        try:
            self.done.emit(self.task(progress=self.progress.emit, **self.kw))
        except Exception as e:
            self.fail.emit(f"{type(e).__name__}: {e}\n\n{traceback.format_exc(limit=4)}")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DocumentOCR — ГПР | АО «СТНГ»")
        self.resize(1400, 880)
        self.ocr = OCREngineManager(languages=["ru", "en"])
        self.source = None
        self.result = None
        self.compress_info = None
        self.worker = None
        self._ui()
        self.statusBar().showMessage("Готов. Откройте PDF со сканом или текстовым слоем.")

    # ---------------------------------------------------------------- UI
    def _ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        top = QFrame()
        top.setFrameStyle(QFrame.Shape.StyledPanel)
        tl = QHBoxLayout(top)
        self.btn_open = QPushButton("Открыть файл")
        self.btn_open.clicked.connect(self.open_file)
        tl.addWidget(self.btn_open)
        tl.addWidget(QLabel("Режим:"))
        self.cmb_mode = QComboBox()
        self.cmb_mode.addItems(["Авто (текст → OCR)", "Только текстовый слой", "Только OCR"])
        tl.addWidget(self.cmb_mode)
        tl.addWidget(QLabel("DPI:"))
        self.spin_dpi = QSpinBox()
        self.spin_dpi.setRange(150, 600)
        self.spin_dpi.setValue(300)
        self.spin_dpi.setSingleStep(50)
        tl.addWidget(self.spin_dpi)
        self.chk_prev = QCheckBox("Превью страниц в HTML")
        self.chk_prev.setChecked(True)
        tl.addWidget(self.chk_prev)
        tl.addStretch()
        self.lbl_engine = QLabel("OCR: " + (", ".join(self.ocr.get_available_engines()) or "нет"))
        tl.addWidget(self.lbl_engine)
        root.addWidget(top)

        act = QHBoxLayout()
        self.btn_run = QPushButton("▶ Распознать и собрать таблицы ГПР")
        self.btn_run.clicked.connect(self.run_recognize)
        self.btn_run.setEnabled(False)
        act.addWidget(self.btn_run)
        self.btn_zip = QPushButton("📦 Экспорт пакета (HTML + XLSX + сжатый PDF с вложениями)")
        self.btn_zip.clicked.connect(self.run_package)
        self.btn_zip.setEnabled(False)
        act.addWidget(self.btn_zip)
        self.btn_compress = QPushButton("🗜 Только сжать PDF")
        self.btn_compress.clicked.connect(self.run_compress)
        self.btn_compress.setEnabled(False)
        act.addWidget(self.btn_compress)
        act.addStretch()
        root.addLayout(act)

        split = QSplitter(Qt.Orientation.Horizontal)
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        g = QGroupBox("Документ")
        gl = QVBoxLayout(g)
        self.lbl_file = QLabel("Файл: не выбран")
        self.lbl_size = QLabel("Размер: —")
        self.lbl_meta = QLabel("Объект: —")
        self.lbl_meta.setWordWrap(True)
        for w in (self.lbl_file, self.lbl_size, self.lbl_meta):
            gl.addWidget(w)
        ll.addWidget(g)
        g2 = QGroupBox("Результат")
        g2l = QVBoxLayout(g2)
        self.lbl_stat = QLabel("Таблиц: —\nСтрок: —\nРасхождений: —")
        self.lbl_compress = QLabel("Сжатие: —")
        self.lbl_compress.setWordWrap(True)
        g2l.addWidget(self.lbl_stat)
        g2l.addWidget(self.lbl_compress)
        ll.addWidget(g2)
        ll.addStretch()
        split.addWidget(left)

        self.tabs = QTabWidget()
        self.tbl = QTableWidget()
        self.tbl.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.tabs.addTab(self.tbl, "Таблицы ГПР")
        self.cmb_tables = QComboBox()
        self.cmb_tables.currentIndexChanged.connect(self.show_table)
        tw = QWidget()
        tv = QVBoxLayout(tw)
        tv.addWidget(self.cmb_tables)
        tv.addWidget(self.tbl)
        self.tabs.removeTab(0)
        self.tabs.addTab(tw, "Таблицы ГПР")
        self.txt = QTextEdit()
        self.txt.setReadOnly(True)
        self.txt.setFont(QFont("Consolas", 10))
        self.tabs.addTab(self.txt, "Текст")
        self.chk = QTextEdit()
        self.chk.setReadOnly(True)
        self.chk.setFont(QFont("Consolas", 10))
        self.tabs.addTab(self.chk, "Проверка сумм")
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.tabs.addTab(self.log, "Лог")
        split.addWidget(self.tabs)
        split.setSizes([320, 1080])
        root.addWidget(split, 1)

        bot = QHBoxLayout()
        self.bar = QProgressBar()
        bot.addWidget(self.bar, 1)
        self.lbl_status = QLabel("Ожидание")
        self.lbl_status.setMinimumWidth(280)
        bot.addWidget(self.lbl_status)
        root.addLayout(bot)

    def _p(self, v, m):
        self.bar.setValue(v)
        self.lbl_status.setText(m)
        self.log.append(f"[{v:3d}%] {m}")

    def _busy(self, on):
        for b in (self.btn_open, self.btn_run, self.btn_zip, self.btn_compress):
            b.setEnabled(not on)
        if not on and self.source:
            self.btn_run.setEnabled(True)
            self.btn_compress.setEnabled(Path(self.source).suffix.lower() == ".pdf")
            self.btn_zip.setEnabled(self.result is not None)

    # ---------------------------------------------------------------- actions
    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Выберите документ", "", SUPPORTED)
        if not path:
            return
        self.source = path
        self.result = None
        self.compress_info = None
        self.lbl_file.setText("Файл: " + Path(path).name)
        self.lbl_size.setText(f"Размер: {os.path.getsize(path) / 1048576:.2f} МБ")
        self.btn_run.setEnabled(True)
        self.btn_compress.setEnabled(Path(path).suffix.lower() == ".pdf")
        self.btn_zip.setEnabled(False)
        self._p(0, "Файл выбран")

    def _mode(self):
        return ["auto", "text", "ocr"][self.cmb_mode.currentIndex()]

    def run_recognize(self):
        if not self.source:
            return
        self._busy(True)
        self.worker = Worker(pipeline.process, path=self.source, ocr=self.ocr,
                             mode=self._mode(), dpi=self.spin_dpi.value(),
                             keep_page_images=self.chk_prev.isChecked())
        self.worker.progress.connect(self._p)
        self.worker.done.connect(self._on_result)
        self.worker.fail.connect(self._on_fail)
        self.worker.start()

    def _on_result(self, res):
        self.result = res
        self._busy(False)
        st = res["stats"]
        self.lbl_stat.setText(f"Таблиц: {st['tables']}\nСтрок: {st['rows']}\n"
                              f"Расхождений: {st['errors']}")
        self.lbl_meta.setText("Объект: " + res.get("meta", {}).get("object", "—"))
        self.txt.setPlainText(res.get("text", ""))
        self.chk.setPlainText(pipeline.build_report(res))
        self.cmb_tables.blockSignals(True)
        self.cmb_tables.clear()
        for t in res["tables"]:
            self.cmb_tables.addItem(f"{t['title']} (стр. {t['page']})")
        self.cmb_tables.blockSignals(False)
        if res["tables"]:
            self.cmb_tables.setCurrentIndex(0)
            self.show_table(0)
        self.btn_zip.setEnabled(True)
        self.tabs.setCurrentIndex(0)
        if st["errors"]:
            QMessageBox.warning(self, "Проверка сумм",
                                f"Таблиц собрано: {st['tables']}\n"
                                f"Строк: {st['rows']}\n"
                                f"Расхождений: {st['errors']}\n\n"
                                "См. вкладку «Проверка сумм».")
        else:
            QMessageBox.information(self, "Готово",
                                    f"Таблиц ГПР: {st['tables']}\nСтрок: {st['rows']}\n"
                                    "Все контрольные суммы сошлись.")

    def _on_fail(self, msg):
        self._busy(False)
        self.log.append(msg)
        self.tabs.setCurrentIndex(3)
        QMessageBox.critical(self, "Ошибка", msg[:1500])

    def show_table(self, idx):
        if not self.result or not (0 <= idx < len(self.result["tables"])):
            return
        t = self.result["tables"][idx]
        cols = t["columns"]
        is_tech = t["kind"] == "tech"
        headers = [t.get("name_header") or "Наименование"]
        if is_tech:
            headers.append("Коэф-т")
        headers.append("ИТОГО кол-во")
        headers += [c["label"] for c in cols]
        if is_tech:
            headers.append("Пик")
        headers += ["Часов/мес", "ИТОГО часы", "Проверка"]
        self.tbl.clear()
        self.tbl.setColumnCount(len(headers))
        self.tbl.setHorizontalHeaderLabels(headers)
        self.tbl.setRowCount(len(t["rows"]))
        for r, row in enumerate(t["rows"]):
            nm = (row.get("code", "") + " " + row["name"]).strip() if is_tech else row["name"]
            vals = [nm]
            if is_tech:
                vals.append(gpr.fmt(row.get("shift")))
            vals.append(gpr.fmt(row.get("total")))
            vals += [gpr.fmt(v) for v in row["values"]]
            if is_tech:
                vals.append(gpr.fmt(row.get("peak")))
            vals += [gpr.fmt(row.get("hours")), gpr.fmt(row.get("sum_hours")),
                     "; ".join(row.get("checks", [])) or "OK"]
            for c, v in enumerate(vals):
                item = QTableWidgetItem(str(v))
                if c:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                if row.get("level", 2) <= 1:
                    f = item.font()
                    f.setBold(True)
                    item.setFont(f)
                if row.get("checks") and c == len(vals) - 1:
                    item.setForeground(Qt.GlobalColor.red)
                self.tbl.setItem(r, c, item)
        self.tbl.setColumnWidth(0, 420)
        for c in range(1, len(headers)):
            self.tbl.setColumnWidth(c, 105)
        self.tbl.setColumnWidth(len(headers) - 1, 320)

    def run_compress(self):
        if not self.source:
            return
        out, _ = QFileDialog.getSaveFileName(
            self, "Сохранить сжатый PDF",
            str(Path(self.source).with_name(Path(self.source).stem + "_compressed.pdf")),
            "PDF (*.pdf)")
        if not out:
            return
        self._busy(True)
        self.worker = Worker(self._compress_task, src=self.source, dst=out)
        self.worker.progress.connect(self._p)
        self.worker.done.connect(self._on_compress)
        self.worker.fail.connect(self._on_fail)
        self.worker.start()

    @staticmethod
    def _compress_task(progress, src, dst):
        info = pio.compress_pdf(src, dst, target_dpi=200, quality=85, progress=progress)
        return {"compress": info, "out": dst}

    def _on_compress(self, res):
        self._busy(False)
        info = res["compress"]
        self.compress_info = info
        self.lbl_compress.setText(f"Сжатие: {info['before']/1048576:.2f} → "
                                  f"{info['after']/1048576:.2f} МБ (−{info['ratio']:.1f}%)")
        QMessageBox.information(self, "Сжатие завершено",
                                f"Было: {info['before']/1048576:.2f} МБ\n"
                                f"Стало: {info['after']/1048576:.2f} МБ\n"
                                f"Уменьшение: {info['ratio']:.1f}%\n\n{res['out']}")

    def run_package(self):
        if not self.result:
            return
        folder = QFileDialog.getExistingDirectory(self, "Папка для результата")
        if not folder:
            return
        self._busy(True)
        self.worker = Worker(self._package_task, out_dir=folder, result=self.result,
                             source=self.source, embed_pages=self.chk_prev.isChecked())
        self.worker.progress.connect(self._p)
        self.worker.done.connect(self._on_package)
        self.worker.fail.connect(self._on_fail)
        self.worker.start()

    @staticmethod
    def _package_task(progress, out_dir, result, source, embed_pages):
        stem = Path(source).stem
        out = Path(out_dir)
        html_p = out / f"{stem}_ГПР.html"
        xlsx_p = out / f"{stem}_ГПР.xlsx"
        json_p = out / f"{stem}_ГПР.json"
        pdf_p = out / f"{stem}_compressed.pdf"

        progress(10, "Формирование HTML")
        exporters.to_html(result, str(html_p),
                          page_images=result.get("page_images") if embed_pages else None)
        progress(30, "Формирование Excel")
        exporters.to_xlsx(result, str(xlsx_p))
        progress(40, "Формирование JSON")
        exporters.to_json(result, str(json_p))

        info = None
        if Path(source).suffix.lower() == ".pdf":
            info = pio.compress_pdf(source, str(pdf_p), target_dpi=200, quality=85,
                                    progress=lambda p, m: progress(40 + int(p * 0.4), m))
            progress(85, "Вложение результатов в PDF")
            pio.attach_files(str(pdf_p), {
                html_p.name: html_p.read_bytes(),
                xlsx_p.name: xlsx_p.read_bytes(),
                json_p.name: json_p.read_bytes(),
            })

        report = pipeline.build_report(result, info)
        (out / f"{stem}_REPORT.txt").write_text(report, encoding="utf-8")
        progress(92, "Сборка ZIP")
        files = [str(html_p), str(xlsx_p), str(json_p), str(out / f"{stem}_REPORT.txt")]
        if pdf_p.exists():
            files.append(str(pdf_p))
        zip_p = out / f"{stem}_result.zip"
        exporters.make_zip(str(zip_p), files, source=source, report=report)
        progress(100, "Пакет готов")
        return {"zip": str(zip_p), "files": files, "compress": info, "html": str(html_p)}

    def _on_package(self, res):
        self._busy(False)
        self.compress_info = res.get("compress")
        if self.compress_info:
            i = self.compress_info
            self.lbl_compress.setText(f"Сжатие: {i['before']/1048576:.2f} → "
                                      f"{i['after']/1048576:.2f} МБ (−{i['ratio']:.1f}%)")
        QMessageBox.information(
            self, "Пакет сформирован",
            "Созданы файлы:\n\n" + "\n".join(Path(f).name for f in res["files"]) +
            f"\n\nZIP (включая исходный файл): {Path(res['zip']).name}"
            "\n\nHTML и XLSX также вложены внутрь сжатого PDF (панель «Вложения» в Acrobat).")


def main():
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
