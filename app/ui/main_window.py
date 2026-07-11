"""Main application window (PySide6).

The window is intentionally thin: it collects input, hands work to
``AssessmentService`` on a background thread, and renders the returned result.
All heavy imports (xgboost / shap / anthropic) live behind the service.
"""

from __future__ import annotations

from datetime import datetime
import traceback

from PySide6.QtCore import QDate, Qt, QThread, Signal
from PySide6.QtGui import QAction, QFont
from PySide6.QtWidgets import (
    QComboBox, QDateEdit, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox, QPlainTextEdit,
    QPushButton, QScrollArea, QSpinBox, QSplitter, QTextBrowser, QVBoxLayout,
    QWidget,
)

from .. import config
from ..models import StudentCase
from ..service import AssessmentResult, AssessmentService
from .report import (
    build_report_html, build_summary_report_html, export_report_pdf,
    placeholder_html, render_result_html,
)


class FnWorker(QThread):
    """Run any callable on a background thread and emit its result or error."""

    done = Signal(object)
    failed = Signal(str)

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self._fn, self._args, self._kwargs = fn, args, kwargs

    def run(self) -> None:  # noqa: D401
        try:
            self.done.emit(self._fn(*self._args, **self._kwargs))
        except Exception as exc:  # surfaced to the UI, never crashes the app
            self.failed.emit(f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}")


def _safe_stem(text: str, fallback: str) -> str:
    """A filesystem-safe file stem from a (possibly Romanian) name."""
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in text.strip())
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    return cleaned or fallback


class BatchImportWorker(QThread):
    """Import a Google-Forms .xlsx, assess every student and write the PDFs.

    Runs the whole offline pipeline on a background thread and reports progress
    back to the UI via a queued signal. Emits ``done`` with a summary dict, or
    ``failed`` with a message; the app never crashes on a bad workbook.
    """

    progress = Signal(int, int, str)   # done, total, phase label
    done = Signal(object)              # {"out_dir", "count", "summary_path"}
    failed = Signal(str)

    def __init__(self, service: AssessmentService, cases: list, out_dir: str):
        super().__init__()
        self._service, self._cases, self._out_dir = service, cases, out_dir

    def run(self) -> None:  # noqa: D401
        import os

        try:
            cases = self._cases
            total = len(cases)

            self.progress.emit(0, total, "Se evaluează elevii…")
            results = self._service.assess_many(
                cases,
                progress=lambda done, tot: self.progress.emit(
                    done, tot, "Se evaluează elevii…"
                ),
            )

            entries: list[tuple[StudentCase, object]] = []
            for index, (case, result) in enumerate(zip(cases, results), start=1):
                evaluation = result.evaluation
                # Use the configured AI provider when a key is available (same as
                # the single-student flow); generate_plan falls back to the local
                # template per student if the cloud call fails.
                plan_text, _source = self._service.generate_plan(evaluation, case)
                html = build_report_html(case, result, plan_text)
                stem = _safe_stem(case.display_name(), f"elev_{index}")
                pdf_path = os.path.join(self._out_dir, f"raport_{index:02d}_{stem}.pdf")
                export_report_pdf(pdf_path, html)
                entries.append((case, evaluation))
                self.progress.emit(index, total, "Se generează planurile și rapoartele PDF…")

            summary_html = build_summary_report_html(
                entries, model_version=self._service.model_version
            )
            summary_path = os.path.join(self._out_dir, "raport_general.pdf")
            export_report_pdf(summary_path, summary_html)

            self.done.emit(
                {"out_dir": self._out_dir, "count": total, "summary_path": summary_path}
            )
        except Exception as exc:  # surfaced to the UI, never crashes the app
            self.failed.emit(f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}")

_DEMO_ANSWERS = {
    "full_name": "Popescu Andrei",
    "birth_date": "2009-05-14",
    "student_class": "IX A",
    "school_name": "Liceul Tehnologic",
    "sex": "Masculin",
    "residential_environment": "Rural",
    "family_situation": "Monoparental",
    "family_situation_other": "",
    "mother_education": "Gimnazial",
    "father_education": "Primar",
    "unexcused_absences_3m": 12,
    "excused_absences_3m": 4,
    "extracurricular_participation": "Nu",
    "previous_module_average": 4.3,
    "low_grades_details": "4 Matematică; 3 Română; 2 Istorie",
    "school_attitude": "Negativă",
    "disciplinary_sanctions": "Avertismente",
    "school_feeling": "Stresat",
    "school_feeling_other": "Se simte copleșit de cerințe.",
    "school_support_goal": "Nu",
    "additional_notes": "Elevul este retras și obosit în ultima perioadă, refuză să participe la activități și pare demotivat.",
}


def _split_full_name(full_name: str) -> tuple[str, str]:
    parts = [part for part in full_name.split() if part]
    if not parts:
        return "", ""
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], " ".join(parts[1:])


def _compose_observation_text(answers: dict[str, object]) -> str:
    parts: list[str] = []
    for key, label in (
        ("family_situation_other", "Situație familială - detalii"),
        ("low_grades_details", "Note sub 5 - detalii"),
        ("school_feeling_other", "Cum se simte la școală - detalii"),
        ("additional_notes", "Observații suplimentare"),
    ):
        value = answers.get(key)
        if value not in (None, ""):
            parts.append(f"{label}: {value}")
    return "\n".join(parts).strip()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(config.APP_TITLE)
        self.resize(1180, 780)

        self.service = AssessmentService()
        self.current_case: StudentCase | None = None
        self.current_result: AssessmentResult | None = None
        self.current_plan_text: str = ""     # raw plan markdown (source for the PDF)
        self._workers: list[FnWorker] = []
        self._question_widgets: dict[str, QWidget] = {}

        self._build_menu()
        self._build_ui()
        self._refresh_llm_status()

    # --- UI construction ---------------------------------------------------
    def _build_menu(self) -> None:
        questionnaire_menu = self.menuBar().addMenu("&Chestionar")
        self.act_import = QAction("Importă răspunsuri Google Forms (.xlsx)…", self)
        self.act_import.triggered.connect(self._on_import_excel)
        questionnaire_menu.addAction(self.act_import)

        settings_menu = self.menuBar().addMenu("&Setări")
        act_kb = QAction("Bază de cunoștințe (.docx)…", self)
        act_kb.triggered.connect(self._on_settings)
        settings_menu.addAction(act_kb)

        model_menu = self.menuBar().addMenu("&Model")
        act_retrain = QAction("Reantrenează modelul", self)
        act_retrain.triggered.connect(self._on_retrain)
        model_menu.addAction(act_retrain)
        act_metrics = QAction("Metrici model…", self)
        act_metrics.triggered.connect(self._on_metrics)
        model_menu.addAction(act_metrics)

        help_menu = self.menuBar().addMenu("&Ajutor")
        act_about = QAction("Despre", self)
        act_about.triggered.connect(self._on_about)
        help_menu.addAction(act_about)

    def _make_question_widget(self, item: config.QuestionnaireItem) -> QWidget:
        if item.key == "timestamp":
            label = QLabel(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            label.setStyleSheet("color:#555;")
            return label
        if item.kind == "date":
            widget = QDateEdit()
            widget.setCalendarPopup(True)
            if isinstance(item.default, str) and item.default:
                qdate = QDate.fromString(item.default, "yyyy-MM-dd")
                widget.setDate(qdate if qdate.isValid() else QDate.currentDate())
            else:
                widget.setDate(QDate.currentDate())
            return widget
        if item.kind == "categorical":
            widget = QComboBox()
            widget.addItems(list(item.categories))
            return widget
        if item.kind == "multiline":
            widget = QPlainTextEdit()
            widget.setFixedHeight(72)
            return widget
        if item.kind == "text":
            widget = QLineEdit()
            if isinstance(item.default, str) and item.default:
                widget.setText(item.default)
            return widget
        if item.step < 1:
            widget = QDoubleSpinBox()
            widget.setRange(item.minimum, item.maximum)
            widget.setSingleStep(item.step)
            widget.setDecimals(1)
            widget.setValue(float(item.default or 0.0))
            return widget
        widget = QSpinBox()
        widget.setRange(int(item.minimum), int(item.maximum))
        widget.setSingleStep(int(item.step))
        widget.setValue(int(item.default or 0))
        return widget

    def _question_value(self, key: str):
        widget = self._question_widgets[key]
        if isinstance(widget, QComboBox):
            return widget.currentText()
        if isinstance(widget, QDateEdit):
            return widget.date().toString("yyyy-MM-dd")
        if isinstance(widget, QDoubleSpinBox):
            return float(widget.value())
        if isinstance(widget, QSpinBox):
            return int(widget.value())
        if isinstance(widget, QPlainTextEdit):
            return widget.toPlainText().strip()
        if isinstance(widget, QLineEdit):
            return widget.text().strip()
        if isinstance(widget, QLabel):
            return widget.text().strip()
        return ""

    def _build_ui(self) -> None:
        splitter = QSplitter(Qt.Horizontal)

        # Left: input form (scrollable).
        form_widget = QWidget()
        form_layout = QVBoxLayout(form_widget)

        identity = QGroupBox("Date de identificare (rămân doar local)")
        idl = QFormLayout(identity)
        identity_fields = {
            "full_name",
            "birth_date",
            "student_class",
            "school_name",
        }
        for item in config.QUESTIONNAIRE_FIELDS:
            if item.key not in identity_fields:
                continue
            widget = self._make_question_widget(item)
            self._question_widgets[item.key] = widget
            idl.addRow(item.label, widget)
        timestamp = next(q for q in config.QUESTIONNAIRE_FIELDS if q.key == "timestamp")
        idl.addRow(timestamp.label, self._make_question_widget(timestamp))
        form_layout.addWidget(identity)

        questionnaire_box = QGroupBox("Chestionar complet")
        ql = QFormLayout(questionnaire_box)
        for item in config.QUESTIONNAIRE_FIELDS:
            if item.key in identity_fields or item.key == "timestamp":
                continue
            widget = self._make_question_widget(item)
            widget.setToolTip(item.help_text)
            self._question_widgets[item.key] = widget
            ql.addRow(item.label, widget)
        self.lbl_stress = QLabel("Stres emoțional (NLP): — (se calculează la evaluare)")
        self.lbl_stress.setStyleSheet("color:#555;")
        ql.addRow(self.lbl_stress)
        form_layout.addWidget(questionnaire_box)

        # Buttons
        btn_row = QHBoxLayout()
        self.btn_demo = QPushButton("Completează exemplu")
        self.btn_demo.clicked.connect(self._fill_demo)
        self.btn_eval = QPushButton("Evaluează riscul")
        self.btn_eval.setStyleSheet("font-weight:bold;")
        self.btn_eval.clicked.connect(self._on_evaluate)
        btn_row.addWidget(self.btn_demo)
        btn_row.addWidget(self.btn_eval)
        form_layout.addLayout(btn_row)

        btn_row2 = QHBoxLayout()
        self.btn_plan = QPushButton("Generează planul de intervenție")
        self.btn_plan.setEnabled(False)
        self.btn_plan.clicked.connect(self._on_generate_plan)
        self.btn_save = QPushButton("Salvează evaluarea")
        self.btn_save.setEnabled(False)
        self.btn_save.clicked.connect(self._on_save)
        btn_row2.addWidget(self.btn_plan)
        btn_row2.addWidget(self.btn_save)
        form_layout.addLayout(btn_row2)

        btn_row3 = QHBoxLayout()
        self.btn_pdf = QPushButton("Salvează raport PDF")
        self.btn_pdf.setEnabled(False)
        self.btn_pdf.clicked.connect(self._on_export_pdf)
        btn_row3.addWidget(self.btn_pdf)
        form_layout.addLayout(btn_row3)
        form_layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(form_widget)
        scroll.setMinimumWidth(430)
        splitter.addWidget(scroll)

        # Right: results + plan.
        right = QWidget()
        right_layout = QVBoxLayout(right)
        self.results = QTextBrowser()
        self.results.setHtml(placeholder_html())
        right_layout.addWidget(self.results, stretch=3)

        plan_box = QGroupBox("Plan de intervenție")
        plan_layout = QVBoxLayout(plan_box)
        self.plan_source = QLabel("Sursă: —")
        self.plan_source.setStyleSheet("color:#555;")
        # QTextBrowser renders the plan's markdown (headings, bold, bullets);
        # the raw markdown is kept in ``self.current_plan_text`` for the PDF.
        self.plan_text = QTextBrowser()
        plan_layout.addWidget(self.plan_source)
        plan_layout.addWidget(self.plan_text)
        right_layout.addWidget(plan_box, stretch=2)
        splitter.addWidget(right)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        self.setCentralWidget(splitter)

        self.statusBar().showMessage("Gata. Modelul se încarcă la prima evaluare.")

    # --- helpers -----------------------------------------------------------
    def _refresh_llm_status(self) -> None:
        from .. import keystore, settings
        provider = config.get_provider(settings.get_active_provider())
        source = keystore.key_source(provider.id)
        if source == "env":
            mode = f"cloud ({provider.label}) — cheie din variabila de mediu"
        elif source == "stored":
            mode = f"cloud ({provider.label}) — cheie salvată"
        else:
            mode = "local (offline)"
        kb = settings.get_knowledge()
        kb_note = f" • bază de cunoștințe: {kb['filename']}" if kb else ""
        self.plan_source.setText(f"Sursă plan: se va folosi modul {mode}.{kb_note}")

    def _on_settings(self) -> None:
        from .settings_dialog import SettingsDialog
        SettingsDialog(self).exec()
        self._refresh_llm_status()

    def _fill_demo(self) -> None:
        for key, val in _DEMO_ANSWERS.items():
            w = self._question_widgets[key]
            if isinstance(w, QComboBox):
                w.setCurrentText(str(val))
            elif isinstance(w, QDateEdit):
                qdate = QDate.fromString(str(val), "yyyy-MM-dd")
                w.setDate(qdate if qdate.isValid() else QDate.currentDate())
            elif isinstance(w, QLineEdit):
                w.setText(str(val))
            elif isinstance(w, QPlainTextEdit):
                w.setPlainText(str(val))
            elif isinstance(w, QDoubleSpinBox):
                w.setValue(float(val))
            elif isinstance(w, QSpinBox):
                w.setValue(int(val))

    def _collect_answers(self) -> dict[str, object]:
        answers: dict[str, object] = {"timestamp": datetime.now().isoformat(timespec="seconds")}
        for key, widget in self._question_widgets.items():
            if isinstance(widget, QComboBox):
                answers[key] = widget.currentText().strip()
            elif isinstance(widget, QDateEdit):
                answers[key] = widget.date().toString("yyyy-MM-dd")
            elif isinstance(widget, QLineEdit):
                answers[key] = widget.text().strip()
            elif isinstance(widget, QPlainTextEdit):
                answers[key] = widget.toPlainText().strip()
            elif isinstance(widget, QDoubleSpinBox):
                answers[key] = float(widget.value())
            elif isinstance(widget, QSpinBox):
                answers[key] = int(widget.value())
            elif isinstance(widget, QLabel):
                answers[key] = widget.text().strip()
        return answers

    def _collect_case(self) -> StudentCase:
        answers = self._collect_answers()
        full_name = str(answers.get("full_name", ""))
        surname, name = _split_full_name(full_name)
        features = dict(answers)
        features["observation_text"] = _compose_observation_text(answers)
        is_urban = answers.get("residential_environment") == "Urban"
        return StudentCase(
            name=name,
            surname=surname,
            student_grade=str(answers.get("student_class", "")).strip(),
            school_name=str(answers.get("school_name", "")).strip(),
            is_urban=is_urban,
            observation_text=str(features["observation_text"]),
            features=features,
        )

    def _start_worker(self, fn, on_done, *args, **kwargs) -> None:
        worker = FnWorker(fn, *args, **kwargs)
        worker.done.connect(on_done)
        worker.failed.connect(self._on_worker_error)
        worker.finished.connect(lambda w=worker: self._workers.remove(w) if w in self._workers else None)
        self._workers.append(worker)
        worker.start()

    def _set_busy(self, busy: bool, message: str = "") -> None:
        self.btn_eval.setEnabled(not busy)
        self.btn_demo.setEnabled(not busy)
        self.act_import.setEnabled(not busy)
        if message:
            self.statusBar().showMessage(message)
        self.setCursor(Qt.BusyCursor if busy else Qt.ArrowCursor)

    # --- actions -----------------------------------------------------------
    def _on_evaluate(self) -> None:
        self.current_case = self._collect_case()
        self.current_plan_text = ""      # a new evaluation invalidates the old plan
        self.plan_text.clear()
        self.plan_source.setText("Sursă: —")
        self._set_busy(True, "Se calculează scorul (model XGBoost + SHAP)…")
        self.btn_plan.setEnabled(False)
        self.btn_save.setEnabled(False)
        self.btn_pdf.setEnabled(False)
        self._start_worker(self.service.assess, self._on_eval_done, self.current_case)

    def _on_eval_done(self, result: AssessmentResult) -> None:
        self.current_result = result
        self.lbl_stress.setText(
            f"Stres emoțional (NLP): {result.nlp.stress_score:.2f} / 2.0  "
            f"— {result.nlp.label}"
        )
        self.results.setHtml(render_result_html(result))
        self.btn_plan.setEnabled(True)
        self.btn_save.setEnabled(True)
        self.btn_pdf.setEnabled(True)
        self._set_busy(False, f"Evaluare completă. Model: {result.evaluation.model_version}")

    def _on_generate_plan(self) -> None:
        if not self.current_result:
            return
        self._set_busy(True, "Se generează planul de intervenție…")
        self.plan_text.setPlainText("Se generează…")
        self._start_worker(
            self.service.generate_plan, self._on_plan_done,
            self.current_result.evaluation,
            self.current_case,
        )

    def _on_plan_done(self, payload) -> None:
        text, source = payload
        self.current_plan_text = text
        self.plan_text.setMarkdown(text)
        self.plan_source.setText(f"Sursă plan: {source}")
        self._set_busy(False, "Plan generat.")

    def _on_save(self) -> None:
        if not (self.current_case and self.current_result):
            return
        try:
            case_id, eval_id = self.service.save(
                self.current_case, self.current_result.evaluation
            )
            self.statusBar().showMessage(
                f"Salvat în baza de date locală (evaluare #{eval_id})."
            )
        except Exception as exc:
            self._on_worker_error(f"{type(exc).__name__}: {exc}")

    def _on_export_pdf(self) -> None:
        if not (self.current_case and self.current_result):
            return
        safe_name = self.current_case.display_name().replace(" ", "_")
        if safe_name in ("", "(elev_fără_nume)"):
            safe_name = "elev"
        default_path = f"raport_{safe_name}.pdf"
        path, _ = QFileDialog.getSaveFileName(
            self, "Salvează raportul ca PDF", default_path, "Fișier PDF (*.pdf)"
        )
        if not path:
            return
        if not path.lower().endswith(".pdf"):
            path += ".pdf"
        try:
            html = build_report_html(
                self.current_case, self.current_result, self.current_plan_text
            )
            export_report_pdf(path, html)
            self.statusBar().showMessage(f"Raport PDF salvat: {path}")
        except Exception as exc:
            self._on_worker_error(f"{type(exc).__name__}: {exc}")

    # --- batch import (Google-Forms .xlsx) ---------------------------------
    def _on_import_excel(self) -> None:
        xlsx_path, _ = QFileDialog.getOpenFileName(
            self,
            "Alege fișierul cu răspunsuri (.xlsx exportat din Google Forms)",
            "",
            "Fișiere Excel (*.xlsx *.xlsm)",
        )
        if not xlsx_path:
            return

        # Load + validate the workbook up front (fast, no model) so a bad file
        # fails immediately and we can show the student count in the prompt.
        from ..excel_import import ExcelImportError, load_cases_from_excel
        try:
            cases = load_cases_from_excel(xlsx_path)
        except ExcelImportError as exc:
            QMessageBox.critical(self, "Import eșuat", str(exc))
            return
        except Exception as exc:
            QMessageBox.critical(self, "Import eșuat", f"{type(exc).__name__}: {exc}")
            return
        if not cases:
            QMessageBox.warning(self, "Import", "Nu s-au găsit elevi în fișier.")
            return

        # Plans use the configured cloud provider when a key exists (same as the
        # single-student flow); warn about the per-student cloud latency first.
        from .. import keystore, settings
        provider = config.get_provider(settings.get_active_provider())
        uses_cloud = bool(keystore.resolve_api_key(provider.id))
        if uses_cloud:
            note = (
                f"Se vor evalua {len(cases)} elevi, iar planurile de intervenție "
                f"vor fi generate prin {provider.label} (cu revenire la planul "
                "local dacă un apel eșuează).\n\n"
                "Fiecare plan necesită un apel în cloud, deci procesul poate dura "
                "câteva minute pentru o clasă întreagă. Continuați?"
            )
        else:
            note = (
                f"Se vor evalua {len(cases)} elevi, iar planurile vor fi generate "
                "local (offline). Continuați?"
            )
        if QMessageBox.question(self, "Confirmare import", note) != QMessageBox.Yes:
            return

        out_dir = QFileDialog.getExistingDirectory(
            self, "Alege folderul unde se salvează rapoartele PDF"
        )
        if not out_dir:
            return

        self._set_busy(True, "Se importă și se evaluează elevii…")
        worker = BatchImportWorker(self.service, cases, out_dir)
        worker.progress.connect(self._on_batch_progress)
        worker.done.connect(self._on_batch_done)
        worker.failed.connect(self._on_batch_error)
        worker.finished.connect(
            lambda w=worker: self._workers.remove(w) if w in self._workers else None
        )
        self._workers.append(worker)
        worker.start()

    def _on_batch_progress(self, done: int, total: int, phase: str) -> None:
        if total:
            self.statusBar().showMessage(f"{phase} ({done}/{total})")
        else:
            self.statusBar().showMessage(phase)

    def _on_batch_done(self, payload: dict) -> None:
        out_dir = payload["out_dir"]
        count = payload["count"]
        self._set_busy(
            False, f"Import complet: {count} rapoarte + raport general în {out_dir}"
        )
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Information)
        box.setWindowTitle("Import finalizat")
        box.setText(
            f"Au fost evaluați {count} elevi.\n\n"
            f"S-au generat {count} rapoarte individuale și un raport general "
            f"(raport_general.pdf) în:\n{out_dir}"
        )
        open_btn = box.addButton("Deschide folderul", QMessageBox.AcceptRole)
        box.addButton("Închide", QMessageBox.RejectRole)
        box.exec()
        if box.clickedButton() is open_btn:
            self._open_folder(out_dir)

    def _on_batch_error(self, message: str) -> None:
        self._set_busy(False, "Import eșuat.")
        QMessageBox.critical(self, "Import eșuat", message)

    @staticmethod
    def _open_folder(path: str) -> None:
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices
        QDesktopServices.openUrl(QUrl.fromLocalFile(path))

    def _on_retrain(self) -> None:
        if QMessageBox.question(
            self, "Reantrenare",
            "Reantrenezi modelul pe date sintetice? (câteva secunde)",
        ) != QMessageBox.Yes:
            return
        self._set_busy(True, "Se reantrenează modelul…")
        self._start_worker(self.service.retrain, self._on_retrained)

    def _on_retrained(self, model) -> None:
        self._set_busy(False, f"Model reantrenat: {model.version}")

    def _on_metrics(self) -> None:
        self._set_busy(True, "Se încarcă metricile modelului…")
        self._start_worker(lambda: self.service.model_metrics, self._show_metrics)

    def _show_metrics(self, metrics) -> None:
        self._set_busy(False, "")
        if not metrics:
            QMessageBox.information(self, "Metrici model", "Metrici indisponibile.")
            return
        msg = (
            f"Versiune: {self.service.model_version}\n"
            f"Acuratețe: {metrics.get('accuracy', float('nan')):.3f}\n"
            f"ROC AUC: {metrics.get('roc_auc', float('nan')):.3f}\n"
            f"Echilibrare înainte SMOTE-NC: {metrics.get('balance_before')}\n"
            f"Echilibrare după SMOTE-NC: {metrics.get('balance_after')}\n\n"
            f"{metrics.get('report_text', '')}"
        )
        box = QMessageBox(self)
        box.setWindowTitle("Metrici model")
        box.setText(msg)
        box.setFont(QFont("Consolas", 9))
        box.exec()

    def _on_about(self) -> None:
        QMessageBox.about(
            self, "Despre",
            f"{config.APP_TITLE}\nv{config.APP_VERSION}\n\n"
            "Asistent predictiv pentru identificarea timpurie („Ziua 14”) a "
            "elevilor cu risc de abandon școlar.\n\n"
            "Proof-of-concept: model XGBoost real + SMOTE-NC + explicații SHAP "
            "autentice, cu plan de intervenție generat local sau prin Claude.\n\n"
            "Datele elevilor rămân local; către cloud se trimite doar scorul "
            "anonimizat și explicația SHAP.\n\n"
            "© Ramona Richițeanu — concept și metodologie de cercetare.",
        )

    def _on_worker_error(self, message: str) -> None:
        self._set_busy(False, "Eroare.")
        QMessageBox.critical(self, "Eroare", message)
