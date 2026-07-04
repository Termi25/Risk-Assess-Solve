"""Main application window (PySide6).

The window is intentionally thin: it collects input, hands work to
``AssessmentService`` on a background thread, and renders the returned result.
All heavy imports (xgboost / shap / anthropic) live behind the service.
"""

from __future__ import annotations

import os
import traceback

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QAction, QFont
from PySide6.QtWidgets import (
    QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
    QLineEdit, QMainWindow, QMessageBox, QPlainTextEdit, QPushButton,
    QScrollArea, QSpinBox, QSplitter, QTextBrowser, QVBoxLayout, QWidget,
)

from .. import config
from ..models import StudentCase
from ..service import AssessmentResult, AssessmentService
from .report import placeholder_html, render_result_html


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


_DEMO_CASE = {
    "Medie_Modul_Anterior": 4.3,
    "Note_Sub_7": 4,
    "Absente_Nemotivate_Zilele_1_13": 12,
    "Studentship_Score": 2,
    "Mediu_Rezidential": "Rural",
    "Parinti_In_Strainatate": "Da",
    "Vulnerabilitate_Financiara": "Ridicata",
}
_DEMO_TEXT = (
    "Elevul este retras și obosit în ultima perioadă, refuză să participe la "
    "activități și pare demotivat. Lipsește frecvent."
)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(config.APP_TITLE)
        self.resize(1180, 780)

        self.service = AssessmentService()
        self.current_case: StudentCase | None = None
        self.current_result: AssessmentResult | None = None
        self._workers: list[FnWorker] = []
        self._feature_widgets: dict[str, QWidget] = {}

        self._build_menu()
        self._build_ui()
        self._refresh_llm_status()

    # --- UI construction ---------------------------------------------------
    def _build_menu(self) -> None:
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

    def _build_ui(self) -> None:
        splitter = QSplitter(Qt.Horizontal)

        # Left: input form (scrollable).
        form_widget = QWidget()
        form_layout = QVBoxLayout(form_widget)

        identity = QGroupBox("Date elev (rămân doar local)")
        idl = QFormLayout(identity)
        self.in_surname = QLineEdit()
        self.in_name = QLineEdit()
        self.in_grade = QLineEdit()
        self.in_school = QLineEdit()
        idl.addRow("Nume", self.in_surname)
        idl.addRow("Prenume", self.in_name)
        idl.addRow("Clasa", self.in_grade)
        idl.addRow("Școala", self.in_school)
        form_layout.addWidget(identity)

        features_box = QGroupBox("Indicatori „Day 14”")
        fl = QFormLayout(features_box)
        for feat in config.FEATURES:
            if feat.key == "Stres_Emotional_NLP":
                continue  # derived from the observation text
            if feat.kind == "categorical":
                widget = QComboBox()
                widget.addItems(list(feat.categories))
            elif feat.step < 1:
                widget = QDoubleSpinBox()
                widget.setRange(feat.minimum, feat.maximum)
                widget.setSingleStep(feat.step)
                widget.setDecimals(1)
                widget.setValue(feat.default)
            else:
                widget = QSpinBox()
                widget.setRange(int(feat.minimum), int(feat.maximum))
                widget.setSingleStep(int(feat.step))
                widget.setValue(int(feat.default))
            widget.setToolTip(feat.help_text)
            self._feature_widgets[feat.key] = widget
            fl.addRow(feat.label, widget)
        form_layout.addWidget(features_box)

        obs_box = QGroupBox("Observații calitative (analizate NLP)")
        obs_layout = QVBoxLayout(obs_box)
        self.in_observation = QPlainTextEdit()
        self.in_observation.setPlaceholderText(
            "Ex: elevul este retras, obosit, refuză să participe…"
        )
        self.in_observation.setFixedHeight(90)
        self.lbl_stress = QLabel("Stres emoțional (NLP): — (se calculează la evaluare)")
        self.lbl_stress.setStyleSheet("color:#555;")
        obs_layout.addWidget(self.in_observation)
        obs_layout.addWidget(self.lbl_stress)
        form_layout.addWidget(obs_box)

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

        plan_box = QGroupBox("Plan de intervenție („Proiectul Podul”)")
        plan_layout = QVBoxLayout(plan_box)
        self.plan_source = QLabel("Sursă: —")
        self.plan_source.setStyleSheet("color:#555;")
        self.plan_text = QPlainTextEdit()
        self.plan_text.setReadOnly(True)
        self.plan_text.setFont(QFont("Consolas", 9))
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
        has_key = bool(os.environ.get(config.LLM_API_KEY_ENV, "").strip())
        mode = "cloud (Claude)" if has_key else "local (offline)"
        from .. import settings
        kb = settings.get_knowledge()
        kb_note = f" • bază de cunoștințe: {kb['filename']}" if kb else ""
        self.plan_source.setText(f"Sursă plan: se va folosi modul {mode}.{kb_note}")

    def _on_settings(self) -> None:
        from .settings_dialog import SettingsDialog
        SettingsDialog(self).exec()
        self._refresh_llm_status()

    def _fill_demo(self) -> None:
        self.in_surname.setText("Popescu")
        self.in_name.setText("Andrei")
        self.in_grade.setText("IX A")
        self.in_school.setText("Liceul Tehnologic")
        for key, val in _DEMO_CASE.items():
            w = self._feature_widgets[key]
            if isinstance(w, QComboBox):
                w.setCurrentText(str(val))
            elif isinstance(w, QDoubleSpinBox):
                w.setValue(float(val))
            elif isinstance(w, QSpinBox):
                w.setValue(int(val))
        self.in_observation.setPlainText(_DEMO_TEXT)

    def _collect_case(self) -> StudentCase:
        features: dict[str, object] = {}
        for key, w in self._feature_widgets.items():
            if isinstance(w, QComboBox):
                features[key] = w.currentText()
            elif isinstance(w, QDoubleSpinBox):
                features[key] = float(w.value())
            elif isinstance(w, QSpinBox):
                features[key] = int(w.value())
        is_urban = features.get("Mediu_Rezidential") == "Urban"
        return StudentCase(
            name=self.in_name.text().strip(),
            surname=self.in_surname.text().strip(),
            student_grade=self.in_grade.text().strip(),
            school_name=self.in_school.text().strip(),
            is_urban=is_urban,
            observation_text=self.in_observation.toPlainText().strip(),
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
        if message:
            self.statusBar().showMessage(message)
        self.setCursor(Qt.BusyCursor if busy else Qt.ArrowCursor)

    # --- actions -----------------------------------------------------------
    def _on_evaluate(self) -> None:
        self.current_case = self._collect_case()
        self._set_busy(True, "Se calculează scorul (model XGBoost + SHAP)…")
        self.btn_plan.setEnabled(False)
        self.btn_save.setEnabled(False)
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
        self._set_busy(False, f"Evaluare completă. Model: {result.evaluation.model_version}")

    def _on_generate_plan(self) -> None:
        if not self.current_result:
            return
        self._set_busy(True, "Se generează planul de intervenție…")
        self.plan_text.setPlainText("Se generează…")
        self._start_worker(
            self.service.generate_plan, self._on_plan_done,
            self.current_result.evaluation,
        )

    def _on_plan_done(self, payload) -> None:
        text, source = payload
        self.plan_text.setPlainText(text)
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
