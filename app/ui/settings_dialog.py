"""Settings dialog — manage the optional knowledge-base (.docx) document."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QDialog, QFileDialog, QGroupBox, QHBoxLayout, QLabel, QMessageBox,
    QPlainTextEdit, QPushButton, QVBoxLayout,
)

from .. import settings
from ..docx_reader import DocxError


class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Setări — Bază de cunoștințe")
        self.resize(680, 520)
        self._build_ui()
        self._refresh()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        intro = QLabel(
            "Atașează un document Word (.docx) — de regulă articolul de "
            "cercetare / metodologia — ca „bază de cunoștințe”. Textul lui este "
            "folosit pentru a ghida generarea planului de intervenție în cloud "
            "(Claude), aliniind planul la metodologia ta."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        privacy = QLabel(
            "ℹ️ Documentul este material metodologic (nu date despre elevi). "
            "Datele elevului rămân întotdeauna locale și anonimizate."
        )
        privacy.setWordWrap(True)
        privacy.setStyleSheet("color:#555; font-size:9pt;")
        layout.addWidget(privacy)

        status_box = QGroupBox("Document curent")
        status_layout = QVBoxLayout(status_box)
        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        status_layout.addWidget(self.status_label)
        layout.addWidget(status_box)

        preview_box = QGroupBox("Previzualizare text extras")
        preview_layout = QVBoxLayout(preview_box)
        self.preview = QPlainTextEdit()
        self.preview.setReadOnly(True)
        preview_layout.addWidget(self.preview)
        layout.addWidget(preview_box, stretch=1)

        btn_row = QHBoxLayout()
        self.btn_choose = QPushButton("Alege document .docx…")
        self.btn_choose.clicked.connect(self._on_choose)
        self.btn_remove = QPushButton("Elimină")
        self.btn_remove.clicked.connect(self._on_remove)
        btn_close = QPushButton("Închide")
        btn_close.clicked.connect(self.accept)
        btn_row.addWidget(self.btn_choose)
        btn_row.addWidget(self.btn_remove)
        btn_row.addStretch(1)
        btn_row.addWidget(btn_close)
        layout.addLayout(btn_row)

    def _refresh(self) -> None:
        kb = settings.get_knowledge()
        if kb:
            self.status_label.setText(
                f"<b>{kb.get('filename', '?')}</b><br>"
                f"Caractere extrase: {kb.get('char_count', 0):,}<br>"
                f"Adăugat: {kb.get('added_at', '')}"
            )
            self.preview.setPlainText(kb.get("text", "")[:4000])
            self.btn_remove.setEnabled(True)
        else:
            self.status_label.setText("<i>Niciun document încărcat.</i>")
            self.preview.setPlainText("")
            self.btn_remove.setEnabled(False)

    def _on_choose(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Alege documentul de cunoștințe", "",
            "Documente Word (*.docx)",
        )
        if not path:
            return
        try:
            entry = settings.set_knowledge_from_docx(path)
        except DocxError as exc:
            QMessageBox.warning(self, "Document invalid", str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Eroare", f"{type(exc).__name__}: {exc}")
            return
        QMessageBox.information(
            self, "Salvat",
            f"Bază de cunoștințe actualizată: {entry['filename']} "
            f"({entry['char_count']:,} caractere).",
        )
        self._refresh()

    def _on_remove(self) -> None:
        settings.clear_knowledge()
        self._refresh()
