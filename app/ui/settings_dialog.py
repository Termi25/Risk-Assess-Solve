"""Settings dialog — cloud provider + API key, and the knowledge-base (.docx)."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QFileDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPlainTextEdit, QPushButton,
    QVBoxLayout,
)

from .. import config, keystore, settings
from ..docx_reader import DocxError


class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Setări — Cloud & Bază de cunoștințe")
        self.resize(680, 640)
        self._build_ui()
        self._refresh()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        layout.addWidget(self._build_api_key_group())

        intro = QLabel(
            "Atașează un document Word (.docx) — de regulă articolul de "
            "cercetare / metodologia — ca „bază de cunoștințe”. Textul lui este "
            "folosit pentru a ghida generarea planului de intervenție în cloud, "
            "aliniind planul la metodologia ta."
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

    # --- Cloud provider & API key ------------------------------------------
    def _build_api_key_group(self) -> QGroupBox:
        box = QGroupBox("Conexiune cloud — furnizor și cheie API")
        v = QVBoxLayout(box)

        info = QLabel(
            "Opțional. Alege un furnizor și introdu cheia lui API pentru a genera "
            "planul de intervenție în cloud. Fără cheie, aplicația rămâne complet "
            "funcțională și folosește generatorul local. Cheile sunt păstrate "
            "securizat în seiful de credențiale al sistemului de operare "
            "(Windows Credential Manager) — niciodată în fișiere text."
        )
        info.setWordWrap(True)
        v.addWidget(info)

        form = QFormLayout()
        self.provider_combo = QComboBox()
        for p in config.LLM_PROVIDERS:
            self.provider_combo.addItem(p.label, p.id)
        idx = self.provider_combo.findData(settings.get_active_provider())
        self.provider_combo.setCurrentIndex(max(0, idx))
        self.provider_combo.currentIndexChanged.connect(self._on_provider_changed)
        form.addRow("Furnizor:", self.provider_combo)
        v.addLayout(form)

        self.key_status = QLabel()
        self.key_status.setWordWrap(True)
        v.addWidget(self.key_status)

        row = QHBoxLayout()
        self.key_input = QLineEdit()
        self.key_input.setEchoMode(QLineEdit.Password)
        self.key_input.returnPressed.connect(self._on_save_key)
        self.chk_show = QCheckBox("Afișează")
        self.chk_show.toggled.connect(self._on_toggle_key_visibility)
        row.addWidget(self.key_input, stretch=1)
        row.addWidget(self.chk_show)
        v.addLayout(row)

        btns = QHBoxLayout()
        self.btn_save_key = QPushButton("Salvează cheia")
        self.btn_save_key.clicked.connect(self._on_save_key)
        self.btn_clear_key = QPushButton("Șterge cheia")
        self.btn_clear_key.clicked.connect(self._on_clear_key)
        btns.addWidget(self.btn_save_key)
        btns.addWidget(self.btn_clear_key)
        btns.addStretch(1)
        v.addLayout(btns)

        return box

    def _current_provider(self) -> config.LLMProvider:
        return config.get_provider(self.provider_combo.currentData())

    def _refresh_key_status(self) -> None:
        provider = self._current_provider()
        self.key_input.setPlaceholderText(provider.key_prefix)

        if not keystore.keyring_available():
            self.key_status.setText(
                "⚠️ Stocarea securizată nu este disponibilă pe acest sistem. "
                f"Poți folosi în schimb variabila de mediu {provider.env_var}."
            )
            self.key_status.setStyleSheet("color:#a15c00;")
            for w in (self.key_input, self.chk_show,
                      self.btn_save_key, self.btn_clear_key):
                w.setEnabled(False)
            return

        for w in (self.key_input, self.chk_show, self.btn_save_key):
            w.setEnabled(True)

        has_stored = bool(keystore.get_stored_key(provider.id))
        source = keystore.key_source(provider.id)
        if source == "env":
            msg = (f"Activ: cheia din variabila de mediu {provider.env_var} "
                   "(are prioritate față de cheia salvată).")
            if has_stored:
                msg += (" O cheie este și salvată în seif, dar este ignorată "
                        "cât timp variabila de mediu există.")
            color = "#0a7d00"
        elif source == "stored":
            msg = "Activ: o cheie este salvată securizat în seiful sistemului. ✓"
            color = "#0a7d00"
        else:
            msg = ("Nicio cheie pentru acest furnizor — se folosește "
                   "generatorul local (offline).")
            color = "#555"
        self.key_status.setText(msg)
        self.key_status.setStyleSheet(f"color:{color};")
        self.btn_clear_key.setEnabled(has_stored)

    def _on_provider_changed(self) -> None:
        settings.set_active_provider(self._current_provider().id)
        self.key_input.clear()
        self.chk_show.setChecked(False)
        self._refresh_key_status()

    def _on_toggle_key_visibility(self, shown: bool) -> None:
        self.key_input.setEchoMode(
            QLineEdit.Normal if shown else QLineEdit.Password
        )

    def _on_save_key(self) -> None:
        provider = self._current_provider()
        key = self.key_input.text().strip()
        if not key:
            QMessageBox.warning(
                self, "Cheie goală",
                "Introdu o cheie API înainte de salvare.",
            )
            return
        try:
            keystore.set_stored_key(provider.id, key)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(
                self, "Eroare", f"Cheia nu a putut fi salvată: {exc}"
            )
            return
        self.key_input.clear()
        self.chk_show.setChecked(False)
        QMessageBox.information(
            self, "Salvat",
            f"Cheia API pentru {provider.label} a fost salvată securizat.",
        )
        self._refresh_key_status()

    def _on_clear_key(self) -> None:
        provider = self._current_provider()
        if QMessageBox.question(
            self, "Confirmă",
            f"Ștergi cheia API salvată pentru {provider.label}?",
        ) != QMessageBox.Yes:
            return
        keystore.clear_stored_key(provider.id)
        self.key_input.clear()
        QMessageBox.information(
            self, "Șters", "Cheia API a fost ștearsă din seif."
        )
        self._refresh_key_status()

    # --- Knowledge base -----------------------------------------------------
    def _refresh(self) -> None:
        self._refresh_key_status()
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
