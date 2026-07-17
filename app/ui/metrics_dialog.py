"""Performance-metrics viewer: latency / tokens / cost, grouped per run.

Reads the shared :class:`app.metrics.MetricsStore` and renders every recorded
run — a single-student generation or an imported-workbook batch — with its
report count, latencies, total cost and cost per report. Also exports the raw
per-report table to CSV (for computing distributions in a paper) and can clear
the accumulated history.
"""

from __future__ import annotations

from html import escape

from PySide6.QtWidgets import (
    QDialog, QFileDialog, QHBoxLayout, QMessageBox, QPushButton, QTextBrowser,
    QVBoxLayout,
)

from .. import metrics

_PRIMARY = "#1f3a5f"
_PANEL = "#f2f4f7"
_BORDER = "#d8dee8"
_MUTED = "#555555"

_RUN_TYPE_LABELS = {
    "single": "Rulare individuală (un elev)",
    "import": "Import fișier (rapoarte multiple)",
}


def _cell(label: str, value: str) -> str:
    return (
        f'<td style="padding:4px 10px 4px 0;">'
        f'<span style="color:{_MUTED}; font-size:8pt;">{escape(label.upper())}</span><br>'
        f'<span style="font-size:11pt;"><b>{value}</b></span></td>'
    )


def _run_html(index: int, run: metrics.RunMetrics) -> str:
    type_label = _RUN_TYPE_LABELS.get(run.run_type, run.run_type)
    header_bits = [escape(run.started_at)]
    if run.label:
        header_bits.append(escape(run.label))
    provider_model = escape(f"{run.provider or '—'} / {run.model or '—'}")

    parts: list[str] = []
    parts.append(
        f'<table width="100%" cellspacing="0" cellpadding="0" '
        f'style="margin-top:14px;"><tr>'
        f'<td width="5" bgcolor="{_PRIMARY}"></td><td width="8"></td>'
        f'<td><span style="color:{_PRIMARY}; font-size:12pt;"><b>#{index} · {escape(type_label)}</b></span><br>'
        f'<span style="color:{_MUTED}; font-size:8.5pt;">{" · ".join(header_bits)} · '
        f'model: {provider_model} · prețuri: {escape(run.pricing_as_of)}</span></td>'
        f'</tr></table>'
    )

    # Cost may be partial if a model has no price in the snapshot.
    priced = run.pricing_available
    cost_total = metrics.fmt_cost(run.total_cost if priced else None)
    per_report = metrics.fmt_cost(run.cost_per_report if priced else None)
    per_cloud_call = metrics.fmt_cost(run.cost_per_cloud_call if priced else None)
    latency_detail = (
        f"{metrics.fmt_seconds(run.avg_latency_s)} medie"
        + (f" ± {run.stdev_latency_s:.2f}" if run.report_count > 1 else "")
    )

    parts.append('<table cellspacing="0" cellpadding="0" style="margin-top:6px;">')
    # Row 1 — activity
    parts.append('<tr>')
    parts.append(_cell("Rapoarte generate", str(run.report_count)))
    parts.append(_cell(
        "Apeluri cloud / local",
        f"{run.cloud_calls} / {run.local_calls + run.fallback_calls}",
    ))
    parts.append(_cell(
        "Tokeni (intrare / ieșire)",
        f"{run.total_input_tokens:,} / {run.total_billed_output_tokens:,}",
    ))
    parts.append('</tr>')
    # Row 2 — latency
    parts.append('<tr>')
    parts.append(_cell("Latență totală", metrics.fmt_seconds(run.total_latency_s)))
    parts.append(_cell("Latență / raport", latency_detail))
    parts.append(_cell("Latență / apel cloud", metrics.fmt_seconds(run.avg_cloud_latency_s)))
    parts.append(_cell(
        "Latență min / max",
        f"{run.min_latency_s:.2f} / {run.max_latency_s:.2f} s",
    ))
    parts.append('</tr>')
    # Row 3 — cost
    parts.append('<tr>')
    parts.append(_cell("Cost total", cost_total))
    parts.append(_cell("Cost / raport", per_report))
    parts.append(_cell("Cost / apel cloud", per_cloud_call))
    parts.append('</tr></table>')

    if not run.pricing_available:
        parts.append(
            f'<p style="color:#b45309; font-size:8.5pt; margin:4px 0 0 0;">'
            f'Cost parțial: modelul nu are un preț în snapshot-ul '
            f'{escape(run.pricing_as_of)}; completează MODEL_PRICING pentru cifre '
            f'complete.</p>'
        )

    # Per-report breakdown table for batch runs.
    if run.report_count > 1:
        rows = [
            '<tr bgcolor="{}"><td style="font-size:8pt; color:{};"><b>#</b></td>'
            '<td style="font-size:8pt; color:{};"><b>Sursă</b></td>'
            '<td style="font-size:8pt; color:{};" align="right"><b>Latență</b></td>'
            '<td style="font-size:8pt; color:{};" align="right"><b>Tok. intr.</b></td>'
            '<td style="font-size:8pt; color:{};" align="right"><b>Tok. ieș.</b></td>'
            '<td style="font-size:8pt; color:{};" align="right"><b>Cost</b></td></tr>'.format(
                _PANEL, *([_PRIMARY] * 6)
            )
        ]
        for i, call in enumerate(run.calls, start=1):
            stripe = "#ffffff" if i % 2 else "#f9fafb"
            rows.append(
                f'<tr bgcolor="{stripe}">'
                f'<td style="font-size:8.5pt;">{i}</td>'
                f'<td style="font-size:8.5pt;">{escape(call.source)}</td>'
                f'<td style="font-size:8.5pt;" align="right">{call.latency_s:.2f} s</td>'
                f'<td style="font-size:8.5pt;" align="right">{call.input_tokens:,}</td>'
                f'<td style="font-size:8.5pt;" align="right">{call.billed_output_tokens:,}</td>'
                f'<td style="font-size:8.5pt;" align="right">{metrics.fmt_cost(call.cost)}</td>'
                '</tr>'
            )
        parts.append(
            '<table width="100%" cellspacing="0" cellpadding="4" border="1" '
            f'style="border-color:{_BORDER}; margin-top:6px;">' + "".join(rows) + '</table>'
        )

    return "".join(parts)


def build_runs_html(runs: list[metrics.RunMetrics]) -> str:
    parts: list[str] = ['<div style="font-family: Segoe UI, Arial, sans-serif; color:#2b2b2b;">']
    parts.append(
        f'<span style="color:{_PRIMARY}; font-size:15pt;"><b>Metrici de performanță pe rulare</b></span>'
    )
    parts.append(
        f'<p style="color:{_MUTED}; font-size:9pt; margin:4px 0 0 0;">'
        'Latența, tokenii (raportați de furnizor) și costul derivat pentru fiecare '
        'apel de generare a planului. Costul este calculat din tokeni cu tabelul '
        f'de prețuri din {escape(metrics.PRICING_AS_OF)} — verifică ratele pentru '
        'modelul și nivelul exact înainte de a le cita într-o lucrare.</p>'
    )

    if not runs:
        parts.append(
            '<p style="color:#888; margin-top:16px;"><i>Nu există rulări '
            'înregistrate încă. Generează un plan sau importă un fișier pentru a '
            'colecta metrici.</i></p></div>'
        )
        return "".join(parts)

    # Overall footer summary across all recorded runs.
    total_reports = sum(r.report_count for r in runs)
    total_cost = sum(r.total_cost for r in runs)
    all_priced = all(r.pricing_available for r in runs)
    parts.append(
        f'<p style="font-size:9.5pt; margin:8px 0 0 0;">'
        f'<b>{len(runs)}</b> rulări · <b>{total_reports}</b> rapoarte · cost cumulat '
        f'<b>{metrics.fmt_cost(total_cost if all_priced else None)}</b></p>'
    )

    for index, run in enumerate(reversed(runs), start=1):
        # Newest first, but keep a stable descending index.
        parts.append(_run_html(len(runs) - index + 1, run))

    parts.append("</div>")
    return "".join(parts)


class MetricsDialog(QDialog):
    """Modal window listing per-run performance metrics with export/clear."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Metrici de performanță")
        self.resize(720, 640)

        layout = QVBoxLayout(self)
        self.view = QTextBrowser()
        layout.addWidget(self.view)

        buttons = QHBoxLayout()
        self.btn_export = QPushButton("Exportă CSV…")
        self.btn_export.clicked.connect(self._on_export)
        self.btn_clear = QPushButton("Golește istoricul")
        self.btn_clear.clicked.connect(self._on_clear)
        btn_close = QPushButton("Închide")
        btn_close.clicked.connect(self.accept)
        buttons.addWidget(self.btn_export)
        buttons.addWidget(self.btn_clear)
        buttons.addStretch(1)
        buttons.addWidget(btn_close)
        layout.addLayout(buttons)

        self._refresh()

    def _refresh(self) -> None:
        runs = metrics.get_store().runs()
        self.view.setHtml(build_runs_html(runs))
        has_runs = bool(runs)
        self.btn_export.setEnabled(has_runs)
        self.btn_clear.setEnabled(has_runs)

    def _on_export(self) -> None:
        runs = metrics.get_store().runs()
        if not runs:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Exportă metricile ca CSV", "metrici_performanta.csv",
            "Fișier CSV (*.csv)",
        )
        if not path:
            return
        if not path.lower().endswith(".csv"):
            path += ".csv"
        try:
            rows = metrics.export_calls_csv(path, runs)
        except Exception as exc:
            QMessageBox.critical(self, "Export eșuat", f"{type(exc).__name__}: {exc}")
            return
        QMessageBox.information(
            self, "Export finalizat", f"S-au exportat {rows} rânduri în:\n{path}"
        )

    def _on_clear(self) -> None:
        if QMessageBox.question(
            self, "Golește istoricul",
            "Ștergi toate metricile de performanță înregistrate? "
            "Acțiunea nu poate fi anulată.",
        ) != QMessageBox.Yes:
            return
        metrics.get_store().clear()
        self._refresh()
