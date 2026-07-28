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
from ..i18n import tr, trf
from ..timing import STAGE_LABELS
from . import theme

# This view renders document-style HTML, so its palette is fixed light and does
# not follow the app theme — see ``theme.document_qss``.
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


def _fmt_ms(seconds: float) -> str:
    """Sub-second stages read better in milliseconds."""
    return f"{seconds * 1000:.0f} ms" if seconds < 1.0 else f"{seconds:.2f} s"


def _stage_breakdown_html(run: metrics.RunMetrics) -> str:
    """Per-stage latency table: where the time in a report actually goes."""
    summary = run.stage_summary()
    if not summary:
        return ""

    rows = [
        '<tr bgcolor="{}"><td style="font-size:8pt; color:{};"><b>{}</b></td>'
        '<td style="font-size:8pt; color:{};" align="right"><b>{}</b></td>'
        '<td style="font-size:8pt; color:{};" align="right"><b>{}</b></td>'
        '<td style="font-size:8pt; color:{};" align="right"><b>N</b></td></tr>'.format(
            _PANEL, _PRIMARY, escape(tr("Etapă")),
            _PRIMARY, escape(tr("Medie")),
            _PRIMARY, escape(tr("Ab. std.")),
            _PRIMARY,
        )
    ]
    # Fixed pipeline order rather than dict order, so the table reads as the
    # sequence a report actually goes through.
    ordered = [s for s in STAGE_LABELS if s in summary]
    for i, stage in enumerate(ordered, start=1):
        mean, stdev, n = summary[stage]
        stripe = "#ffffff" if i % 2 else "#f9fafb"
        rows.append(
            f'<tr bgcolor="{stripe}">'
            f'<td style="font-size:8.5pt;">{escape(tr(STAGE_LABELS[stage]))}</td>'
            f'<td style="font-size:8.5pt;" align="right">{_fmt_ms(mean)}</td>'
            f'<td style="font-size:8.5pt;" align="right">'
            f'{_fmt_ms(stdev) if n > 1 else "—"}</td>'
            f'<td style="font-size:8.5pt;" align="right">{n}</td>'
            '</tr>'
        )
    # The LLM call, for scale: it is the reason the local stages look small.
    rows.append(
        f'<tr bgcolor="{_PANEL}">'
        f'<td style="font-size:8.5pt;"><b>{escape(tr("Apel LLM (plan de intervenție)"))}'
        f'</b></td>'
        f'<td style="font-size:8.5pt;" align="right"><b>'
        f'{metrics.fmt_seconds(run.avg_latency_s)}</b></td>'
        f'<td style="font-size:8.5pt;" align="right">'
        f'{metrics.fmt_seconds(run.stdev_latency_s) if run.report_count > 1 else "—"}</td>'
        f'<td style="font-size:8.5pt;" align="right">{run.report_count}</td>'
        '</tr>'
    )

    parts = [
        f'<p style="color:{_MUTED}; font-size:8.5pt; margin:10px 0 2px 0;">'
        f'<b>{escape(tr("Defalcarea latenței pe etape"))}</b> '
        f'{escape(tr("(medie per raport)"))}</p>',
        '<table width="100%" cellspacing="0" cellpadding="4" border="1" '
        f'style="border-color:{_BORDER};">' + "".join(rows) + '</table>',
    ]

    totals = (
        f'{escape(tr("Nucleu xAI local (NLP + predicție + SHAP + LIME)"))}: '
        f'<b>{_fmt_ms(run.avg_xai_s)}</b> · '
        f'{escape(tr("Total local / raport"))}: <b>{_fmt_ms(run.avg_local_s)}</b> · '
        f'{escape(tr("Total end-to-end / raport"))}: '
        f'<b>{metrics.fmt_seconds(run.avg_total_report_s)}</b>'
    )
    if run.model_load_s is not None:
        totals += (
            f' · {escape(tr("Încărcare model (o singură dată)"))}: '
            f'<b>{metrics.fmt_seconds(run.model_load_s)}</b>'
        )
    parts.append(
        f'<p style="font-size:8.5pt; color:{_MUTED}; margin:4px 0 0 0;">{totals}</p>'
    )
    return "".join(parts)


def _run_html(index: int, run: metrics.RunMetrics) -> str:
    type_label = tr(_RUN_TYPE_LABELS.get(run.run_type, run.run_type))
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
        trf("{mean} medie", mean=metrics.fmt_seconds(run.avg_latency_s))
        + (f" ± {run.stdev_latency_s:.2f}" if run.report_count > 1 else "")
    )

    parts.append('<table cellspacing="0" cellpadding="0" style="margin-top:6px;">')
    # Row 1 — activity
    parts.append('<tr>')
    parts.append(_cell(tr("Rapoarte generate"), str(run.report_count)))
    parts.append(_cell(
        tr("Apeluri cloud / local"),
        f"{run.cloud_calls} / {run.local_calls + run.fallback_calls}",
    ))
    parts.append(_cell(
        tr("Tokeni (intrare / ieșire)"),
        f"{run.total_input_tokens:,} / {run.total_billed_output_tokens:,}",
    ))
    parts.append('</tr>')
    # Row 2 — latency (LLM call; the local stages are broken out below)
    parts.append('<tr>')
    parts.append(_cell(tr("Latență totală"), metrics.fmt_seconds(run.total_latency_s)))
    parts.append(_cell(tr("Latență LLM / raport"), latency_detail))
    parts.append(_cell(tr("Latență / apel cloud"),
                       metrics.fmt_seconds(run.avg_cloud_latency_s)))
    parts.append(_cell(
        tr("Latență min / max"),
        f"{run.min_latency_s:.2f} / {run.max_latency_s:.2f} s",
    ))
    parts.append('</tr>')
    # Row 3 — cost
    parts.append('<tr>')
    parts.append(_cell(tr("Cost total"), cost_total))
    parts.append(_cell(tr("Cost / raport"), per_report))
    parts.append(_cell(tr("Cost / apel cloud"), per_cloud_call))
    parts.append('</tr></table>')

    if not run.pricing_available:
        parts.append(
            f'<p style="color:#b45309; font-size:8.5pt; margin:4px 0 0 0;">'
            + escape(trf(
                "Cost parțial: modelul nu are un preț în snapshot-ul {as_of}; "
                "completează MODEL_PRICING pentru cifre complete.",
                as_of=run.pricing_as_of,
            ))
            + '</p>'
        )

    parts.append(_stage_breakdown_html(run))

    # Per-report breakdown table for batch runs.
    if run.report_count > 1:
        rows = [
            '<tr bgcolor="{}"><td style="font-size:8pt; color:{};"><b>#</b></td>'
            '<td style="font-size:8pt; color:{};"><b>{}</b></td>'
            '<td style="font-size:8pt; color:{};" align="right"><b>{}</b></td>'
            '<td style="font-size:8pt; color:{};" align="right"><b>{}</b></td>'
            '<td style="font-size:8pt; color:{};" align="right"><b>{}</b></td>'
            '<td style="font-size:8pt; color:{};" align="right"><b>{}</b></td></tr>'.format(
                _PANEL, _PRIMARY,
                _PRIMARY, escape(tr("Sursă")),
                _PRIMARY, escape(tr("Latență")),
                _PRIMARY, escape(tr("Tok. intr.")),
                _PRIMARY, escape(tr("Tok. ieș.")),
                _PRIMARY, escape(tr("Cost")),
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
        f'<span style="color:{_PRIMARY}; font-size:15pt;"><b>'
        f'{escape(tr("Metrici de performanță pe rulare"))}</b></span>'
    )
    parts.append(
        f'<p style="color:{_MUTED}; font-size:9pt; margin:4px 0 0 0;">'
        + escape(trf(
            "Latența, tokenii (raportați de furnizor) și costul derivat pentru "
            "fiecare apel de generare a planului, plus defalcarea pe etape a "
            "pipeline-ului local (NLP, predicție, SHAP, LIME, randare PDF). "
            "Costul este calculat din tokeni cu tabelul de prețuri din {as_of} — "
            "verifică ratele pentru model.",
            as_of=metrics.PRICING_AS_OF,
        ))
        + '</p>'
    )

    if not runs:
        parts.append(
            '<p style="color:#888; margin-top:16px;"><i>'
            + escape(tr("Nu există rulări înregistrate încă. Generează un plan "
                        "sau importă un fișier pentru a colecta metrici."))
            + '</i></p></div>'
        )
        return "".join(parts)

    # Overall footer summary across all recorded runs.
    total_reports = sum(r.report_count for r in runs)
    total_cost = sum(r.total_cost for r in runs)
    all_priced = all(r.pricing_available for r in runs)
    parts.append(
        '<p style="font-size:9.5pt; margin:8px 0 0 0;">'
        + trf("<b>{runs}</b> rulări · <b>{reports}</b> rapoarte · "
              "cost cumulat <b>{cost}</b>",
              runs=len(runs), reports=total_reports,
              cost=metrics.fmt_cost(total_cost if all_priced else None))
        + '</p>'
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
        self.setWindowTitle(tr("Metrici de performanță"))
        self.resize(720, 640)

        layout = QVBoxLayout(self)
        self.view = QTextBrowser()
        self.view.setStyleSheet(theme.document_qss())
        layout.addWidget(self.view)

        buttons = QHBoxLayout()
        self.btn_export = QPushButton(tr("Exportă CSV…"))
        self.btn_export.clicked.connect(self._on_export)
        self.btn_clear = QPushButton(tr("Golește istoricul"))
        self.btn_clear.clicked.connect(self._on_clear)
        btn_close = QPushButton(tr("Închide"))
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
            self, tr("Exportă metricile ca CSV"), "metrici_performanta.csv",
            tr("Fișier CSV (*.csv)"),
        )
        if not path:
            return
        if not path.lower().endswith(".csv"):
            path += ".csv"
        try:
            rows = metrics.export_calls_csv(path, runs)
        except Exception as exc:
            QMessageBox.critical(
                self, tr("Export eșuat"), f"{type(exc).__name__}: {exc}"
            )
            return
        QMessageBox.information(
            self, tr("Export finalizat"),
            trf("S-au exportat {rows} rânduri în:\n{path}", rows=rows, path=path),
        )

    def _on_clear(self) -> None:
        if QMessageBox.question(
            self, tr("Golește istoricul"),
            tr("Ștergi toate metricile de performanță înregistrate? "
               "Acțiunea nu poate fi anulată."),
        ) != QMessageBox.Yes:
            return
        metrics.get_store().clear()
        self._refresh()
