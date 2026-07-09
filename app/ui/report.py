"""Render an assessment result as Qt-rich-text HTML (score + SHAP + NLP).

Uses only the HTML subset supported by QTextDocument (tables with fixed-width,
background-coloured cells) so no plotting library is needed at runtime.
"""

from __future__ import annotations

from datetime import datetime
from html import escape

from ..models import StudentCase
from ..service import AssessmentResult

_BAND_COLORS = {
    "Scăzut": "#2e8b57",   # green
    "Mediu": "#e08e0b",    # amber
    "Ridicat": "#d64550",  # red
}
_POS_COLOR = "#d64550"     # raises risk
_NEG_COLOR = "#2e8b57"     # lowers risk
_DOMAIN_COLOR = "#3b7dd8"  # sub-score bars
_MAX_BAR_PX = 170


def _band_color(band: str) -> str:
    return _BAND_COLORS.get(band, "#666666")


def _bar_cell(width_px: int, color: str) -> str:
    width_px = max(2, int(width_px))
    return (
        f'<td width="{_MAX_BAR_PX + 4}">'
        f'<table cellspacing="0" cellpadding="0"><tr>'
        f'<td width="{width_px}" bgcolor="{color}">&nbsp;</td>'
        f"</tr></table></td>"
    )


def render_result_html(result: AssessmentResult) -> str:
    ev = result.evaluation
    nlp = result.nlp
    band_color = _band_color(ev.risk_band)

    parts: list[str] = []
    parts.append('<div style="font-family: Segoe UI, Arial, sans-serif;">')

    # --- Headline gauge ---
    parts.append(
        f'<table width="100%" cellpadding="8"><tr>'
        f'<td bgcolor="{band_color}" width="150" align="center">'
        f'<span style="color:#ffffff; font-size:30pt;"><b>{ev.aggregate_score:.0f}%</b></span><br>'
        f'<span style="color:#ffffff; font-size:11pt;">Risc {escape(ev.risk_band)}</span>'
        f"</td>"
        f'<td valign="middle">'
        f'<span style="font-size:11pt;">Probabilitate de abandon estimată de model: '
        f"<b>{ev.probability:.3f}</b><br>"
        f"Valoare de referință (bază SHAP): {ev.base_value:.3f}<br>"
        f'<span style="color:#888;">Model: {escape(ev.model_version)}</span></span>'
        f"</td></tr></table>"
    )

    # --- SHAP explanation ---
    parts.append('<h3>Explicație xAI (SHAP)</h3>')
    parts.append(
        '<p style="color:#555; font-size:9pt;">Contribuția fiecărui factor la '
        "scorul final, în puncte procentuale (roșu = crește riscul, verde = "
        "reduce riscul). Suma contribuțiilor + valoarea de bază = scorul model.</p>"
    )
    drivers = ev.top_drivers(len(ev.attributions))
    max_pts = max((abs(a.points) for a in drivers), default=1.0) or 1.0
    parts.append('<table width="100%" cellpadding="3">')
    for a in drivers:
        color = _POS_COLOR if a.shap_value >= 0 else _NEG_COLOR
        width = abs(a.points) / max_pts * _MAX_BAR_PX
        sign = "+" if a.shap_value >= 0 else ""
        parts.append(
            "<tr>"
            f'<td width="210">{escape(a.label)}<br>'
            f'<span style="color:#888; font-size:8pt;">valoare: {escape(a.value_display)}</span></td>'
            + _bar_cell(width, color)
            + f'<td width="70" align="right"><b><span style="color:{color};">{sign}{a.points:.1f} pp</span></b></td>'
            "</tr>"
        )
    parts.append("</table>")

    # --- Sub-scores (domain shares) ---
    parts.append('<h3>Sub-scoruri pe domenii</h3>')
    parts.append(
        '<p style="color:#555; font-size:9pt;">Ponderea fiecărui domeniu în '
        "explicația totală (%).</p>"
    )
    parts.append('<table width="100%" cellpadding="3">')
    for s in ev.sub_scores:
        width = s.value / 100.0 * _MAX_BAR_PX
        parts.append(
            "<tr>"
            f'<td width="210">{escape(s.name)}</td>'
            + _bar_cell(width, _DOMAIN_COLOR)
            + f'<td width="70" align="right"><b>{s.value:.0f}%</b></td>'
            "</tr>"
        )
    parts.append("</table>")

    # --- NLP analysis ---
    parts.append('<h3>Analiză text (NLP)</h3>')
    neg = ", ".join(escape(t) for t in nlp.negative_terms) or "—"
    pos = ", ".join(escape(t) for t in nlp.positive_terms) or "—"
    parts.append(
        f'<p><b>{escape(nlp.label)}</b><br>'
        f"Scor stres emoțional (feature model): <b>{nlp.stress_score:.2f}</b> / 2.0<br>"
        f"Valență: {nlp.valence:+.2f}<br>"
        f'<span style="color:{_POS_COLOR};">Termeni negativi:</span> {neg}<br>'
        f'<span style="color:{_NEG_COLOR};">Termeni pozitivi:</span> {pos}</p>'
    )

    parts.append("</div>")
    return "".join(parts)


_PLACEHOLDER = (
    '<div style="font-family: Segoe UI, Arial; color:#888; padding:20px;">'
    "<h3>Niciun rezultat încă</h3>"
    "<p>Completează datele elevului în stânga și apasă "
    "<b>„Evaluează riscul”</b>.</p>"
    "<p>Scorul este calculat local de un model XGBoost real, iar explicația "
    "provine din valori SHAP autentice — nu dintr-o simulare a unui LLM.</p>"
    "</div>"
)


def placeholder_html() -> str:
    return _PLACEHOLDER


# --- Full PDF report --------------------------------------------------------
def _personal_data_html(case: StudentCase) -> str:
    """A label→value table of the identifying details + questionnaire answers."""
    from .. import config

    parts: list[str] = []
    parts.append(
        '<table width="100%" cellpadding="5" cellspacing="0" border="1" '
        'style="border-color:#cccccc;">'
    )
    for item in config.QUESTIONNAIRE_FIELDS:
        if item.key == "timestamp":
            continue
        value = case.features.get(item.key, "")
        if value in (None, ""):
            continue
        parts.append(
            "<tr>"
            f'<td width="42%" bgcolor="#f2f4f7"><b>{escape(item.label)}</b></td>'
            f"<td>{escape(str(value))}</td>"
            "</tr>"
        )
    parts.append("</table>")
    return "".join(parts)


def build_report_html(case: StudentCase, result: AssessmentResult, plan_text: str) -> str:
    """Assemble the full report (personal data + risk assessment + plan) as HTML."""
    ev = result.evaluation
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")

    parts: list[str] = []
    parts.append('<div style="font-family: Segoe UI, Arial, sans-serif;">')
    parts.append(
        '<h1 style="font-size:16pt;">Raport de evaluare a riscului de abandon școlar</h1>'
    )
    parts.append(
        f'<p style="color:#555; font-size:9pt;">Generat: {escape(generated)} • '
        f"Model: {escape(ev.model_version)}</p>"
    )
    parts.append('<hr>')

    parts.append('<h2 style="font-size:13pt;">1. Date personale și chestionar</h2>')
    parts.append(_personal_data_html(case))

    parts.append('<h2 style="font-size:13pt;">2. Evaluarea riscului</h2>')
    parts.append(render_result_html(result))

    parts.append('<h2 style="font-size:13pt;">3. Plan de intervenție</h2>')
    if plan_text and plan_text.strip():
        source = ev.action_plan_source
        if source:
            parts.append(
                f'<p style="color:#555; font-size:9pt;">Sursă: {escape(source)}</p>'
            )
        body = escape(plan_text.strip()).replace("\n", "<br>")
        parts.append(
            '<div style="font-family: Segoe UI, Arial, sans-serif; font-size:10pt; '
            f'white-space:pre-wrap;">{body}</div>'
        )
    else:
        parts.append(
            '<p style="color:#888;"><i>Planul de intervenție nu a fost generat.</i></p>'
        )

    parts.append("</div>")
    return "".join(parts)


def export_report_pdf(path: str, html: str) -> None:
    """Render ``html`` into an A4 PDF at ``path`` using Qt's built-in PDF writer."""
    from PySide6.QtCore import QMarginsF
    from PySide6.QtGui import QPageLayout, QPageSize, QPdfWriter, QTextDocument

    writer = QPdfWriter(str(path))
    writer.setPageSize(QPageSize(QPageSize.A4))
    writer.setPageMargins(QMarginsF(15, 15, 15, 15), QPageLayout.Millimeter)

    doc = QTextDocument()
    doc.setHtml(html)
    doc.print_(writer)
