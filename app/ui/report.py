"""Render an assessment result as Qt-rich-text HTML (score + SHAP + NLP).

Uses only the HTML subset supported by QTextDocument (tables with fixed-width,
background-coloured cells) so no plotting library is needed at runtime. The
layout mirrors the research figure "xAI Analysis and Personalized Intervention
Plan": four sections — risk profile, xAI explanation, intervention plan and
success indicators — with the 4-tier risk level, its urgency and the
Studentship (engagement) score surfaced up front.
"""

from __future__ import annotations

from datetime import datetime
from html import escape

from .. import config
from ..models import RiskEvaluation, StudentCase
from ..service import AssessmentResult

_POS_COLOR = "#d64550"     # raises risk
_NEG_COLOR = "#2e8b57"     # lowers risk
_DOMAIN_COLOR = "#3b7dd8"  # sub-score bars
_TRACK_COLOR = "#e9edf2"   # empty part of a gauge/bar track
_MAX_BAR_PX = 170


def _bar_cell(width_px: int, color: str) -> str:
    width_px = max(2, int(width_px))
    return (
        f'<td width="{_MAX_BAR_PX + 4}">'
        f'<table cellspacing="0" cellpadding="0"><tr>'
        f'<td width="{width_px}" bgcolor="{color}">&nbsp;</td>'
        f"</tr></table></td>"
    )


def _studentship_color(score: float) -> str:
    """Green/amber/red by engagement level (low engagement = red)."""
    if score <= 3.0:
        return _POS_COLOR
    if score <= 6.0:
        return "#e0a30b"
    return _NEG_COLOR


def _studentship_gauge_html(score: float) -> str:
    """A horizontal gauge for the Studentship (engagement) score, out of 10."""
    score = max(0.0, min(10.0, float(score)))
    color = _studentship_color(score)
    fill = int(round(score / 10.0 * _MAX_BAR_PX))
    return (
        '<table cellspacing="0" cellpadding="2"><tr>'
        '<td width="150" style="font-size:10pt;"><b>Scor Studentship (implicare)</b></td>'
        f'<td width="{_MAX_BAR_PX + 4}">'
        f'<table cellspacing="0" cellpadding="0" bgcolor="{_TRACK_COLOR}"><tr>'
        f'<td width="{max(2, fill)}" bgcolor="{color}">&nbsp;</td>'
        f'<td width="{max(1, _MAX_BAR_PX - fill)}">&nbsp;</td>'
        "</tr></table></td>"
        f'<td width="60" align="right"><b>{score:g}/10</b></td>'
        "</tr></table>"
    )


# --- Section 1: Student risk profile ---------------------------------------
def _risk_profile_html(ev: RiskEvaluation) -> str:
    tier = config.tier_for_band(ev.risk_band)
    parts: list[str] = []
    parts.append('<h2 style="font-size:13pt;">Secțiunea 1 — Profilul de risc al elevului</h2>')
    parts.append('<table width="100%" cellpadding="8" cellspacing="0"><tr>')

    # Left: the risk-level badge, colour-coded by tier.
    parts.append(
        f'<td width="190" bgcolor="{tier.color}" align="center" valign="middle">'
        f'<span style="color:{tier.text_color}; font-size:8pt;">NIVEL DE RISC</span><br>'
        f'<span style="color:{tier.text_color}; font-size:22pt;"><b>{escape(ev.risk_band.upper())}</b></span><br>'
        f'<span style="color:{tier.text_color}; font-size:11pt;">Scor model: {ev.aggregate_score:.0f}%</span>'
        "</td>"
    )

    # Right: urgency, engagement gauge and the main risk indicators.
    parts.append('<td valign="top">')
    parts.append(
        '<p style="font-size:10pt; margin:0 0 4px 0;"><b>Urgență intervenție:</b> '
        f'<span style="color:{tier.color};">●</span> {escape(ev.urgency)}</p>'
    )
    parts.append(_studentship_gauge_html(ev.studentship_score))
    parts.append('<p style="font-size:10pt; margin:6px 0 2px 0;"><b>Indicatori principali de risc:</b></p>')
    indicators = ev.critical_indicators[:5]
    if indicators:
        parts.append('<ul style="margin-top:2px;">')
        for ind in indicators:
            parts.append(f'<li style="font-size:9.5pt;">{escape(ind)}</li>')
        parts.append("</ul>")
    else:
        parts.append(
            '<p style="color:#2e8b57; font-size:9.5pt;">Niciun factor major de risc identificat.</p>'
        )
    parts.append("</td></tr></table>")
    return "".join(parts)


# --- Section 2: xAI explanation --------------------------------------------
def _why_bullets(ev: RiskEvaluation) -> list[str]:
    """The top risk-raising drivers, phrased as short justifications (HTML)."""
    positive = [a for a in ev.top_drivers(len(ev.attributions)) if a.shap_value > 0][:5]
    if not positive:
        return [
            "Profilul elevului este preponderent protectiv; niciun factor nu "
            "crește semnificativ riscul."
        ]
    return [
        f"<b>{escape(a.label)}</b> — crește riscul cu <b>+{a.points:.1f} pp</b> "
        f"(valoare: {escape(a.value_display)})"
        for a in positive
    ]


def _shap_html(ev: RiskEvaluation) -> str:
    parts: list[str] = []
    parts.append("<h3>Contribuția factorilor (SHAP)</h3>")
    parts.append(
        '<p style="color:#555; font-size:9pt;">Contribuția fiecărui factor la '
        "scorul final, în puncte procentuale (roșu = crește riscul, verde = "
        "reduce riscul). Suma contribuțiilor + valoarea de bază "
        f"({ev.base_value:.3f}) = probabilitatea modelului.</p>"
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
    return "".join(parts)


def _subscores_html(ev: RiskEvaluation) -> str:
    parts: list[str] = []
    parts.append("<h3>Sub-scoruri pe domenii</h3>")
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
    return "".join(parts)


def _nlp_html(result: AssessmentResult) -> str:
    nlp = result.nlp
    neg = ", ".join(escape(t) for t in nlp.negative_terms) or "—"
    pos = ", ".join(escape(t) for t in nlp.positive_terms) or "—"
    return (
        "<h3>Analiză text (NLP)</h3>"
        f"<p><b>{escape(nlp.label)}</b><br>"
        f"Scor stres emoțional (feature model): <b>{nlp.stress_score:.2f}</b> / 2.0<br>"
        f"Valență: {nlp.valence:+.2f}<br>"
        f'<span style="color:{_POS_COLOR};">Termeni negativi:</span> {neg}<br>'
        f'<span style="color:{_NEG_COLOR};">Termeni pozitivi:</span> {pos}</p>'
    )


def _xai_html(result: AssessmentResult) -> str:
    ev = result.evaluation
    parts: list[str] = []
    parts.append(
        '<h2 style="font-size:13pt;">Secțiunea 2 — Explicație xAI '
        "(de ce acest nivel de risc)</h2>"
    )
    parts.append(
        '<p style="color:#555; font-size:9pt;">Factorii care justifică nivelul '
        "de risc, în ordinea importanței pentru model:</p>"
    )
    parts.append('<ul>')
    for bullet in _why_bullets(ev):
        parts.append(f'<li style="font-size:10pt;">{bullet}</li>')
    parts.append("</ul>")
    parts.append(_shap_html(ev))
    parts.append(_subscores_html(ev))
    parts.append(_nlp_html(result))
    parts.append(
        f'<p style="color:#888; font-size:8pt;">Probabilitate model: '
        f"{ev.probability:.3f} • valoare de referință (bază SHAP): "
        f"{ev.base_value:.3f} • model: {escape(ev.model_version)}</p>"
    )
    return "".join(parts)


def render_result_html(result: AssessmentResult) -> str:
    """Sections 1 + 2 — the on-screen result view and the report's core."""
    parts: list[str] = ['<div style="font-family: Segoe UI, Arial, sans-serif;">']
    parts.append(_risk_profile_html(result.evaluation))
    parts.append(_xai_html(result))
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


# --- Section 3: Personalized intervention plan -----------------------------
def _plan_html(plan_text: str, source: str) -> str:
    parts: list[str] = []
    parts.append(
        '<h2 style="font-size:13pt;">Secțiunea 3 — Plan personalizat de intervenție</h2>'
    )
    parts.append(
        '<p style="color:#555; font-size:9pt;">Contract de implicare al elevului: '
        "„Proiectul Podul”.</p>"
    )
    if plan_text and plan_text.strip():
        if source:
            parts.append(
                f'<p style="color:#555; font-size:9pt;">Sursă: {escape(source)}</p>'
            )
        body = escape(plan_text.strip()).replace("\n", "<br>")
        parts.append(
            '<table width="100%" cellpadding="8" cellspacing="0" bgcolor="#fbf5ee" '
            'border="1" style="border-color:#e6c9a8;"><tr><td>'
            '<div style="font-family: Segoe UI, Arial, sans-serif; font-size:10pt;">'
            f"{body}</div></td></tr></table>"
        )
    else:
        parts.append(
            '<p style="color:#888;"><i>Planul de intervenție nu a fost generat.</i></p>'
        )
    return "".join(parts)


# --- Section 4: Success indicators -----------------------------------------
def _success_indicators_html(ev: RiskEvaluation) -> str:
    current = max(0.0, min(10.0, ev.studentship_score))
    target = min(10.0, round(current + 3.0))
    parts: list[str] = []
    parts.append(
        '<h2 style="font-size:13pt;">Secțiunea 4 — Indicatori de succes (4 săptămâni)</h2>'
    )
    parts.append(
        '<table width="100%" cellpadding="8" cellspacing="0" bgcolor="#eef7f0" '
        'border="1" style="border-color:#bfe0c8;"><tr><td>'
        '<ul style="margin:0; font-size:10pt;">'
        "<li>Absențe: sub 2 absențe nemotivate pe săptămână.</li>"
        f"<li>Implicare: creșterea scorului Studentship de la <b>{current:g}/10</b> "
        f"la <b>{target:g}/10</b>.</li>"
        "<li>Participare: cel puțin un moment / o activitate școlară activă pe săptămână.</li>"
        "<li>Atitudine: trecere spre „neutru / pozitiv” față de școală.</li>"
        "<li>Reevaluarea scorului de risc la finalul celor 4 săptămâni.</li>"
        "</ul></td></tr></table>"
    )
    return "".join(parts)


def build_report_html(case: StudentCase, result: AssessmentResult, plan_text: str) -> str:
    """Assemble the full report (identity + 4 report sections) as HTML."""
    ev = result.evaluation
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")

    parts: list[str] = []
    parts.append('<div style="font-family: Segoe UI, Arial, sans-serif;">')
    parts.append(
        '<h1 style="font-size:16pt;">Analiză xAI și plan personalizat de intervenție</h1>'
    )
    parts.append(
        f'<p style="color:#555; font-size:9pt;">Elev: {escape(case.display_name())} • '
        f"Generat: {escape(generated)} • Model: {escape(ev.model_version)}</p>"
    )
    parts.append("<hr>")

    parts.append('<h2 style="font-size:13pt;">Date de identificare și chestionar</h2>')
    parts.append(_personal_data_html(case))

    parts.append(_risk_profile_html(ev))
    parts.append(_xai_html(result))
    parts.append(_plan_html(plan_text, ev.action_plan_source))
    parts.append(_success_indicators_html(ev))

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
