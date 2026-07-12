"""Render an assessment result as Qt-rich-text HTML (score + SHAP + NLP).

Two consumers share the same building blocks:

* the on-screen result view (``render_result_html``) — sections 1 + 2 only;
* the exported PDF (``build_report_html`` + ``export_report_pdf``) — a polished
  three-page report:
      Page 1  Datele introduse        (identity + full questionnaire)
      Page 2  Evaluarea riscului      (risk profile + xAI explanation)
      Page 3  Planul de recomandare   (intervention plan + success indicators)

Everything is expressed with the HTML/CSS subset ``QTextDocument`` understands
(tables with fixed widths and background-coloured cells, inline font styling,
and ``page-break-before`` to split the pages) so no browser engine or plotting
library has to be bundled into the executable. The layout mirrors the research
figure "xAI Analysis and Personalized Intervention Plan".
"""

from __future__ import annotations

import re
from datetime import datetime
from html import escape

from .. import config
from ..models import RiskEvaluation, StudentCase
from ..service import AssessmentResult

# --- Design system ----------------------------------------------------------
_PRIMARY = "#1f3a5f"       # deep navy — accents, headings
_PANEL = "#f2f4f7"         # light panel / label / header-band background
_PANEL_ALT = "#f9fafb"     # zebra-stripe alternate row
_BORDER = "#d8dee8"        # hairline borders
_INK = "#2b2b2b"           # body text
_MUTED = "#2b2b2b"         # secondary text

_POS_COLOR = "#d64550"     # raises risk
_NEG_COLOR = "#2e8b57"     # lowers risk
_DOMAIN_COLOR = "#3b7dd8"  # sub-score bars
_TRACK_COLOR = "#e9edf2"   # empty part of a gauge/bar track
_MAX_BAR_PX = 170

_REPORT_TITLE = "Analiză xAI și plan personalizat de intervenție"
_REPORT_SUBTITLE = "Raport de evaluare a riscului de abandon școlar"


def _bar_cell(width_px: int, color: str, cell_px: int = _MAX_BAR_PX) -> str:
    width_px = max(2, int(width_px))
    return (
        f'<td width="{cell_px + 4}">'
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


# --- Shared chrome: page header band + spacer -------------------------------
def _page_header_html(section_label: str, *, cover: bool = False) -> str:
    """A title band shown at the top of every report page.

    A light panel with black text and a navy left-accent rule (no white text).
    The cover variant (page 1) is taller and shows the full report title; inner
    pages show a slim running band with the section name on the right.
    """
    if cover:
        inner = (
            '<td valign="middle">'
            f'<span style="color:{_INK}; font-size:16pt;"><b>{escape(_REPORT_TITLE)}</b></span><br>'
            f'<span style="color:{_MUTED}; font-size:9.5pt;">{escape(_REPORT_SUBTITLE)}</span>'
            "</td>"
            '<td width="120" align="right" valign="middle">'
            f'<span style="color:{_MUTED}; font-size:8pt;">CONFIDENȚIAL</span>'
            "</td>"
        )
        pad = 12
    else:
        inner = (
            '<td valign="middle">'
            f'<span style="color:{_INK}; font-size:10.5pt;"><b>{escape(_REPORT_TITLE)}</b></span>'
            "</td>"
            '<td align="right" valign="middle">'
            f'<span style="color:{_MUTED}; font-size:9pt;">{escape(section_label)}</span>'
            "</td>"
        )
        pad = 7
    return (
        '<table width="100%" cellpadding="0" cellspacing="0"><tr>'
        f'<td width="6" bgcolor="{_PRIMARY}"></td>'
        f'<td><table width="100%" cellpadding="{pad}" cellspacing="0" bgcolor="{_PANEL}">'
        f"<tr>{inner}</tr></table></td>"
        "</tr></table>"
    )


def _spacer(height: int = 14) -> str:
    return f'<table width="100%"><tr><td height="{height}"></td></tr></table>'


def _section_title(text: str) -> str:
    """A consistent section heading with a coloured left accent rule."""
    return (
        '<table width="100%" cellspacing="0" cellpadding="0"><tr>'
        f'<td width="5" bgcolor="{_PRIMARY}"></td>'
        '<td width="8"></td>'
        f'<td><span style="color:{_PRIMARY}; font-size:13pt;"><b>{escape(text)}</b></span></td>'
        "</tr></table>"
    )


def _meta_strip_html(case: StudentCase, ev: RiskEvaluation, generated: str) -> str:
    """A light info strip (elev / clasă / școală / generat / model) under the cover."""
    grade = case.student_grade or str(case.features.get("student_class", "") or "—")
    school = case.school_name or str(case.features.get("school_name", "") or "—")
    environment = "Urban" if case.is_urban else "Rural"
    cells = [
        ("Elev", case.display_name()),
        ("Clasa", grade),
        ("Școala", school),
        ("Mediu", environment),
        ("Generat", generated),
        ("Model", ev.model_version or "—"),
    ]
    row_label, row_value = [], []
    for label, value in cells:
        row_label.append(
            f'<td bgcolor="{_PANEL}" style="font-size:7.5pt; color:{_MUTED};">'
            f"{escape(label.upper())}</td>"
        )
        row_value.append(
            f'<td style="font-size:9.5pt; color:{_INK};"><b>{escape(str(value))}</b></td>'
        )
    return (
        '<table width="100%" cellpadding="6" cellspacing="0" border="1" '
        f'style="border-color:{_BORDER};"><tr>'
        + "".join(row_label)
        + "</tr><tr>"
        + "".join(row_value)
        + "</tr></table>"
    )


# --- Section 1: Student risk profile ---------------------------------------
def _risk_profile_html(ev: RiskEvaluation) -> str:
    tier = config.tier_for_band(ev.risk_band)
    parts: list[str] = []
    parts.append(_section_title("Profilul de risc al elevului"))
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


def _shap_html(ev: RiskEvaluation, compact: bool = False) -> str:
    bar_px = 78 if compact else _MAX_BAR_PX
    label_w = 118 if compact else 210
    val_w = 46 if compact else 70
    fs = "8pt" if compact else "9.5pt"
    parts: list[str] = []
    parts.append("<h3>Contribuția factorilor (SHAP)</h3>")
    if compact:
        parts.append(
            '<p style="color:#555; font-size:8pt;">Contribuția în puncte '
            "procentuale (roșu = crește riscul, verde = reduce riscul).</p>"
        )
    else:
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
        width = abs(a.points) / max_pts * bar_px
        sign = "+" if a.shap_value >= 0 else ""
        parts.append(
            "<tr>"
            f'<td width="{label_w}" style="font-size:{fs};">{escape(a.label)}<br>'
            f'<span style="color:#888; font-size:7.5pt;">valoare: {escape(a.value_display)}</span></td>'
            + _bar_cell(width, color, bar_px)
            + f'<td width="{val_w}" align="right" style="font-size:{fs};">'
            f'<b><span style="color:{color};">{sign}{a.points:.1f} pp</span></b></td>'
            "</tr>"
        )
    parts.append("</table>")
    return "".join(parts)


def _subscores_html(ev: RiskEvaluation, compact: bool = False) -> str:
    bar_px = 78 if compact else _MAX_BAR_PX
    label_w = 118 if compact else 210
    val_w = 46 if compact else 70
    fs = "8pt" if compact else "9.5pt"
    parts: list[str] = []
    parts.append("<h3>Sub-scoruri pe domenii</h3>")
    parts.append(
        f'<p style="color:#555; font-size:{("8pt" if compact else "9pt")};">'
        "Ponderea fiecărui domeniu în explicația totală (%).</p>"
    )
    parts.append('<table width="100%" cellpadding="3">')
    for s in ev.sub_scores:
        width = s.value / 100.0 * bar_px
        parts.append(
            "<tr>"
            f'<td width="{label_w}" style="font-size:{fs};">{escape(s.name)}</td>'
            + _bar_cell(width, _DOMAIN_COLOR, bar_px)
            + f'<td width="{val_w}" align="right" style="font-size:{fs};"><b>{s.value:.0f}%</b></td>'
            "</tr>"
        )
    parts.append("</table>")
    return "".join(parts)


def _two_column_html(left: str, right: str) -> str:
    """Place two blocks side by side (used to fit SHAP + sub-scores on one row)."""
    return (
        '<table width="100%" cellspacing="0" cellpadding="0"><tr>'
        f'<td width="53%" valign="top">{left}</td>'
        '<td width="3%"></td>'
        f'<td width="44%" valign="top">{right}</td>'
        "</tr></table>"
    )


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


def _xai_html(result: AssessmentResult, two_column: bool = False) -> str:
    ev = result.evaluation
    parts: list[str] = []
    parts.append(_section_title("Explicație xAI (de ce acest nivel de risc)"))
    parts.append(
        '<p style="color:#555; font-size:9pt;">Factorii care justifică nivelul '
        "de risc, în ordinea importanței pentru model:</p>"
    )
    parts.append('<ul>')
    for bullet in _why_bullets(ev):
        parts.append(f'<li style="font-size:10pt;">{bullet}</li>')
    parts.append("</ul>")
    if two_column:
        # SHAP + sub-scores side by side so page 2 stays on a single sheet.
        parts.append(
            _two_column_html(_shap_html(ev, compact=True),
                             _subscores_html(ev, compact=True))
        )
    else:
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


# --- Page 1: Datele introduse ----------------------------------------------
def _personal_data_html(case: StudentCase) -> str:
    """A zebra-striped label→value table of the questionnaire answers."""
    parts: list[str] = []
    parts.append(
        '<table width="100%" cellpadding="6" cellspacing="0" border="1" '
        f'style="border-color:{_BORDER};">'
    )
    row = 0
    for item in config.QUESTIONNAIRE_FIELDS:
        if item.key == "timestamp":
            continue
        value = case.features.get(item.key, "")
        if value in (None, ""):
            continue
        stripe = _PANEL_ALT if row % 2 else "#ffffff"
        parts.append(
            f'<tr bgcolor="{stripe}">'
            f'<td width="44%" bgcolor="{_PANEL}" style="font-size:9.5pt; color:{_INK};">'
            f"<b>{escape(item.label)}</b></td>"
            f'<td style="font-size:9.5pt; color:{_INK};">{escape(str(value))}</td>'
            "</tr>"
        )
        row += 1
    if row == 0:
        parts.append(
            '<tr><td style="font-size:9.5pt; color:#888;">'
            "<i>Nu au fost introduse date de chestionar.</i></td></tr>"
        )
    parts.append("</table>")
    return "".join(parts)


# --- Section 3: Personalized intervention plan -----------------------------
def _markdown_to_html(md_text: str) -> str:
    """Convert the plan's markdown into an HTML fragment for the report.

    Uses Qt's CommonMark parser (``QTextDocument.setMarkdown``) so the model's
    headings, bold/italic and bullet/numbered lists render properly rather than
    showing raw ``**`` / ``-`` markers. Returns the inner ``<body>`` HTML, which
    inherits the surrounding report font. Falls back to escaped text with line
    breaks if Qt is unavailable, so the report can still be built without a GUI.
    """
    text = (md_text or "").strip()
    if not text:
        return ""
    # Defensive: a bare ``<br>`` inside a Markdown table cell makes Qt's parser
    # truncate the row and drop every block after the table (blanking the plan).
    # Self-closing ``<br/>`` parses cleanly, so normalise all variants to it.
    text = re.sub(r"(?i)<br\s*/?>", "<br/>", text)
    try:
        from PySide6.QtGui import QTextDocument
        doc = QTextDocument()
        doc.setMarkdown(text)  # GitHub dialect is Qt's default
        html = doc.toHtml()
    except Exception:
        return escape(text).replace("\n", "<br>")
    # Qt always wraps the content in <body …>…</body>; keep only the inner HTML
    # so it inherits the report's font instead of Qt's default "Sans Serif".
    lower = html.lower()
    open_tag = lower.find("<body")
    start = html.find(">", open_tag) + 1 if open_tag != -1 else 0
    end = lower.rfind("</body>")
    if end == -1:
        end = len(html)
    return html[start:end].strip() or escape(text).replace("\n", "<br>")


def _plan_html(plan_text: str, source: str) -> str:
    parts: list[str] = []
    parts.append(_section_title("Plan personalizat de intervenție"))
    parts.append(
        '<p style="color:#555; font-size:9pt;">Contract de implicare al elevului: '
        "„Proiectul Podul”.</p>"
    )
    if plan_text and plan_text.strip():
        if source:
            parts.append(
                f'<p style="color:#555; font-size:9pt;">Sursă: {escape(source)}</p>'
            )
        body = _markdown_to_html(plan_text)
        parts.append(
            '<table width="100%" cellpadding="10" cellspacing="0" bgcolor="#fbf5ee" '
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
    parts.append(_section_title("Indicatori de succes (4 săptămâni)"))
    parts.append(
        '<table width="100%" cellpadding="10" cellspacing="0" bgcolor="#eef7f0" '
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


def _page_break() -> str:
    return '<div style="page-break-before:always;"></div>'


def build_report_html(case: StudentCase, result: AssessmentResult, plan_text: str) -> str:
    """Assemble the full three-page report as HTML.

    Each page is a block that starts with the running header band; pages 2 and 3
    are preceded by ``page-break-before`` so ``QTextDocument`` paginates them onto
    their own sheets.
    """
    ev = result.evaluation
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")

    parts: list[str] = ['<div style="font-family: Segoe UI, Arial, sans-serif;">']

    # --- Page 1 — Datele introduse -----------------------------------------
    parts.append(_page_header_html("", cover=True))
    parts.append(_spacer(10))
    parts.append(_meta_strip_html(case, ev, generated))
    parts.append(_spacer(16))
    parts.append(_section_title("Datele introduse"))
    parts.append(
        '<p style="color:#555; font-size:9pt;">Date de identificare și răspunsurile '
        "din chestionar, așa cum au fost introduse pentru această evaluare.</p>"
    )
    parts.append(_personal_data_html(case))

    # --- Page 2 — Evaluarea riscului ---------------------------------------
    parts.append(_page_break())
    parts.append(_page_header_html("Evaluarea riscului"))
    parts.append(_spacer(12))
    parts.append(_spacer(4))
    parts.append(_risk_profile_html(ev))
    parts.append(_spacer(6))
    parts.append(_xai_html(result, two_column=True))

    # --- Page 3 — Planul de recomandare ------------------------------------
    parts.append(_page_break())
    parts.append(_page_header_html("Planul de recomandare"))
    parts.append(_spacer(12))
    parts.append(_spacer(4))
    parts.append(_plan_html(plan_text, ev.action_plan_source))
    parts.append(_spacer(14))
    parts.append(_success_indicators_html(ev))

    parts.append("</div>")
    return "".join(parts)


# --- General / summary report (class-level intervention prioritization) -----
_SUMMARY_TITLE = "Raport general — prioritizarea intervențiilor"
_SUMMARY_SUBTITLE = "Evaluarea riscului de abandon școlar la nivel de grup"

# Severity rank of each band (higher = more urgent); drives the priority order.
_TIER_SEVERITY: dict[str, int] = {t.band: i for i, t in enumerate(config.RISK_TIERS)}


def _summary_header_html() -> str:
    """Cover band for the group report (mirrors the per-student cover)."""
    inner = (
        '<td valign="middle">'
        f'<span style="color:{_INK}; font-size:16pt;"><b>{escape(_SUMMARY_TITLE)}</b></span><br>'
        f'<span style="color:{_MUTED}; font-size:9.5pt;">{escape(_SUMMARY_SUBTITLE)}</span>'
        "</td>"
        '<td width="120" align="right" valign="middle">'
        f'<span style="color:{_MUTED}; font-size:8pt;">CONFIDENȚIAL</span>'
        "</td>"
    )
    return (
        '<table width="100%" cellpadding="0" cellspacing="0"><tr>'
        f'<td width="6" bgcolor="{_PRIMARY}"></td>'
        f'<td><table width="100%" cellpadding="12" cellspacing="0" bgcolor="{_PANEL}">'
        f"<tr>{inner}</tr></table></td>"
        "</tr></table>"
    )


def _summary_meta_strip_html(
    school_name: str, count: int, generated: str, model_version: str
) -> str:
    cells = [
        ("Școala", school_name or "—"),
        ("Elevi evaluați", str(count)),
        ("Generat", generated),
        ("Model", model_version or "—"),
    ]
    row_label = "".join(
        f'<td bgcolor="{_PANEL}" style="font-size:7.5pt; color:{_MUTED};">'
        f"{escape(label.upper())}</td>"
        for label, _ in cells
    )
    row_value = "".join(
        f'<td style="font-size:9.5pt; color:{_INK};"><b>{escape(str(value))}</b></td>'
        for _, value in cells
    )
    return (
        '<table width="100%" cellpadding="6" cellspacing="0" border="1" '
        f'style="border-color:{_BORDER};"><tr>{row_label}</tr>'
        f"<tr>{row_value}</tr></table>"
    )


def _risk_distribution_html(entries: list[tuple[StudentCase, RiskEvaluation]]) -> str:
    """Count per risk tier with a colour-coded bar (Critic first)."""
    total = len(entries) or 1
    counts: dict[str, int] = {}
    for _, ev in entries:
        counts[ev.risk_band] = counts.get(ev.risk_band, 0) + 1

    parts: list[str] = [_section_title("Distribuția pe niveluri de risc")]
    parts.append('<table width="100%" cellpadding="3" cellspacing="0">')
    for tier in reversed(config.RISK_TIERS):  # Critic -> Moderat
        n = counts.get(tier.band, 0)
        width = int(round(n / total * _MAX_BAR_PX))
        pct = n / total * 100.0
        parts.append(
            "<tr>"
            f'<td width="150" style="font-size:9.5pt;"><b>{escape(tier.band)}</b><br>'
            f'<span style="color:#888; font-size:7.5pt;">{escape(tier.urgency)}</span></td>'
            + _bar_cell(width, tier.color)
            + f'<td width="90" style="font-size:9.5pt;">'
            f'<b>{n}</b> elevi <span style="color:#888;">({pct:.0f}%)</span></td>'
            "</tr>"
        )
    parts.append("</table>")
    return "".join(parts)


def _priority_table_html(entries: list[tuple[StudentCase, RiskEvaluation]]) -> str:
    """Ranked table the teacher uses to decide who to help first."""
    ordered = sorted(
        entries,
        key=lambda item: (
            -_TIER_SEVERITY.get(item[1].risk_band, 0),
            -item[1].aggregate_score,
            item[1].studentship_score,
            item[0].display_name(),
        ),
    )

    head = (
        f'<tr bgcolor="#ffffff">'
        + "".join(
            f'<td style="color:#ffffff; font-size:8pt;"><b>{escape(h)}</b></td>'
            for h in (
                "#", "Elev", "Clasa", "Nivel de risc", "Scor",
                "Urgență", "Implicare", "Factori principali",
            )
        )
        + "</tr>"
    )

    rows: list[str] = []
    for rank, (case, ev) in enumerate(ordered, start=1):
        tier = config.tier_for_band(ev.risk_band)
        stripe = _PANEL_ALT if rank % 2 == 0 else "#ffffff"
        indicators = "; ".join(ev.critical_indicators[:2]) or "—"
        rows.append(
            f'<tr bgcolor="{stripe}">'
            f'<td align="center" style="font-size:9pt;"><b>{rank}</b></td>'
            f'<td style="font-size:9pt;"><b>{escape(case.display_name())}</b></td>'
            f'<td style="font-size:8.5pt;">{escape(case.student_grade or "—")}</td>'
            f'<td bgcolor="{tier.color}" align="center" style="font-size:8.5pt; color:{tier.text_color};">'
            f"<b>{escape(ev.risk_band)}</b></td>"
            f'<td align="right" style="font-size:9pt;"><b>{ev.aggregate_score:.0f}%</b></td>'
            f'<td style="font-size:8.5pt;">{escape(ev.urgency)}</td>'
            f'<td align="center" style="font-size:8.5pt;">{ev.studentship_score:g}/10</td>'
            f'<td style="font-size:8pt;">{escape(indicators)}</td>'
            "</tr>"
        )

    return (
        _section_title("Prioritizarea intervențiilor")
        + '<p style="color:#555; font-size:9pt;">Elevii sunt ordonați după urgența '
        "intervenției: mai întâi nivelul de risc (Critic → Moderat), apoi scorul "
        "modelului; la risc egal, o implicare (Studentship) mai scăzută urcă în "
        "prioritate.</p>"
        '<table width="100%" cellpadding="6" cellspacing="0" border="1" '
        f'style="border-color:{_BORDER};">{head}{"".join(rows)}</table>'
    )


def build_summary_report_html(
    entries: list[tuple[StudentCase, RiskEvaluation]],
    *,
    school_name: str = "",
    model_version: str = "",
    generated: str | None = None,
) -> str:
    """Assemble the group report: distribution + a ranked prioritization table.

    ``entries`` is a list of ``(case, evaluation)`` pairs (one per imported
    student). The report is the teacher's decision aid — a single view over the
    whole class from which to triage who needs help first.
    """
    generated = generated or datetime.now().strftime("%Y-%m-%d %H:%M")
    if not school_name:
        school_name = next(
            (c.school_name for c, _ in entries if c.school_name), ""
        )

    parts: list[str] = ['<div style="font-family: Segoe UI, Arial, sans-serif;">']
    parts.append(_summary_header_html())
    parts.append(_spacer(10))
    parts.append(_summary_meta_strip_html(school_name, len(entries), generated, model_version))
    parts.append(_spacer(16))
    parts.append(_risk_distribution_html(entries))
    parts.append(_spacer(16))
    parts.append(_priority_table_html(entries))
    parts.append(_spacer(12))
    parts.append(
        '<p style="color:#888; font-size:8pt;">Scorurile și explicațiile provin '
        "dintr-un model XGBoost real cu atribuiri SHAP autentice. Acest raport "
        "sintetizează evaluările individuale; pentru fiecare elev există un raport "
        "detaliat separat.</p>"
    )
    parts.append("</div>")
    return "".join(parts)


def export_report_pdf(path: str, html: str) -> None:
    """Render ``html`` into an A4 PDF at ``path`` with page-numbered footers.

    The document is drawn page-by-page through a ``QPainter`` (rather than
    ``QTextDocument.print_``) so a confidentiality note and a "Pagina X din Y"
    footer can be painted on every sheet. ``page-break-before`` in the HTML still
    controls where the report's three logical pages fall.
    """
    from PySide6.QtCore import QMarginsF, QRectF, QSizeF, Qt
    from PySide6.QtGui import (
        QColor, QFont, QPageLayout, QPageSize, QPainter, QPdfWriter, QTextDocument,
    )

    writer = QPdfWriter(str(path))
    writer.setPageSize(QPageSize(QPageSize.A4))
    writer.setPageMargins(QMarginsF(14, 14, 14, 16), QPageLayout.Millimeter)
    writer.setResolution(150)

    res = writer.resolution()
    page_px = writer.pageLayout().paintRectPixels(res)
    footer_h = int(res * 0.38)          # ~0.38 in reserved for the footer band
    body_w = float(page_px.width())
    body_h = float(page_px.height() - footer_h)

    doc = QTextDocument()
    doc.setPageSize(QSizeF(body_w, body_h))
    # Force dark text on the (light) paper regardless of the OS/app colour
    # scheme: a QTextDocument otherwise inherits the application palette, so on
    # Windows dark mode any element without an explicit colour would render
    # white. Inline colours (SHAP red/green, tier badge, muted greys) still win.
    doc.setDefaultStyleSheet(
        "body, div, p, span, td, th, li, ul, ol, h1, h2, h3, b, i, strong, em "
        f"{{ color: {_INK}; }}"
    )
    doc.setHtml(html)

    painter = QPainter(writer)
    try:
        page_count = max(1, doc.pageCount())
        footer_font = QFont("Segoe UI", 7)
        generated = datetime.now().strftime("%Y-%m-%d %H:%M")
        for i in range(page_count):
            if i > 0:
                writer.newPage()

            # Body: draw this page's horizontal slice of the document.
            painter.save()
            painter.translate(0, -i * body_h)
            painter.setClipRect(QRectF(0, i * body_h, body_w, body_h))
            doc.drawContents(painter)
            painter.restore()

            # Footer: separator rule + confidentiality note + page number.
            painter.save()
            line_y = int(body_h + footer_h * 0.30)
            painter.setPen(QColor(_BORDER))
            painter.drawLine(0, line_y, int(body_w), line_y)
            painter.setFont(footer_font)
            painter.setPen(QColor(_MUTED))
            footer_rect = QRectF(0, body_h + footer_h * 0.34, body_w, footer_h * 0.6)
            painter.drawText(
                footer_rect,
                int(Qt.AlignLeft | Qt.AlignVCenter),
                f"Confidențial — uz educațional • Generat {generated}",
            )
            painter.drawText(
                footer_rect,
                int(Qt.AlignRight | Qt.AlignVCenter),
                f"Pagina {i + 1} din {page_count}",
            )
            painter.restore()
    finally:
        painter.end()
