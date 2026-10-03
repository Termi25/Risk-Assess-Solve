"""Render an assessment result as Qt-rich-text HTML (score + SHAP + LIME + NLP).

Two consumers share the same building blocks:

* the on-screen result view (``render_result_html``) — sections 1–3 only;
* the exported PDF (``build_report_html`` + ``export_report_pdf``) — a polished
  three-page report:
      Page 1  Datele introduse        (identity + full questionnaire)
      Page 2  Evaluarea riscului      (risk profile, SHAP, sub-scores, then the
                                       LIME rule list + its fidelity, then NLP)
      Page 3  Planul de recomandare   (intervention plan + success indicators)

  The LIME block is simply absent when ``lime`` is unavailable. Page 2 carries
  the most content and flows onto a continuation sheet when the rule list is
  long; only the explicit breaks above start a new headed page.

Everything is expressed with the HTML/CSS subset ``QTextDocument`` understands
(tables with fixed widths and background-coloured cells, inline font styling,
and ``page-break-before`` to split the pages) so no browser engine or plotting
library has to be bundled into the executable. The layout mirrors the research
figure "xAI Analysis and Personalized Intervention Plan".
"""

from __future__ import annotations

import re
import textwrap
import unicodedata
from datetime import datetime
from html import escape

from .. import config
from ..i18n import tr, tr_band, tr_value, trf
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
        f'<td width="150" style="font-size:10pt;">'
        f'<b>{escape(tr("Scor Studentship (implicare)"))}</b></td>'
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
            f'<span style="color:{_INK}; font-size:16pt;"><b>{escape(tr(_REPORT_TITLE))}</b></span><br>'
            f'<span style="color:{_MUTED}; font-size:9.5pt;">{escape(tr(_REPORT_SUBTITLE))}</span>'
            "</td>"
            '<td width="120" align="right" valign="middle">'
            f'<span style="color:{_MUTED}; font-size:8pt;">{escape(tr("CONFIDENȚIAL"))}</span>'
            "</td>"
        )
        pad = 12
    else:
        inner = (
            '<td valign="middle">'
            f'<span style="color:{_INK}; font-size:10.5pt;"><b>{escape(tr(_REPORT_TITLE))}</b></span>'
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
    environment = tr_value("Urban" if case.is_urban else "Rural")
    cells = [
        (tr("Elev"), case.display_name()),
        (tr("Clasa"), grade),
        (tr("Școala"), school),
        (tr("Mediu"), environment),
        (tr("Generat"), generated),
        (tr("Model"), ev.model_version or "—"),
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
    parts.append(_section_title(tr("Profilul de risc al elevului")))
    parts.append('<table width="100%" cellpadding="8" cellspacing="0"><tr>')

    # Left: the risk-level badge, colour-coded by tier.
    parts.append(
        f'<td width="190" bgcolor="{tier.color}" align="center" valign="middle">'
        f'<span style="color:{tier.text_color}; font-size:8pt;">'
        f'{escape(tr("NIVEL DE RISC"))}</span><br>'
        f'<span style="color:{tier.text_color}; font-size:22pt;">'
        f'<b>{escape(tr_band(ev.risk_band).upper())}</b></span><br>'
        f'<span style="color:{tier.text_color}; font-size:11pt;">'
        f'{escape(trf("Scor model: {score:.0f}%", score=ev.aggregate_score))}</span>'
        "</td>"
    )

    # Right: urgency, engagement gauge and the main risk indicators.
    parts.append('<td valign="top">')
    parts.append(
        f'<p style="font-size:10pt; margin:0 0 4px 0;">'
        f'<b>{escape(tr("Urgență intervenție:"))}</b> '
        f'<span style="color:{tier.color};">●</span> {escape(tr_band(ev.urgency))}</p>'
    )
    parts.append(_studentship_gauge_html(ev.studentship_score))
    parts.append(
        f'<p style="font-size:10pt; margin:6px 0 2px 0;">'
        f'<b>{escape(tr("Indicatori principali de risc:"))}</b></p>'
    )
    indicators = ev.critical_indicators[:5]
    if indicators:
        parts.append('<ul style="margin-top:2px;">')
        for ind in indicators:
            parts.append(f'<li style="font-size:9.5pt;">{escape(ind)}</li>')
        parts.append("</ul>")
    else:
        parts.append(
            f'<p style="color:#2e8b57; font-size:9.5pt;">'
            f'{escape(tr("Niciun factor major de risc identificat."))}</p>'
        )
    parts.append("</td></tr></table>")
    return "".join(parts)


# --- Section 2: xAI explanation --------------------------------------------
def _why_bullets(ev: RiskEvaluation) -> list[str]:
    """The top risk-raising drivers, phrased as short justifications (HTML)."""
    positive = [a for a in ev.top_drivers(len(ev.attributions)) if a.shap_value > 0][:5]
    if not positive:
        return [tr(
            "Profilul elevului este preponderent protectiv; niciun factor nu "
            "crește semnificativ riscul."
        )]
    return [
        trf("<b>{label}</b> — crește riscul cu <b>+{points:.1f} pp</b> "
            "(valoare: {value})",
            label=escape(tr(a.label)), points=a.points,
            value=escape(tr_value(a.value_display)))
        for a in positive
    ]


def _shap_html(ev: RiskEvaluation, compact: bool = False) -> str:
    bar_px = 78 if compact else _MAX_BAR_PX
    label_w = 118 if compact else 210
    val_w = 46 if compact else 70
    fs = "8pt" if compact else "9.5pt"
    parts: list[str] = []
    parts.append(f"<h3>{escape(tr('Contribuția factorilor (SHAP)'))}</h3>")
    if compact:
        parts.append(
            '<p style="color:#555; font-size:8pt;">'
            + escape(tr("Contribuția în puncte procentuale (roșu = crește "
                        "riscul, verde = reduce riscul)."))
            + "</p>"
        )
    else:
        parts.append(
            '<p style="color:#555; font-size:9pt;">'
            + escape(trf(
                "Contribuția fiecărui factor la scorul final, în puncte "
                "procentuale (roșu = crește riscul, verde = reduce riscul). "
                "Suma contribuțiilor + valoarea de bază ({base:.3f}) = "
                "probabilitatea modelului.",
                base=ev.base_value))
            + "</p>"
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
            f'<td width="{label_w}" style="font-size:{fs};">{escape(tr(a.label))}<br>'
            f'<span style="color:#888; font-size:7.5pt;">{escape(tr("valoare"))}: '
            f'{escape(tr_value(a.value_display))}</span></td>'
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
    parts.append(f"<h3>{escape(tr('Sub-scoruri pe domenii'))}</h3>")
    parts.append(
        f'<p style="color:#555; font-size:{("8pt" if compact else "9pt")};">'
        + escape(tr("Ponderea fiecărui domeniu în explicația totală (%)."))
        + "</p>"
    )
    parts.append('<table width="100%" cellpadding="3">')
    for s in ev.sub_scores:
        width = s.value / 100.0 * bar_px
        parts.append(
            "<tr>"
            f'<td width="{label_w}" style="font-size:{fs};">{escape(tr(s.name))}</td>'
            + _bar_cell(width, _DOMAIN_COLOR, bar_px)
            + f'<td width="{val_w}" align="right" style="font-size:{fs};"><b>{s.value:.0f}%</b></td>'
            "</tr>"
        )
    parts.append("</table>")
    return "".join(parts)


def _fidelity_color(r2: float) -> str:
    """Green/amber/red by how well the local surrogate fits this student."""
    if r2 >= 0.70:
        return _NEG_COLOR
    if r2 >= 0.40:
        return "#e67e22"
    return _POS_COLOR


def _lime_html(result: AssessmentResult) -> str:
    """Section 3 — the individual risk profile fitted by LIME.

    Deliberately phrased as rules rather than percentages: unlike the SHAP block,
    these weights are local surrogate coefficients and do not add up to the
    model's probability, so showing them as "pp of the score" would misread them.
    """
    lime = result.lime
    if lime is None or not lime.conditions:
        return ""

    ev = result.evaluation
    parts: list[str] = []
    parts.append(_section_title(tr("Profil individual de risc (LIME)")))
    parts.append(
        '<p style="color:#555; font-size:9pt;">'
        # Contains inline <i>/<b> markup, so it is deliberately not escaped.
        + tr("Regulile care descriu situația <i>acestui</i> elev, așa cum le-a "
             "identificat un model local aproximativ (LIME), antrenat în jurul "
             "cazului său. Spre deosebire de analiza SHAP, ponderile de mai jos "
             "<b>nu se adună</b> la scorul final — ele arată ce anume "
             "diferențiază local acest elev, nu din ce se compune procentul.")
        + "</p>"
    )

    conditions = lime.top_conditions(len(lime.conditions))
    max_w = max((abs(c.weight) for c in conditions), default=1.0) or 1.0
    parts.append('<table width="100%" cellpadding="3">')
    for c in conditions:
        color = _POS_COLOR if c.weight >= 0 else _NEG_COLOR
        width = abs(c.weight) / max_w * _MAX_BAR_PX
        sign = "+" if c.weight >= 0 else ""
        parts.append(
            "<tr>"
            f'<td width="250" style="font-size:9.5pt;">{escape(c.condition)}<br>'
            f'<span style="color:#888; font-size:7.5pt;">{escape(c.direction)}</span></td>'
            + _bar_cell(width, color, _MAX_BAR_PX)
            + f'<td width="70" align="right" style="font-size:9.5pt;">'
            f'<b><span style="color:{color};">{sign}{c.influence:.1f}</span></b></td>'
            "</tr>"
        )
    parts.append("</table>")

    # Fidelity panel — a weak local fit is shown, not hidden.
    # The panel colour goes on the ``<td>``, not the ``<table>``: QTextBrowser
    # (the on-screen view) ignores a table-level ``bgcolor`` and renders the
    # panel white, while the PDF writer honours it — so the two views disagreed.
    # Cell-level background is respected by both, as in ``_meta_strip_html``.
    fid_color = _fidelity_color(lime.fidelity_r2)
    parts.append(_spacer(8))
    fidelity = trf(
        "Fidelitatea explicației locale: {label}",
        label=f'<span style="color:{fid_color};">{escape(lime.fidelity_label)}</span>',
    )
    parts.append(
        f'<table width="100%" cellspacing="0" cellpadding="7">'
        f'<tr><td bgcolor="{_PANEL}">'
        f'<span style="font-size:9pt;"><b>{fidelity}</b> '
        + escape(trf("(R² = {r2:.2f} pe {samples} perturbări)",
                     r2=lime.fidelity_r2, samples=lime.num_samples))
        + "</span><br>"
        f'<span style="color:#555; font-size:8.5pt;">'
        + trf("Modelul local aproximează probabilitatea la <b>{local:.3f}</b>, "
              "față de <b>{actual:.3f}</b> cât indică modelul real (diferență: "
              "{gap:.3f}). Cu cât R² este mai mic și diferența mai mare, cu atât "
              "regulile de mai sus trebuie citite mai prudent — decizia rămâne "
              "a cadrului didactic.",
              local=lime.local_prediction, actual=ev.probability,
              gap=lime.local_gap)
        + "</span></td></tr></table>"
    )
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
        f"<h3>{escape(tr('Analiză text (NLP)'))}</h3>"
        f"<p><b>{escape(nlp.label)}</b><br>"
        + trf("Scor stres emoțional (feature model): <b>{score:.2f}</b> / 2.0",
              score=nlp.stress_score)
        + "<br>"
        + escape(trf("Valență: {valence:+.2f}", valence=nlp.valence))
        + "<br>"
        f'<span style="color:{_POS_COLOR};">{escape(tr("Termeni negativi:"))}</span> '
        f"{neg}<br>"
        f'<span style="color:{_NEG_COLOR};">{escape(tr("Termeni pozitivi:"))}</span> '
        f"{pos}</p>"
    )


def _xai_html(
    result: AssessmentResult, two_column: bool = False, lime_block: str = ""
) -> str:
    """The xAI section: why-bullets, SHAP, sub-scores, then NLP.

    ``lime_block`` (the report path) is inserted directly under the sub-scores,
    keeping the local rule list next to the domain weights it refines. The
    on-screen view leaves it empty and appends LIME after the whole section.
    """
    ev = result.evaluation
    parts: list[str] = []
    parts.append(_section_title(tr("Explicație xAI (de ce acest nivel de risc)")))
    parts.append(
        '<p style="color:#555; font-size:9pt;">'
        + escape(tr("Factorii care justifică nivelul de risc, în ordinea "
                    "importanței pentru model:"))
        + "</p>"
    )
    parts.append('<ul>')
    for bullet in _why_bullets(ev):
        parts.append(f'<li style="font-size:10pt;">{bullet}</li>')
    parts.append("</ul>")
    if two_column:
        # SHAP + sub-scores side by side to keep the page compact.
        parts.append(
            _two_column_html(_shap_html(ev, compact=True),
                             _subscores_html(ev, compact=True))
        )
    else:
        parts.append(_shap_html(ev))
        parts.append(_subscores_html(ev))
    if lime_block:
        parts.append(_spacer(10))
        parts.append(lime_block)
    parts.append(_nlp_html(result))
    parts.append(
        '<p style="color:#888; font-size:8pt;">'
        + escape(trf("Probabilitate model: {probability:.3f} • valoare de "
                     "referință (bază SHAP): {base:.3f} • model: {version}",
                     probability=ev.probability, base=ev.base_value,
                     version=ev.model_version))
        + "</p>"
    )
    return "".join(parts)


def render_result_html(result: AssessmentResult) -> str:
    """Sections 1–3 — the on-screen result view and the report's core."""
    parts: list[str] = ['<div style="font-family: Segoe UI, Arial, sans-serif;">']
    parts.append(_risk_profile_html(result.evaluation))
    parts.append(_xai_html(result))
    parts.append(_lime_html(result))
    parts.append("</div>")
    return "".join(parts)


def placeholder_html() -> str:
    """Built per call, not cached: the language can change between calls."""
    return (
        '<div style="font-family: Segoe UI, Arial; color:#888; padding:20px;">'
        f"<h3>{escape(tr('Niciun rezultat încă'))}</h3>"
        "<p>"
        + trf("Completează datele elevului în stânga și apasă "
              "<b>„Evaluează riscul”</b>.")
        + "</p><p>"
        + escape(tr("Scorul este calculat local de un model XGBoost real, iar "
                    "explicația provine din valori SHAP autentice — nu dintr-o "
                    "simulare a unui LLM."))
        + "</p></div>"
    )


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
            f"<b>{escape(tr(item.label))}</b></td>"
            f'<td style="font-size:9.5pt; color:{_INK};">'
            f"{escape(tr_value(str(value)))}</td>"
            "</tr>"
        )
        row += 1
    if row == 0:
        parts.append(
            '<tr><td style="font-size:9.5pt; color:#888;"><i>'
            + escape(tr("Nu au fost introduse date de chestionar."))
            + "</i></td></tr>"
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
    parts.append(_section_title(tr("Plan personalizat de intervenție")))
    parts.append(
        '<p style="color:#555; font-size:9pt;">'
        + escape(tr("Contract de implicare al elevului: „Proiectul Podul”."))
        + "</p>"
    )
    if plan_text and plan_text.strip():
        if source:
            parts.append(
                '<p style="color:#555; font-size:9pt;">'
                + escape(trf("Sursă: {source}", source=source))
                + "</p>"
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
            '<p style="color:#888;"><i>'
            + escape(tr("Planul de intervenție nu a fost generat."))
            + "</i></p>"
        )
    return "".join(parts)


# --- Section 4: Success indicators -----------------------------------------
# Both plan sources are required to end with the same section — item 4 of
# ``llm_client.SYSTEM_PROMPT`` for the cloud plan, the "4. Indicatori de succes"
# block for the local template. The report also renders that section on its own,
# so it has to be lifted out of the plan body or it prints twice.
# Both languages must be recognised: the cloud plan is written in the interface
# language, so an English plan headed "4. Success indicators" has to be lifted
# out too — otherwise the list prints twice, once in the body and once in the
# dedicated panel, which is the exact regression this split exists to prevent.
_SUCCESS_HEADING_RE = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?:\*\*)?\s*\d*\.?\s*"
    r"(?:indicatori de succes|success indicators)"
)
# What terminates the block: a markdown heading, a horizontal rule, or the local
# template's "— Bază de calcul (date anonimizate) —" footer.
_BLOCK_END_RE = re.compile(r"^\s*(?:#{1,6}\s|[—–]\s|-{3,}|={3,})")


def _deaccent(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))


def _split_success_indicators(plan_text: str) -> tuple[str, str]:
    """Split the plan into (body without indicators, indicators section).

    The heading must *start* the line — a passing mention of "indicatori de
    succes" inside the risk summary must not tear the plan in half.
    An empty second element means the plan had no such section, and the generic
    checklist below applies instead.
    """
    text = (plan_text or "").strip()
    if not text:
        return "", ""
    lines = text.splitlines()

    start = next(
        (i for i, line in enumerate(lines)
         if _SUCCESS_HEADING_RE.match(_deaccent(line))),
        None,
    )
    if start is None:
        return text, ""

    end = next(
        (j for j in range(start + 1, len(lines)) if _BLOCK_END_RE.match(lines[j])),
        len(lines),
    )
    # Dedent as a block: the local template indents its bullets by three spaces,
    # and stripping only the first line would leave item 1 at column 0 with the
    # rest indented — which the Markdown parser may read as two separate lists.
    section = textwrap.dedent("\n".join(lines[start + 1:end])).strip()
    remaining = "\n".join(lines[:start] + lines[end:]).strip()
    # A heading with nothing under it is not worth relocating.
    return (remaining, section) if section else (text, "")


def _success_indicators_html(ev: RiskEvaluation, section_md: str = "") -> str:
    """The dedicated success-indicators panel.

    ``section_md`` is the block lifted out of the plan; when the plan supplied
    one, it is shown here *instead of* the generic checklist, so the tailored
    indicators survive de-duplication rather than being discarded.
    """
    parts: list[str] = []
    parts.append(_section_title(tr("Indicatori de succes (4 săptămâni)")))
    if section_md.strip():
        body = _markdown_to_html(section_md)
    else:
        current = max(0.0, min(10.0, ev.studentship_score))
        target = min(10.0, round(current + 3.0))
        body = (
            '<ul style="margin:0; font-size:10pt;">'
            f"<li>{escape(tr('Absențe: sub 2 absențe nemotivate pe săptămână.'))}</li>"
            "<li>"
            + trf("Implicare: creșterea scorului Studentship de la "
                  "<b>{current:g}/10</b> la <b>{target:g}/10</b>.",
                  current=current, target=target)
            + "</li><li>"
            + escape(tr("Participare: cel puțin un moment / o activitate "
                        "școlară activă pe săptămână."))
            + "</li><li>"
            + escape(tr("Atitudine: trecere spre „neutru / pozitiv” față de școală."))
            + "</li><li>"
            + escape(tr("Reevaluarea scorului de risc la finalul celor 4 săptămâni."))
            + "</li></ul>"
        )
    parts.append(
        '<table width="100%" cellpadding="10" cellspacing="0" '
        'border="1" style="border-color:#bfe0c8;"><tr>'
        '<td bgcolor="#eef7f0">'
        f'<div style="font-family: Segoe UI, Arial, sans-serif; font-size:10pt;">'
        f"{body}</div></td></tr></table>"
    )
    return "".join(parts)


def _page_break() -> str:
    return '<div style="page-break-before:always;"></div>'


def build_report_html(case: StudentCase, result: AssessmentResult, plan_text: str) -> str:
    """Assemble the full report as HTML.

    Each page is a block that starts with the running header band, preceded by
    ``page-break-before`` so ``QTextDocument`` paginates it onto its own sheet.
    The risk page carries the LIME profile under the sub-scores, so it flows onto
    a continuation sheet when the rule list is long.
    """
    ev = result.evaluation
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    # The plan's own success-indicators section is rendered separately below;
    # leaving it in the body would print the same list twice.
    plan_body, success_section = _split_success_indicators(plan_text)

    parts: list[str] = ['<div style="font-family: Segoe UI, Arial, sans-serif;">']

    # --- Page 1 — Datele introduse -----------------------------------------
    parts.append(_page_header_html("", cover=True))
    parts.append(_spacer(10))
    parts.append(_meta_strip_html(case, ev, generated))
    parts.append(_spacer(16))
    parts.append(_section_title(tr("Datele introduse")))
    parts.append(
        '<p style="color:#555; font-size:9pt;">'
        + escape(tr("Date de identificare și răspunsurile din chestionar, așa "
                    "cum au fost introduse pentru această evaluare."))
        + "</p>"
    )
    parts.append(_personal_data_html(case))

    # --- Page 2 — Evaluarea riscului ---------------------------------------
    # The LIME rule list sits under the sub-scores rather than on its own sheet:
    # it refines the same domain weights, and a teacher reads the two together.
    parts.append(_page_break())
    parts.append(_page_header_html(tr("Evaluarea riscului")))
    parts.append(_spacer(12))
    parts.append(_spacer(4))
    parts.append(_risk_profile_html(ev))
    parts.append(_spacer(6))
    parts.append(_xai_html(result, two_column=True, lime_block=_lime_html(result)))

    # --- Page 3 — Planul de recomandare ------------------------------------
    parts.append(_page_break())
    parts.append(_page_header_html(tr("Planul de recomandare")))
    parts.append(_spacer(12))
    parts.append(_spacer(4))
    parts.append(_plan_html(plan_body, ev.action_plan_source))
    parts.append(_spacer(14))
    parts.append(_success_indicators_html(ev, success_section))

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
        f'<span style="color:{_INK}; font-size:16pt;"><b>{escape(tr(_SUMMARY_TITLE))}</b></span><br>'
        f'<span style="color:{_MUTED}; font-size:9.5pt;">{escape(tr(_SUMMARY_SUBTITLE))}</span>'
        "</td>"
        '<td width="120" align="right" valign="middle">'
        f'<span style="color:{_MUTED}; font-size:8pt;">{escape(tr("CONFIDENȚIAL"))}</span>'
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
        (tr("Școala"), school_name or "—"),
        (tr("Elevi evaluați"), str(count)),
        (tr("Generat"), generated),
        (tr("Model"), model_version or "—"),
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

    parts: list[str] = [_section_title(tr("Distribuția pe niveluri de risc"))]
    parts.append('<table width="100%" cellpadding="3" cellspacing="0">')
    for tier in reversed(config.RISK_TIERS):  # Critic -> Scăzut
        n = counts.get(tier.band, 0)
        width = int(round(n / total * _MAX_BAR_PX))
        pct = n / total * 100.0
        parts.append(
            "<tr>"
            f'<td width="150" style="font-size:9.5pt;">'
            f'<b>{escape(tr_band(tier.band))}</b><br>'
            f'<span style="color:#888; font-size:7.5pt;">'
            f'{escape(tr_band(tier.urgency))}</span></td>'
            + _bar_cell(width, tier.color)
            + '<td width="90" style="font-size:9.5pt;">'
            + trf('<b>{count}</b> elevi <span style="color:#888;">({pct:.0f}%)</span>',
                  count=n, pct=pct)
            + "</td></tr>"
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
                "#", tr("Elev"), tr("Clasa"), tr("Nivel de risc"), tr("Scor"),
                tr("Urgență"), tr("Implicare"), tr("Factori principali"),
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
            f"<b>{escape(tr_band(ev.risk_band))}</b></td>"
            f'<td align="right" style="font-size:9pt;"><b>{ev.aggregate_score:.0f}%</b></td>'
            f'<td style="font-size:8.5pt;">{escape(tr_band(ev.urgency))}</td>'
            f'<td align="center" style="font-size:8.5pt;">{ev.studentship_score:g}/10</td>'
            f'<td style="font-size:8pt;">{escape(indicators)}</td>'
            "</tr>"
        )

    return (
        _section_title(tr("Prioritizarea intervențiilor"))
        + '<p style="color:#555; font-size:9pt;">'
        + escape(tr("Elevii sunt ordonați după urgența intervenției: mai întâi "
                    "nivelul de risc (Critic → Scăzut), apoi scorul modelului; "
                    "la risc egal, o implicare (Studentship) mai scăzută urcă în "
                    "prioritate."))
        + "</p>"
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
        '<p style="color:#888; font-size:8pt;">'
        + escape(tr("Scorurile și explicațiile provin dintr-un model XGBoost "
                    "real cu atribuiri SHAP autentice. Acest raport sintetizează "
                    "evaluările individuale; pentru fiecare elev există un raport "
                    "detaliat separat."))
        + "</p>"
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
