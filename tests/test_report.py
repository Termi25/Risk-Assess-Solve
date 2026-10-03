"""Tests for the report layout: section order, de-duplication, panel styling."""

from __future__ import annotations

import pytest

from app.ui.report import _split_success_indicators as split


LOCAL_PLAN = """PLANUL DE INTERVENȚIE: PROIECTUL PODUL
==========================================

1. Rezumatul riscului
   Elevul se află în banda de risc «Critic» (80%).

4. Indicatori de succes pe 4 săptămâni
   - Săpt. 1: 0 absențe nemotivate noi; semnarea contractului.
   - Săpt. 2: participare la ≥80% dintre orele de sprijin.
   - Săpt. 4: reevaluarea scorului de risc.

— Bază de calcul (date anonimizate) —
Scor agregat de risc: 80.3%"""


# --- splitting the plan's success-indicators section ------------------------
def test_split_lifts_the_local_templates_indicator_block():
    body, section = split(LOCAL_PLAN)
    assert "Săpt. 1" in section and "Săpt. 4" in section
    assert "Indicatori de succes" not in body       # heading left with the block
    assert "Săpt. 1" not in body


def test_split_keeps_the_calculation_footer_in_the_plan():
    """The block ends at the '—' footer; the audit trail stays in the plan."""
    body, _ = split(LOCAL_PLAN)
    assert "Bază de calcul" in body
    assert "Scor agregat de risc" in body


def test_split_dedents_the_block_as_a_unit():
    """Mixed indentation would read as two lists once converted to Markdown."""
    _, section = split(LOCAL_PLAN)
    lines = [ln for ln in section.splitlines() if ln.strip()]
    assert all(ln.startswith("- ") for ln in lines), section


def test_split_handles_markdown_headings_from_the_cloud_plan():
    plan = (
        "## 1. Rezumatul riscului\nElevul prezinta risc ridicat.\n\n"
        "## 4. Indicatori de succes pe 4 saptamani\n"
        "- Reducerea absentelor sub 2/saptamana\n- Cresterea mediei la 6.0"
    )
    body, section = split(plan)
    assert "Reducerea absentelor" in section
    assert "Indicatori de succes" not in body
    assert "Rezumatul riscului" in body


def test_split_ignores_a_passing_mention_mid_sentence():
    """The heading must start the line, or a stray mention tears the plan apart."""
    plan = (
        "1. Rezumatul riscului\n"
        "   Vom stabili indicatori de succes masurabili pentru acest elev.\n\n"
        "2. Contract educational\n   - Elev: prezenta zilnica."
    )
    body, section = split(plan)
    assert section == ""
    assert body == plan.strip()               # nothing removed


def test_split_leaves_a_plan_without_indicators_untouched():
    plan = "1. Rezumat\n   Text simplu."
    assert split(plan) == (plan, "")


def test_split_handles_an_empty_plan():
    assert split("") == ("", "")
    assert split(None) == ("", "")


def test_split_ignores_a_heading_with_no_content_under_it():
    """A bare heading is not worth relocating — keep the plan as written."""
    plan = "1. Rezumat\n   Text.\n\n4. Indicatori de succes pe 4 saptamani\n"
    body, section = split(plan)
    assert section == ""
    assert "Indicatori de succes" in body


# --- assembled report -------------------------------------------------------
@pytest.fixture(scope="module")
def report_html(trained_model, sample_workbook):
    """A full report for a workbook-imported case, rendered once."""
    from PySide6.QtWidgets import QApplication
    from app.excel_import import load_cases_from_excel
    from app.service import AssessmentService
    from app.ui.report import build_report_html

    QApplication.instance() or QApplication([])   # QTextDocument needs an app
    case = load_cases_from_excel(sample_workbook)[0]
    service = AssessmentService()
    service._model = trained_model[0]
    result = service.assess(case)
    plan = service.local_plan(result.evaluation, case)
    return build_report_html(case, result, plan), result


def test_success_indicators_appear_exactly_once(report_html):
    """Regression: the plan body and the dedicated panel both printed them."""
    html, _ = report_html
    assert html.count("Indicatori de succes") == 1
    assert html.count("Săpt. 1") == 1


def test_lime_sits_between_the_subscores_and_the_nlp_section(report_html):
    html, result = report_html
    assert result.lime is not None            # otherwise the check is vacuous
    i_sub = html.find("Sub-scoruri pe domenii")
    i_lime = html.find("Profil individual de risc (LIME)")
    i_nlp = html.find("Analiză text (NLP)")
    i_plan = html.find("Plan personalizat")
    assert -1 < i_sub < i_lime < i_nlp < i_plan


def test_lime_no_longer_gets_its_own_headed_page(report_html):
    """Two breaks: cover -> risc, risc -> plan. LIME flows inside the risk page."""
    html, _ = report_html
    assert html.count("page-break-before") == 2


def test_fidelity_panel_colours_the_cell_not_the_table(report_html):
    """QTextBrowser ignores a table-level bgcolor and renders the panel white."""
    import re

    html, _ = report_html
    assert re.search(r'<td bgcolor="#f2f4f7">\s*<span[^>]*><b>Fidelitatea', html)


def test_report_names_the_construct_studentship_only(report_html):
    """The report used "implicare" (engagement) and "Studentship" for the same
    score, plus a separate "Implicare" domain; only "Studentship" remains."""
    html, _ = report_html
    assert "Implicare" not in html
    assert "implicare" not in html
    assert "Scor Studentship" in html


def test_report_states_the_scope_of_the_studentship_score(report_html):
    html, _ = report_html
    assert "Nu este scala Studentship evaluată de profesor" in html
