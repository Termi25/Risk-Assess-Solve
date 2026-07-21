"""Tests pinning what may and may not reach the cloud provider.

These guard a privacy contract, not a formatting preference: the payload built
here is the only student-derived data that ever leaves the machine, so each
assertion below corresponds to a decision recorded in ``llm_client``'s module
docstring. A change that makes one of these fail is a change to what the app
discloses about a minor, and should be made deliberately.
"""

from __future__ import annotations

import pytest

from app.llm_client import (
    _parental_education_index,
    _questionnaire_context_lines,
    anonymized_summary,
)


IDENTIFIABLE_CASE = {
    "full_name": "Popescu Ion",
    "school_name": "Liceul Teoretic Test",
    "birth_date": "2008-03-04",
    "sex": "Feminin",
    "residential_environment": "Rural",
    "family_situation": "Tutore / plasament",
    "mother_education": "Primar",
    "father_education": "Gimnazial",
    "unexcused_absences_3m": 40,
    "previous_module_average": 4.2,
    "school_attitude": "Negativă",
    "low_grades_details": "4 la matematică, 3 la română",
    "additional_notes": "Elevul locuiește cu bunica, situație materială dificilă.",
}


@pytest.fixture(scope="module")
def evaluation():
    from app.explainability import evaluate
    from app.scoring_engine import train_model
    model, _ = train_model(n_samples=400, seed=7, cv_folds=0)
    return evaluate(model, IDENTIFIABLE_CASE)


# --- direct identifiers -----------------------------------------------------
def test_summary_excludes_direct_identifiers(evaluation):
    summary = anonymized_summary(evaluation, IDENTIFIABLE_CASE)
    assert "Popescu" not in summary
    assert "Ion" not in summary
    assert "Liceul Teoretic Test" not in summary
    assert "2008-03-04" not in summary


def test_summary_excludes_free_text(evaluation):
    """Free-text answers stay local — they can contain anything a teacher typed."""
    summary = anonymized_summary(evaluation, IDENTIFIABLE_CASE)
    assert "bunica" not in summary
    assert "matematică" not in summary


# --- quasi-identifiers ------------------------------------------------------
def test_cloud_context_drops_age_and_sex():
    lines = " ".join(_questionnaire_context_lines(IDENTIFIABLE_CASE))
    assert "Vârsta" not in lines
    assert "Feminin" not in lines
    assert "Sex" not in lines


def test_cloud_context_coarsens_family_situation():
    lines = " ".join(_questionnaire_context_lines(IDENTIFIABLE_CASE))
    assert "Structura familiei: tutore / altă situație" in lines
    # The precise 4-way category must not survive verbatim.
    assert "Situația familială" not in lines


@pytest.mark.parametrize(
    ("answer", "bucket"),
    [
        ("Ambii părinți", "ambii părinți"),
        ("Monoparental", "un singur adult"),
        ("Părinți divortați/separați", "un singur adult"),
        ("Tutore / plasament", "tutore / altă situație"),
        ("Altă situație", "tutore / altă situație"),
    ],
)
def test_every_family_answer_maps_to_a_bucket(answer, bucket):
    case = dict(IDENTIFIABLE_CASE, family_situation=answer)
    lines = " ".join(_questionnaire_context_lines(case))
    assert f"Structura familiei: {bucket}" in lines


def test_cloud_context_merges_parental_education_into_one_index():
    lines = " ".join(_questionnaire_context_lines(IDENTIFIABLE_CASE))
    assert "Indice educație parentală" in lines
    # Neither parent's raw level, nor the per-parent labels, may appear.
    assert "Educația mamei" not in lines
    assert "Educația tatălui" not in lines


def test_parental_education_index_averages_both_parents():
    # _EDUCATION_RISK: Primar 0.5, Gimnazial 0.2 -> mean 0.35
    assert _parental_education_index(
        {"Educatie_Mama": "Primar", "Educatie_Tata": "Gimnazial"}
    ) == pytest.approx(0.35)
    # Universitar -0.3 both -> -0.3
    assert _parental_education_index(
        {"Educatie_Mama": "Universitar", "Educatie_Tata": "Universitar"}
    ) == pytest.approx(-0.3)


def test_parental_education_index_handles_one_missing_parent():
    assert _parental_education_index(
        {"Educatie_Mama": "Primar", "Educatie_Tata": ""}
    ) == pytest.approx(0.5)
    assert _parental_education_index({}) is None


def test_unknown_education_level_defaults_to_the_high_risk_score():
    """An unrecognised label must not silently read as low risk."""
    assert _parental_education_index(
        {"Educatie_Mama": "ceva necunoscut", "Educatie_Tata": "ceva necunoscut"}
    ) == pytest.approx(0.5)


# --- what is deliberately kept ---------------------------------------------
def test_cloud_context_keeps_residential_environment():
    """Two values only, and it drives the commuting-student intervention."""
    lines = " ".join(_questionnaire_context_lines(IDENTIFIABLE_CASE))
    assert "Mediul de proveniență: Rural" in lines


def test_cloud_context_keeps_the_actionable_signals():
    lines = " ".join(_questionnaire_context_lines(IDENTIFIABLE_CASE))
    for expected in (
        "Absențe nemotivate",
        "Media modulului anterior",
        "Atitudinea față de școală",
        "Scor Studentship",
    ):
        assert expected in lines


# --- the local path keeps full detail --------------------------------------
def test_local_context_retains_full_detail():
    """The local plan never leaves the machine, so it is not de-identified."""
    lines = " ".join(
        _questionnaire_context_lines(IDENTIFIABLE_CASE, deidentified=False)
    )
    assert "Vârsta aproximativă" in lines
    assert "Feminin" in lines
    assert "Educația mamei: Primar" in lines
    assert "Situația familială: Tutore / plasament" in lines


def test_deidentified_is_the_default():
    """A caller that forgets the flag must get the safe payload, not a leak."""
    default = _questionnaire_context_lines(IDENTIFIABLE_CASE)
    explicit = _questionnaire_context_lines(IDENTIFIABLE_CASE, deidentified=True)
    assert default == explicit


# --- missing answers must not be invented -----------------------------------
def test_unanswered_categorical_is_reported_as_unspecified():
    """Regression: an unanswered categorical falls back to Feature.default (0.0).

    It previously reached the LLM as a bare "0.0". Decoding that to the first
    category would instead assert an answer the teacher never gave.
    """
    sparse = {
        "unexcused_absences_3m": 10,
        "previous_module_average": 5.0,
    }
    lines = " ".join(_questionnaire_context_lines(sparse))
    assert "Participare extrașcolară: nespecificat" in lines
    assert "Sancțiuni / avertismente: nespecificat" in lines
    assert ": 0.0" not in lines


def test_answered_categorical_is_reported_verbatim():
    lines = " ".join(_questionnaire_context_lines(IDENTIFIABLE_CASE))
    assert "Atitudinea față de școală: Negativă" in lines
    assert "nespecificat" not in lines.split("Atitudinea")[1][:40]
