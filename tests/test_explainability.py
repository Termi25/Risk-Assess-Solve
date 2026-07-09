"""Tests for the SHAP explainability layer."""

from app import config
from app.explainability import classify_risk, compute_attributions, evaluate
from tests.conftest import HIGH_RISK, LOW_RISK


def test_shap_is_additive(trained_model):
    """base_value + sum(shap) must reconstruct the model's real probability."""
    model, _ = trained_model
    proba = model.predict_probability(HIGH_RISK)
    base, attrs = compute_attributions(model, HIGH_RISK)
    reconstructed = base + sum(a.shap_value for a in attrs)
    assert abs(reconstructed - proba) < 1e-3


def test_all_features_attributed(trained_model):
    model, _ = trained_model
    _, attrs = compute_attributions(model, HIGH_RISK)
    keys = {a.feature_key for a in attrs}
    assert keys == set(config.FEATURE_KEYS)


def test_absences_are_a_positive_driver_for_high_risk(trained_model):
    model, _ = trained_model
    _, attrs = compute_attributions(model, HIGH_RISK)
    by_key = {a.feature_key: a for a in attrs}
    # Heavy unmotivated absences should push risk up, not down.
    assert by_key["Absente_Nemotivate_Zilele_1_13"].shap_value > 0


def test_risk_bands(trained_model):
    model, _ = trained_model
    # HIGH_RISK is an extreme, compounded profile -> top of the 4-tier scale.
    assert evaluate(model, HIGH_RISK).risk_band in {"Ridicat", "Critic"}
    assert evaluate(model, LOW_RISK).risk_band == "Moderat"


def test_urgency_matches_band(trained_model):
    model, _ = trained_model
    for case in (HIGH_RISK, LOW_RISK):
        ev = evaluate(model, case)
        assert ev.urgency == config.tier_for_band(ev.risk_band).urgency
        assert ev.urgency in {"Monitorizare", "Medie", "Ridicată", "Maximă"}


def test_studentship_score_is_surfaced(trained_model):
    model, _ = trained_model
    high = evaluate(model, HIGH_RISK)
    low = evaluate(model, LOW_RISK)
    assert 0.0 <= high.studentship_score <= 10.0
    assert high.studentship_score <= 4.0     # disengaged profile
    assert low.studentship_score >= 7.0      # engaged profile


def test_critical_indicators_track_risk(trained_model):
    model, _ = trained_model
    assert evaluate(model, HIGH_RISK).critical_indicators      # non-empty
    assert evaluate(model, LOW_RISK).critical_indicators == []


def test_severe_profile_escalates_to_critic():
    # Compounded severe factors must classify as Critic even below p=0.85.
    mf = {
        "Absente_Nemotivate_Zilele_1_13": 30,
        "Medie_Modul_Anterior": 3.5,
        "Studentship_Score": 1,
        "Sanctiuni_Avertismente": "Sancțiuni",
        "Cum_te_Simti_La_Scoala": "Izolat",
    }
    assert classify_risk(0.70, mf).band == "Critic"
    # A plain high-probability case with no severe factors stays Ridicat.
    assert classify_risk(0.70, {"Medie_Modul_Anterior": 8.0}).band == "Ridicat"


def test_value_display_reflects_questionnaire_answers(trained_model):
    """Regression: SHAP 'valoare' must show the student's real inputs even when
    the caller passes questionnaire-keyed answers rather than model keys."""
    model, _ = trained_model
    raw = {
        "unexcused_absences_3m": 17,
        "previous_module_average": 4.2,
        "school_attitude": "Negativă",
    }
    _, attrs = compute_attributions(model, raw)
    by_key = {a.feature_key: a for a in attrs}
    assert by_key["Absente_Nemotivate_Zilele_1_13"].value_display == "17"
    assert by_key["Atitudine_Scoala"].value_display == "Negativă"


def test_sub_scores_sum_to_about_100(trained_model):
    model, _ = trained_model
    ev = evaluate(model, HIGH_RISK)
    total = sum(s.value for s in ev.sub_scores)
    assert 99.0 <= total <= 101.0


def test_evaluate_populates_fields(trained_model):
    model, _ = trained_model
    ev = evaluate(model, LOW_RISK)
    assert 0.0 <= ev.probability <= 1.0
    assert ev.model_version
    assert len(ev.attributions) == len(config.FEATURE_KEYS)
