"""Tests for the SHAP explainability layer."""

from app import config
from app.explainability import compute_attributions, evaluate
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
    assert evaluate(model, HIGH_RISK).risk_band == "Ridicat"
    assert evaluate(model, LOW_RISK).risk_band == "Scăzut"


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
