"""Tests for the LIME local-explanation layer.

These assert the *contract* the report depends on (shape, key mapping, fidelity
reporting, graceful degradation), not specific weights — LIME fits a surrogate
on sampled perturbations, so individual coefficients are noisy by construction.
"""

from __future__ import annotations

import pytest

from app import config
from app.lime_explainer import _humanise, explain_case
from app.models import LimeExplanation

lime = pytest.importorskip("lime", reason="lime is an optional dependency")


@pytest.fixture(scope="module")
def model():
    from app.scoring_engine import train_model
    trained, _metrics = train_model(n_samples=400, seed=7, cv_folds=0)
    return trained


HIGH_RISK_CASE = {
    "unexcused_absences_3m": 45,
    "previous_module_average": 4.1,
    "school_attitude": "Negativă",
    "disciplinary_sanctions": "Sancțiuni",
    "school_feeling": "Izolat",
    "school_support_goal": "Nu",
    "extracurricular_participation": "Nu",
}


def test_returns_explanation_with_conditions(model):
    exp = explain_case(model, HIGH_RISK_CASE, num_samples=1000)
    assert isinstance(exp, LimeExplanation)
    assert exp.conditions, "expected at least one local rule"
    assert len(exp.conditions) <= config.FEATURE_KEYS.__len__()


def test_conditions_map_to_real_feature_keys(model):
    exp = explain_case(model, HIGH_RISK_CASE, num_samples=1000)
    for cond in exp.conditions:
        assert cond.feature_key in config.FEATURE_KEYS
        assert cond.label == config.feature(cond.feature_key).label
        assert cond.condition, "condition text must not be empty"


def test_condition_text_is_humanised_not_raw_keys(model):
    """The report shows these verbatim, so raw column names must not leak."""
    exp = explain_case(model, HIGH_RISK_CASE, num_samples=1000)
    joined = " ".join(c.condition for c in exp.conditions)
    assert "Medie_Modul_Anterior" not in joined
    assert "Absente_Nemotivate_Zilele_1_13" not in joined


def test_fidelity_is_reported(model):
    """A weak local fit must be visible, not silently trusted."""
    exp = explain_case(model, HIGH_RISK_CASE, num_samples=1000)
    assert 0.0 <= exp.fidelity_r2 <= 1.0
    assert exp.fidelity_label in {"bună", "moderată", "slabă"}
    assert 0.0 <= exp.local_prediction <= 1.0
    assert exp.local_gap == pytest.approx(
        abs(exp.local_prediction - exp.model_probability)
    )


def test_model_probability_matches_the_scoring_engine(model):
    """LIME must explain the same row the scoring engine actually scored."""
    exp = explain_case(model, HIGH_RISK_CASE, num_samples=1000)
    assert exp.model_probability == pytest.approx(
        model.predict_probability(HIGH_RISK_CASE), abs=1e-6
    )


def test_accepts_composed_features_idempotently(model):
    """Raw questionnaire answers and composed features describe the same row."""
    from app.scoring_engine import compose_model_features
    composed = compose_model_features(HIGH_RISK_CASE)
    exp = explain_case(model, composed, num_samples=1000)
    assert exp.model_probability == pytest.approx(
        model.predict_probability(HIGH_RISK_CASE), abs=1e-6
    )


def test_weights_are_not_expected_to_be_additive(model):
    """Guards the documented contract: LIME weights are NOT a decomposition.

    If a future change makes them additive, the report copy (which tells the
    teacher they do not sum to the score) has to change with it.
    """
    exp = explain_case(model, HIGH_RISK_CASE, num_samples=1000)
    total = exp.intercept + sum(c.weight for c in exp.conditions)
    # The surrogate reconstructs its OWN local prediction, not the model's.
    assert total == pytest.approx(exp.local_prediction, abs=0.05)


def test_humanise_replaces_keys_with_labels():
    assert "Medie_Modul_Anterior" not in _humanise("Medie_Modul_Anterior <= 6.20")
    assert config.feature("Medie_Modul_Anterior").label in _humanise(
        "Medie_Modul_Anterior <= 6.20"
    )


def test_humanise_prefers_longest_key_first():
    """``Absente_Motivate_3_Luni`` must not be mangled by a shorter overlap."""
    out = _humanise("Absente_Motivate_3_Luni > 12.00")
    assert config.feature("Absente_Motivate_3_Luni").label in out


@pytest.mark.parametrize(
    "raw",
    [
        "Medie_Modul_Anterior <= 6.20",
        "6.20 < Medie_Modul_Anterior <= 7.10",
        "Absente_Motivate_3_Luni >= 4.00",
    ],
)
def test_humanise_keeps_comparison_operators_intact(raw):
    """Regression: spacing the "=" must not split "<=" into "< =".""" ""
    out = _humanise(raw)
    assert "< =" not in out
    assert "> =" not in out


def test_humanise_spaces_categorical_equality():
    assert " = Izolat" in _humanise("Cum_te_Simti_La_Scoala=Izolat")


def test_missing_lime_degrades_to_none(model, monkeypatch):
    """Without the optional dependency the app must still assess, sans section."""
    import app.lime_explainer as mod
    monkeypatch.setattr(mod, "_EXPLAINER_CACHE", {})

    def _boom(*_a, **_k):
        raise ImportError("lime not installed")

    monkeypatch.setattr(mod, "_build_explainer", _boom)
    assert mod.explain_case(model, HIGH_RISK_CASE) is None
