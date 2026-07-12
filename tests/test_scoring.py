"""Tests for the XGBoost + SMOTE-NC scoring engine."""

import pandas as pd

from app import config
from app.scoring_engine import (
    encode_case_features,
    generate_synthetic_dataset,
    load_model,
    save_model,
    compose_model_features,
    _count_low_grades,
)
from tests.conftest import HIGH_RISK, LOW_RISK


def test_count_low_grades_numeric():
    # Distinct numeric grades below 5; passing grades (>=5) are ignored.
    assert _count_low_grades("4 Matematică; 3 Română; 2 Istorie") == 3
    assert _count_low_grades("2 la mate, 4.5 la romană") == 2
    assert _count_low_grades("10 la mate, 9 la română") == 0


def test_count_low_grades_qualitative_passing_marks_are_zero():
    """Primary-school passing marks and subject names must count as zero.

    Regression: the old fallback split such text on commas and mis-read
    "Suficient, limba română, matematică" as three grades below 5, which then
    depressed the Studentship score and inflated the model's risk.
    """
    assert _count_low_grades("Suficient, limba romana, matematica") == 0
    assert _count_low_grades("Suficient la toate materiile") == 0
    assert _count_low_grades("Bine la toate") == 0
    assert _count_low_grades("Nu e cazul") == 0
    assert _count_low_grades("") == 0


def test_count_low_grades_qualitative_insufficient_counts():
    # "Insuficient" is the one primary-school mark below the passing line.
    assert _count_low_grades("Insuficient la matematică") == 1
    assert _count_low_grades("Insuficient, Insuficient la română") == 2


def test_synthetic_dataset_has_both_classes():
    df = generate_synthetic_dataset(400, seed=1)
    counts = df[config.TARGET_COLUMN].value_counts()
    assert set(counts.index) == {0, 1}
    assert counts.min() >= 5


def test_encode_case_features_shape_and_codes():
    X = encode_case_features(HIGH_RISK)
    assert list(X.columns) == list(config.FEATURE_KEYS)
    assert X.shape == (1, len(config.FEATURE_KEYS))
    # "Rural" -> 1, "Monoparental" -> 1, "Sancțiuni" -> 2
    assert X["Mediu_Rezidential"].iloc[0] == 1
    assert X["Situatie_Familiala"].iloc[0] == 1
    assert X["Sanctiuni_Avertismente"].iloc[0] == 2


def test_compose_model_features_derives_age_and_studentship():
    features = compose_model_features(HIGH_RISK)
    assert "Age_Years" in features
    assert 0 <= features["Age_Years"] <= 30
    assert features["Studentship_Score"] <= 4


def test_compose_model_features_is_idempotent():
    """Composing an already-composed dict must not reset derived values."""
    once = compose_model_features(HIGH_RISK)
    twice = compose_model_features(once)
    assert twice == once


def test_smote_balances_training_set(trained_model):
    _, metrics = trained_model
    after = metrics.balance_after
    assert after["0"] == after["1"]  # perfectly balanced


def test_model_is_reasonably_accurate(trained_model):
    # The graded (probabilistic) labels carry irreducible Bayes error by design —
    # the model trades the old deterministic-label near-perfection for calibrated
    # probabilities that spread across the four risk tiers. AUC confirms genuine
    # discrimination; accuracy sits at a realistic dropout-model level.
    _, metrics = trained_model
    assert metrics.accuracy >= 0.75
    assert metrics.roc_auc >= 0.75


def test_probabilities_in_range_and_ordered(trained_model):
    model, _ = trained_model
    p_high = model.predict_probability(HIGH_RISK)
    p_low = model.predict_probability(LOW_RISK)
    assert 0.0 <= p_low <= 1.0
    assert 0.0 <= p_high <= 1.0
    assert p_high > p_low
    assert p_high > 0.5
    assert p_low < 0.5


def test_model_save_load_roundtrip(trained_model, tmp_path):
    model, _ = trained_model
    mpath = tmp_path / "m.json"
    meta = tmp_path / "m.meta.json"
    save_model(model, mpath, meta)
    assert mpath.exists() and meta.exists()

    reloaded = load_model(mpath, meta)
    assert reloaded.version == model.version
    p1 = model.predict_probability(HIGH_RISK)
    p2 = reloaded.predict_probability(HIGH_RISK)
    assert abs(p1 - p2) < 1e-6
