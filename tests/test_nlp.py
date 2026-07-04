"""Tests for the lexicon-based NLP valence/stress engine."""

from app.nlp_engine import analyze


def test_empty_text_is_neutral():
    r = analyze("")
    assert r.stress_score == 0.0
    assert r.valence == 0.0


def test_negative_text_raises_stress():
    r = analyze("Elevul este retras, obosit si demotivat. Refuza sa participe.")
    assert r.stress_score > 0.5
    assert r.valence < 0
    assert "retras" in r.negative_terms


def test_positive_text_low_stress():
    r = analyze("Elev motivat, implicat si sociabil, cu progres vizibil.")
    assert r.valence > 0
    assert r.stress_score < 0.5


def test_stress_is_bounded():
    r = analyze(("abandon " * 50))
    assert 0.0 <= r.stress_score <= 2.0


def test_diacritics_are_handled():
    with_dia = analyze("Elevul este obosit și retras")
    without = analyze("Elevul este obosit si retras")
    assert with_dia.stress_score == without.stress_score


def test_negation_flips_polarity():
    negated = analyze("Elevul nu este motivat")
    assert negated.valence <= 0
