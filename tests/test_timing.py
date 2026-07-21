"""Tests for the per-stage latency instrumentation (app.timing).

The point of these is that the numbers reaching the research CSV mean what the
paper will claim they mean: unmeasured stays unmeasured, one-time construction
costs stay out of the per-student means, and a stage that raised still reports
the time it consumed.
"""

from __future__ import annotations

import pytest

from app.timing import STAGE_FIELDS, StageTimings, measure


def test_record_accumulates_rather_than_overwrites():
    t = StageTimings()
    t.record("shap_s", 0.10)
    t.record("shap_s", 0.05)
    assert t.shap_s == pytest.approx(0.15)


def test_unmeasured_stage_stays_none():
    """None and 0.0 must not be conflated — see the module docstring."""
    t = StageTimings()
    t.record("nlp_s", 0.001)
    assert t.nlp_s == pytest.approx(0.001)
    assert t.shap_s is None
    assert t.measured() == {"nlp_s": pytest.approx(0.001)}


def test_record_rejects_an_unknown_stage():
    with pytest.raises(ValueError):
        StageTimings().record("not_a_stage", 1.0)


def test_measure_context_manager_records_elapsed_time():
    t = StageTimings()
    with measure(t, "lime_s"):
        sum(range(100_000))
    assert t.lime_s is not None and t.lime_s > 0.0


def test_measure_with_none_sink_is_a_noop():
    """Instrumented code stays callable without forcing measurement."""
    with measure(None, "shap_s"):
        pass  # must not raise


def test_measure_records_even_when_the_block_raises():
    """A stage that failed still consumed time; hiding it would bias the mean."""
    t = StageTimings()
    with pytest.raises(RuntimeError):
        with measure(t, "shap_s"):
            raise RuntimeError("boom")
    assert t.shap_s is not None and t.shap_s >= 0.0


def test_totals_exclude_one_time_init_costs():
    """Explainer construction must not inflate the per-student figures."""
    t = StageTimings(
        nlp_s=0.001, predict_s=0.002, shap_s=0.30, lime_s=0.05, pdf_s=0.20,
        shap_init_s=1.50, lime_init_s=0.80,
    )
    assert t.xai_total_s == pytest.approx(0.353)      # nlp + predict + shap + lime
    assert t.local_total_s == pytest.approx(0.553)    # + pdf
    assert t.init_total_s == pytest.approx(2.30)      # the one-time fixtures only


def test_totals_are_none_when_nothing_was_measured():
    t = StageTimings()
    assert t.xai_total_s is None
    assert t.local_total_s is None
    assert t.init_total_s is None


def test_totals_ignore_unmeasured_stages():
    t = StageTimings(nlp_s=0.001, shap_s=0.30)   # predict/lime never ran
    assert t.xai_total_s == pytest.approx(0.301)


def test_copy_is_detached():
    """A stored metric must not be mutated by later work — see attach_stages."""
    original = StageTimings(shap_s=0.30)
    clone = original.copy()
    original.record("pdf_s", 0.20)
    assert clone.shap_s == pytest.approx(0.30)
    assert clone.pdf_s is None


def test_stage_fields_cover_the_pipeline_order():
    assert STAGE_FIELDS == ("nlp_s", "predict_s", "shap_s", "lime_s", "pdf_s")


# --- integration: the real pipeline actually fills these in -----------------
def test_assess_populates_stage_timings(trained_model):
    """A real assessment measures NLP, prediction, SHAP and LIME."""
    from app.models import StudentCase
    from app.service import AssessmentService
    from tests.conftest import HIGH_RISK

    model, _ = trained_model
    service = AssessmentService()
    service._model = model                      # skip the load; timings are the point

    case = StudentCase(
        name="Test", surname="Elev",
        observation_text="Elevul este retras și demotivat.",
        features=dict(HIGH_RISK),
    )
    result = service.assess(case)

    t = result.timings
    assert t.nlp_s is not None and t.nlp_s >= 0.0
    assert t.predict_s is not None and t.predict_s > 0.0
    assert t.shap_s is not None and t.shap_s > 0.0
    # pdf_s belongs to the report writer, not the assessment.
    assert t.pdf_s is None
    assert t.xai_total_s is not None


def test_explainer_construction_is_timed_separately(trained_model):
    """The first case of a process pays explainer construction; later ones don't.

    If this regressed, the one-time cost would land in ``shap_s`` and inflate the
    per-student mean reported in the paper.
    """
    from app import explainability

    model, _ = trained_model
    explainability._EXPLAINER_CACHE.pop(id(model.clf), None)   # force a cache miss
    from tests.conftest import LOW_RISK

    first = StageTimings()
    explainability.compute_attributions(model, dict(LOW_RISK), first)
    assert first.shap_init_s is not None and first.shap_init_s > 0.0

    second = StageTimings()
    explainability.compute_attributions(model, dict(LOW_RISK), second)
    assert second.shap_init_s is None          # cache hit — nothing rebuilt
    assert second.shap_s is not None and second.shap_s > 0.0


def test_shap_warmup_keeps_the_first_students_cost_off_their_report(trained_model):
    """The explainer's lazy first-call setup must be paid during construction.

    Measured on a cold process: without the warm-up in ``_get_explainer`` the
    first student's SHAP step costs ~3.9 s against ~48 ms for everyone after —
    an 80x outlier that would dominate any per-student mean, and a 4-second UI
    stall. The warm-up must also use a composed case row: a background row
    short-circuits against its own masker and leaves ~1.2 s unpaid for the first
    real student.

    Only the outlier relation is asserted, not the magnitude of ``shap_init_s``.
    SHAP's lazy setup is process-global rather than per-explainer, so once any
    earlier test has paid it, a rebuilt explainer here constructs in ~50 ms —
    the cost is real on a cold start but not reproducible within a session.
    """
    from app import explainability

    model, _ = trained_model
    explainability._EXPLAINER_CACHE.pop(id(model.clf), None)
    from tests.conftest import HIGH_RISK, LOW_RISK

    first = StageTimings()
    explainability.compute_attributions(model, dict(LOW_RISK), first)
    second = StageTimings()
    explainability.compute_attributions(model, dict(HIGH_RISK), second)

    assert first.shap_init_s is not None      # construction was attributed
    # The first student is not an outlier against the second. A generous bound:
    # the regression this guards is 80x, so 5x separates it from machine noise.
    assert first.shap_s < second.shap_s * 5
