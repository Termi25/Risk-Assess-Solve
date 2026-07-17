"""Tests for the runtime performance-metrics module (latency / tokens / cost)."""

from __future__ import annotations

from app import metrics
from app.metrics import CallMetric, MetricsStore, RunMetrics


def _cloud_call(latency=1.0, inp=1000, out=500, think=200, model="gemini-2.5-pro"):
    return CallMetric.build(
        provider_id="gemini", model=model, source="cloud", ok=True,
        latency_s=latency,
        usage={"input_tokens": inp, "output_tokens": out, "thinking_tokens": think},
    )


def test_cost_usd_uses_pricing_snapshot():
    # gemini-2.5-pro = (1.25 in, 10.00 out) per 1M tokens; thinking billed as output.
    cost = metrics.cost_usd("gemini-2.5-pro", input_tokens=1_000_000, billed_output_tokens=1_000_000)
    assert cost == 1.25 + 10.00


def test_cost_usd_unknown_model_is_none():
    assert metrics.cost_usd("some-unlisted-model", 1000, 1000) is None


def test_call_metric_build_derives_cost_from_measured_tokens():
    call = _cloud_call(inp=1000, out=500, think=200)
    assert call.billed_output_tokens == 700           # visible output + thinking
    assert call.total_tokens == 1700
    expected = 1000 / 1e6 * 1.25 + 700 / 1e6 * 10.0
    assert abs(call.cost - expected) < 1e-12


def test_call_metric_local_has_zero_cost():
    call = CallMetric.build(
        provider_id="gemini", model="gemini-2.5-pro", source="local",
        ok=False, latency_s=0.01,
    )
    assert call.cost == 0.0
    assert call.total_tokens == 0


def test_call_metric_unknown_model_cost_is_none():
    call = _cloud_call(model="mystery-model")
    assert call.cost is None


def test_run_aggregates_latency_cost_and_per_report():
    calls = [_cloud_call(latency=2.0), _cloud_call(latency=4.0)]
    run = RunMetrics.for_calls("import", calls, label="clasa.xlsx")

    assert run.run_type == "import"
    assert run.provider == "gemini"
    assert run.model == "gemini-2.5-pro"
    assert run.report_count == 2
    assert run.cloud_calls == 2
    assert run.total_latency_s == 6.0
    assert run.avg_latency_s == 3.0
    assert run.min_latency_s == 2.0
    assert run.max_latency_s == 4.0
    assert run.pricing_available is True
    assert abs(run.cost_per_report - run.total_cost / 2) < 1e-12
    assert run.total_input_tokens == 2000
    assert run.total_billed_output_tokens == 1400


def test_cost_and_latency_per_cloud_call_exclude_local_fallback():
    # Two students: one billed cloud call, one local fallback (no API call).
    cloud = _cloud_call(latency=3.0, inp=1000, out=500, think=200)
    fallback = CallMetric.build(
        provider_id="gemini", model="gemini-2.5-pro", source="local-fallback",
        ok=False, latency_s=0.2, error="RuntimeError: boom",
    )
    run = RunMetrics.for_calls("import", [cloud, fallback])

    assert run.report_count == 2
    assert run.cloud_calls == 1
    # cloud_cost equals the total (the fallback is $0)…
    assert abs(run.cloud_cost - run.total_cost) < 1e-12
    # …but the denominators differ: per-report spreads over both students,
    # per-cloud-call counts only the one billed call (so it is 2x here).
    assert abs(run.cost_per_report - run.total_cost / 2) < 1e-12
    assert abs(run.cost_per_cloud_call - run.total_cost / 1) < 1e-12
    # latency per cloud call ignores the instant local fallback.
    assert run.avg_cloud_latency_s == 3.0
    assert run.avg_latency_s == 1.6


def test_cloud_figures_zero_when_no_cloud_calls():
    run = RunMetrics.for_calls("single", [CallMetric.build(
        provider_id="gemini", model="gemini-2.5-pro", source="local",
        ok=False, latency_s=0.01,
    )])
    assert run.cloud_calls == 0
    assert run.cost_per_cloud_call == 0.0
    assert run.avg_cloud_latency_s == 0.0


def test_run_pricing_partial_when_model_unpriced():
    run = RunMetrics.for_calls("single", [_cloud_call(model="mystery-model")])
    assert run.pricing_available is False


def test_run_dict_round_trip():
    run = RunMetrics.for_calls("import", [_cloud_call(), _cloud_call(latency=3.0)], label="x.xlsx")
    restored = RunMetrics.from_dict(run.to_dict())
    assert restored.run_type == run.run_type
    assert restored.label == run.label
    assert restored.report_count == run.report_count
    assert abs(restored.total_cost - run.total_cost) < 1e-12
    assert restored.calls[1].latency_s == 3.0


def test_store_persists_and_reloads(tmp_path):
    path = tmp_path / "runs.jsonl"
    store = MetricsStore(path=path)
    store.add(RunMetrics.for_calls("single", [_cloud_call()], label="Elev Test"))
    assert len(store.runs()) == 1

    reloaded = MetricsStore(path=path)          # a fresh process reading the log
    assert len(reloaded.runs()) == 1
    assert reloaded.runs()[0].label == "Elev Test"

    reloaded.clear()
    assert reloaded.runs() == []
    assert not path.exists()


def test_store_skips_corrupt_lines(tmp_path):
    path = tmp_path / "runs.jsonl"
    good = RunMetrics.for_calls("single", [_cloud_call()])
    import json
    path.write_text(
        "not-json\n" + json.dumps(good.to_dict()) + "\n", encoding="utf-8"
    )
    store = MetricsStore(path=path)
    assert len(store.runs()) == 1


def test_export_calls_csv(tmp_path):
    path = tmp_path / "metrics.csv"
    runs = [
        RunMetrics.for_calls("single", [_cloud_call()], label="Elev"),
        RunMetrics.for_calls("import", [_cloud_call(), _cloud_call()], label="clasa.xlsx"),
    ]
    rows = metrics.export_calls_csv(str(path), runs)
    assert rows == 3                            # 1 + 2 calls
    content = path.read_text(encoding="utf-8-sig")
    assert "cost_usd" in content.splitlines()[0]
    assert "clasa.xlsx" in content


def test_generate_action_plan_records_local_metric(trained_model, monkeypatch):
    """With no API key, generate_action_plan records a local (0-cost) call."""
    from app.explainability import evaluate
    from app.llm_client import generate_action_plan
    from tests.conftest import HIGH_RISK

    monkeypatch.setattr("app.keystore.resolve_api_key", lambda pid: None)
    model, _ = trained_model
    evaluation = evaluate(model, HIGH_RISK)

    calls: list = []
    plan, source = generate_action_plan(
        evaluation, questionnaire_answers=HIGH_RISK, api_key=None,
        knowledge_text=None, metric_out=calls,
    )
    assert source == "local template"
    assert len(calls) == 1
    assert calls[0].source == "local"
    assert calls[0].ok is False
    assert calls[0].cost == 0.0
