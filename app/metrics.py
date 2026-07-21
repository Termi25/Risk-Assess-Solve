from __future__ import annotations

import csv
import json
import statistics
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from . import config
from .timing import ALL_TIMING_FIELDS, StageTimings

# --- Pricing snapshot -------------------------------------------------------
# USD per 1,000,000 tokens as ``model id -> (input_rate, output_rate)``.
# Thinking/reasoning tokens are billed at the output rate. These are a dated
# snapshot for the paper's cost derivation — confirm them against the provider's
# current price list (and the right context-window tier) before publishing.
PRICING_AS_OF = "2026-07"
MODEL_PRICING: dict[str, tuple[float, float]] = {
    # Gemini (Google) — standard ≤200k-token context tier.
    "gemini-2.5-pro": (1.25, 10.00),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-1.5-pro": (1.25, 5.00),
    # Claude (Anthropic).
    "claude-opus-4-8": (15.00, 75.00),
    "claude-sonnet-5": (3.00, 15.00),
}

METRICS_FILENAME = "metrics_runs.jsonl"


def cost_usd(model: str, input_tokens: int, billed_output_tokens: int) -> Optional[float]:
    """USD cost of a call from its measured tokens, or ``None`` if the model's
    price is not in :data:`MODEL_PRICING` (so cost can't be stated honestly)."""
    rates = MODEL_PRICING.get(model)
    if rates is None:
        return None
    in_rate, out_rate = rates
    return (input_tokens / 1_000_000.0) * in_rate + (
        billed_output_tokens / 1_000_000.0
    ) * out_rate


# --- Per-call and per-run records -------------------------------------------
@dataclass
class CallMetric:
    """Measurements for one report: the plan-generation call plus its local stages.

    ``latency_s`` covers only the LLM call. The surrounding local pipeline (NLP,
    prediction, SHAP, LIME, PDF) is measured separately and attached as
    ``stages``, so latency can be reported as a breakdown — see
    :meth:`attach_stages` and :mod:`app.timing`.
    """

    provider: str = ""          # "gemini" | "claude"
    model: str = ""             # exact model id used
    source: str = "cloud"       # "cloud" | "local" | "local-fallback"
    ok: bool = False            # cloud call succeeded (False => local fallback)
    latency_s: float = 0.0      # wall-clock duration of the LLM call alone
    input_tokens: int = 0       # prompt tokens (provider-reported)
    output_tokens: int = 0      # visible answer tokens (provider-reported)
    thinking_tokens: int = 0    # reasoning tokens, billed at the output rate
    cost: Optional[float] = None  # USD; None only when the model price is unknown
    error: str = ""             # exception summary when the cloud call failed
    # Per-stage local timings for the report this call produced. ``None`` when
    # the caller did not measure them (the library API and tests need not).
    stages: Optional[StageTimings] = None

    @property
    def billed_output_tokens(self) -> int:
        return self.output_tokens + self.thinking_tokens

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.billed_output_tokens

    @property
    def total_report_s(self) -> float:
        """End-to-end wall clock for this report: local stages + the LLM call.

        This is the number a teacher actually waits through, and the one to
        quote as "time per report"; ``latency_s`` alone understates it and
        ``stages.local_total_s`` alone omits the network.
        """
        local = self.stages.local_total_s if self.stages else None
        return self.latency_s + (local or 0.0)

    def attach_stages(self, timings: Optional[StageTimings]) -> None:
        """Record the local pipeline timings for this report.

        Stores a *copy*: this metric is serialized to the JSONL log as soon as
        the run is added, and a live reference would let a later stage mutate an
        already-persisted record, putting the in-memory history out of step with
        the file on disk.
        """
        self.stages = timings.copy() if timings is not None else None

    @classmethod
    def build(
        cls,
        *,
        provider_id: str,
        model: str,
        source: str,
        ok: bool,
        latency_s: float,
        usage: Optional[dict] = None,
        error: str = "",
    ) -> "CallMetric":
        """Assemble a metric, deriving cost from the measured token usage.

        ``usage`` is the provider-agnostic dict produced by ``llm_client``
        (``input_tokens`` / ``output_tokens`` / ``thinking_tokens``). Local and
        fallback calls make no billable request, so their cost is 0.0.
        """
        usage = usage or {}
        inp = int(usage.get("input_tokens", 0) or 0)
        out = int(usage.get("output_tokens", 0) or 0)
        think = int(usage.get("thinking_tokens", 0) or 0)
        if source == "cloud" and ok:
            cost = cost_usd(model, inp, out + think)
        else:
            cost = 0.0
        return cls(
            provider=provider_id, model=model, source=source, ok=ok,
            latency_s=float(latency_s), input_tokens=inp, output_tokens=out,
            thinking_tokens=think, cost=cost, error=error,
        )

    @classmethod
    def from_dict(cls, data: dict) -> "CallMetric":
        """Rebuild from a logged dict, restoring the nested stage timings.

        ``asdict`` flattens :class:`StageTimings` into a plain dict, so it has to
        be reconstructed here; a bare ``cls(**data)`` would leave ``stages`` as a
        dict and every timing property would fail on it.
        """
        payload = dict(data)
        raw_stages = payload.pop("stages", None)
        call = cls(**payload)
        if raw_stages:
            call.stages = StageTimings(**{
                k: v for k, v in raw_stages.items() if k in ALL_TIMING_FIELDS
            })
        return call


def single_run_label(risk_band: str) -> str:
    """Non-identifying label for a single-student run.

    The metrics store measures *the LLM call* — latency, tokens, cost — not the
    student, so it has no legitimate need for a name. This log is mirrored to
    ``metrics_runs.jsonl`` and exported by :func:`export_calls_csv`, and that CSV
    is meant to be shared as research data; a name in the ``label`` column would
    travel with it. The risk band is used instead: it carries no identity but
    still lets cost and latency be broken down by tier, which is the analysis
    the label is actually for.
    """
    band = (risk_band or "").strip()
    return f"Evaluare individuală · {band}" if band else "Evaluare individuală"


@dataclass
class RunMetrics:
    """All the calls made during one run, plus the run's identity."""

    run_type: str                          # "single" | "import"
    started_at: str = field(
        default_factory=lambda: datetime.now().isoformat(timespec="seconds")
    )
    provider: str = ""
    model: str = ""
    pricing_as_of: str = PRICING_AS_OF
    # Free-text run descriptor. MUST NOT contain personal data — it is written to
    # the JSONL log and to the exported research CSV. Use ``single_run_label()``
    # for single runs; import runs carry the source workbook name.
    label: str = ""
    calls: list[CallMetric] = field(default_factory=list)
    # One-time cost of loading (or training) the model for this process. It is a
    # startup cost amortised across every report in the run, so it is held here
    # rather than being averaged into any single report's latency.
    model_load_s: Optional[float] = None

    @classmethod
    def for_calls(
        cls, run_type: str, calls: list[CallMetric], *, label: str = "",
        model_load_s: Optional[float] = None,
    ) -> "RunMetrics":
        provider = calls[0].provider if calls else ""
        model = calls[0].model if calls else ""
        return cls(
            run_type=run_type, provider=provider, model=model,
            label=label, calls=list(calls), model_load_s=model_load_s,
        )

    # --- aggregates --------------------------------------------------------
    @property
    def report_count(self) -> int:
        return len(self.calls)

    @property
    def cloud_calls(self) -> int:
        return sum(1 for c in self.calls if c.source == "cloud" and c.ok)

    @property
    def fallback_calls(self) -> int:
        return sum(1 for c in self.calls if c.source == "local-fallback")

    @property
    def local_calls(self) -> int:
        return sum(1 for c in self.calls if c.source == "local")

    @property
    def total_latency_s(self) -> float:
        return sum(c.latency_s for c in self.calls)

    @property
    def avg_latency_s(self) -> float:
        return self.total_latency_s / self.report_count if self.report_count else 0.0

    @property
    def min_latency_s(self) -> float:
        return min((c.latency_s for c in self.calls), default=0.0)

    @property
    def max_latency_s(self) -> float:
        return max((c.latency_s for c in self.calls), default=0.0)

    @property
    def stdev_latency_s(self) -> float:
        values = [c.latency_s for c in self.calls]
        return statistics.stdev(values) if len(values) > 1 else 0.0

    @property
    def pricing_available(self) -> bool:
        """True when every call has a known price (so the total is complete)."""
        return all(c.cost is not None for c in self.calls)

    @property
    def total_cost(self) -> float:
        return sum(c.cost for c in self.calls if c.cost is not None)

    @property
    def cost_per_report(self) -> float:
        return self.total_cost / self.report_count if self.report_count else 0.0

    # --- cloud-only view (excludes local / fallback calls) -----------------
    # ``cost_per_report`` averages over every report generated (local fallbacks
    # cost $0, so they pull the mean down). ``cost_per_cloud_call`` and
    # ``avg_cloud_latency_s`` isolate the actual billed API calls — the figure to
    # quote for "cost/latency per Gemini call".
    @property
    def cloud_cost(self) -> float:
        return sum(
            c.cost for c in self.calls
            if c.source == "cloud" and c.ok and c.cost is not None
        )

    @property
    def cost_per_cloud_call(self) -> float:
        return self.cloud_cost / self.cloud_calls if self.cloud_calls else 0.0

    @property
    def avg_cloud_latency_s(self) -> float:
        latencies = [c.latency_s for c in self.calls if c.source == "cloud" and c.ok]
        return sum(latencies) / len(latencies) if latencies else 0.0

    # --- latency breakdown (local pipeline stages) --------------------------
    # ``latency_s`` above is the LLM call only. These aggregate the local stages
    # measured around it, so a run can be reported as
    # "NLP x ms + SHAP y ms + LIME z ms + plan w s" rather than one number.
    def stage_summary(self) -> dict[str, tuple[float, float, int]]:
        """Per-stage ``{stage: (mean_s, stdev_s, n)}`` over the calls that measured it.

        Only calls that actually recorded a stage contribute to it, so the
        one-time explainer construction (paid by the first report of a process)
        reports ``n = 1`` instead of being diluted across the whole batch.
        """
        buckets: dict[str, list[float]] = {}
        for call in self.calls:
            if call.stages is None:
                continue
            for stage, value in call.stages.measured().items():
                buckets.setdefault(stage, []).append(value)
        return {
            stage: (
                sum(values) / len(values),
                statistics.stdev(values) if len(values) > 1 else 0.0,
                len(values),
            )
            for stage, values in buckets.items()
        }

    @property
    def has_stage_timings(self) -> bool:
        return any(c.stages is not None for c in self.calls)

    @property
    def avg_local_s(self) -> float:
        """Mean local pipeline time per report (everything except the LLM call)."""
        values = [
            c.stages.local_total_s for c in self.calls
            if c.stages is not None and c.stages.local_total_s is not None
        ]
        return sum(values) / len(values) if values else 0.0

    @property
    def avg_xai_s(self) -> float:
        """Mean explainable-AI core time per report (NLP + predict + SHAP + LIME)."""
        values = [
            c.stages.xai_total_s for c in self.calls
            if c.stages is not None and c.stages.xai_total_s is not None
        ]
        return sum(values) / len(values) if values else 0.0

    @property
    def avg_total_report_s(self) -> float:
        """Mean end-to-end wall clock per report (local stages + LLM call)."""
        if not self.calls:
            return 0.0
        return sum(c.total_report_s for c in self.calls) / len(self.calls)

    @property
    def init_total_s(self) -> float:
        """One-time explainer construction across the run (SHAP + LIME fixtures)."""
        return sum(
            c.stages.init_total_s for c in self.calls
            if c.stages is not None and c.stages.init_total_s is not None
        )

    @property
    def total_input_tokens(self) -> int:
        return sum(c.input_tokens for c in self.calls)

    @property
    def total_billed_output_tokens(self) -> int:
        return sum(c.billed_output_tokens for c in self.calls)

    # --- (de)serialization -------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "RunMetrics":
        calls = [CallMetric.from_dict(c) for c in data.get("calls", [])]
        return cls(
            run_type=data["run_type"],
            started_at=data.get("started_at", ""),
            provider=data.get("provider", ""),
            model=data.get("model", ""),
            pricing_as_of=data.get("pricing_as_of", PRICING_AS_OF),
            label=data.get("label", ""),
            calls=calls,
            model_load_s=data.get("model_load_s"),
        )


# --- Persistent store -------------------------------------------------------
class MetricsStore:
    """In-memory list of runs, mirrored to a JSONL log in the user-data dir."""

    def __init__(self, path: Optional[Path] = None):
        self._path = Path(path) if path else (config.user_data_dir() / METRICS_FILENAME)
        self._lock = threading.Lock()
        self._runs: list[RunMetrics] = []
        self._load()

    def _load(self) -> None:
        try:
            with open(self._path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        self._runs.append(RunMetrics.from_dict(json.loads(line)))
                    except Exception:
                        continue  # skip a corrupt line, keep the rest
        except FileNotFoundError:
            pass

    def add(self, run: RunMetrics) -> None:
        with self._lock:
            self._runs.append(run)
            try:
                with open(self._path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(run.to_dict(), ensure_ascii=False) + "\n")
            except Exception:
                pass  # metrics are best-effort; never break the app over them

    def runs(self) -> list[RunMetrics]:
        with self._lock:
            return list(self._runs)

    def clear(self) -> None:
        with self._lock:
            self._runs.clear()
            try:
                if self._path.exists():
                    self._path.unlink()
            except Exception:
                pass


_STORE: Optional[MetricsStore] = None


def get_store() -> MetricsStore:
    """The shared, lazily-created metrics store for this process."""
    global _STORE
    if _STORE is None:
        _STORE = MetricsStore()
    return _STORE


# --- CSV export -------------------------------------------------------------
CSV_COLUMNS = (
    "run_index", "run_type", "started_at", "label", "provider", "model",
    "pricing_as_of", "report_index", "source", "ok",
    # ``latency_s`` is the LLM call alone; the stage columns below break down the
    # local pipeline around it, and the two totals combine them.
    "latency_s",
    *ALL_TIMING_FIELDS,
    "xai_total_s", "local_total_s", "total_report_s", "model_load_s",
    "input_tokens", "output_tokens", "thinking_tokens",
    "billed_output_tokens", "cost_usd",
)


def _fmt_optional(value: Optional[float], precision: int = 4) -> str:
    """Format a duration, or an empty cell when it was never measured.

    Deliberately not ``0.000``: an un-instrumented stage and a stage that ran in
    under a millisecond are different findings, and a research CSV must not
    conflate them.
    """
    return "" if value is None else f"{value:.{precision}f}"


def export_calls_csv(path: str, runs: list[RunMetrics]) -> int:
    """Write one row per call (report) to ``path``; returns the row count.

    This is the granular table a paper needs to compute per-report distributions
    (mean ± SD of latency, per-stage breakdown, and cost). Encoded UTF-8 with BOM
    so Excel opens the Romanian labels correctly.
    """
    rows = 0
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh)
        writer.writerow(CSV_COLUMNS)
        for run_index, run in enumerate(runs, start=1):
            for report_index, call in enumerate(run.calls, start=1):
                stages = call.stages
                writer.writerow([
                    run_index, run.run_type, run.started_at, run.label,
                    call.provider, call.model, run.pricing_as_of, report_index,
                    call.source, int(call.ok), f"{call.latency_s:.3f}",
                    *(
                        _fmt_optional(getattr(stages, name) if stages else None)
                        for name in ALL_TIMING_FIELDS
                    ),
                    _fmt_optional(stages.xai_total_s if stages else None),
                    _fmt_optional(stages.local_total_s if stages else None),
                    f"{call.total_report_s:.4f}",
                    # Repeated on every row of a run so the column survives a
                    # per-report pivot; it is a run-level constant, not a sum.
                    _fmt_optional(run.model_load_s, precision=3),
                    call.input_tokens, call.output_tokens, call.thinking_tokens,
                    call.billed_output_tokens,
                    "" if call.cost is None else f"{call.cost:.6f}",
                ])
                rows += 1
    return rows


# --- Display helpers --------------------------------------------------------
def fmt_seconds(value: float) -> str:
    return f"{value:.2f} s"


def fmt_cost(value: Optional[float]) -> str:
    return "—" if value is None else f"${value:.4f}"
