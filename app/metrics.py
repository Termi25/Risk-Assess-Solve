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
    """Measurements for one plan-generation call (one report)."""

    provider: str = ""          # "gemini" | "claude"
    model: str = ""             # exact model id used
    source: str = "cloud"       # "cloud" | "local" | "local-fallback"
    ok: bool = False            # cloud call succeeded (False => local fallback)
    latency_s: float = 0.0      # wall-clock duration of the call
    input_tokens: int = 0       # prompt tokens (provider-reported)
    output_tokens: int = 0      # visible answer tokens (provider-reported)
    thinking_tokens: int = 0    # reasoning tokens, billed at the output rate
    cost: Optional[float] = None  # USD; None only when the model price is unknown
    error: str = ""             # exception summary when the cloud call failed

    @property
    def billed_output_tokens(self) -> int:
        return self.output_tokens + self.thinking_tokens

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.billed_output_tokens

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
    label: str = ""                        # student name / source workbook name
    calls: list[CallMetric] = field(default_factory=list)

    @classmethod
    def for_calls(
        cls, run_type: str, calls: list[CallMetric], *, label: str = ""
    ) -> "RunMetrics":
        provider = calls[0].provider if calls else ""
        model = calls[0].model if calls else ""
        return cls(
            run_type=run_type, provider=provider, model=model,
            label=label, calls=list(calls),
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
        calls = [CallMetric(**c) for c in data.get("calls", [])]
        return cls(
            run_type=data["run_type"],
            started_at=data.get("started_at", ""),
            provider=data.get("provider", ""),
            model=data.get("model", ""),
            pricing_as_of=data.get("pricing_as_of", PRICING_AS_OF),
            label=data.get("label", ""),
            calls=calls,
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
    "pricing_as_of", "report_index", "source", "ok", "latency_s",
    "input_tokens", "output_tokens", "thinking_tokens",
    "billed_output_tokens", "cost_usd",
)


def export_calls_csv(path: str, runs: list[RunMetrics]) -> int:
    """Write one row per call (report) to ``path``; returns the row count.

    This is the granular table a paper needs to compute per-report distributions
    (mean ± SD of latency and cost). Encoded UTF-8 with BOM so Excel opens the
    Romanian labels correctly.
    """
    rows = 0
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh)
        writer.writerow(CSV_COLUMNS)
        for run_index, run in enumerate(runs, start=1):
            for report_index, call in enumerate(run.calls, start=1):
                writer.writerow([
                    run_index, run.run_type, run.started_at, run.label,
                    call.provider, call.model, run.pricing_as_of, report_index,
                    call.source, int(call.ok), f"{call.latency_s:.3f}",
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
