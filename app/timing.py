"""Per-stage wall-clock timing for the assessment pipeline.

:mod:`app.metrics` measures the LLM call — the one stage that is network-bound
and billed. This module measures everything *around* it, so a run can be
reported as a breakdown (NLP -> prediction -> SHAP -> LIME -> plan -> PDF)
rather than as a single opaque duration.

Two design points that matter for the numbers to be publishable:

*Unmeasured is not zero.* Every field is ``Optional[float]`` and defaults to
``None``. A stage that was never instrumented stays ``None`` and is excluded
from every mean; writing ``0.000`` into a research CSV would assert a
measurement that was never taken.

*One-time costs are separated from per-student ones.* Both explainers cache an
expensive fixture on first use — SHAP its background masker
(:func:`app.explainability._get_explainer`), LIME its discretiser
(:func:`app.lime_explainer._build_explainer`). Student #1 of a batch pays that
construction and students 2..N do not, so folding it into ``shap_s`` would
inflate the per-student mean by a constant that shrinks as the batch grows.
It is recorded as ``shap_init_s`` / ``lime_init_s`` instead, and reported
separately.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, fields, replace
from typing import Iterator, Optional

# Per-student stages, in pipeline order. ``pdf_s`` is last because the report is
# only rendered once the plan text exists.
STAGE_FIELDS: tuple[str, ...] = (
    "nlp_s",
    "predict_s",
    "shap_s",
    "lime_s",
    "pdf_s",
)

# One-time, per-process fixtures. Excluded from the per-student totals.
INIT_FIELDS: tuple[str, ...] = (
    "shap_init_s",
    "lime_init_s",
)

ALL_TIMING_FIELDS: tuple[str, ...] = STAGE_FIELDS + INIT_FIELDS

# Human-readable names for the UI and for paper table rows.
STAGE_LABELS: dict[str, str] = {
    "nlp_s": "NLP (text observațional)",
    "predict_s": "Predicție XGBoost",
    "shap_s": "Explicație SHAP",
    "lime_s": "Explicație LIME",
    "pdf_s": "Randare raport PDF",
    "shap_init_s": "Construcție explainer SHAP (o singură dată)",
    "lime_init_s": "Construcție explainer LIME (o singură dată)",
}


@dataclass
class StageTimings:
    """Wall-clock seconds per pipeline stage for one student's report.

    ``None`` means the stage was not measured in this run — see the module
    docstring. Durations accumulate, so a stage entered twice reports the sum.
    """

    nlp_s: Optional[float] = None
    predict_s: Optional[float] = None
    shap_s: Optional[float] = None
    lime_s: Optional[float] = None
    pdf_s: Optional[float] = None
    shap_init_s: Optional[float] = None
    lime_init_s: Optional[float] = None

    def record(self, stage: str, seconds: float) -> None:
        """Add ``seconds`` to ``stage``, treating an unset stage as 0."""
        if stage not in ALL_TIMING_FIELDS:
            raise ValueError(f"unknown timing stage: {stage!r}")
        current = getattr(self, stage)
        setattr(self, stage, float(seconds) + (current or 0.0))

    def measured(self) -> dict[str, float]:
        """Only the stages that were actually timed, in declaration order."""
        return {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if getattr(self, f.name) is not None
        }

    @property
    def xai_total_s(self) -> Optional[float]:
        """NLP + prediction + SHAP + LIME — the local explainable-AI core.

        This is the figure behind the claim that the whole explanation pipeline
        runs on the teacher's machine; it deliberately excludes report rendering
        (presentation, not inference) and the one-time explainer construction.
        """
        return self._sum(("nlp_s", "predict_s", "shap_s", "lime_s"))

    @property
    def local_total_s(self) -> Optional[float]:
        """Every measured per-student stage, including PDF rendering.

        Excludes the LLM call (network-bound, tracked as
        :attr:`app.metrics.CallMetric.latency_s`) and the one-time init costs.
        """
        return self._sum(STAGE_FIELDS)

    @property
    def init_total_s(self) -> Optional[float]:
        """One-time explainer construction paid by the first case of a process."""
        return self._sum(INIT_FIELDS)

    def _sum(self, names: tuple[str, ...]) -> Optional[float]:
        values = [getattr(self, n) for n in names if getattr(self, n) is not None]
        return sum(values) if values else None

    def copy(self) -> "StageTimings":
        """A detached copy.

        Callers store timings on a metric that is serialized immediately; keeping
        a live reference would let a later stage (a PDF exported after the run
        was written to the JSONL log) mutate an already-persisted record and put
        the in-memory history out of step with the file.
        """
        return replace(self)


@contextmanager
def measure(timings: Optional[StageTimings], stage: str) -> Iterator[None]:
    """Time the block and add it to ``timings`` under ``stage``.

    A ``None`` sink makes this a no-op, so instrumented code paths stay callable
    from tests and from the library API without forcing measurement on anyone.
    The duration is recorded even when the block raises: a stage that failed
    still consumed the time, and hiding that would bias the mean downwards.
    """
    if timings is None:
        yield
        return
    start = time.perf_counter()
    try:
        yield
    finally:
        timings.record(stage, time.perf_counter() - start)
