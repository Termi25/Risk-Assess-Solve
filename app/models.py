"""Plain domain dataclasses.

These mirror the README data model at the level the proof-of-concept needs:
a ``StudentCase`` bundles the identifying details, the free-text observations,
and the structured "Day 14" feature values a teacher enters; a
``RiskEvaluation`` holds the computed, explainable output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class StudentCase:
    """Everything a teacher enters for one evaluation."""

    name: str = ""
    surname: str = ""
    school_name: str = ""
    student_grade: str = ""          # e.g. "IX A"
    is_urban: bool = True

    # Free-text qualitative observation (fed to the NLP engine).
    observation_text: str = ""

    # Structured model inputs keyed by feature key (config.FEATURE_KEYS).
    # Categorical values are stored as display strings ("Rural", "Da", ...).
    features: dict[str, object] = field(default_factory=dict)

    id: Optional[int] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    def display_name(self) -> str:
        full = f"{self.surname} {self.name}".strip()
        return full or "(elev fără nume)"


@dataclass
class FeatureAttribution:
    """One SHAP feature contribution, in probability-points of dropout risk."""

    feature_key: str
    label: str
    value_display: str          # the student's value for this feature, as shown
    shap_value: float           # signed contribution (+ raises risk, - lowers it)

    @property
    def direction(self) -> str:
        return "crește riscul" if self.shap_value >= 0 else "reduce riscul"

    @property
    def points(self) -> float:
        """Contribution expressed in percentage points (rounded for display)."""
        return round(self.shap_value * 100.0, 1)


@dataclass
class SubScore:
    """A grouped, interpretable component of the aggregate risk."""

    name: str
    value: float  # 0..100


@dataclass
class RiskEvaluation:
    """Computed, explainable output for one student case."""

    probability: float                       # model P(dropout) in [0, 1]
    aggregate_score: float                    # probability * 100, rounded
    risk_band: str                            # "Moderat" | "Mediu" | "Ridicat" | "Critic"
    base_value: float                         # SHAP expected value (baseline P)
    urgency: str = ""                         # "Monitorizare" | "Medie" | "Ridicată" | "Maximă"
    studentship_score: float = 0.0            # engagement score 0..10 (surfaced in the report)
    critical_indicators: list[str] = field(default_factory=list)  # short xAI risk statements
    attributions: list[FeatureAttribution] = field(default_factory=list)
    sub_scores: list[SubScore] = field(default_factory=list)
    action_plan_text: str = ""
    action_plan_source: str = ""              # "cloud (Claude)" | "local template"
    model_version: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    id: Optional[int] = None
    student_case_id: Optional[int] = None

    def top_drivers(self, n: int = 5) -> list[FeatureAttribution]:
        return sorted(self.attributions, key=lambda a: abs(a.shap_value), reverse=True)[:n]
