"""Assessment orchestration: NLP -> scoring -> SHAP -> LIME -> (optional) plan -> save.

Keeps the GUI thin. All the heavy imports (xgboost, shap, lime) live behind this
service so the window can construct instantly and load the model lazily.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .models import LimeExplanation, RiskEvaluation, StudentCase
from .nlp_engine import NlpResult, analyze


@dataclass
class AssessmentResult:
    nlp: NlpResult
    evaluation: RiskEvaluation
    # Local LIME profile for this student. ``None`` when ``lime`` is not
    # installed or the surrogate failed — the report then omits that section.
    lime: Optional[LimeExplanation] = None


class AssessmentService:
    """Loads the model once, then evaluates cases and persists results."""

    def __init__(self):
        self._model = None  # loaded lazily on first use

    # Imports are deferred so importing this module stays cheap/fast.
    def ensure_model(self):
        if self._model is None:
            from .scoring_engine import get_or_train_model
            self._model = get_or_train_model()
        return self._model

    def retrain(self):
        """Force a fresh training run and cache the new model."""
        from .scoring_engine import get_or_train_model
        self._model = get_or_train_model(force_retrain=True)
        return self._model

    @property
    def model_version(self) -> str:
        return self.ensure_model().version

    @property
    def model_metrics(self) -> Optional[dict]:
        return self.ensure_model().meta.metrics

    def assess(self, case: StudentCase, with_lime: bool = True) -> AssessmentResult:
        """Run the full local pipeline for one case.

        ``with_lime`` computes the individual LIME risk profile alongside the
        SHAP decomposition (~10 ms). It is separable because the two answer
        different questions — see :mod:`app.lime_explainer`.
        """
        from .explainability import evaluate_case
        model = self.ensure_model()
        nlp = analyze(case.observation_text)
        evaluation = evaluate_case(model, case, nlp)

        lime_profile = None
        if with_lime:
            from .lime_explainer import explain_case
            features = dict(case.features)
            features["Stres_Emotional_NLP"] = round(nlp.stress_score, 3)
            lime_profile = explain_case(model, features)

        return AssessmentResult(nlp=nlp, evaluation=evaluation, lime=lime_profile)

    def assess_many(
        self,
        cases: list[StudentCase],
        progress: Optional[callable] = None,
    ) -> list[AssessmentResult]:
        """Assess a batch of cases, loading the model once.

        ``progress(done, total)`` — if given — is called after each case so the
        GUI can drive a progress bar. Runs entirely offline.
        """
        self.ensure_model()
        results: list[AssessmentResult] = []
        total = len(cases)
        for index, case in enumerate(cases, start=1):
            results.append(self.assess(case))
            if progress is not None:
                progress(index, total)
        return results

    def local_plan(self, evaluation: RiskEvaluation, case: StudentCase | None = None) -> str:
        """Offline intervention plan (no API call) — used for batch reports."""
        from .llm_client import local_action_plan
        text, source = local_action_plan(
            evaluation,
            questionnaire_answers=(case.features if case else None),
            observation_text=(case.observation_text if case else None),
        )
        evaluation.action_plan_text = text
        evaluation.action_plan_source = source
        return text

    def generate_plan(
        self,
        evaluation: RiskEvaluation,
        case: StudentCase | None = None,
        api_key: Optional[str] = None,
        metric_out: Optional[list] = None,
    ) -> tuple[str, str]:
        """Generate the intervention plan and, if ``metric_out`` is given, append
        this call's performance metric (latency / tokens / cost) to it."""
        from .llm_client import generate_action_plan
        from .settings import get_knowledge_text
        text, source = generate_action_plan(
            evaluation,
            api_key=api_key,
            knowledge_text=get_knowledge_text() or None,
            questionnaire_answers=(case.features if case else None),
            observation_text=(case.observation_text if case else None),
            metric_out=metric_out,
        )
        evaluation.action_plan_text = text
        evaluation.action_plan_source = source
        return text, source

    def save(self, case: StudentCase, evaluation: RiskEvaluation) -> tuple[int, int]:
        from .database import Database
        with Database() as db:
            return db.save_assessment(case, evaluation)
