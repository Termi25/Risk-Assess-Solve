"""Assessment orchestration: NLP -> scoring -> SHAP -> (optional) plan -> save.

Keeps the GUI thin. All the heavy imports (xgboost, shap) live behind this
service so the window can construct instantly and load the model lazily.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .models import RiskEvaluation, StudentCase
from .nlp_engine import NlpResult, analyze


@dataclass
class AssessmentResult:
    nlp: NlpResult
    evaluation: RiskEvaluation


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

    def assess(self, case: StudentCase) -> AssessmentResult:
        from .explainability import evaluate_case
        model = self.ensure_model()
        nlp = analyze(case.observation_text)
        evaluation = evaluate_case(model, case, nlp)
        return AssessmentResult(nlp=nlp, evaluation=evaluation)

    def generate_plan(
        self,
        evaluation: RiskEvaluation,
        case: StudentCase | None = None,
        api_key: Optional[str] = None,
    ) -> tuple[str, str]:
        from .llm_client import generate_action_plan
        from .settings import get_knowledge_text
        text, source = generate_action_plan(
            evaluation,
            api_key=api_key,
            knowledge_text=get_knowledge_text() or None,
            questionnaire_answers=(case.features if case else None),
            observation_text=(case.observation_text if case else None),
        )
        evaluation.action_plan_text = text
        evaluation.action_plan_source = source
        return text, source

    def save(self, case: StudentCase, evaluation: RiskEvaluation) -> tuple[int, int]:
        from .database import Database
        with Database() as db:
            return db.save_assessment(case, evaluation)
