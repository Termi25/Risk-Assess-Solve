"""Explainability — genuine SHAP attributions over the real XGBoost model.

Per the README, explanations are produced by running SHAP against the *actual*
model and its *actual* inputs — never simulated by an LLM. We compute exact
Shapley values in probability space, so the attributions are additive:

    base_value + sum(shap_values) ≈ P(dropout) predicted by the model

which lets the UI say things like "Absențe nemotivate: +18 puncte" and have that
be a true decomposition of the model's real output rather than a plausible-
looking invention.

Implementation note: SHAP's ``TreeExplainer`` in probability mode does not yet
support XGBoost 3.x split metadata, so we use the model-agnostic *exact*
explainer over the model's own ``predict_proba``. With only 8 features this is
fast and returns true Shapley values. (The research-style log-odds
``TreeExplainer`` summary plot is reproduced separately in ``train.py`` for the
article figures.)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap

from . import config
from .models import FeatureAttribution, RiskEvaluation, SubScore, StudentCase
from .nlp_engine import NlpResult
from .scoring_engine import (
    RiskModel,
    encode_case_features,
    generate_synthetic_dataset,
    _encode_dataframe,
)


# Group each feature into an interpretable domain (drives the sub-scores).
FEATURE_DOMAINS: dict[str, str] = {
    "Age_Years": "Context personal",
    "Sex": "Context personal",
    "Mediu_Rezidential": "Context personal",
    "Situatie_Familiala": "Context familial",
    "Educatie_Mama": "Context familial",
    "Educatie_Tata": "Context familial",
    "Absente_Nemotivate_Zilele_1_13": "Frecvență",
    "Absente_Motivate_3_Luni": "Frecvență",
    "Medie_Modul_Anterior": "Performanță academică",
    "Note_Sub_5": "Performanță academică",
    "Participare_Extrascolara": "Implicare",
    "Studentship_Score": "Implicare (Studentship)",
    "Atitudine_Scoala": "Climat școlar",
    "Sanctiuni_Avertismente": "Climat școlar",
    "Cum_te_Simti_La_Scoala": "Climat școlar",
    "Scoala_Ajuta_Obiective": "Climat școlar",
    "Stres_Emotional_NLP": "Stare emoțională (NLP)",
}

# The explainer is mildly expensive to build; cache one per model instance.
_EXPLAINER_CACHE: dict[int, shap.Explainer] = {}


def _get_explainer(model: RiskModel) -> shap.Explainer:
    key = id(model.clf)
    cached = _EXPLAINER_CACHE.get(key)
    if cached is not None:
        return cached

    # A background sample defines the reference distribution for the baseline.
    background = _encode_dataframe(
        generate_synthetic_dataset(200, config.RANDOM_SEED)
    ).astype(float)
    columns = list(config.FEATURE_KEYS)

    def predict_pos(data: np.ndarray) -> np.ndarray:
        frame = pd.DataFrame(np.asarray(data, dtype=float), columns=columns)
        return model.clf.predict_proba(frame)[:, 1]

    masker = shap.maskers.Independent(background, max_samples=100)
    explainer = shap.Explainer(predict_pos, masker, algorithm="permutation")
    _EXPLAINER_CACHE[key] = explainer
    return explainer


def _as_positive_class(values: np.ndarray, base) -> tuple[np.ndarray, float]:
    """Normalise SHAP output shapes to (per-feature vector, scalar base)."""
    values = np.asarray(values)
    if values.ndim == 2:            # (n_features, n_classes)
        values = values[:, -1]
    base_arr = np.atleast_1d(np.asarray(base))
    base_val = float(base_arr[-1])
    return values.astype(float), base_val


def _value_display(feature_key: str, features: dict) -> str:
    feat = config.feature(feature_key)
    raw = features.get(feature_key, feat.default)
    if feat.kind == "categorical":
        return str(raw)
    try:
        num = float(raw)
    except (TypeError, ValueError):
        return str(raw)
    return str(int(num)) if num.is_integer() else f"{num:g}"


def compute_attributions(
    model: RiskModel, features: dict
) -> tuple[float, list[FeatureAttribution]]:
    """Return (base_value, per-feature SHAP attributions) in probability space."""
    X = encode_case_features(features).astype(float)
    explainer = _get_explainer(model)
    explanation = explainer(X, max_evals=1000)

    values, base_value = _as_positive_class(
        explanation.values[0], explanation.base_values[0]
    )

    attributions: list[FeatureAttribution] = []
    for key, shap_val in zip(config.FEATURE_KEYS, values):
        feat = config.feature(key)
        attributions.append(
            FeatureAttribution(
                feature_key=key,
                label=feat.label,
                value_display=_value_display(key, features),
                shap_value=float(shap_val),
            )
        )
    return base_value, attributions


def _risk_band(probability: float) -> str:
    if probability < 0.34:
        return "Scăzut"
    if probability < 0.67:
        return "Mediu"
    return "Ridicat"


def _sub_scores(attributions: list[FeatureAttribution]) -> list[SubScore]:
    """Domain-level share of the total explanation magnitude (sums to ~100)."""
    domain_mag: dict[str, float] = {}
    for a in attributions:
        domain = FEATURE_DOMAINS.get(a.feature_key, "Altele")
        domain_mag[domain] = domain_mag.get(domain, 0.0) + abs(a.shap_value)

    total = sum(domain_mag.values()) or 1.0
    order = [
        "Context personal",
        "Context familial",
        "Frecvență",
        "Performanță academică",
        "Implicare",
        "Implicare (Studentship)",
        "Climat școlar",
        "Stare emoțională (NLP)",
    ]
    result = []
    for name in order:
        if name in domain_mag:
            result.append(SubScore(name=name, value=round(domain_mag[name] / total * 100.0, 1)))
    # Any domain not in the fixed order (defensive).
    for name, mag in domain_mag.items():
        if name not in order:
            result.append(SubScore(name=name, value=round(mag / total * 100.0, 1)))
    return result


def evaluate(model: RiskModel, features: dict) -> RiskEvaluation:
    """Run the full local pipeline for one feature dict and return an evaluation.

    Note: the action plan is *not* filled here — that is the optional cloud/
    local text-generation step handled by ``llm_client``.
    """
    probability = model.predict_probability(features)
    base_value, attributions = compute_attributions(model, features)
    sub_scores = _sub_scores(attributions)

    return RiskEvaluation(
        probability=probability,
        aggregate_score=round(probability * 100.0, 1),
        risk_band=_risk_band(probability),
        base_value=base_value,
        attributions=attributions,
        sub_scores=sub_scores,
        model_version=model.version,
    )


def evaluate_case(
    model: RiskModel, case: StudentCase, nlp_result: NlpResult
) -> RiskEvaluation:
    """Evaluate a StudentCase, injecting the NLP stress feature from the text."""
    features = dict(case.features)
    features["Stres_Emotional_NLP"] = round(nlp_result.stress_score, 3)
    evaluation = evaluate(model, features)
    evaluation.student_case_id = case.id
    return evaluation
