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
    compose_model_features,
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
    # Compose first so the displayed "valoare" reflects the student's actual
    # model inputs — the real app passes questionnaire-keyed answers, whose
    # values live under different keys until they are mapped here.
    model_features = compose_model_features(features)
    X = encode_case_features(model_features).astype(float)
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
                value_display=_value_display(key, model_features),
                shap_value=float(shap_val),
            )
        )
    return base_value, attributions


# --- Risk classification (4 tiers + urgency, per the research prioritization)
# Thresholds that make a factor a *severe* contributor. A base "Ridicat" case is
# escalated to "Critic" when enough severe factors compound (mirroring the
# figure's CRITICAL alert: extreme absences + academic failure + disengagement).
_SEVERE_UNEXCUSED = 20        # unexcused absences (3 luni) signalling a crisis
_LOW_STUDENTSHIP = 2.0        # engagement at or below this is severe
_ACADEMIC_FAIL = 5.0          # average below the pass threshold
_CRISIS_STRESS = 1.3          # NLP emotional-stress level signalling a crisis


def _num(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _critical_indicators(mf: dict) -> list[str]:
    """Short, human-readable risk statements (the figure's 'Critical Indicators').

    Ordered by severity so the report can show the most important few first.
    ``mf`` is a composed model-feature dict.
    """
    out: list[str] = []
    unexcused = _num(mf.get("Absente_Nemotivate_Zilele_1_13"))
    if unexcused >= _SEVERE_UNEXCUSED:
        out.append(f"Absenteism cronic ({int(unexcused)} absențe nemotivate / 3 luni)")
    elif unexcused > 12:
        out.append(f"Absențe nemotivate ridicate ({int(unexcused)} / 3 luni)")

    studentship = _num(mf.get("Studentship_Score"), 10.0)
    if studentship <= _LOW_STUDENTSHIP:
        out.append(f"Implicare (Studentship) foarte scăzută ({studentship:g}/10)")

    medie = _num(mf.get("Medie_Modul_Anterior"), 10.0)
    if medie < _ACADEMIC_FAIL:
        out.append(f"Medie sub pragul de promovare ({medie:g})")

    note_sub5 = _num(mf.get("Note_Sub_5"))
    if note_sub5 >= 3:
        out.append(f"Note multiple sub 5 ({int(note_sub5)})")

    feeling = str(mf.get("Cum_te_Simti_La_Scoala", ""))
    stress = _num(mf.get("Stres_Emotional_NLP"))
    if feeling in {"Stresat", "Izolat"} or stress >= _CRISIS_STRESS:
        out.append("Stare emoțională vulnerabilă (stres / izolare)")

    if str(mf.get("Sanctiuni_Avertismente", "")) == "Sancțiuni":
        out.append("Sancțiuni disciplinare active")

    if str(mf.get("Atitudine_Scoala", "")) == "Negativă":
        out.append("Atitudine negativă față de școală")

    if str(mf.get("Scoala_Ajuta_Obiective", "")) == "Nu":
        out.append("Nu percepe sprijin pentru obiectivele personale")

    if str(mf.get("Participare_Extrascolara", "")) == "Nu":
        out.append("Fără participare extrașcolară")

    if str(mf.get("Situatie_Familiala", "")) not in {"Ambii părinți", ""}:
        out.append("Sprijin familial redus (situație monoparentală / tutore)")

    return out


def _severe_factor_count(probability: float, mf: dict) -> int:
    """How many crisis-level factors compound for this student."""
    severe = 0
    if _num(mf.get("Absente_Nemotivate_Zilele_1_13")) >= _SEVERE_UNEXCUSED:
        severe += 1
    if _num(mf.get("Medie_Modul_Anterior"), 10.0) < _ACADEMIC_FAIL:
        severe += 1
    if _num(mf.get("Studentship_Score"), 10.0) <= _LOW_STUDENTSHIP:
        severe += 1
    if str(mf.get("Sanctiuni_Avertismente", "")) == "Sancțiuni":
        severe += 1
    if (
        str(mf.get("Cum_te_Simti_La_Scoala", "")) in {"Stresat", "Izolat"}
        or _num(mf.get("Stres_Emotional_NLP")) >= _CRISIS_STRESS
    ):
        severe += 1
    return severe


def classify_risk(probability: float, model_features: dict) -> config.RiskTier:
    """Return the 4-tier risk level (with urgency), escalating to Critic when
    a high-risk case is compounded by several severe factors."""
    base = config.tier_for_probability(probability)
    if base.band == "Ridicat" and _severe_factor_count(probability, model_features) >= 3:
        return config.tier_for_band("Critic")
    return base


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
    model_features = compose_model_features(features)
    probability = model.predict_probability(model_features)
    base_value, attributions = compute_attributions(model, model_features)
    sub_scores = _sub_scores(attributions)
    tier = classify_risk(probability, model_features)

    return RiskEvaluation(
        probability=probability,
        aggregate_score=round(probability * 100.0, 1),
        risk_band=tier.band,
        urgency=tier.urgency,
        studentship_score=_num(model_features.get("Studentship_Score"), 0.0),
        critical_indicators=_critical_indicators(model_features),
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
