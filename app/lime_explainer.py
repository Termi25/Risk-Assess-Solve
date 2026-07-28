"""LIME — the individual student's local risk profile.

Division of labour with :mod:`app.explainability` (SHAP):

    SHAP  ->  *global* feature importance (the research figure) and the additive
              per-case decomposition that feeds the domain sub-scores and the
              anonymised LLM payload.
    LIME  ->  the *individual* student's risk profile: a short, human-readable
              rule list ("media modulului ≤ 6.20 → crește riscul") that a
              teacher can read off the report without knowing what a Shapley
              value is.

Why the two are not interchangeable
-----------------------------------
SHAP values here are an exact additive decomposition in probability space:
``base_value + Σ shap ≈ P(dropout)``. LIME is a *local surrogate* — it perturbs
the student's row, weights the perturbations by proximity, and fits a sparse
linear model to that neighbourhood. Its coefficients answer "what locally
distinguishes this student" rather than "how is this exact probability
composed", and they do **not** sum to the model's output. They must therefore
never be substituted into the sub-scores or the LLM prompt, both of which rely
on additivity.

Because a local surrogate can fit badly, every explanation carries its own
fidelity (R² over the perturbed neighbourhood) plus the surrogate's local
prediction next to the model's real probability. A rule list with a weak fit is
still shown, but it is shown *labelled as weak* rather than silently trusted.

The dependency is imported lazily and the whole module degrades to ``None`` when
``lime`` is absent, so the packaged app keeps working without it.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from . import config
from .models import LimeCondition, LimeExplanation
from .scoring_engine import (
    RiskModel,
    _encode_dataframe,
    compose_model_features,
    encode_case_features,
    generate_synthetic_dataset,
)
from .timing import StageTimings, measure

# Perturbations drawn per explanation. 5000 is the value used across the LIME
# literature; with 17 features and a batched XGBoost predict it costs ~10 ms.
NUM_SAMPLES = 5000
# Rules shown per student. Enough to cover the drivers without turning the
# report page into a wall of conditions.
NUM_FEATURES = 8
# Rows of the synthetic distribution used to fit LIME's discretiser.
BACKGROUND_SAMPLES = 800

_CLASS_NAMES = ("Menținere", "Risc abandon")

# Every categorical value the model knows, for translating the value side of a
# LIME equality rule. Built once from the feature schema.
_CATEGORY_VALUES: tuple[str, ...] = tuple(
    {value for feature in config.FEATURES for value in feature.categories}
)

# One explainer per model instance — fitting the discretiser is the expensive
# part and it depends only on the background distribution, not on the student.
_EXPLAINER_CACHE: dict[int, object] = {}


def _humanise(condition: str) -> str:
    """Replace raw feature keys inside a LIME rule with their display labels.

    LIME phrases rules using the ``feature_names`` it was given (the model's
    internal column names), e.g. ``"Medie_Modul_Anterior <= 6.20"``. Feeding it
    the display labels directly would break the index mapping we rely on, so the
    substitution happens here instead, longest key first so no key that is a
    prefix of another is partially replaced.
    """
    from .i18n import tr, tr_value

    for key in sorted(config.FEATURE_KEYS, key=len, reverse=True):
        if key in condition:
            condition = condition.replace(key, tr(config.feature(key).label))
    # Categorical rules read "Label=Valoare"; translate the value side too, or an
    # English rule list would still say "Attitude towards school=Negativă".
    for value in sorted(_CATEGORY_VALUES, key=len, reverse=True):
        if value in condition:
            condition = condition.replace(value, tr_value(value))
    # LIME writes categorical equalities as "Label=Value"; a spaced "=" reads
    # better next to the numeric rules. The lookarounds keep the "=" of a "<="
    # or ">=" comparison intact — spacing those would render "< =".
    return re.sub(r"(?<![<>=!])\s*=\s*(?!=)", " = ", condition).strip()


def _build_explainer(model: RiskModel, seed: int, timings: StageTimings | None = None):
    """Fit (and cache) a ``LimeTabularExplainer`` over the training distribution."""
    key = id(model.clf)
    cached = _EXPLAINER_CACHE.get(key)
    if cached is not None:
        return cached

    from lime.lime_tabular import LimeTabularExplainer

    # Cache miss: fitting the discretiser over the background distribution is the
    # expensive one-time cost, kept out of the per-student ``lime_s`` so a batch's
    # first student does not carry it alone. See :mod:`app.timing`.
    with measure(timings, "lime_init_s"):
        background = _encode_dataframe(
            generate_synthetic_dataset(BACKGROUND_SAMPLES, seed)
        ).astype(float).to_numpy()

        categorical_indices = list(config.CATEGORICAL_INDICES)
        explainer = LimeTabularExplainer(
            background,
            feature_names=list(config.FEATURE_KEYS),
            categorical_features=categorical_indices,
            # Lets LIME print "Participare extrașcolară = Nu" instead of "= 2.0".
            categorical_names={
                i: list(config.FEATURES[i].categories) for i in categorical_indices
            },
            class_names=list(_CLASS_NAMES),
            mode="classification",
            discretize_continuous=True,
            random_state=seed,
        )
        _EXPLAINER_CACHE[key] = explainer
    return explainer


def explain_case(
    model: RiskModel,
    features: dict,
    *,
    num_features: int = NUM_FEATURES,
    num_samples: int = NUM_SAMPLES,
    seed: int = config.RANDOM_SEED,
    timings: StageTimings | None = None,
) -> LimeExplanation | None:
    """Local LIME profile for one student, or ``None`` if ``lime`` is unavailable.

    ``features`` accepts either raw questionnaire answers or an already-composed
    model-feature dict — both are normalised through ``compose_model_features``,
    exactly as the SHAP path does, so the two explanations always describe the
    same input row.

    ``timings`` (optional) collects the surrogate's wall-clock cost, split into
    the one-time explainer construction and this student's fit.
    """
    try:
        explainer = _build_explainer(model, seed, timings)
    except ImportError:
        return None  # lime not installed — the report simply omits the section

    model_features = compose_model_features(features)
    columns = list(config.FEATURE_KEYS)
    row = encode_case_features(model_features).astype(float).to_numpy()[0]

    def predict_fn(data: np.ndarray) -> np.ndarray:
        frame = pd.DataFrame(np.asarray(data, dtype=float), columns=columns)
        return model.clf.predict_proba(frame)

    try:
        with measure(timings, "lime_s"):
            explanation = explainer.explain_instance(
                row,
                predict_fn,
                num_features=num_features,
                num_samples=num_samples,
                labels=(1,),
            )
    except Exception:
        return None  # a failed surrogate must never sink an assessment

    # as_map() gives (feature_index, weight); as_list() gives the matching rule
    # strings in the same order. Zipping them keeps the key mapping exact even
    # after the labels are humanised.
    weights = explanation.as_map().get(1, [])
    rules = explanation.as_list(label=1)
    conditions: list[LimeCondition] = []
    for (index, weight), (condition, _w) in zip(weights, rules):
        key = columns[int(index)]
        conditions.append(
            LimeCondition(
                feature_key=key,
                label=config.feature(key).label,
                condition=_humanise(str(condition)),
                weight=float(weight),
            )
        )

    return LimeExplanation(
        conditions=conditions,
        intercept=float(explanation.intercept[1]),
        local_prediction=float(np.ravel(explanation.local_pred)[0]),
        model_probability=float(model.clf.predict_proba(
            pd.DataFrame([row], columns=columns)
        )[0, 1]),
        fidelity_r2=float(explanation.score),
        num_samples=int(num_samples),
    )
