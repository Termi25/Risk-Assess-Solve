"""Risk scoring engine — the REAL "Day 14" model.

This is the ``Scoring engine`` box from the README: a genuine XGBoost classifier
trained on class-balanced data (SMOTE-NC), producing a dropout-risk probability.
It is *not* an LLM pretending to be a model — the numbers come from a trained
gradient-boosted tree ensemble, exactly as in the research methodology.

Pipeline (mirrors the research script):
    synthetic SIIIR-like data  ->  encode categoricals  ->  train/test split
    ->  SMOTE-NC on the training set  ->  XGBoost  ->  metrics.

The trained booster is persisted as XGBoost-native JSON so it can be bundled
into the packaged exe and loaded without retraining on every launch.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from xgboost import XGBClassifier

from . import config


# --- Encoding ---------------------------------------------------------------
_QUESTIONNAIRE_TO_MODEL_KEYS = {
    "sex": "Sex",
    "residential_environment": "Mediu_Rezidential",
    "family_situation": "Situatie_Familiala",
    "mother_education": "Educatie_Mama",
    "father_education": "Educatie_Tata",
    "unexcused_absences_3m": "Absente_Nemotivate_Zilele_1_13",
    "excused_absences_3m": "Absente_Motivate_3_Luni",
    "extracurricular_participation": "Participare_Extrascolara",
    "previous_module_average": "Medie_Modul_Anterior",
    "school_attitude": "Atitudine_Scoala",
    "disciplinary_sanctions": "Sanctiuni_Avertismente",
    "school_feeling": "Cum_te_Simti_La_Scoala",
    "school_support_goal": "Scoala_Ajuta_Obiective",
}


def _coerce_float(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        if isinstance(value, bool):
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _parse_birth_date(value) -> date | None:
    if not value:
        return None
    if isinstance(value, date):
        return value
    if isinstance(value, datetime):
        return value.date()
    try:
        return datetime.fromisoformat(str(value)).date()
    except ValueError:
        try:
            return date.fromisoformat(str(value))
        except ValueError:
            return None


def _age_from_birth_date(value) -> float | None:
    birth = _parse_birth_date(value)
    if birth is None:
        return None
    today = date.today()
    years = today.year - birth.year - ((today.month, today.day) < (birth.month, birth.day))
    return float(max(0, years))


# Answers to Q13 that mean "no grades below 5".
_NO_LOW_GRADES = {
    "", "-", "—", "nu", "n/a", "na", "nu e cazul", "nu este cazul",
    "niciuna", "niciun", "nicio", "nimic", "0",
}
# Qualitative marks used in Romanian primary school (clasele 0–IV): only
# "Insuficient" is below the passing line — "Suficient"/"Bine"/"Foarte bine"
# all pass, so they must NOT be counted as grades below 5.
_FAILING_QUALIFIER_RE = re.compile(r"insuficient", re.IGNORECASE)


def _count_low_grades(value) -> int:
    """Number of grades below 5 described in the free-text answer to Q13.

    Robust to the two ways teachers fill this in:
      * numeric grades ("4 Matematică; 3 Română") -> distinct values below 5;
      * qualitative primary-school marks -> only "Insuficient" counts as a
        failing mark ("Suficient"/"Bine"/"Foarte bine" are passing).

    Descriptive text that names neither a grade below 5 nor an "Insuficient"
    mark (e.g. "Suficient, limba română, matematică") counts as zero. The older
    fallback split such text on commas and mis-read the passing mark and the
    subject names as three separate failing grades.
    """
    if not value:
        return 0
    text = str(value).strip()
    if text.lower() in _NO_LOW_GRADES:
        return 0

    # 1. Explicit numeric grades below 5 (distinct; Romanian grades are 1–10).
    grades: set[int] = set()
    for match in re.findall(r"\b\d+(?:[.,]\d+)?\b", text):
        try:
            grade = float(match.replace(",", "."))
        except ValueError:
            continue
        if 1.0 <= grade < 5.0:
            grades.add(int(grade))
    if grades:
        return len(grades)

    # 2. No numeric grade: count qualitative failing marks ("Insuficient").
    #    Passing marks and plain subject lists therefore yield zero.
    return len(_FAILING_QUALIFIER_RE.findall(text))


def _studentship_score(values: dict) -> float:
    participation = str(values.get("Participare_Extrascolara", values.get("extracurricular_participation", "")))
    attitude = str(values.get("Atitudine_Scoala", values.get("school_attitude", "")))
    feeling = str(values.get("Cum_te_Simti_La_Scoala", values.get("school_feeling", "")))
    support = str(values.get("Scoala_Ajuta_Obiective", values.get("school_support_goal", "")))
    sanctions = str(values.get("Sanctiuni_Avertismente", values.get("disciplinary_sanctions", "")))
    unexcused = _coerce_float(values.get("Absente_Nemotivate_Zilele_1_13", values.get("unexcused_absences_3m", 0.0)))
    excused = _coerce_float(values.get("Absente_Motivate_3_Luni", values.get("excused_absences_3m", 0.0)))
    low_grades = int(values.get("Note_Sub_5", _count_low_grades(values.get("low_grades_details"))))

    score = 0.0
    score += {"Da, frecvent": 2.0, "Ocazional": 1.0, "Nu": 0.0}.get(participation, 1.0)
    score += {"Pozitivă": 2.0, "Neutră": 1.0, "Negativă": 0.0}.get(attitude, 1.0)
    score += {"Bine": 2.0, "Neutru": 1.0, "Stresat": 0.0, "Izolat": 0.0, "Altul": 1.0}.get(feeling, 1.0)
    score += {"Da": 2.0, "Parțial": 1.0, "Nu": 0.0}.get(support, 1.0)
    score += {"Nu": 2.0, "Avertismente": 1.0, "Sancțiuni": 0.0}.get(sanctions, 1.0)
    score -= min(2.0, unexcused / 12.0)
    score -= min(1.0, excused / 20.0)
    score -= min(2.0, low_grades / 3.0)
    return float(max(0.0, min(10.0, round(score, 1))))


def compose_model_features(features: dict) -> dict[str, object]:
    """Derive the model feature set from the full questionnaire payload.

    Accepts either raw questionnaire answers (keyed by questionnaire keys) or an
    already-composed model-feature dict, and is idempotent for the latter: any
    model key explicitly present in the input is preserved instead of being
    re-derived. This lets a composed dict flow safely through both the encoder
    and the SHAP layer without ``Note_Sub_5`` / ``Studentship_Score`` resetting.
    """
    raw = dict(features)
    prepared: dict[str, object] = {}

    # 1. Map questionnaire answers onto model keys.
    for raw_key, model_key in _QUESTIONNAIRE_TO_MODEL_KEYS.items():
        value = raw.get(raw_key)
        if value not in (None, ""):
            prepared[model_key] = value

    # 2. Preserve any model keys already provided directly (idempotency): these
    #    take precedence over the derivations below.
    for feat in config.FEATURES:
        if feat.key not in prepared and raw.get(feat.key) not in (None, ""):
            prepared[feat.key] = raw[feat.key]

    # 3. Derive the remaining values only when still missing.
    if "Age_Years" not in prepared:
        age = _age_from_birth_date(raw.get("birth_date"))
        if age is not None:
            prepared["Age_Years"] = age

    if "Note_Sub_5" not in prepared:
        prepared["Note_Sub_5"] = _count_low_grades(raw.get("low_grades_details"))

    if "Studentship_Score" not in prepared:
        prepared["Studentship_Score"] = _studentship_score({**raw, **prepared})

    for feat in config.FEATURES:
        prepared.setdefault(feat.key, feat.default)
    return prepared


def encode_case_features(features: dict) -> pd.DataFrame:
    """Turn a raw feature dict into a single-row, model-ready DataFrame.

    Raw questionnaire answers are converted into the model feature set first,
    then categorical values are mapped to the integer codes the model was
    trained on. Column order is fixed to ``config.FEATURE_KEYS``.
    """
    features = compose_model_features(features)
    row: dict[str, float] = {}
    for feat in config.FEATURES:
        raw = features.get(feat.key)
        if feat.kind == "categorical":
            codes = config.CATEGORY_CODES[feat.key]
            if isinstance(raw, (int, float)) and not isinstance(raw, bool):
                row[feat.key] = float(int(raw))
            else:
                row[feat.key] = float(codes.get(str(raw), 0))
        else:
            row[feat.key] = float(raw if raw is not None else feat.default)
    return pd.DataFrame([row], columns=list(config.FEATURE_KEYS))


def _encode_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for key, mapping in config.CATEGORY_CODES.items():
        out[key] = out[key].map(mapping).astype(int)
    return out[list(config.FEATURE_KEYS)]


# --- Synthetic data (SIIIR-like) -------------------------------------------
def generate_synthetic_dataset(
    n_samples: int = config.TRAIN_SAMPLES,
    seed: int = config.RANDOM_SEED,
) -> pd.DataFrame:
    """Generate questionnaire-shaped student records with the non-linear 'Day 14' rules.

    Replaces the real SIIIR export used in production. Real deployment would
    swap this for ``pd.read_csv(<anonymised export>)`` with the same columns.
    """
    rng = np.random.default_rng(seed)

    age = rng.integers(11, 20, n_samples)
    sex = rng.choice(["Feminin", "Masculin", "Altul / prefer să nu spun"], n_samples, p=[0.49, 0.49, 0.02])
    mediu = rng.choice(["Urban", "Rural"], n_samples, p=[0.45, 0.55])
    familie = rng.choice(
        ["Ambii părinți", "Monoparental", "Tutore / plasament", "Altă situație"],
        n_samples,
        p=[0.62, 0.23, 0.1, 0.05],
    )
    mama = rng.choice(
        ["Primar", "Gimnazial", "Liceal", "Postliceal", "Universitar", "Necunoscut"],
        n_samples,
        p=[0.08, 0.23, 0.32, 0.12, 0.22, 0.03],
    )
    tata = rng.choice(
        ["Primar", "Gimnazial", "Liceal", "Postliceal", "Universitar", "Necunoscut"],
        n_samples,
        p=[0.09, 0.24, 0.31, 0.11, 0.22, 0.03],
    )
    # Absences on the questionnaire's 3-month, per-class-hour scale (right-skewed:
    # most students low, a disengaged tail into the hundreds). The old 0–21 range
    # made any real 3-month total out-of-distribution and saturated the model.
    absente_ne = np.clip(rng.gamma(1.4, 34.0, n_samples), 0, 300).round().astype(int)
    absente_mot = np.clip(rng.gamma(1.5, 22.0, n_samples), 0, 200).round().astype(int)
    extrac = rng.choice(["Da, frecvent", "Ocazional", "Nu"], n_samples, p=[0.36, 0.34, 0.3])
    medie = np.clip(rng.normal(7.1, 1.4, n_samples), 1.0, 10.0).round(1)
    note_sub_5 = rng.integers(0, 6, n_samples)
    attitude = rng.choice(["Pozitivă", "Neutră", "Negativă"], n_samples, p=[0.48, 0.34, 0.18])
    sanctions = rng.choice(["Nu", "Avertismente", "Sancțiuni"], n_samples, p=[0.74, 0.18, 0.08])
    feeling = rng.choice(["Bine", "Neutru", "Stresat", "Izolat", "Altul"], n_samples, p=[0.38, 0.28, 0.17, 0.11, 0.06])
    help_goal = rng.choice(["Da", "Parțial", "Nu"], n_samples, p=[0.42, 0.33, 0.25])

    studentship = []
    stress = []
    for i in range(n_samples):
        score = 0.0
        score += {"Da, frecvent": 2.0, "Ocazional": 1.0, "Nu": 0.0}[extrac[i]]
        score += {"Pozitivă": 2.0, "Neutră": 1.0, "Negativă": 0.0}[attitude[i]]
        score += {"Bine": 2.0, "Neutru": 1.0, "Stresat": 0.0, "Izolat": 0.0, "Altul": 1.0}[feeling[i]]
        score += {"Da": 2.0, "Parțial": 1.0, "Nu": 0.0}[help_goal[i]]
        score += {"Nu": 2.0, "Avertismente": 1.0, "Sancțiuni": 0.0}[sanctions[i]]
        score -= min(2.0, absente_ne[i] / 12.0)
        score -= min(1.0, absente_mot[i] / 20.0)
        score -= min(2.0, note_sub_5[i] / 3.0)
        studentship.append(float(max(0.0, min(10.0, round(score, 1)))))

        stress_base = 0.15
        stress_base += 0.55 if feeling[i] in {"Stresat", "Izolat"} else 0.0
        stress_base += 0.35 if sanctions[i] != "Nu" else 0.0
        stress_base += 0.2 if attitude[i] == "Negativă" else 0.0
        stress_base += 0.2 if help_goal[i] == "Nu" else 0.0
        stress.append(float(np.clip(rng.normal(stress_base, 0.18), 0.0, 2.0)))

    df = pd.DataFrame(
        {
            "Age_Years": age,
            "Sex": sex,
            "Mediu_Rezidential": mediu,
            "Situatie_Familiala": familie,
            "Educatie_Mama": mama,
            "Educatie_Tata": tata,
            "Absente_Nemotivate_Zilele_1_13": absente_ne,
            "Absente_Motivate_3_Luni": absente_mot,
            "Participare_Extrascolara": extrac,
            "Medie_Modul_Anterior": medie,
            "Note_Sub_5": note_sub_5,
            "Studentship_Score": studentship,
            "Atitudine_Scoala": attitude,
            "Sanctiuni_Avertismente": sanctions,
            "Cum_te_Simti_La_Scoala": feeling,
            "Scoala_Ajuta_Obiective": help_goal,
            "Stres_Emotional_NLP": np.round(stress, 2),
        }
    )

    # Graded "Day 14" risk: the dropout probability is a smooth (logistic)
    # function of the drivers, so the trained model yields a calibrated spread
    # across the four risk tiers instead of the old hard-threshold rules, which
    # saturated every student with more than 12 absences to ~100%.
    #
    # The coefficients (calibrated in _RISK_LOGIT) encode the intended structure:
    # absences dominate only at *extreme* levels (a saturating term), while a
    # good module average, a high Studentship score and low emotional stress are genuine
    # protective factors. This reproduces the counter-intuitive but correct
    # ordering where a student with more absences but a solid average and only
    # stress-driven disengagement ranks *below* one with fewer absences but
    # sanctions, no support and total disinterest.
    prob = _risk_probability(df)
    df[config.TARGET_COLUMN] = (rng.random(n_samples) < prob).astype(int)
    return df


# Calibrated logistic risk model behind the synthetic labels. Kept as a module
# constant so the same structure documents the "Day 14" scoring rationale.
_RISK_LOGIT = {
    "intercept": -1.877,
    "unexcused": 1.074,   # x saturating min(3.5, unexcused/70) — extreme absences dominate
    "excused": 0.044,     # x saturating min(1.0, excused/80); justified absences weigh little
    "avg_dev": -1.137,    # per point of (module average - 6.5); a strong protective factor
    "studentship": -0.098,
    "attitude": 0.337,    # Negativă +1 / Neutră 0 / Pozitivă -1
    "participation": 0.356,   # Nu +1 / Ocazional 0 / Da, frecvent -1
    "sanctions": 0.626,   # Nu 0 / Avertismente +1 / Sancțiuni +2
    "support": 0.490,     # Nu +1 / Parțial 0 / Da -1
    "stress": 0.316,      # Stres_Emotional_NLP in [0, 2]
    "family": 0.247,      # Ambii părinți -1 / Monoparental +0.5 / Tutore +1 / Altă +0.7
    "education": 0.233,   # mean of mother/father education risk
    "age": 0.085,         # per year above 16 (older-in-cohort → higher risk)
}
_ATTITUDE_RISK = {"Negativă": 1.0, "Neutră": 0.0, "Pozitivă": -1.0}
_PARTICIPATION_RISK = {"Nu": 1.0, "Ocazional": 0.0, "Da, frecvent": -1.0}
_SANCTIONS_RISK = {"Nu": 0.0, "Avertismente": 1.0, "Sancțiuni": 2.0}
_SUPPORT_RISK = {"Nu": 1.0, "Parțial": 0.0, "Da": -1.0}
_FAMILY_RISK = {"Ambii părinți": -1.0, "Monoparental": 0.5, "Tutore / plasament": 1.0, "Altă situație": 0.7}
_EDUCATION_RISK = {
    "Necunoscut": 0.5, "Primar": 0.5, "Gimnazial": 0.2,
    "Liceal": 0.0, "Postliceal": -0.3, "Universitar": -0.3,
}


def _risk_probability(df: pd.DataFrame) -> np.ndarray:
    """Dropout probability P(abandon) for each row, from the calibrated logit."""
    b = _RISK_LOGIT
    ne = df["Absente_Nemotivate_Zilele_1_13"].to_numpy(dtype=float)
    mot = df["Absente_Motivate_3_Luni"].to_numpy(dtype=float)
    edu = (
        df["Educatie_Mama"].map(_EDUCATION_RISK).to_numpy(dtype=float)
        + df["Educatie_Tata"].map(_EDUCATION_RISK).to_numpy(dtype=float)
    ) / 2.0
    z = (
        b["intercept"]
        + b["unexcused"] * np.minimum(3.5, ne / 70.0)
        + b["excused"] * np.minimum(1.0, mot / 80.0)
        + b["avg_dev"] * (df["Medie_Modul_Anterior"].to_numpy(dtype=float) - 6.5)
        + b["studentship"] * df["Studentship_Score"].to_numpy(dtype=float)
        + b["attitude"] * df["Atitudine_Scoala"].map(_ATTITUDE_RISK).to_numpy(dtype=float)
        + b["participation"] * df["Participare_Extrascolara"].map(_PARTICIPATION_RISK).to_numpy(dtype=float)
        + b["sanctions"] * df["Sanctiuni_Avertismente"].map(_SANCTIONS_RISK).to_numpy(dtype=float)
        + b["support"] * df["Scoala_Ajuta_Obiective"].map(_SUPPORT_RISK).to_numpy(dtype=float)
        + b["stress"] * df["Stres_Emotional_NLP"].to_numpy(dtype=float)
        + b["family"] * df["Situatie_Familiala"].map(_FAMILY_RISK).to_numpy(dtype=float)
        + b["education"] * edu
        + b["age"] * np.maximum(0.0, df["Age_Years"].to_numpy(dtype=float) - 16.0)
    )
    return 1.0 / (1.0 + np.exp(-z))


# --- Trained model wrapper --------------------------------------------------
@dataclass
class TrainingMetrics:
    accuracy: float
    roc_auc: float
    n_before: int
    n_after: int
    balance_before: dict[str, int]
    balance_after: dict[str, int]
    report_text: str
    # --- extended conventional metrics (the dropout-literature standard) ----
    # All computed on the un-resampled held-out test set, so they describe the
    # model's behaviour on the real class balance (SMOTE-NC touches only train).
    pr_auc: float = float("nan")            # average precision (PR-AUC)
    balanced_accuracy: float = float("nan")
    precision_dropout: float = float("nan")  # positive ("Abandon") class
    recall_dropout: float = float("nan")     # = sensitivity
    specificity: float = float("nan")        # recall of the negative class
    f1_dropout: float = float("nan")
    g_mean: float = float("nan")             # sqrt(sensitivity * specificity)
    mcc: float = float("nan")                # Matthews correlation coefficient
    brier: float = float("nan")             # calibration error (lower is better)
    confusion: list = field(default_factory=list)   # [[tn, fp], [fn, tp]]
    # --- stratified k-fold CV (SMOTE-NC re-fit inside each fold; mean ± SD) --
    cv_folds: int = 0
    cv_accuracy_mean: float = float("nan")
    cv_accuracy_std: float = float("nan")
    cv_roc_auc_mean: float = float("nan")
    cv_roc_auc_std: float = float("nan")
    cv_pr_auc_mean: float = float("nan")
    cv_pr_auc_std: float = float("nan")
    cv_f1_mean: float = float("nan")
    cv_f1_std: float = float("nan")
    # --- reproduction parameters (so metric images regenerate the exact split)
    n_samples: int = 0
    seed: int = 0


@dataclass
class ModelMeta:
    model_version: str
    trained_at: str
    feature_keys: list[str] = field(default_factory=lambda: list(config.FEATURE_KEYS))
    categorical_indices: list[int] = field(default_factory=lambda: list(config.CATEGORICAL_INDICES))
    metrics: dict | None = None


class RiskModel:
    """A trained XGBoost classifier plus the metadata SHAP and the UI need."""

    def __init__(self, clf: XGBClassifier, meta: ModelMeta):
        self.clf = clf
        self.meta = meta

    @property
    def version(self) -> str:
        return self.meta.model_version

    def predict_probability(self, features: dict) -> float:
        """Return P(dropout) in [0, 1] for one raw feature dict."""
        X = encode_case_features(features)
        proba = self.clf.predict_proba(X)[0, 1]
        return float(proba)

    def predict_frame(self, X: pd.DataFrame) -> np.ndarray:
        return self.clf.predict_proba(X)[:, 1]


# --- Training ---------------------------------------------------------------
# Test-set fraction and CV fold count. TEST_SIZE is a module constant so the
# train/test partition can be reproduced identically for the metric plots
# (``holdout_predictions``), and CV_FOLDS documents the k reported in the paper.
TEST_SIZE = 0.25
CV_FOLDS = 5


def _make_classifier(seed: int) -> XGBClassifier:
    """The single source of truth for the model's hyper-parameters, shared by
    the final fit and the cross-validation pipeline so both agree."""
    return XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=seed,
    )


def _prepare_xy(n_samples: int, seed: int) -> tuple[pd.DataFrame, pd.Series]:
    """Regenerate the encoded synthetic dataset (features X, target y)."""
    df = generate_synthetic_dataset(n_samples, seed)
    X = _encode_dataframe(df[list(config.FEATURE_KEYS)])
    y = df[config.TARGET_COLUMN].astype(int)
    return X, y


def _smote_k(minority: int) -> int:
    """A safe ``k_neighbors`` for SMOTE-NC: strictly below the minority count."""
    return max(1, min(5, minority - 1))


def _cross_val_metrics(X: pd.DataFrame, y: pd.Series, seed: int, folds: int) -> dict:
    """Stratified k-fold CV with SMOTE-NC re-fit *inside* each fold.

    Resampling lives in an imbalanced-learn pipeline so it only ever sees each
    fold's training partition — evaluating on untouched, real-balance folds and
    avoiding the optimistic leakage of oversampling before the split. Returns
    mean ± SD for accuracy, ROC-AUC, PR-AUC and F1 (empty on failure)."""
    try:
        from imblearn.pipeline import Pipeline as ImbPipeline

        # Guard k against the smallest minority a training fold can hold.
        fold_minority = int(y.value_counts().min() * (folds - 1) / folds)
        pipe = ImbPipeline([
            ("smote", SMOTENC(
                categorical_features=config.CATEGORICAL_INDICES,
                random_state=seed,
                k_neighbors=_smote_k(fold_minority),
            )),
            ("clf", _make_classifier(seed)),
        ])
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        scores = cross_validate(
            pipe, X, y, cv=cv,
            scoring={
                "accuracy": "accuracy",
                "roc_auc": "roc_auc",
                "pr_auc": "average_precision",
                "f1": "f1",
            },
        )
    except Exception:
        return {}

    def pair(metric: str) -> tuple[float, float]:
        arr = np.asarray(scores[f"test_{metric}"], dtype=float)
        return float(arr.mean()), float(arr.std())

    acc_m, acc_s = pair("accuracy")
    auc_m, auc_s = pair("roc_auc")
    pr_m, pr_s = pair("pr_auc")
    f1_m, f1_s = pair("f1")
    return {
        "cv_folds": folds,
        "cv_accuracy_mean": acc_m, "cv_accuracy_std": acc_s,
        "cv_roc_auc_mean": auc_m, "cv_roc_auc_std": auc_s,
        "cv_pr_auc_mean": pr_m, "cv_pr_auc_std": pr_s,
        "cv_f1_mean": f1_m, "cv_f1_std": f1_s,
    }


def train_model(
    n_samples: int = config.TRAIN_SAMPLES,
    seed: int = config.RANDOM_SEED,
    cv_folds: int = CV_FOLDS,
) -> tuple[RiskModel, TrainingMetrics]:
    """Train the XGBoost dropout-risk model with SMOTE-NC balancing.

    Besides the headline accuracy/ROC-AUC, this reports the conventional
    imbalanced-classification suite used across the dropout literature (PR-AUC,
    balanced accuracy, sensitivity/specificity, G-mean, MCC, Brier calibration
    and the confusion matrix), plus a stratified ``cv_folds``-fold cross-
    validation (pass ``cv_folds=0`` to skip it). All test-set figures are on the
    un-resampled hold-out."""
    X, y = _prepare_xy(n_samples, seed)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=seed, stratify=y
    )

    balance_before = {str(k): int(v) for k, v in y_train.value_counts().items()}
    n_before = int(len(y_train))

    # SMOTE-NC needs k_neighbors < size of the minority class.
    minority = int(y_train.value_counts().min())
    smote = SMOTENC(
        categorical_features=config.CATEGORICAL_INDICES,
        random_state=seed,
        k_neighbors=_smote_k(minority),
    )
    resampled = smote.fit_resample(X_train, y_train)
    X_res = resampled[0]
    y_res = resampled[1]

    balance_after = {str(k): int(v) for k, v in pd.Series(np.asarray(y_res)).value_counts().items()}
    n_after = int(len(y_res))

    clf = _make_classifier(seed)
    clf.fit(X_res, y_res)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    accuracy = float(accuracy_score(y_test, y_pred))
    try:
        roc_auc = float(roc_auc_score(y_test, y_proba))
    except ValueError:
        roc_auc = float("nan")
    report_text = str(classification_report(y_test, y_pred, zero_division=0))

    # Confusion matrix (fixed [0, 1] order) and the derived rate metrics.
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in cm.ravel())
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    sensitivity = tp / (tp + fn) if (tp + fn) else float("nan")
    if np.isnan(sensitivity) or np.isnan(specificity):
        g_mean = float("nan")
    else:
        g_mean = float(np.sqrt(max(0.0, sensitivity) * max(0.0, specificity)))
    try:
        pr_auc = float(average_precision_score(y_test, y_proba))
    except ValueError:
        pr_auc = float("nan")

    metrics = TrainingMetrics(
        accuracy=accuracy,
        roc_auc=roc_auc,
        n_before=n_before,
        n_after=n_after,
        balance_before=balance_before,
        balance_after=balance_after,
        report_text=report_text,
        pr_auc=pr_auc,
        balanced_accuracy=float(balanced_accuracy_score(y_test, y_pred)),
        precision_dropout=float(precision_score(y_test, y_pred, pos_label=1, zero_division=0)),
        recall_dropout=float(sensitivity),
        specificity=float(specificity),
        f1_dropout=float(f1_score(y_test, y_pred, pos_label=1, zero_division=0)),
        g_mean=g_mean,
        mcc=float(matthews_corrcoef(y_test, y_pred)),
        brier=float(brier_score_loss(y_test, y_proba)),
        confusion=[[tn, fp], [fn, tp]],
        n_samples=int(n_samples),
        seed=int(seed),
    )

    if cv_folds and cv_folds > 1:
        for key, value in _cross_val_metrics(X, y, seed, cv_folds).items():
            setattr(metrics, key, value)

    version = f"xgb-day14-n{n_samples}-s{seed}-{datetime.now():%Y%m%d}"
    meta = ModelMeta(
        model_version=version,
        trained_at=datetime.now().isoformat(timespec="seconds"),
        metrics=asdict(metrics),
    )
    return RiskModel(clf, meta), metrics


def holdout_predictions(
    model: RiskModel,
    n_samples: int = config.TRAIN_SAMPLES,
    seed: int = config.RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reproduce the held-out test set and the model's predictions on it.

    Deterministic: the same ``seed`` regenerates the same synthetic data and the
    same stratified split used during training, so the returned arrays match the
    reported test metrics exactly. This is what the metric plots draw from,
    without persisting any arrays. Returns ``(y_true, y_pred, y_proba)``."""
    X, y = _prepare_xy(n_samples, seed)
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=seed, stratify=y
    )
    y_proba = model.clf.predict_proba(X_test)[:, 1]
    y_pred = model.clf.predict(X_test)
    return y_test.to_numpy(), np.asarray(y_pred), np.asarray(y_proba, dtype=float)


# --- Persistence ------------------------------------------------------------
def save_model(model: RiskModel, model_path: Path, meta_path: Path) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.clf.save_model(str(model_path))
    meta_path.write_text(json.dumps(asdict(model.meta), indent=2, ensure_ascii=False), encoding="utf-8")


def load_model(model_path: Path, meta_path: Path) -> RiskModel:
    clf = XGBClassifier()
    clf.load_model(str(model_path))
    meta_dict = json.loads(meta_path.read_text(encoding="utf-8"))
    meta = ModelMeta(
        model_version=meta_dict.get("model_version", "unknown"),
        trained_at=meta_dict.get("trained_at", ""),
        feature_keys=meta_dict.get("feature_keys", list(config.FEATURE_KEYS)),
        categorical_indices=meta_dict.get("categorical_indices", list(config.CATEGORICAL_INDICES)),
        metrics=meta_dict.get("metrics"),
    )
    return RiskModel(clf, meta)


def _model_schema_matches(model: RiskModel) -> bool:
    return list(model.meta.feature_keys) == list(config.FEATURE_KEYS)


def get_or_train_model(force_retrain: bool = False) -> RiskModel:
    """Return a usable model, training + caching one on first run.

    Resolution order:
      1. A model bundled inside the app (shipped with the exe).
      2. A model previously trained into the user's data directory.
      3. Train a fresh model, save it to the user directory, and return it.
    """
    if not force_retrain:
        if config.bundled_model_path().exists() and config.bundled_meta_path().exists():
            bundled = load_model(config.bundled_model_path(), config.bundled_meta_path())
            if _model_schema_matches(bundled):
                return bundled
        if config.user_model_path().exists() and config.user_meta_path().exists():
            user_model = load_model(config.user_model_path(), config.user_meta_path())
            if _model_schema_matches(user_model):
                return user_model

    model, _ = train_model()
    save_model(model, config.user_model_path(), config.user_meta_path())
    return model
