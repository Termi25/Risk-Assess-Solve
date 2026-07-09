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
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
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


def _count_low_grades(value) -> int:
    if not value:
        return 0
    text = str(value)
    grades: set[int] = set()
    for match in re.findall(r"\b\d+(?:[.,]\d+)?\b", text):
        try:
            grade = float(match.replace(",", "."))
        except ValueError:
            continue
        if grade < 5.0:
            grades.add(int(grade))
    if grades:
        return len(grades)
    parts = [chunk.strip() for chunk in re.split(r"[;\n,|]+", text) if chunk.strip()]
    return len(parts)


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
    """Derive the model feature set from the full questionnaire payload."""
    raw = dict(features)
    prepared: dict[str, object] = {}

    for raw_key, model_key in _QUESTIONNAIRE_TO_MODEL_KEYS.items():
        value = raw.get(raw_key)
        if value not in (None, ""):
            prepared[model_key] = value

    if "Age_Years" not in prepared:
        age = _age_from_birth_date(raw.get("birth_date"))
        if age is not None:
            prepared["Age_Years"] = age

    if "Note_Sub_5" not in prepared:
        prepared["Note_Sub_5"] = _count_low_grades(raw.get("low_grades_details"))

    if "Studentship_Score" not in prepared:
        prepared["Studentship_Score"] = _studentship_score({**raw, **prepared})

    for feat in config.FEATURES:
        if feat.key in raw and feat.key not in prepared:
            prepared[feat.key] = raw[feat.key]

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
    absente_ne = rng.integers(0, 22, n_samples)
    absente_mot = rng.integers(0, 12, n_samples)
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

    # Non-linear risk definition (the "Day 14" model), extended so the NLP
    # stress signal is a genuine driver rather than decorative.
    df[config.TARGET_COLUMN] = np.where(
        (df["Absente_Nemotivate_Zilele_1_13"] > 12)
        | ((df["Medie_Modul_Anterior"] < 5.5) & (df["Studentship_Score"] <= 4))
        | ((df["Note_Sub_5"] >= 3) & (df["Sanctiuni_Avertismente"] != "Nu"))
        | ((df["Scoala_Ajuta_Obiective"] == "Nu") & (df["Atitudine_Scoala"] == "Negativă") & (df["Participare_Extrascolara"] == "Nu"))
        | (
            (df["Situatie_Familiala"] != "Ambii părinți")
            & (df["Absente_Nemotivate_Zilele_1_13"] > 7)
            & (df["Stres_Emotional_NLP"] >= 0.8)
        )
        | ((df["Cum_te_Simti_La_Scoala"].isin(["Stresat", "Izolat"])) & (df["Stres_Emotional_NLP"] >= 1.1) & (df["Studentship_Score"] <= 5))
        | ((df["Age_Years"] >= 17) & (df["Medie_Modul_Anterior"] < 6.0) & (df["Absente_Nemotivate_Zilele_1_13"] > 5)),
        1,
        0,
    )
    return df


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
def train_model(
    n_samples: int = config.TRAIN_SAMPLES,
    seed: int = config.RANDOM_SEED,
) -> tuple[RiskModel, TrainingMetrics]:
    """Train the XGBoost dropout-risk model with SMOTE-NC balancing."""
    df = generate_synthetic_dataset(n_samples, seed)

    X = _encode_dataframe(df[list(config.FEATURE_KEYS)])
    y = df[config.TARGET_COLUMN].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )

    balance_before = {str(k): int(v) for k, v in y_train.value_counts().items()}
    n_before = int(len(y_train))

    # SMOTE-NC needs k_neighbors < size of the minority class.
    minority = int(y_train.value_counts().min())
    k_neighbors = max(1, min(5, minority - 1))
    smote = SMOTENC(
        categorical_features=config.CATEGORICAL_INDICES,
        random_state=seed,
        k_neighbors=k_neighbors,
    )
    resampled = smote.fit_resample(X_train, y_train)
    X_res = resampled[0]
    y_res = resampled[1]

    balance_after = {str(k): int(v) for k, v in pd.Series(np.asarray(y_res)).value_counts().items()}
    n_after = int(len(y_res))

    clf = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=seed,
    )
    clf.fit(X_res, y_res)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    accuracy = float(accuracy_score(y_test, y_pred))
    try:
        roc_auc = float(roc_auc_score(y_test, y_proba))
    except ValueError:
        roc_auc = float("nan")
    report_text = str(classification_report(y_test, y_pred, zero_division=0))

    metrics = TrainingMetrics(
        accuracy=accuracy,
        roc_auc=roc_auc,
        n_before=n_before,
        n_after=n_after,
        balance_before=balance_before,
        balance_after=balance_after,
        report_text=report_text,
    )

    version = f"xgb-day14-n{n_samples}-s{seed}-{datetime.now():%Y%m%d}"
    meta = ModelMeta(
        model_version=version,
        trained_at=datetime.now().isoformat(timespec="seconds"),
        metrics=asdict(metrics),
    )
    return RiskModel(clf, meta), metrics


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
