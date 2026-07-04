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
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from . import config


# --- Encoding ---------------------------------------------------------------
def encode_case_features(features: dict) -> pd.DataFrame:
    """Turn a raw feature dict into a single-row, model-ready DataFrame.

    Categorical values arrive as display strings ("Rural", "Da", ...) and are
    mapped to the integer codes the model was trained on. Column order is
    fixed to ``config.FEATURE_KEYS``.
    """
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
    """Generate SIIIR-like student records with the non-linear 'Day 14' rules.

    Replaces the real SIIIR export used in production. Real deployment would
    swap this for ``pd.read_csv(<anonymised export>)`` with the same columns.
    """
    rng = np.random.default_rng(seed)

    medie = rng.uniform(4.0, 9.5, n_samples)
    note_sub_7 = rng.integers(0, 6, n_samples)
    absente = rng.integers(0, 15, n_samples)

    # Studentship: sum of five 0..2 presence components (0..10).
    presence = rng.integers(0, 3, size=(n_samples, 5)).sum(axis=1)

    # NLP-derived emotional stress, 0..2 (what nlp_engine would output).
    stres = np.round(rng.beta(1.5, 4.0, n_samples) * 2.0, 2)

    mediu = rng.choice(["Urban", "Rural"], n_samples, p=[0.4, 0.6])
    parinti = rng.choice(["Nu", "Da"], n_samples, p=[0.75, 0.25])
    vulnerab = rng.choice(
        ["Scazuta", "Medie", "Ridicata"], n_samples, p=[0.2, 0.4, 0.4]
    )

    df = pd.DataFrame(
        {
            "Medie_Modul_Anterior": medie,
            "Note_Sub_7": note_sub_7,
            "Absente_Nemotivate_Zilele_1_13": absente,
            "Studentship_Score": presence,
            "Stres_Emotional_NLP": stres,
            "Mediu_Rezidential": mediu,
            "Parinti_In_Strainatate": parinti,
            "Vulnerabilitate_Financiara": vulnerab,
        }
    )

    # Non-linear risk definition (the "Day 14" model), extended so the NLP
    # stress signal is a genuine driver rather than decorative.
    df[config.TARGET_COLUMN] = np.where(
        (df["Absente_Nemotivate_Zilele_1_13"] > 10)
        | ((df["Medie_Modul_Anterior"] < 5.0) & (df["Studentship_Score"] <= 3))
        | (
            (df["Vulnerabilitate_Financiara"] == "Ridicata")
            & (df["Parinti_In_Strainatate"] == "Da")
            & (df["Absente_Nemotivate_Zilele_1_13"] > 7)
        )
        | ((df["Stres_Emotional_NLP"] >= 1.5) & (df["Studentship_Score"] <= 4)),
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
    X_res, y_res = smote.fit_resample(X_train, y_train)

    balance_after = {str(k): int(v) for k, v in pd.Series(y_res).value_counts().items()}
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
    report_text = classification_report(y_test, y_pred, zero_division=0)

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


def get_or_train_model(force_retrain: bool = False) -> RiskModel:
    """Return a usable model, training + caching one on first run.

    Resolution order:
      1. A model bundled inside the app (shipped with the exe).
      2. A model previously trained into the user's data directory.
      3. Train a fresh model, save it to the user directory, and return it.
    """
    if not force_retrain:
        if config.bundled_model_path().exists() and config.bundled_meta_path().exists():
            return load_model(config.bundled_model_path(), config.bundled_meta_path())
        if config.user_model_path().exists() and config.user_meta_path().exists():
            return load_model(config.user_model_path(), config.user_meta_path())

    model, _ = train_model()
    save_model(model, config.user_model_path(), config.user_meta_path())
    return model
