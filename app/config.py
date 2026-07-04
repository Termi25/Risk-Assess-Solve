"""Central configuration: feature schema, paths, and app constants.

The feature schema mirrors the research methodology's "Day 14" early-warning
model (previous-module average, unmotivated absences in the first 13 school
days, the *studentship* engagement composite, and the socio-economic context),
plus one NLP-derived emotional-stress feature that stands in for the BERT
qualitative signal described in the README.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

APP_NAME = "RiskSolvingApp"
APP_TITLE = "Evaluare Risc Abandon Școlar & Plan de Intervenție"
APP_VERSION = "0.1.0"

# --- Model / target ---------------------------------------------------------
# Binary target: 1 = elev cu risc de abandon (dropout risk), 0 = menținere.
TARGET_COLUMN = "Abandon"


# --- Feature schema ---------------------------------------------------------
@dataclass(frozen=True)
class Feature:
    """One model input, with the metadata the GUI and encoders need."""

    key: str                 # column name fed to the model (matches research)
    label: str               # human-readable label for the GUI
    kind: str                # "numeric" | "categorical"
    minimum: float = 0.0     # for numeric widgets
    maximum: float = 10.0
    step: float = 1.0
    default: float = 0.0
    # For categorical features: ordered display choices mapped to integer codes.
    categories: tuple[str, ...] = field(default_factory=tuple)
    help_text: str = ""


# Order is significant: this is exactly the column order fed to XGBoost/SHAP.
FEATURES: tuple[Feature, ...] = (
    Feature(
        key="Medie_Modul_Anterior",
        label="Media modulului anterior",
        kind="numeric",
        minimum=1.0, maximum=10.0, step=0.1, default=7.0,
        help_text="Media notelor din modulul precedent (1–10).",
    ),
    Feature(
        key="Note_Sub_7",
        label="Număr note sub 7",
        kind="numeric",
        minimum=0, maximum=20, step=1, default=1,
        help_text="Câte note sub 7 a primit elevul în perioada de referință.",
    ),
    Feature(
        key="Absente_Nemotivate_Zilele_1_13",
        label="Absențe nemotivate (zilele 1–13)",
        kind="numeric",
        minimum=0, maximum=40, step=1, default=2,
        help_text="Absențe nemotivate în fereastra de avertizare timpurie 'Day 14'.",
    ),
    Feature(
        key="Studentship_Score",
        label="Scor Studentship (implicare 0–10)",
        kind="numeric",
        minimum=0, maximum=10, step=1, default=6,
        help_text="Suma componentelor de prezență cognitivă și socială (0–10).",
    ),
    Feature(
        key="Stres_Emotional_NLP",
        label="Stres emoțional (NLP, 0–2)",
        kind="numeric",
        minimum=0.0, maximum=2.0, step=0.1, default=0.0,
        help_text="Sub-scor derivat automat din observațiile calitative (BERT-style).",
    ),
    Feature(
        key="Mediu_Rezidential",
        label="Mediu rezidențial",
        kind="categorical",
        categories=("Urban", "Rural"),
        help_text="Mediul de proveniență al elevului.",
    ),
    Feature(
        key="Parinti_In_Strainatate",
        label="Părinți în străinătate",
        kind="categorical",
        categories=("Nu", "Da"),
        help_text="Cel puțin un părinte plecat la muncă în străinătate.",
    ),
    Feature(
        key="Vulnerabilitate_Financiara",
        label="Vulnerabilitate financiară",
        kind="categorical",
        categories=("Scazuta", "Medie", "Ridicata"),
        help_text="Nivelul vulnerabilității socio-economice a familiei.",
    ),
)

FEATURE_KEYS: tuple[str, ...] = tuple(f.key for f in FEATURES)

# Categorical column indices (positions in FEATURE_KEYS) — needed by SMOTE-NC.
CATEGORICAL_INDICES: list[int] = [
    i for i, f in enumerate(FEATURES) if f.kind == "categorical"
]

# Map each categorical feature's display value -> integer code, in schema order.
CATEGORY_CODES: dict[str, dict[str, int]] = {
    f.key: {name: code for code, name in enumerate(f.categories)}
    for f in FEATURES
    if f.kind == "categorical"
}


def feature(key: str) -> Feature:
    for f in FEATURES:
        if f.key == key:
            return f
    raise KeyError(key)


# --- Paths ------------------------------------------------------------------
def _bundle_dir() -> Path:
    """Directory of bundled read-only resources.

    Under PyInstaller the app is unpacked to ``sys._MEIPASS``; in a normal
    checkout it's the package directory.
    """
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        return Path(meipass) / "app"
    return Path(__file__).resolve().parent


def user_data_dir() -> Path:
    """Writable per-user directory (DB, freshly trained model, config)."""
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_DATA_HOME")
    if base:
        path = Path(base) / APP_NAME
    else:
        path = Path.home() / f".{APP_NAME.lower()}"
    path.mkdir(parents=True, exist_ok=True)
    return path


BUNDLED_ARTIFACTS_DIR = _bundle_dir() / "artifacts"
MODEL_FILENAME = "risk_model.json"
MODEL_META_FILENAME = "risk_model.meta.json"
DB_FILENAME = "risk_app.db"
SETTINGS_FILENAME = "settings.json"


def bundled_model_path() -> Path:
    return BUNDLED_ARTIFACTS_DIR / MODEL_FILENAME


def bundled_meta_path() -> Path:
    return BUNDLED_ARTIFACTS_DIR / MODEL_META_FILENAME


def user_model_path() -> Path:
    return user_data_dir() / MODEL_FILENAME


def user_meta_path() -> Path:
    return user_data_dir() / MODEL_META_FILENAME


def database_path() -> Path:
    return user_data_dir() / DB_FILENAME


def settings_path() -> Path:
    return user_data_dir() / SETTINGS_FILENAME


# --- LLM (optional cloud step) ---------------------------------------------
# Generous headroom: with adaptive thinking, reasoning tokens share this budget.
LLM_MAX_TOKENS = 4000


@dataclass(frozen=True)
class LLMProvider:
    """One cloud action-plan provider (Claude or Gemini)."""

    id: str            # stable key: "claude" | "gemini"
    label: str         # human-readable name for the GUI
    env_var: str       # environment variable checked for an override key
    key_prefix: str    # placeholder hint shown in the settings dialog
    default_model: str # model used for generation

    @property
    def keyring_service(self) -> str:
        """OS-vault service name — one entry per provider."""
        return f"{APP_NAME}/{self.id}"


# The app works with either provider; the active one is chosen in Settings.
LLM_PROVIDERS: tuple[LLMProvider, ...] = (
    LLMProvider(
        id="claude",
        label="Claude (Anthropic)",
        env_var="ANTHROPIC_API_KEY",
        key_prefix="sk-ant-…",
        default_model="claude-opus-4-8",
    ),
    LLMProvider(
        id="gemini",
        label="Gemini (Google)",
        env_var="GEMINI_API_KEY",
        key_prefix="AIza…",
        default_model="gemini-2.5-pro",
    ),
)
DEFAULT_LLM_PROVIDER = "claude"
# Shared username under each provider's keyring service entry. The key is kept
# in the OS credential vault (Windows Credential Manager), never in settings.json.
LLM_KEYRING_USERNAME = "api_key"


def get_provider(provider_id: str) -> LLMProvider:
    for p in LLM_PROVIDERS:
        if p.id == provider_id:
            return p
    raise KeyError(provider_id)


# Cap on how much knowledge-base text is sent to the cloud (keeps token cost
# bounded); the stored document is truncated to this many characters at prompt
# build time only.
KNOWLEDGE_MAX_CHARS = 24000


# --- Training (synthetic SIIIR-like data) ----------------------------------
TRAIN_SAMPLES = 1200
RANDOM_SEED = 42
