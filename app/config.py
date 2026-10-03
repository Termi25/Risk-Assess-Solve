"""Central configuration: questionnaire schema, feature schema, paths, and app constants.

The app now separates the full questionnaire collected from the teacher from
the smaller model feature set used by the scoring engine. The questionnaire
covers the complete report input, while the model uses a derived subset of the
answers plus one NLP-derived emotional-stress feature.
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


@dataclass(frozen=True)
class QuestionnaireItem:
    """One questionnaire field collected from the teacher or student."""

    key: str
    label: str
    kind: str                # "text" | "multiline" | "date" | "categorical" | "numeric"
    minimum: float = 0.0
    maximum: float = 10.0
    step: float = 1.0
    default: object = ""
    categories: tuple[str, ...] = field(default_factory=tuple)
    help_text: str = ""
    multiline: bool = False


# Questionnaire order is significant for the UI.
QUESTIONNAIRE_FIELDS: tuple[QuestionnaireItem, ...] = (
    QuestionnaireItem(
        key="timestamp",
        label="Marcaj de timp",
        kind="date",
        help_text="Se completează automat la salvare.",
        default="",
    ),
    QuestionnaireItem(
        key="full_name",
        label="1. Nume și Prenume",
        kind="text",
        default="",
    ),
    QuestionnaireItem(
        key="birth_date",
        label="2. Data nașterii",
        kind="date",
        default="2009-01-01",
    ),
    QuestionnaireItem(
        key="student_class",
        label="3. Clasa",
        kind="text",
        default="",
    ),
    QuestionnaireItem(
        key="school_name",
        label="4. Școala",
        kind="text",
        default="",
    ),
    QuestionnaireItem(
        key="sex",
        label="5. Sexul",
        kind="categorical",
        categories=("Feminin", "Masculin", "Altul / prefer să nu spun"),
    ),
    QuestionnaireItem(
        key="residential_environment",
        label="6. Mediul de proveniență",
        kind="categorical",
        categories=("Urban", "Rural"),
    ),
    QuestionnaireItem(
        key="family_situation",
        label="7. Situația familială",
        kind="categorical",
        categories=("Ambii părinți", "Monoparental", "Tutore / plasament","Părinți divortați/separați", "Altă situație"),
    ),
    QuestionnaireItem(
        key="family_situation_other",
        label="7.a Dacă ai precizat altă situație la întrebarea de mai sus, descrie pe scurt",
        kind="multiline",
        multiline=True,
    ),
    QuestionnaireItem(
        key="mother_education",
        label="8.a Nivelul de educație al părinților (mama)",
        kind="categorical",
        categories=("Primar", "Gimnazial", "Liceal", "Postliceal", "Universitar", "Nu se aplică/Necunoscut"),
    ),
    QuestionnaireItem(
        key="father_education",
        label="8.b Nivelul de educație al părinților (tata)",
        kind="categorical",
        categories=("Primar", "Gimnazial", "Liceal", "Postliceal", "Universitar", "Nu se aplică/Necunoscut"),
    ),
    QuestionnaireItem(
        key="unexcused_absences_3m",
        label="9. Numărul absențelor nemotivate în ultimele 3 luni",
        kind="numeric",
        minimum=0,
        maximum=60,
        step=1,
        default=0,
    ),
    QuestionnaireItem(
        key="excused_absences_3m",
        label="10. Numărul absențelor motivate în ultimele 3 luni",
        kind="numeric",
        minimum=0,
        step=1,
        default=0,
    ),
    QuestionnaireItem(
        key="extracurricular_participation",
        label="11. Participarea la activități extrașcolare",
        kind="categorical",
        categories=("Da, frecvent", "Ocazional", "Nu"),
    ),
    QuestionnaireItem(
        key="previous_module_average",
        label="12. Media generale pe modulul anterior",
        kind="numeric",
        minimum=1.0,
        maximum=10.0,
        step=0.1,
        default=7.0,
    ),
    QuestionnaireItem(
        key="low_grades_details",
        label="13. Note mai mici de 5 obținute la discipline în ultimul semestru (de specificat nota si materia)",
        kind="multiline",
        multiline=True,
    ),
    QuestionnaireItem(
        key="school_attitude",
        label="14. Cum ți-ai descrie atitudinea față de școală?",
        kind="categorical",
        categories=("Pozitivă", "Neutră", "Negativă"),
    ),
    QuestionnaireItem(
        key="disciplinary_sanctions",
        label="15. Ai primit sancțiuni sau avertismente disciplinare în ultimul an",
        kind="categorical",
        categories=("Nu", "Avertismente", "Sancțiuni"),
    ),
    QuestionnaireItem(
        key="school_feeling",
        label="16. Cum te simți în general la școală?",
        kind="categorical",
        categories=("Bine", "Neutru", "Stresat", "Izolat", "Altul"),
    ),
    QuestionnaireItem(
        key="school_feeling_other",
        label="16.a Dacă ai menționat altele, descrie pe scurt.",
        kind="multiline",
        multiline=True,
    ),
    QuestionnaireItem(
        key="school_support_goal",
        label="17. Consideri că școala te ajută să îți atingi obiectivele personale?",
        kind="categorical",
        categories=("Da", "Parțial", "Nu"),
    ),
    QuestionnaireItem(
        key="additional_notes",
        label="Observații suplimentare",
        kind="multiline",
        multiline=True,
    ),
)


# Order is significant: this is exactly the column order fed to XGBoost/SHAP.
FEATURES: tuple[Feature, ...] = (
    Feature(
        key="Age_Years",
        label="Vârsta (ani)",
        kind="numeric",
        minimum=10.0,
        maximum=100.0,
        step=1.0,
        default=15.0,
        help_text="Vârsta aproximativă calculată din data nașterii.",
    ),
    Feature(
        key="Sex",
        label="Sexul",
        kind="categorical",
        categories=("Feminin", "Masculin", "Altul / prefer să nu spun"),
        help_text="Răspunsul la întrebarea privind sexul.",
    ),
    Feature(
        key="Medie_Modul_Anterior",
        label="Media modulului anterior",
        kind="numeric",
        minimum=1.0, maximum=10.0, step=0.1, default=7.0,
        help_text="Media notelor din modulul precedent (1–10).",
    ),
    Feature(
        key="Mediu_Rezidential",
        label="Mediu rezidențial",
        kind="categorical",
        categories=("Urban", "Rural"),
        help_text="Mediul de proveniență al elevului.",
    ),
    Feature(
        key="Situatie_Familiala",
        label="Situație familială",
        kind="categorical",
        categories=("Ambii părinți", "Monoparental", "Tutore / plasament", "Altă situație"),
        help_text="Situația familială declarată.",
    ),
    Feature(
        key="Educatie_Mama",
        label="Educația mamei",
        kind="categorical",
        categories=("Primar", "Gimnazial", "Liceal", "Postliceal", "Universitar", "Necunoscut"),
        help_text="Nivelul de educație al mamei.",
    ),
    Feature(
        key="Educatie_Tata",
        label="Educația tatălui",
        kind="categorical",
        categories=("Primar", "Gimnazial", "Liceal", "Postliceal", "Universitar", "Necunoscut"),
        help_text="Nivelul de educație al tatălui.",
    ),
    Feature(
        key="Absente_Nemotivate_Zilele_1_13",
        label="Absențe nemotivate (3 luni)",
        kind="numeric",
        minimum=0, maximum=1000, step=1, default=0,
        help_text="Numărul de absențe nemotivate din ultimele 3 luni.",
    ),
    Feature(
        key="Absente_Motivate_3_Luni",
        label="Absențe motivate (3 luni)",
        kind="numeric",
        minimum=0, maximum=1000, step=1, default=0,
        help_text="Numărul de absențe motivate din ultimele 3 luni.",
    ),
    Feature(
        key="Participare_Extrascolara",
        label="Participare extrașcolară",
        kind="categorical",
        categories=("Da, frecvent", "Ocazional", "Nu"),
        help_text="Participarea la activități extrașcolare.",
    ),
    Feature(
        key="Note_Sub_5",
        label="Număr note sub 5",
        kind="numeric",
        minimum=0, maximum=20, step=1, default=0,
        help_text="Numărul notelor mai mici de 5 menționate în răspunsul la întrebarea 13.",
    ),
    Feature(
        key="Studentship_Score",
        label="Scor Studentship (implicare 0–10)",
        kind="numeric",
        minimum=0, maximum=10, step=1, default=6,
        help_text="Scor compozit derivat din participare, atitudine, școală și sancțiuni.",
    ),
    Feature(
        key="Atitudine_Scoala",
        label="Atitudinea față de școală",
        kind="categorical",
        categories=("Pozitivă", "Neutră", "Negativă"),
        help_text="Răspunsul la întrebarea despre atitudinea față de școală.",
    ),
    Feature(
        key="Sanctiuni_Avertismente",
        label="Sancțiuni / avertismente",
        kind="categorical",
        categories=("Nu", "Avertismente", "Sancțiuni"),
        help_text="Dacă elevul a primit sancțiuni sau avertismente disciplinare.",
    ),
    Feature(
        key="Cum_te_Simti_La_Scoala",
        label="Cum se simte la școală",
        kind="categorical",
        categories=("Bine", "Neutru", "Stresat", "Izolat", "Altul"),
        help_text="Cum se simte elevul în general la școală.",
    ),
    Feature(
        key="Scoala_Ajuta_Obiective",
        label="Școala ajută obiectivele personale",
        kind="categorical",
        categories=("Da", "Parțial", "Nu"),
        help_text="Dacă elevul consideră că școala îl ajută să își atingă obiectivele.",
    ),
    Feature(
        key="Stres_Emotional_NLP",
        label="Stres emoțional (NLP, 0–2)",
        kind="numeric",
        minimum=0.0,
        maximum=2.0,
        step=0.1,
        default=0.0,
        help_text="Sub-scor derivat automat din textul calitativ.",
    ),
)

FEATURE_KEYS: tuple[str, ...] = tuple(f.key for f in FEATURES)
QUESTIONNAIRE_KEYS: tuple[str, ...] = tuple(q.key for q in QUESTIONNAIRE_FIELDS)

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


# --- Risk classification (4-tier prioritization, per the research figures) --
# The intervention-prioritization table distinguishes four risk levels, each
# with a matching intervention urgency and a colour used across the report:
#   Scăzut   -> Monitorizare (green)   — low risk, keep watching
#   Mediu    -> Medie         (yellow) — targeted support
#   Ridicat  -> Ridicată      (orange) — active intervention
#   Critic   -> Maximă        (red)    — immediate, compounded-risk crisis
# ``min_probability`` is the lower bound of the model-probability band; a Ridicat
# case is escalated to Critic by the severity rules in ``explainability``.
@dataclass(frozen=True)
class RiskTier:
    band: str            # short risk-level name shown on the badge
    urgency: str         # matching intervention urgency
    min_probability: float
    color: str           # badge / dot colour
    text_color: str      # readable text colour on ``color``


# Probability bands calibrated to the graded scoring model (see scoring_engine
# ``_RISK_LOGIT``): the model produces a smooth spread rather than saturating, so
# the tiers sit at the natural break-points of that distribution (base dropout
# rate ≈ 0.22) instead of the near-1.0 values the old hard-rule model emitted.
RISK_TIERS: tuple[RiskTier, ...] = (
    RiskTier("Scăzut",  "Monitorizare", 0.00, "#2e8b57", "#ffffff"),
    RiskTier("Mediu",   "Medie",        0.20, "#f1c40f", "#3a3a3a"),
    RiskTier("Ridicat", "Ridicată",     0.42, "#e67e22", "#ffffff"),
    RiskTier("Critic",  "Maximă",       0.65, "#d64550", "#ffffff"),
)
RISK_TIER_BY_BAND: dict[str, RiskTier] = {t.band: t for t in RISK_TIERS}


def tier_for_probability(probability: float) -> RiskTier:
    """Base risk tier for a model probability (before severity escalation)."""
    chosen = RISK_TIERS[0]
    for tier in RISK_TIERS:
        if probability >= tier.min_probability:
            chosen = tier
    return chosen


# Band names used by earlier versions, still present in saved evaluations.
# The lowest tier was called "Moderat" until it was renamed to "Scăzut" (Low),
# which matches the green / monitoring-only meaning of the tier.
LEGACY_BAND_NAMES: dict[str, str] = {"Moderat": "Scăzut"}


def normalize_band(band: str) -> str:
    """Current name for a band, mapping legacy names (``Moderat`` -> ``Scăzut``)."""
    return LEGACY_BAND_NAMES.get(band, band)


def tier_for_band(band: str) -> RiskTier:
    """Look up a tier by band name; falls back to the lowest tier."""
    return RISK_TIER_BY_BAND.get(normalize_band(band), RISK_TIERS[0])


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
# Output-token ceiling. Reasoning ("thinking") tokens share this budget on both
# providers, so it must comfortably fit both the internal reasoning and the full
# written plan — otherwise the plan is truncated (or empty) and silently falls
# back to the local template, making batch reports inconsistent.
LLM_MAX_TOKENS = 8192
# Gemini 2.5 counts thinking tokens against ``max_output_tokens``. Bounding the
# thinking budget guarantees the written plan always has room and keeps plan
# length/quality consistent across a batch of per-student calls.
GEMINI_THINKING_BUDGET = 2048


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
TRAIN_SAMPLES = 2800
RANDOM_SEED = 42
