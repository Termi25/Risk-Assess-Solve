"""Local application settings (JSON in the user data dir).

Currently holds the optional **knowledge base**: a .docx document (typically the
research methodology article) whose extracted text is used to ground the cloud
action-plan generation — the same idea as a Gemini Gem's knowledge pool.

The document text is methodology, not student data, so sending it to the cloud
alongside the anonymized score is consistent with the app's privacy model.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from . import config
from .docx_reader import extract_text


def load_settings() -> dict:
    path = config.settings_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def save_settings(data: dict) -> None:
    config.settings_path().write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# --- LLM provider -----------------------------------------------------------
# Which cloud provider (Claude / Gemini) the app uses. This is a preference,
# not a secret, so it lives in settings.json; the API keys live in the OS vault.
def get_active_provider() -> str:
    """Return the configured provider id, defaulting if unset/unknown."""
    pid = load_settings().get("llm_provider", config.DEFAULT_LLM_PROVIDER)
    try:
        config.get_provider(pid)
    except KeyError:
        return config.DEFAULT_LLM_PROVIDER
    return pid


def set_active_provider(provider_id: str) -> None:
    config.get_provider(provider_id)  # validate before persisting
    data = load_settings()
    data["llm_provider"] = provider_id
    save_settings(data)


# --- Appearance & language --------------------------------------------------
# Both are pure interface preferences. Notably the language does *not* change
# what is stored or fed to the model — the questionnaire's category values stay
# Romanian in the database and in the feature frame (see :mod:`app.i18n`), so a
# report exported today stays comparable with one exported before the switch.
def get_theme() -> str:
    from .ui.theme import DEFAULT_THEME, THEMES

    code = load_settings().get("theme", DEFAULT_THEME)
    return code if any(c == code for c, _ in THEMES) else DEFAULT_THEME


def set_theme(code: str) -> None:
    data = load_settings()
    data["theme"] = code
    save_settings(data)


def get_language() -> str:
    from .i18n import DEFAULT_LANGUAGE, LANGUAGES

    code = load_settings().get("language", DEFAULT_LANGUAGE)
    return code if any(c == code for c, _ in LANGUAGES) else DEFAULT_LANGUAGE


def set_language(code: str) -> None:
    data = load_settings()
    data["language"] = code
    save_settings(data)


# --- Knowledge base ---------------------------------------------------------
def get_knowledge() -> Optional[dict]:
    """Return the stored knowledge-base metadata + text, or None."""
    return load_settings().get("knowledge")


def get_knowledge_text() -> str:
    kb = get_knowledge()
    return (kb or {}).get("text", "") if kb else ""


def set_knowledge_from_docx(path: str | Path) -> dict:
    """Extract text from a .docx and store it as the knowledge base.

    Returns the stored metadata dict. Raises ``DocxError`` on bad input.
    """
    p = Path(path)
    text = extract_text(p)
    entry = {
        "filename": p.name,
        "source_path": str(p),
        "added_at": datetime.now().isoformat(timespec="seconds"),
        "char_count": len(text),
        "text": text,
    }
    data = load_settings()
    data["knowledge"] = entry
    save_settings(data)
    return entry


def clear_knowledge() -> None:
    data = load_settings()
    if "knowledge" in data:
        del data["knowledge"]
        save_settings(data)
