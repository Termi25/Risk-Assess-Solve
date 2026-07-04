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
