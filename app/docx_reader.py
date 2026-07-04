"""Minimal, dependency-free .docx text extractor.

A .docx file is a ZIP archive; the body text lives in ``word/document.xml`` as
``<w:t>`` runs inside ``<w:p>`` paragraphs. We extract paragraph text with the
standard library only (``zipfile`` + ``xml.etree``) so the app gains a
knowledge-base import feature without pulling in python-docx and inflating the
packaged exe.
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


class DocxError(ValueError):
    """Raised when a file cannot be read as a .docx document."""


def extract_text(path: str | Path) -> str:
    """Return the plain-text body of a .docx file.

    Raises ``DocxError`` with a user-facing (Romanian) message on any problem.
    """
    p = Path(path)
    if not p.exists():
        raise DocxError(f"Fișierul nu există: {p}")
    if p.suffix.lower() != ".docx":
        raise DocxError(
            "Sunt acceptate doar fișiere .docx (Word). Pentru .doc mai vechi, "
            "salvați documentul ca .docx."
        )

    try:
        with zipfile.ZipFile(p) as zf:
            if "word/document.xml" not in zf.namelist():
                raise DocxError("Fișierul .docx nu conține word/document.xml.")
            xml_bytes = zf.read("word/document.xml")
    except zipfile.BadZipFile as exc:
        raise DocxError("Fișierul nu este un document .docx valid.") from exc

    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as exc:
        raise DocxError("Conținutul documentului nu a putut fi analizat.") from exc

    paragraphs: list[str] = []
    for para in root.iter(f"{_W}p"):
        runs = [node.text for node in para.iter(f"{_W}t") if node.text]
        paragraphs.append("".join(runs))

    text = "\n".join(paragraphs).strip()
    if not text:
        raise DocxError("Documentul nu conține text extractibil.")
    return text
