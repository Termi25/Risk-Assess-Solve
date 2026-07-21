"""Tests for the .docx knowledge-base import and settings persistence."""

import zipfile

import pytest

from app import docx_reader, settings
from app.docx_reader import DocxError
from tests.conftest import HIGH_RISK

_W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def _make_docx(path, paragraphs):
    body = "".join(
        f"<w:p><w:r><w:t>{p}</w:t></w:r></w:p>" for p in paragraphs
    )
    doc_xml = (
        f'<?xml version="1.0"?><w:document xmlns:w="{_W}"><w:body>'
        f"{body}</w:body></w:document>"
    )
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0"?><Types '
            'xmlns="http://schemas.openxmlformats.org/package/2006/content-types"/>',
        )
        z.writestr("word/document.xml", doc_xml)
    return path


def test_extract_text_from_docx(tmp_path):
    docx = _make_docx(tmp_path / "a.docx", ["Prima linie.", "A doua linie."])
    text = docx_reader.extract_text(docx)
    assert "Prima linie." in text
    assert "A doua linie." in text
    assert text.count("\n") == 1  # two paragraphs -> one newline between


def test_extract_text_rejects_non_docx(tmp_path):
    txt = tmp_path / "note.txt"
    txt.write_text("hello", encoding="utf-8")
    with pytest.raises(DocxError):
        docx_reader.extract_text(txt)


def test_extract_text_rejects_bad_zip(tmp_path):
    fake = tmp_path / "broken.docx"
    fake.write_bytes(b"not a zip")
    with pytest.raises(DocxError):
        docx_reader.extract_text(fake)


def test_extract_text_missing_file(tmp_path):
    with pytest.raises(DocxError):
        docx_reader.extract_text(tmp_path / "missing.docx")


def test_settings_knowledge_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "app.config.settings_path", lambda: tmp_path / "settings.json"
    )
    assert settings.get_knowledge() is None
    assert settings.get_knowledge_text() == ""

    docx = _make_docx(tmp_path / "kb.docx", ["Metodologie Day 14 cu SHAP."])
    entry = settings.set_knowledge_from_docx(docx)
    assert entry["filename"] == "kb.docx"
    assert entry["char_count"] > 0

    assert "SHAP" in settings.get_knowledge_text()
    assert settings.get_knowledge()["filename"] == "kb.docx"

    settings.clear_knowledge()
    assert settings.get_knowledge() is None


def test_grounded_system_prompt_includes_doc():
    from app.llm_client import SYSTEM_PROMPT, _build_system_prompt

    assert _build_system_prompt(None) == SYSTEM_PROMPT
    grounded = _build_system_prompt("TEXT-UNIC-DE-REFERINTA")
    assert "TEXT-UNIC-DE-REFERINTA" in grounded
    assert len(grounded) > len(SYSTEM_PROMPT)


def test_local_plan_uses_questionnaire_context(trained_model, monkeypatch):
    from app.explainability import evaluate
    from app.llm_client import anonymized_summary, generate_action_plan

    monkeypatch.setattr("app.keystore.resolve_api_key", lambda pid: None)

    model, _ = trained_model
    evaluation = evaluate(model, HIGH_RISK)
    summary = anonymized_summary(evaluation, HIGH_RISK)
    assert "Context chestionar (anonimizat):" in summary
    # The cloud payload carries family structure only as a coarse bucket; the
    # precise category stays local. See tests/test_deidentification.py.
    assert "Structura familiei" in summary
    assert "Situația familială" not in summary
    assert "Nume" not in summary

    plan, source = generate_action_plan(
        evaluation,
        questionnaire_answers=HIGH_RISK,
        api_key=None,
        knowledge_text=None,
    )
    assert source == "local template"
    assert "Context relevant din chestionar" in plan
    assert "Situația familială" in plan
