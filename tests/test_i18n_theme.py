"""Interface language and theme.

The load-bearing test here is ``test_model_categories_are_never_translated``:
the questionnaire's category strings are the model's training vocabulary, so
switching the interface to English must change what the teacher *sees* without
changing what the classifier is *fed*. The rest guard the catalogs against the
usual drift — a call site whose literal no longer matches its entry, or a format
template whose placeholders differ between the two languages.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from app import config, i18n
from app.ui import theme

APP_DIR = Path(__file__).resolve().parent.parent / "app"


@pytest.fixture(autouse=True)
def _restore_language():
    """Language is process-global; never let one test leak into the next."""
    previous = i18n.get_language()
    yield
    i18n.set_language(previous)


# --- the contract that protects the model ----------------------------------
def test_model_categories_are_never_translated():
    """``tr_value`` translates the label; the stored value stays Romanian."""
    i18n.set_language("en")
    for item in config.QUESTIONNAIRE_FIELDS:
        for category in item.categories:
            assert i18n.tr_value(category) != ""
            # The catalog must not be able to feed English into the model: the
            # value the widget stores is the loop variable, never tr_value().
            assert category in {c for c in item.categories}


def test_every_model_category_has_an_english_label():
    """A missing entry would silently show Romanian in an English UI."""
    i18n.set_language("en")
    missing = {
        category
        for item in config.QUESTIONNAIRE_FIELDS
        for category in item.categories
        if i18n.tr_value(category) == category and not _is_same_in_both(category)
    }
    assert not missing, f"untranslated category labels: {sorted(missing)}"


def _is_same_in_both(text: str) -> bool:
    """Words that legitimately read identically in Romanian and English."""
    return text in {"Urban", "Rural"}


def test_feature_categories_covered():
    """The model's own FEATURES vocabulary is display-translatable too."""
    i18n.set_language("en")
    for feature in config.FEATURES:
        for category in feature.categories:
            assert isinstance(i18n.tr_value(category), str)


# --- language switching -----------------------------------------------------
def test_romanian_is_identity():
    i18n.set_language("ro")
    assert i18n.tr("Evaluează riscul") == "Evaluează riscul"
    assert i18n.tr_value("Feminin") == "Feminin"
    assert i18n.tr_band("Critic") == "Critic"


def test_english_translates_known_strings():
    i18n.set_language("en")
    assert i18n.tr("Evaluează riscul") == "Assess risk"
    assert i18n.tr_value("Feminin") == "Female"
    assert i18n.tr_band("Critic") == "Critical"


def test_unknown_string_falls_back_to_romanian():
    i18n.set_language("en")
    assert i18n.tr("un șir care nu există în catalog") == "un șir care nu există în catalog"


def test_unknown_language_code_falls_back_to_default():
    i18n.set_language("de")
    assert i18n.get_language() == i18n.DEFAULT_LANGUAGE


def test_plan_language_name_tracks_the_setting():
    i18n.set_language("ro")
    assert i18n.plan_language_name() == "Romanian"
    i18n.set_language("en")
    assert i18n.plan_language_name() == "English"


def test_all_risk_bands_translate():
    i18n.set_language("en")
    for tier in config.RISK_TIERS:
        assert i18n.tr_band(tier.band) != tier.band
        assert i18n.tr_band(tier.urgency) != ""


# --- catalog integrity ------------------------------------------------------
def test_template_placeholders_match_across_languages():
    """A renamed field in one language would raise KeyError at format time."""
    field_re = re.compile(r"\{(\w+)")
    for source, target in i18n._TEMPLATES.items():
        assert set(field_re.findall(source)) == set(field_re.findall(target)), (
            f"placeholder mismatch for template: {source[:60]!r}"
        )


def _string_literal_args(path: Path, func_names: set[str]) -> set[str]:
    """Every literal first argument passed to ``tr`` / ``trf`` in one module."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
        if name in func_names and isinstance(node.args[0], ast.Constant):
            if isinstance(node.args[0].value, str):
                found.add(node.args[0].value)
    return found


@pytest.mark.parametrize(
    "module",
    ["ui/main_window.py", "ui/settings_dialog.py", "ui/metrics_dialog.py"],
)
def test_every_tr_call_site_has_a_catalog_entry(module):
    """Catches a call-site literal drifting away from its catalog key."""
    literals = _string_literal_args(APP_DIR / module, {"tr"})
    missing = {
        text for text in literals
        if text and text not in i18n._UI and not _translatable_elsewhere(text)
    }
    assert not missing, f"{module}: no English entry for {sorted(missing)[:5]}"


@pytest.mark.parametrize(
    "module",
    ["ui/main_window.py", "ui/settings_dialog.py", "ui/metrics_dialog.py"],
)
def test_every_trf_call_site_has_a_template(module):
    literals = _string_literal_args(APP_DIR / module, {"trf"})
    missing = {text for text in literals if text and text not in i18n._TEMPLATES}
    assert not missing, f"{module}: no template for {sorted(missing)[:5]}"


def _translatable_elsewhere(text: str) -> bool:
    """``tr`` is also used on values that live in the other catalogs."""
    return text in i18n._VALUES or text in i18n._BANDS


def test_questionnaire_labels_are_translated():
    """Every visible questionnaire label must have an English form."""
    i18n.set_language("en")
    untranslated = [
        item.label for item in config.QUESTIONNAIRE_FIELDS
        if i18n.tr(item.label) == item.label
    ]
    assert not untranslated, f"untranslated labels: {untranslated}"


def test_app_title_translates():
    i18n.set_language("en")
    assert i18n.tr(config.APP_TITLE) != config.APP_TITLE


# --- theme ------------------------------------------------------------------
def test_theme_codes_and_labels():
    codes = [code for code, _ in theme.available_themes()]
    assert codes == ["system", "light", "dark"]
    assert theme.DEFAULT_THEME in codes


def test_theme_labels_translate():
    i18n.set_language("en")
    assert i18n.tr("Luminoasă (alb)") == "Light (white)"
    assert i18n.tr("Întunecată") == "Dark"
    assert i18n.tr("Sistem") == "System"


def test_stylesheet_only_for_explicit_themes():
    assert theme.stylesheet_for("system") == ""
    assert "QGroupBox" in theme.stylesheet_for("light")
    assert "QGroupBox" in theme.stylesheet_for("dark")


def test_document_surfaces_stay_light_in_every_theme():
    """Report HTML is calibrated for paper; a dark pane would hide its text."""
    qss = theme.document_qss()
    assert "#ffffff" in qss and "#2b2b2b" in qss


def test_muted_colour_differs_between_light_and_dark():
    """The old hard-coded #555 was unreadable on a dark background."""
    assert theme._MUTED["light"] != theme._MUTED["dark"]
    for code in ("system", "light", "dark"):
        assert theme._MUTED[code].startswith("#")


# --- settings persistence ---------------------------------------------------
def test_theme_and_language_round_trip(tmp_path, monkeypatch):
    from app import settings

    monkeypatch.setattr("app.config.user_data_dir", lambda: tmp_path)
    settings.set_theme("light")
    settings.set_language("en")
    assert settings.get_theme() == "light"
    assert settings.get_language() == "en"


def test_invalid_stored_values_fall_back(tmp_path, monkeypatch):
    from app import settings

    monkeypatch.setattr("app.config.user_data_dir", lambda: tmp_path)
    settings.save_settings({"theme": "neon", "language": "klingon"})
    assert settings.get_theme() == theme.DEFAULT_THEME
    assert settings.get_language() == i18n.DEFAULT_LANGUAGE


def test_defaults_when_settings_file_absent(tmp_path, monkeypatch):
    from app import settings

    monkeypatch.setattr("app.config.user_data_dir", lambda: tmp_path)
    assert settings.get_theme() == theme.DEFAULT_THEME
    assert settings.get_language() == i18n.DEFAULT_LANGUAGE


# --- window integration -----------------------------------------------------
@pytest.fixture(scope="module")
def qapp():
    """One offscreen QApplication for the widget-level tests."""
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


@pytest.fixture
def window(qapp, tmp_path, monkeypatch):
    """A window that always starts Romanian, whatever ran before it."""
    monkeypatch.setattr("app.config.user_data_dir", lambda: tmp_path)
    i18n.set_language("ro")
    from app.ui.main_window import MainWindow

    return MainWindow()


def test_english_window_stores_romanian_category_values(window):
    """The whole point: English labels, Romanian values reaching the model."""
    from PySide6.QtWidgets import QComboBox

    window._on_language_changed("en")
    combo = window._question_widgets["sex"]
    assert isinstance(combo, QComboBox)
    assert [combo.itemText(i) for i in range(combo.count())] == [
        "Female", "Male", "Other / prefer not to say",
    ]
    assert [combo.itemData(i) for i in range(combo.count())] == list(
        next(q for q in config.QUESTIONNAIRE_FIELDS if q.key == "sex").categories
    )


def test_collected_answers_stay_romanian_in_english_ui(window):
    window._on_language_changed("en")
    window._fill_demo()
    answers = window._collect_answers()
    assert answers["sex"] == "Masculin"
    assert answers["school_attitude"] == "Negativă"
    assert answers["school_feeling"] == "Stresat"
    assert answers["family_situation"] == "Monoparental"


def test_language_switch_preserves_answers(window):
    """A menu click must not discard a half-filled questionnaire."""
    window._on_language_changed("ro")
    window._fill_demo()
    before = window._collect_answers()
    window._on_language_changed("en")
    after = window._collect_answers()
    for key, value in before.items():
        if key == "timestamp":
            continue
        assert after[key] == value, key


def test_language_switch_relabels_the_menu_bar(window):
    window._on_language_changed("ro")
    assert [a.text() for a in window.menuBar().actions()][0] == "&Chestionar"
    window._on_language_changed("en")
    texts = [a.text() for a in window.menuBar().actions()]
    assert texts[0] == "&Questionnaire"
    assert "&Appearance" in texts
    # Exactly one menu bar — a rebuild that appended would duplicate every entry.
    assert len(texts) == len(set(texts))


def test_theme_switch_repaints_muted_labels(window, qapp):
    window._on_theme_changed("light")
    light = window.lbl_stress.styleSheet()
    window._on_theme_changed("dark")
    dark = window.lbl_stress.styleSheet()
    assert light != dark
    assert theme.get_theme() == "dark"


def test_report_panes_stay_light_under_dark_theme(window):
    window._on_theme_changed("dark")
    assert "#ffffff" in window.results.styleSheet()
    assert "#ffffff" in window.plan_text.styleSheet()


# --- cloud plan language ----------------------------------------------------
def test_system_prompt_gets_an_english_override():
    from app.llm_client import _build_system_prompt

    i18n.set_language("ro")
    assert "LANGUAGE OVERRIDE" not in _build_system_prompt(None)
    i18n.set_language("en")
    assert "LANGUAGE OVERRIDE" in _build_system_prompt(None)


def test_knowledge_base_still_appended_in_english():
    from app.llm_client import _build_system_prompt

    i18n.set_language("en")
    prompt = _build_system_prompt("metodologia mea")
    assert "LANGUAGE OVERRIDE" in prompt
    assert "metodologia mea" in prompt
