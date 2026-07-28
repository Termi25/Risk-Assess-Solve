"""Application UI theme (system / light / dark).

The app previously shipped no stylesheet at all, so it inherited whatever the OS
was set to — on a Windows 11 machine in dark mode the window rendered dark while
the few hard-coded ``color:#555`` labels stayed near-black and became unreadable.
Themes are applied through Fusion + an explicit ``QPalette`` rather than a raw
stylesheet, because Fusion is the only Qt style that honours a custom palette on
every widget class; the native Windows style ignores it for several of them.

Only the **application UI** is themed. The PDF report ([app/ui/report.py]) keeps
its own fixed light palette — it is a printed document, and its colours are
calibrated for paper and for the SHAP/LIME legends, not for the screen.
"""

from __future__ import annotations

# code -> Romanian display label (translated through i18n at menu build time)
THEMES: tuple[tuple[str, str], ...] = (
    ("system", "Sistem"),
    ("light", "Luminoasă (alb)"),
    ("dark", "Întunecată"),
)
DEFAULT_THEME = "system"

_current: str = DEFAULT_THEME

# Captured on the first apply so "system" can restore the original look.
_original_style: str | None = None
_original_palette = None

# Secondary/label text per theme — replaces the old hard-coded "#555".
_MUTED = {"system": "#555555", "light": "#5a6472", "dark": "#a8b0bb"}
_WARN = {"system": "#a15c00", "light": "#a15c00", "dark": "#e0a145"}
_OK = {"system": "#0a7d00", "light": "#0a7d00", "dark": "#5fd35a"}


def available_themes() -> tuple[tuple[str, str], ...]:
    return THEMES


def get_theme() -> str:
    return _current


def muted_color() -> str:
    """Hex colour for secondary text under the active theme."""
    return _MUTED.get(_current, _MUTED["system"])


def warning_color() -> str:
    return _WARN.get(_current, _WARN["system"])


def success_color() -> str:
    return _OK.get(_current, _OK["system"])


def _light_palette():
    from PySide6.QtGui import QColor, QPalette

    p = QPalette()
    window, ink = QColor("#ffffff"), QColor("#1f1f1f")
    p.setColor(QPalette.Window, window)
    p.setColor(QPalette.WindowText, ink)
    p.setColor(QPalette.Base, QColor("#ffffff"))
    p.setColor(QPalette.AlternateBase, QColor("#f4f6f9"))
    p.setColor(QPalette.Text, ink)
    p.setColor(QPalette.PlaceholderText, QColor("#8a919c"))
    p.setColor(QPalette.Button, QColor("#f2f4f7"))
    p.setColor(QPalette.ButtonText, ink)
    p.setColor(QPalette.ToolTipBase, QColor("#ffffff"))
    p.setColor(QPalette.ToolTipText, ink)
    p.setColor(QPalette.BrightText, QColor("#d64550"))
    p.setColor(QPalette.Link, QColor("#1f3a5f"))
    p.setColor(QPalette.Highlight, QColor("#1f3a5f"))
    p.setColor(QPalette.HighlightedText, QColor("#ffffff"))
    _apply_disabled(p, QColor("#9aa1ac"))
    return p


def _dark_palette():
    from PySide6.QtGui import QColor, QPalette

    p = QPalette()
    ink = QColor("#e6e8ea")
    p.setColor(QPalette.Window, QColor("#1e1f22"))
    p.setColor(QPalette.WindowText, ink)
    p.setColor(QPalette.Base, QColor("#17181a"))
    p.setColor(QPalette.AlternateBase, QColor("#232529"))
    p.setColor(QPalette.Text, ink)
    p.setColor(QPalette.PlaceholderText, QColor("#7c848f"))
    p.setColor(QPalette.Button, QColor("#2a2c31"))
    p.setColor(QPalette.ButtonText, ink)
    p.setColor(QPalette.ToolTipBase, QColor("#2a2c31"))
    p.setColor(QPalette.ToolTipText, ink)
    p.setColor(QPalette.BrightText, QColor("#ff6b6b"))
    p.setColor(QPalette.Link, QColor("#6fa8ff"))
    p.setColor(QPalette.Highlight, QColor("#3b7dd8"))
    p.setColor(QPalette.HighlightedText, QColor("#ffffff"))
    _apply_disabled(p, QColor("#6b727c"))
    return p


def _apply_disabled(palette, colour) -> None:
    from PySide6.QtGui import QPalette

    for role in (QPalette.WindowText, QPalette.Text, QPalette.ButtonText):
        palette.setColor(QPalette.Disabled, role, colour)


# Minimal stylesheet: Fusion draws group boxes and the menu bar with a flat
# background that reads as unfinished against a pure-white window, so only those
# few surfaces are nudged. Everything else comes from the palette.
_QSS = """
QGroupBox {{
    border: 1px solid {border};
    border-radius: 6px;
    margin-top: 10px;
    padding-top: 8px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
    color: {title};
    font-weight: bold;
}}
QMenuBar, QStatusBar {{ background: {chrome}; }}
QTextBrowser, QPlainTextEdit, QLineEdit {{
    border: 1px solid {border};
    border-radius: 4px;
}}
"""


def document_qss() -> str:
    """Style for the panes that preview *document* content.

    The report preview, the plan pane and the metrics viewer all render the same
    HTML that goes into the PDF, whose palette is calibrated for paper (see
    ``report._INK``). Letting them inherit a dark palette would put that
    near-black body text on a near-black background, so document surfaces stay
    white in every theme — the same convention PDF readers use in dark mode.
    """
    return "background:#ffffff; color:#2b2b2b;"


def stylesheet_for(code: str) -> str:
    if code == "light":
        return _QSS.format(border="#d8dee8", title="#1f3a5f", chrome="#f7f9fc")
    if code == "dark":
        return _QSS.format(border="#3a3d44", title="#9dc0ff", chrome="#232529")
    return ""


def apply_theme(app, code: str) -> str:
    """Apply a theme to the running ``QApplication``. Returns the code used.

    Safe to call repeatedly — switching themes at runtime restyles every open
    window, so no restart is needed.
    """
    global _current, _original_style, _original_palette

    if not any(c == code for c, _ in THEMES):
        code = DEFAULT_THEME

    if _original_palette is None:
        _original_palette = app.palette()
        style = app.style()
        _original_style = style.objectName() if style is not None else None

    if code == "system":
        if _original_style:
            app.setStyle(_original_style)
        app.setPalette(_original_palette)
        app.setStyleSheet("")
    else:
        app.setStyle("Fusion")
        app.setPalette(_light_palette() if code == "light" else _dark_palette())
        app.setStyleSheet(stylesheet_for(code))

    _current = code
    return code


def load_theme(app) -> str:
    """Apply the theme stored in settings.json (called once at startup)."""
    from ..settings import get_theme as stored

    return apply_theme(app, stored())
