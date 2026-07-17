"""Render the model's conventional evaluation metrics as PNG images.

Creates a ``metrici_model/`` folder — next to the generated PDF reports — holding
the conventional dropout-prediction evaluation suite the literature reports, so
the research paper can compare like-for-like:

    * matrice_confuzie.png       — TN/FP/FN/TP on the held-out test set
    * curba_roc.png              — ROC with AUC
    * curba_precizie_recall.png  — PR curve with average precision (PR-AUC)
    * curba_calibrare.png        — reliability diagram with the Brier score
    * shap_global.png            — global SHAP feature attributions (xAI)
    * sumar_metrici.png          — a table of every scalar metric + k-fold CV

Every figure is best-effort: a single failing plot is skipped rather than
aborting report generation. Uses the non-interactive ``Agg`` backend, so it is
safe to call from a background (non-GUI) thread.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import config
from .scoring_engine import RiskModel, holdout_predictions

SUBDIR_NAME = "metrici_model"

# Chart text is English (the figures are meant for an English-language paper);
# the surrounding app UI stays Romanian.
# 0 = retention, 1 = dropout risk (the model's positive class).
_CLASS_LABELS = ["Retention", "Dropout risk"]

# Human-readable English feature names for the SHAP plot, keyed by the model's
# internal column names (config.FEATURE_KEYS). Falls back to the raw key.
_FEATURE_LABELS_EN = {
    "Age_Years": "Age (years)",
    "Sex": "Sex",
    "Medie_Modul_Anterior": "Previous module average",
    "Mediu_Rezidential": "Residential area",
    "Situatie_Familiala": "Family situation",
    "Educatie_Mama": "Mother's education",
    "Educatie_Tata": "Father's education",
    "Absente_Nemotivate_Zilele_1_13": "Unexcused absences (3 mo.)",
    "Absente_Motivate_3_Luni": "Excused absences (3 mo.)",
    "Participare_Extrascolara": "Extracurricular participation",
    "Note_Sub_5": "Grades below 5 (count)",
    "Studentship_Score": "Studentship score",
    "Atitudine_Scoala": "Attitude toward school",
    "Sanctiuni_Avertismente": "Disciplinary sanctions",
    "Cum_te_Simti_La_Scoala": "How they feel at school",
    "Scoala_Ajuta_Obiective": "School supports goals",
    "Stres_Emotional_NLP": "Emotional stress (NLP)",
}


# --- number formatting for the summary table --------------------------------
def _fmt(value) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "—"
    return "—" if np.isnan(v) else f"{v:.3f}"


def _fmt_pm(mean, std) -> str:
    try:
        m, s = float(mean), float(std)
    except (TypeError, ValueError):
        return "—"
    return "—" if np.isnan(m) else f"{m:.3f} ± {s:.3f}"


# --- individual figures ------------------------------------------------------
def _confusion(plt, path: Path, y_true, y_pred) -> None:
    from sklearn.metrics import ConfusionMatrixDisplay

    fig, ax = plt.subplots(figsize=(5.0, 4.3))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, labels=[0, 1], display_labels=_CLASS_LABELS,
        cmap="Blues", ax=ax, colorbar=False,
    )
    ax.set_title("Confusion matrix (test set)")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _roc(plt, path: Path, y_true, y_proba) -> None:
    from sklearn.metrics import RocCurveDisplay, roc_auc_score

    fig, ax = plt.subplots(figsize=(5.0, 4.3))
    RocCurveDisplay.from_predictions(y_true, y_proba, ax=ax, name="XGBoost")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)
    auc = roc_auc_score(y_true, y_proba)
    ax.set_title(f"ROC curve — AUC = {auc:.3f}")
    ax.set_xlabel("False-positive rate (1 − specificity)")
    ax.set_ylabel("True-positive rate (sensitivity)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _precision_recall(plt, path: Path, y_true, y_proba) -> None:
    from sklearn.metrics import PrecisionRecallDisplay, average_precision_score

    fig, ax = plt.subplots(figsize=(5.0, 4.3))
    PrecisionRecallDisplay.from_predictions(y_true, y_proba, ax=ax, name="XGBoost")
    ap = average_precision_score(y_true, y_proba)
    base = float(np.mean(y_true))
    ax.axhline(base, linestyle="--", color="grey", linewidth=1,
               label=f"base rate = {base:.2f}")
    ax.set_title(f"Precision–recall curve — PR-AUC = {ap:.3f}")
    ax.set_xlabel("Recall (sensitivity)")
    ax.set_ylabel("Precision")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _calibration(plt, path: Path, y_true, y_proba) -> None:
    from sklearn.calibration import CalibrationDisplay
    from sklearn.metrics import brier_score_loss

    fig, ax = plt.subplots(figsize=(5.0, 4.3))
    CalibrationDisplay.from_predictions(y_true, y_proba, n_bins=10, ax=ax, name="XGBoost")
    brier = brier_score_loss(y_true, y_proba)
    ax.set_title(f"Calibration plot — Brier = {brier:.3f}")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed positive fraction")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _summary_table(plt, path: Path, model: RiskModel, metrics: dict) -> None:
    rows = [
        ("Accuracy", _fmt(metrics.get("accuracy"))),
        ("Balanced accuracy", _fmt(metrics.get("balanced_accuracy"))),
        ("ROC-AUC", _fmt(metrics.get("roc_auc"))),
        ("PR-AUC (average precision)", _fmt(metrics.get("pr_auc"))),
        ("Precision (dropout)", _fmt(metrics.get("precision_dropout"))),
        ("Recall / sensitivity (dropout)", _fmt(metrics.get("recall_dropout"))),
        ("Specificity", _fmt(metrics.get("specificity"))),
        ("F1 (dropout)", _fmt(metrics.get("f1_dropout"))),
        ("G-mean", _fmt(metrics.get("g_mean"))),
        ("MCC (Matthews)", _fmt(metrics.get("mcc"))),
        ("Brier (calibration, ↓)", _fmt(metrics.get("brier"))),
    ]
    folds = int(metrics.get("cv_folds") or 0)
    if folds:
        rows.append((f"CV accuracy ({folds}-fold)",
                     _fmt_pm(metrics.get("cv_accuracy_mean"), metrics.get("cv_accuracy_std"))))
        rows.append((f"CV ROC-AUC ({folds}-fold)",
                     _fmt_pm(metrics.get("cv_roc_auc_mean"), metrics.get("cv_roc_auc_std"))))
        rows.append((f"CV PR-AUC ({folds}-fold)",
                     _fmt_pm(metrics.get("cv_pr_auc_mean"), metrics.get("cv_pr_auc_std"))))
        rows.append((f"CV F1 ({folds}-fold)",
                     _fmt_pm(metrics.get("cv_f1_mean"), metrics.get("cv_f1_std"))))

    fig, ax = plt.subplots(figsize=(6.8, 0.34 * (len(rows) + 1) + 0.7))
    ax.axis("off")
    ax.set_title(f"Model metrics — {model.version}", fontsize=11, pad=10, loc="left")
    # Fill the axes with the table (bbox) so there is no centering whitespace.
    table = ax.table(
        cellText=rows, colLabels=["Metric", "Value"],
        colWidths=[0.72, 0.28], cellLoc="left", bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#d8dee8")
        if r == 0:
            cell.set_facecolor("#1f3a5f")
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f2f4f7")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _shap_summary(plt, path: Path, model: RiskModel, n_samples: int, seed: int) -> None:
    """Global SHAP summary in log-odds space (mirrors the train.py figure)."""
    import shap

    from .scoring_engine import _encode_dataframe, generate_synthetic_dataset

    df = generate_synthetic_dataset(n_samples, seed)
    X = _encode_dataframe(df[list(config.FEATURE_KEYS)]).astype(float)
    explainer = shap.TreeExplainer(model.clf, feature_perturbation="tree_path_dependent")
    shap_values = explainer.shap_values(X)

    feature_names = [_FEATURE_LABELS_EN.get(k, k) for k in X.columns]
    plt.figure(figsize=(10, 5))
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
    plt.title("Global xAI analysis (SHAP) — dropout risk factors", fontsize=12, pad=15)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# --- public entry point ------------------------------------------------------
def save_metric_images(
    model: RiskModel, out_dir, *, subdir: str = SUBDIR_NAME,
) -> tuple[Path, list[Path]]:
    """Render the full metric-image suite into ``<out_dir>/<subdir>/``.

    Returns ``(folder, created_paths)``. Best-effort per figure: a plot that
    fails (e.g. a degenerate calibration bin) is skipped so the rest still land.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    target = Path(out_dir) / subdir
    target.mkdir(parents=True, exist_ok=True)

    metrics = model.meta.metrics or {}
    n_samples = int(metrics.get("n_samples") or config.TRAIN_SAMPLES)
    seed = int(metrics.get("seed") or config.RANDOM_SEED)
    y_true, y_pred, y_proba = holdout_predictions(model, n_samples=n_samples, seed=seed)

    figures = (
        ("matrice_confuzie.png", lambda p: _confusion(plt, p, y_true, y_pred)),
        ("curba_roc.png", lambda p: _roc(plt, p, y_true, y_proba)),
        ("curba_precizie_recall.png", lambda p: _precision_recall(plt, p, y_true, y_proba)),
        ("curba_calibrare.png", lambda p: _calibration(plt, p, y_true, y_proba)),
        ("sumar_metrici.png", lambda p: _summary_table(plt, p, model, metrics)),
        ("shap_global.png", lambda p: _shap_summary(plt, p, model, n_samples, seed)),
    )

    created: list[Path] = []
    for name, draw in figures:
        path = target / name
        try:
            draw(path)
            created.append(path)
        except Exception:
            plt.close("all")  # skip this figure, keep the rest
    return target, created
