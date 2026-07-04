"""Train the dropout-risk model and persist it into the app bundle.

Run this before packaging so the exe ships with a ready-to-use model:

    python train.py                 # train + save app/artifacts/risk_model.json
    python train.py --plot fig.png  # also export the research-style SHAP figure
    python train.py --samples 2000  # override synthetic sample size

The saved artifact is what ``config.bundled_model_path()`` resolves to, so the
packaged application loads it instantly instead of training on first launch.
"""

from __future__ import annotations

import argparse
import sys

from app import config
from app.scoring_engine import (
    _encode_dataframe,
    generate_synthetic_dataset,
    save_model,
    train_model,
)


def _print_metrics(metrics) -> None:
    print("=" * 60)
    print("REZULTATE ANTRENARE MODEL (EWS 'Day 14')")
    print("=" * 60)
    print(f"Acuratețe (test):     {metrics.accuracy:.3f}")
    print(f"ROC AUC (test):       {metrics.roc_auc:.3f}")
    print(f"Distribuție înainte de SMOTE-NC: {metrics.balance_before} "
          f"(n={metrics.n_before})")
    print(f"Distribuție după SMOTE-NC:       {metrics.balance_after} "
          f"(n={metrics.n_after})")
    print("\nRaport de clasificare:\n")
    print(metrics.report_text)


def _export_shap_plot(model, path: str, samples: int, seed: int) -> None:
    """Reproduce the research-style global SHAP summary plot (log-odds space).

    Uses shap.TreeExplainer with the tree-path-dependent perturbation, matching
    the methodology figures. Requires matplotlib (dev/reporting dependency).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    df = generate_synthetic_dataset(samples, seed)
    X = _encode_dataframe(df[list(config.FEATURE_KEYS)]).astype(float)
    explainer = shap.TreeExplainer(model.clf, feature_perturbation="tree_path_dependent")
    shap_values = explainer.shap_values(X)

    plt.figure(figsize=(10, 5))
    shap.summary_plot(shap_values, X, show=False)
    plt.title("Analiză xAI globală (SHAP) — factori de risc de abandon", fontsize=12, pad=15)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\nFigură SHAP salvată: {path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Antrenează modelul de risc de abandon.")
    parser.add_argument("--samples", type=int, default=config.TRAIN_SAMPLES)
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    parser.add_argument("--plot", metavar="PATH", default=None,
                        help="Exportă graficul SHAP summary (necesită matplotlib).")
    args = parser.parse_args(argv)

    model, metrics = train_model(n_samples=args.samples, seed=args.seed)
    _print_metrics(metrics)

    save_model(model, config.bundled_model_path(), config.bundled_meta_path())
    print(f"\nModel salvat în: {config.bundled_model_path()}")
    print(f"Versiune model:   {model.version}")

    if args.plot:
        _export_shap_plot(model, args.plot, args.samples, args.seed)

    return 0


if __name__ == "__main__":
    sys.exit(main())
