"""Application entry point: create the Qt app and show the main window."""

from __future__ import annotations

import os
import sys


def main() -> int:
    from PySide6.QtWidgets import QApplication

    from .config import APP_NAME, APP_TITLE
    from .ui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationDisplayName(APP_TITLE)

    window = MainWindow()
    window.show()

    # Smoke-test hook: construct the window AND run one real assessment (model
    # load + SHAP) so the packaged exe proves it can load xgboost/shap/PySide6
    # and resolve the bundled model. Exits cleanly with a non-zero code on error.
    if os.environ.get("RISKAPP_SELFTEST"):
        print("RISKAPP_SELFTEST: window constructed OK", flush=True)
        try:
            from .models import StudentCase
            from .service import AssessmentService

            case = StudentCase(
                observation_text="Elevul este retras, obosit și demotivat.",
                features={
                    "full_name": "Popescu Andrei",
                    "birth_date": "2009-05-14",
                    "student_class": "IX A",
                    "school_name": "Liceul Tehnologic",
                    "sex": "Masculin",
                    "Mediu_Rezidential": "Rural",
                    "Situatie_Familiala": "Monoparental",
                    "Educatie_Mama": "Gimnazial",
                    "Educatie_Tata": "Primar",
                    "Absente_Nemotivate_Zilele_1_13": 12,
                    "Absente_Motivate_3_Luni": 4,
                    "Participare_Extrascolara": "Nu",
                    "Medie_Modul_Anterior": 4.3,
                    "Note_Sub_5": 3,
                    "Studentship_Score": 2,
                    "Atitudine_Scoala": "Negativă",
                    "Sanctiuni_Avertismente": "Avertismente",
                    "Cum_te_Simti_La_Scoala": "Stresat",
                    "Scoala_Ajuta_Obiective": "Nu",
                },
            )
            result = AssessmentService().assess(case)
            ev = result.evaluation
            recon = ev.base_value + sum(a.shap_value for a in ev.attributions)
            print(
                f"RISKAPP_SELFTEST: assess OK band={ev.risk_band} "
                f"score={ev.aggregate_score:.1f} shap_additivity_gap="
                f"{abs(recon - ev.probability):.5f} model={ev.model_version}",
                flush=True,
            )
        except Exception as exc:  # frozen-bundle import/runtime failure
            print(f"RISKAPP_SELFTEST: FAILED {type(exc).__name__}: {exc}", flush=True)
            return 2
        return 0

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
