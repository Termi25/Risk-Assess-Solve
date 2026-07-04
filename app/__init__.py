"""Student Risk Assessment & Intervention Planner — proof of concept.

Package layout
--------------
- ``config``          : constants, feature schema, paths.
- ``models``          : plain domain dataclasses (mirror the README data model).
- ``database``        : local SQLite persistence.
- ``nlp_engine``      : BERT-style qualitative valence / emotional-stress signal.
- ``scoring_engine``  : real XGBoost + SMOTE-NC risk model (the "Day 14" model).
- ``explainability``  : genuine SHAP TreeExplainer attributions.
- ``llm_client``      : optional cloud step (Anthropic) with offline fallback.
- ``ui``              : PySide6 desktop interface.

The machine-learning core (``scoring_engine`` + ``explainability``) has **no**
GUI dependency and can be imported and tested on its own.
"""

__version__ = "0.1.0"
