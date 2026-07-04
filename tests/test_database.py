"""Tests for the SQLite persistence layer."""

from app.database import Database
from app.explainability import evaluate
from app.models import StudentCase
from tests.conftest import HIGH_RISK


def _case() -> StudentCase:
    feats = {k: v for k, v in HIGH_RISK.items() if k != "Stres_Emotional_NLP"}
    return StudentCase(
        name="Test", surname="Elev", student_grade="IX A", school_name="Liceu",
        observation_text="text de test", features=feats,
    )


def test_save_and_reload_assessment(trained_model, tmp_path):
    model, _ = trained_model
    case = _case()
    features = dict(case.features)
    features["Stres_Emotional_NLP"] = 1.5
    ev = evaluate(model, features)

    db = Database(tmp_path / "t.db")
    try:
        case_id, eval_id = db.save_assessment(case, ev)
        assert case_id > 0 and eval_id > 0

        rows = db.recent_assessments()
        assert len(rows) == 1
        assert rows[0]["risk_band"] == ev.risk_band

        loaded_case, loaded_ev = db.load_evaluation(eval_id)
        assert loaded_case.surname == "Elev"
        assert abs(loaded_ev.probability - ev.probability) < 1e-9
        assert len(loaded_ev.attributions) == len(ev.attributions)
        assert len(loaded_ev.sub_scores) == len(ev.sub_scores)
    finally:
        db.close()


def test_schema_created_on_fresh_db(tmp_path):
    db = Database(tmp_path / "fresh.db")
    try:
        assert db.recent_assessments() == []
    finally:
        db.close()
