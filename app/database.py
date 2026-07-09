"""Local SQLite persistence.

Stores student cases and their computed risk evaluations locally, matching the
README's "keep processing on-device" principle. The README calls for the DB to
be *encrypted at rest*; a production build would layer SQLCipher (or the OS
credential store for the key) underneath this same interface. For the
proof-of-concept the schema and access pattern are what matter, so this uses the
standard-library ``sqlite3`` driver.

Nothing here imports the ML or GUI layers, so it can be tested in isolation.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from . import config
from .models import FeatureAttribution, RiskEvaluation, StudentCase, SubScore


_SCHEMA = """
CREATE TABLE IF NOT EXISTS student_cases (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    name             TEXT NOT NULL DEFAULT '',
    surname          TEXT NOT NULL DEFAULT '',
    school_name      TEXT NOT NULL DEFAULT '',
    student_grade    TEXT NOT NULL DEFAULT '',
    is_urban         INTEGER NOT NULL DEFAULT 1,
    observation_text TEXT NOT NULL DEFAULT '',
    features_json    TEXT NOT NULL DEFAULT '{}',
    created_at       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS risk_evaluations (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    student_case_id    INTEGER NOT NULL REFERENCES student_cases(id) ON DELETE CASCADE,
    probability        REAL NOT NULL,
    aggregate_score    REAL NOT NULL,
    risk_band          TEXT NOT NULL,
    base_value         REAL NOT NULL,
    attributions_json  TEXT NOT NULL DEFAULT '[]',
    sub_scores_json    TEXT NOT NULL DEFAULT '[]',
    action_plan_text   TEXT NOT NULL DEFAULT '',
    action_plan_source TEXT NOT NULL DEFAULT '',
    model_version      TEXT NOT NULL DEFAULT '',
    created_at         TEXT NOT NULL
);
"""


class Database:
    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else config.database_path()
        self.conn = sqlite3.connect(str(self.path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    # --- writes ------------------------------------------------------------
    def save_case(self, case: StudentCase) -> int:
        cur = self.conn.execute(
            """INSERT INTO student_cases
               (name, surname, school_name, student_grade, is_urban,
                observation_text, features_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                case.name, case.surname, case.school_name, case.student_grade,
                1 if case.is_urban else 0, case.observation_text,
                json.dumps(case.features, ensure_ascii=False), case.created_at,
            ),
        )
        self.conn.commit()
        case.id = int(cur.lastrowid)
        return case.id

    def save_evaluation(self, evaluation: RiskEvaluation) -> int:
        cur = self.conn.execute(
            """INSERT INTO risk_evaluations
               (student_case_id, probability, aggregate_score, risk_band,
                base_value, attributions_json, sub_scores_json,
                action_plan_text, action_plan_source, model_version, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                evaluation.student_case_id, evaluation.probability,
                evaluation.aggregate_score, evaluation.risk_band,
                evaluation.base_value,
                json.dumps([asdict(a) for a in evaluation.attributions], ensure_ascii=False),
                json.dumps([asdict(s) for s in evaluation.sub_scores], ensure_ascii=False),
                evaluation.action_plan_text, evaluation.action_plan_source,
                evaluation.model_version, evaluation.created_at,
            ),
        )
        self.conn.commit()
        evaluation.id = int(cur.lastrowid)
        return evaluation.id

    def save_assessment(self, case: StudentCase, evaluation: RiskEvaluation) -> tuple[int, int]:
        case_id = self.save_case(case)
        evaluation.student_case_id = case_id
        eval_id = self.save_evaluation(evaluation)
        return case_id, eval_id

    # --- reads -------------------------------------------------------------
    def recent_assessments(self, limit: int = 100) -> list[dict]:
        rows = self.conn.execute(
            """SELECT e.id AS eval_id, e.aggregate_score, e.risk_band,
                      e.action_plan_source, e.model_version, e.created_at,
                      c.name, c.surname, c.student_grade, c.school_name
               FROM risk_evaluations e
               JOIN student_cases c ON c.id = e.student_case_id
               ORDER BY e.id DESC
               LIMIT ?""",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def load_evaluation(self, eval_id: int) -> Optional[tuple[StudentCase, RiskEvaluation]]:
        row = self.conn.execute(
            """SELECT e.*, c.name, c.surname, c.school_name, c.student_grade,
                      c.is_urban, c.observation_text, c.features_json
               FROM risk_evaluations e
               JOIN student_cases c ON c.id = e.student_case_id
               WHERE e.id = ?""",
            (eval_id,),
        ).fetchone()
        if row is None:
            return None

        case = StudentCase(
            id=row["student_case_id"],
            name=row["name"], surname=row["surname"],
            school_name=row["school_name"], student_grade=row["student_grade"],
            is_urban=bool(row["is_urban"]),
            observation_text=row["observation_text"],
            features=json.loads(row["features_json"]),
        )
        evaluation = RiskEvaluation(
            id=row["id"], student_case_id=row["student_case_id"],
            probability=row["probability"], aggregate_score=row["aggregate_score"],
            risk_band=row["risk_band"], base_value=row["base_value"],
            urgency=config.tier_for_band(row["risk_band"]).urgency,
            attributions=[FeatureAttribution(**a) for a in json.loads(row["attributions_json"])],
            sub_scores=[SubScore(**s) for s in json.loads(row["sub_scores_json"])],
            action_plan_text=row["action_plan_text"],
            action_plan_source=row["action_plan_source"],
            model_version=row["model_version"],
            created_at=row["created_at"],
        )
        return case, evaluation

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
