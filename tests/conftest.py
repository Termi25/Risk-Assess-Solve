"""Shared pytest fixtures."""

from __future__ import annotations

import warnings

import pytest

warnings.filterwarnings("ignore")


@pytest.fixture(scope="session")
def trained_model():
    """Train one small model for the whole test session (fast)."""
    from app.scoring_engine import train_model
    model, metrics = train_model(n_samples=500, seed=7)
    return model, metrics


HIGH_RISK = {
    "full_name": "Popescu Andrei",
    "birth_date": "2009-05-14",
    "student_class": "IX A",
    "school_name": "Liceul Tehnologic",
    "sex": "Masculin",
    "Mediu_Rezidential": "Rural",
    "Situatie_Familiala": "Monoparental",
    "Educatie_Mama": "Gimnazial",
    "Educatie_Tata": "Primar",
    "Absente_Nemotivate_Zilele_1_13": 95,
    "Absente_Motivate_3_Luni": 15,
    "Participare_Extrascolara": "Nu",
    "Medie_Modul_Anterior": 4.0,
    "low_grades_details": "4 Matematică; 3 Română; 2 Istorie",
    "Note_Sub_5": 3,
    "Studentship_Score": 1,
    "Atitudine_Scoala": "Negativă",
    "Sanctiuni_Avertismente": "Sancțiuni",
    "Cum_te_Simti_La_Scoala": "Stresat",
    "Scoala_Ajuta_Obiective": "Nu",
    "additional_notes": "Elevul este retras și obosit în ultima perioadă, refuză să participe la activități și pare demotivat.",
}

LOW_RISK = {
    "full_name": "Ionescu Maria",
    "birth_date": "2010-09-02",
    "student_class": "VIII B",
    "school_name": "Școala Gimnazială Nr. 1",
    "sex": "Feminin",
    "Mediu_Rezidential": "Urban",
    "Situatie_Familiala": "Ambii părinți",
    "Educatie_Mama": "Universitar",
    "Educatie_Tata": "Universitar",
    "Absente_Nemotivate_Zilele_1_13": 0,
    "Absente_Motivate_3_Luni": 1,
    "Participare_Extrascolara": "Da, frecvent",
    "Medie_Modul_Anterior": 9.2,
    "low_grades_details": "",
    "Note_Sub_5": 0,
    "Studentship_Score": 10,
    "Atitudine_Scoala": "Pozitivă",
    "Sanctiuni_Avertismente": "Nu",
    "Cum_te_Simti_La_Scoala": "Bine",
    "Scoala_Ajuta_Obiective": "Da",
    "additional_notes": "Motivată și implicată.",
}
