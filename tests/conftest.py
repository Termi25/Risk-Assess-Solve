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
    "Medie_Modul_Anterior": 4.0,
    "Note_Sub_7": 5,
    "Absente_Nemotivate_Zilele_1_13": 13,
    "Studentship_Score": 1,
    "Stres_Emotional_NLP": 1.9,
    "Mediu_Rezidential": "Rural",
    "Parinti_In_Strainatate": "Da",
    "Vulnerabilitate_Financiara": "Ridicata",
}

LOW_RISK = {
    "Medie_Modul_Anterior": 9.2,
    "Note_Sub_7": 0,
    "Absente_Nemotivate_Zilele_1_13": 0,
    "Studentship_Score": 10,
    "Stres_Emotional_NLP": 0.0,
    "Mediu_Rezidential": "Urban",
    "Parinti_In_Strainatate": "Nu",
    "Vulnerabilitate_Financiara": "Scazuta",
}
