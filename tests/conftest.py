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


# Column headers copied verbatim from a real Google-Forms export, including its
# irregular spacing ("14.Cum", "8. b Nivelul", trailing blanks). ``excel_import``
# normalises these, so reproducing them exactly is what keeps that normalisation
# under test.
_FORM_HEADERS = (
    "Marcaj de timp",
    "1.  Nume și Prenume ",
    "2. Data nașterii ",
    "3. Clasa ",
    "4. Școala",
    "5.  Sexul",
    "6.  Mediul de proveniență ",
    "7. Situația familială",
    "7.a Dacă ai precizat altă situaţie la întrebarea de mai sus, descrie pe scurt",
    "8.a Nivelul de educație al părinților (mama)",
    "8. b Nivelul de educație al părinților (tata)",
    "9. Numărul absențelor nemotivate în ultimele 3 luni",
    "10. Numărul absențelor motivate în ultimele 3 luni",
    "11. Participarea la activități extrașcolare",
    "12. Media generale pe modulul anterior",
    "13. Note mai mici de 5 obținute la discipline în ultimul semestru "
    "(de specificat nota si materia)",
    "14.Cum ți-ai descrie atitudinea față de școală? ",
    "15. Ai primit sancțiuni sau avertismente disciplinare în ultimul an",
    "16. Cum te simți în general la școală?",
    "16. a Daca ai mentionat altele, descrie pe scurt.",
    "17. Consideri că școala te ajută să îți atingi obiectivele personale?",
)

# One high-risk and one low-risk fabricated respondent. Deliberately invented:
# the real questionnaire covers four minors at a named school, which is personal
# data under GDPR even with pseudonymised names, so it is git-ignored and must
# never become a committed fixture. See the `*.xlsx` rule in .gitignore.
_FORM_ROWS = (
    (
        "2026-03-02 09:14:22", "Elev Sintetic Unu", "2009-05-14", "IX A",
        "Liceul Tehnologic de Test", "Masculin", "Rural", "Monoparental", "",
        "Gimnazial", "Primar", 95, 15, "Nu", 4.0,
        "4 Matematică; 3 Română; 2 Istorie", "Negativă", "Sancțiuni",
        "Stresat", "Se simte copleșit de cerințe.", "Nu",
    ),
    (
        "2026-03-02 09:31:07", "Elev Sintetic Doi", "2010-09-02", "VIII B",
        "Școala Gimnazială de Test", "Feminin", "Urban", "Ambii părinți", "",
        "Universitar", "Universitar", 0, 1, "Da, frecvent", 9.2, "",
        "Pozitivă", "Nu", "Bine", "", "Da",
    ),
)


@pytest.fixture(scope="session")
def sample_workbook(tmp_path_factory):
    """A synthetic Google-Forms responses .xlsx, written fresh for the session."""
    import openpyxl

    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Răspunsuri la formular 1"
    sheet.append(list(_FORM_HEADERS))
    for row in _FORM_ROWS:
        sheet.append(list(row))
    path = tmp_path_factory.mktemp("workbook") / "raspunsuri_test.xlsx"
    workbook.save(path)
    return str(path)


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
