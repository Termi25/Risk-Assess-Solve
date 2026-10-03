"""Interface language (Romanian / English).

The app was written Romanian-first, so the *source string is the key*: ``tr()``
looks the Romanian text up in the English catalog and returns it unchanged when
there is no entry. A missing translation therefore degrades to Romanian rather
than to an empty label or a raised key error.

**The model's category vocabulary is never translated.** Values such as
``"Feminin"`` or ``"Ambii părinți"`` are the literal strings the classifier was
trained on (see ``scoring_engine._FAMILY_RISK`` and friends), so they must reach
the feature frame in Romanian regardless of interface language. ``tr_value()``
exists purely to *display* a category; callers keep the Romanian value as the
widget's ``userData`` and read it back with ``currentData()``. Translating the
stored value instead of the label would silently change what the model is fed.
"""

from __future__ import annotations

from . import config

LANGUAGES: tuple[tuple[str, str], ...] = (
    ("ro", "Română"),
    ("en", "English"),
)
DEFAULT_LANGUAGE = "ro"

_current: str = DEFAULT_LANGUAGE


def available_languages() -> tuple[tuple[str, str], ...]:
    return LANGUAGES


def get_language() -> str:
    """The active language code, loaded from settings on first use."""
    return _current


def set_language(code: str) -> None:
    """Set the in-process language. Persisting is the caller's job."""
    global _current
    _current = code if any(c == code for c, _ in LANGUAGES) else DEFAULT_LANGUAGE


def load_language() -> str:
    """Initialise the language from settings.json (called once at startup)."""
    from .settings import get_language as stored
    set_language(stored())
    return _current


def tr(text: str) -> str:
    """Translate an interface string. Romanian input, Romanian fallback."""
    if _current == "ro":
        return text
    return _UI.get(text, text)


def tr_value(value: str) -> str:
    """Translate a *displayed* category label only — never the stored value."""
    if _current == "ro":
        return value
    return _VALUES.get(value, value)


def tr_band(band: str) -> str:
    """Translate a risk-band name (``Scăzut`` / ``Mediu`` / …) for display."""
    band = config.normalize_band(band)
    if _current == "ro":
        return band
    return _BANDS.get(band, band)


def plan_language_name() -> str:
    """Language the LLM should write the intervention plan in."""
    return "English" if _current == "en" else "Romanian"


# --- Catalogs ---------------------------------------------------------------
# Model category values. Display-only: the stored value stays Romanian.
_VALUES: dict[str, str] = {
    # sex
    "Feminin": "Female",
    "Masculin": "Male",
    "Altul / prefer să nu spun": "Other / prefer not to say",
    # residential environment
    "Urban": "Urban",
    "Rural": "Rural",
    # family situation
    "Ambii părinți": "Both parents",
    "Monoparental": "Single parent",
    "Tutore / plasament": "Guardian / foster care",
    "Părinți divortați/separați": "Divorced / separated parents",
    "Altă situație": "Other situation",
    # parental education
    "Primar": "Primary",
    "Gimnazial": "Lower secondary",
    "Liceal": "Upper secondary",
    "Postliceal": "Post-secondary",
    "Universitar": "University",
    "Nu se aplică/Necunoscut": "Not applicable / unknown",
    "Necunoscut": "Unknown",
    # extracurricular participation
    "Da, frecvent": "Yes, frequently",
    "Ocazional": "Occasionally",
    # attitude
    "Pozitivă": "Positive",
    "Neutră": "Neutral",
    "Negativă": "Negative",
    # sanctions
    "Nu": "No",
    "Avertismente": "Warnings",
    "Sancțiuni": "Sanctions",
    # school feeling
    "Bine": "Good",
    "Neutru": "Neutral",
    "Stresat": "Stressed",
    "Izolat": "Isolated",
    "Altul": "Other",
    # school support
    "Da": "Yes",
    "Parțial": "Partly",
}

# Risk bands and urgency levels.
_BANDS: dict[str, str] = {
    "Scăzut": "Low",
    "Mediu": "Medium",
    "Ridicat": "High",
    "Critic": "Critical",
    "Monitorizare": "Monitoring",
    "Medie": "Medium",
    "Ridicată": "High",
    "Maximă": "Maximum",
}

# Interface strings.
_UI: dict[str, str] = {
    # --- app / window ---
    "Evaluare Risc Abandon Școlar & Plan de Intervenție":
        "School Dropout Risk Assessment & Intervention Plan",
    # --- menus ---
    "&Chestionar": "&Questionnaire",
    "Importă răspunsuri Google Forms (.xlsx)…":
        "Import Google Forms responses (.xlsx)…",
    "&Setări": "&Settings",
    "Bază de cunoștințe (.docx)…": "Knowledge base (.docx)…",
    "&Aspect": "&Appearance",
    "Temă": "Theme",
    "Limbă": "Language",
    "Sistem": "System",
    "Luminoasă (alb)": "Light (white)",
    "Întunecată": "Dark",
    "&Model": "&Model",
    "Reantrenează modelul": "Retrain the model",
    "Metrici model…": "Model metrics…",
    "Salvează graficele metricilor…": "Save metric charts…",
    "&Performanță": "&Performance",
    "Metrici pe rulare (latență / cost)…": "Per-run metrics (latency / cost)…",
    "&Ajutor": "&Help",
    "Despre": "About",
    # --- main form ---
    "Date de identificare (rămân doar local)":
        "Identification data (stays local only)",
    "Chestionar complet": "Full questionnaire",
    "Stres emoțional (NLP): — (se calculează la evaluare)":
        "Emotional stress (NLP): — (computed at assessment)",
    "Completează exemplu": "Fill in example",
    "Evaluează riscul": "Assess risk",
    "Generează planul de intervenție": "Generate intervention plan",
    "Salvează evaluarea": "Save assessment",
    "Salvează raport PDF": "Save PDF report",
    "Plan de intervenție": "Intervention plan",
    "Sursă: —": "Source: —",
    "Gata. Modelul se încarcă la prima evaluare.":
        "Ready. The model loads on the first assessment.",
    # --- questionnaire labels ---
    "Marcaj de timp": "Timestamp",
    "Se completează automat la salvare.": "Filled in automatically on save.",
    "1. Nume și Prenume": "1. Full name",
    "2. Data nașterii": "2. Date of birth",
    "3. Clasa": "3. Class",
    "4. Școala": "4. School",
    "5. Sexul": "5. Sex",
    "6. Mediul de proveniență": "6. Residential environment",
    "7. Situația familială": "7. Family situation",
    "7.a Dacă ai precizat altă situație la întrebarea de mai sus, descrie pe scurt":
        "7.a If you selected another situation above, describe it briefly",
    "8.a Nivelul de educație al părinților (mama)":
        "8.a Parental education level (mother)",
    "8.b Nivelul de educație al părinților (tata)":
        "8.b Parental education level (father)",
    "9. Numărul absențelor nemotivate în ultimele 3 luni":
        "9. Unexcused absences in the last 3 months",
    "10. Numărul absențelor motivate în ultimele 3 luni":
        "10. Excused absences in the last 3 months",
    "11. Participarea la activități extrașcolare":
        "11. Participation in extracurricular activities",
    "12. Media generale pe modulul anterior":
        "12. Overall average for the previous module",
    "13. Note mai mici de 5 obținute la discipline în ultimul semestru (de specificat nota si materia)":
        "13. Grades below 5 in the last semester (specify grade and subject)",
    "14. Cum ți-ai descrie atitudinea față de școală?":
        "14. How would you describe your attitude towards school?",
    "15. Ai primit sancțiuni sau avertismente disciplinare în ultimul an":
        "15. Have you received disciplinary sanctions or warnings in the last year",
    "16. Cum te simți în general la școală?":
        "16. How do you generally feel at school?",
    "16.a Dacă ai menționat altele, descrie pe scurt.":
        "16.a If you selected other, describe it briefly.",
    "17. Consideri că școala te ajută să îți atingi obiectivele personale?":
        "17. Do you think school helps you reach your personal goals?",
    "Observații suplimentare": "Additional notes",
    # --- status / messages ---
    "Se evaluează elevii…": "Assessing students…",
    "Se generează planurile și rapoartele PDF…":
        "Generating plans and PDF reports…",
    "Se generează graficele metricilor…": "Generating metric charts…",
    "Se evaluează…": "Assessing…",
    "Se generează…": "Generating…",
    "Se generează planul…": "Generating the plan…",
    "Eroare": "Error",
    "Salvat": "Saved",
    "Șters": "Deleted",
    "Confirmă": "Confirm",
    "Document invalid": "Invalid document",
    "Cheie goală": "Empty key",
    "local (offline)": "local (offline)",
    # --- settings dialog ---
    "Setări — Cloud & Bază de cunoștințe": "Settings — Cloud & Knowledge base",
    "Conexiune cloud — furnizor și cheie API":
        "Cloud connection — provider and API key",
    "Furnizor:": "Provider:",
    "Afișează": "Show",
    "Salvează cheia": "Save key",
    "Șterge cheia": "Delete key",
    "Document curent": "Current document",
    "Previzualizare text extras": "Extracted text preview",
    "Alege document .docx…": "Choose .docx document…",
    "Elimină": "Remove",
    "Închide": "Close",
    "Alege documentul de cunoștințe": "Choose the knowledge document",
    "Documente Word (*.docx)": "Word documents (*.docx)",
    "<i>Niciun document încărcat.</i>": "<i>No document loaded.</i>",
    "Introdu o cheie API înainte de salvare.":
        "Enter an API key before saving.",
    "Cheia API a fost ștearsă din seif.":
        "The API key has been deleted from the vault.",
    "Atașează un document Word (.docx) — de regulă articolul de "
    "cercetare / metodologia — ca „bază de cunoștințe”. Textul lui este "
    "folosit pentru a ghida generarea planului de intervenție în cloud, "
    "aliniind planul la metodologia ta.":
        "Attach a Word document (.docx) — typically the research article / "
        "methodology — as a “knowledge base”. Its text is used to ground "
        "cloud generation of the intervention plan, aligning the plan with your "
        "methodology.",
    "ℹ️ Documentul este material metodologic (nu date despre elevi). "
    "Datele elevului rămân întotdeauna locale și anonimizate.":
        "ℹ️ The document is methodological material (not student data). Student "
        "data always stays local and anonymized.",
    "Opțional. Alege un furnizor și introdu cheia lui API pentru a genera "
    "planul de intervenție în cloud. Fără cheie, aplicația rămâne complet "
    "funcțională și folosește generatorul local. Cheile sunt păstrate "
    "securizat în seiful de credențiale al sistemului de operare "
    "(Windows Credential Manager) — niciodată în fișiere text.":
        "Optional. Choose a provider and enter its API key to generate the "
        "intervention plan in the cloud. Without a key the app stays fully "
        "functional and uses the local generator. Keys are stored securely in "
        "the operating system's credential vault (Windows Credential Manager) "
        "— never in text files.",
    "Activ: o cheie este salvată securizat în seiful sistemului. ✓":
        "Active: a key is stored securely in the system vault. ✓",
    "Nicio cheie pentru acest furnizor — se folosește "
    "generatorul local (offline).":
        "No key for this provider — the local (offline) generator is used.",
    " O cheie este și salvată în seif, dar este ignorată "
    "cât timp variabila de mediu există.":
        " A key is also stored in the vault, but it is ignored while the "
        "environment variable exists.",
    "Documente Word (*.docx)": "Word documents (*.docx)",
    # --- assessment / plan flow ---
    "Se calculează scorul (model XGBoost + SHAP)…":
        "Computing the score (XGBoost model + SHAP)…",
    "Se generează planul de intervenție…": "Generating the intervention plan…",
    "Plan generat.": "Plan generated.",
    "Eroare.": "Error.",
    "Salvează raportul ca PDF": "Save the report as PDF",
    "Fișier PDF (*.pdf)": "PDF file (*.pdf)",
    "Deschide folderul": "Open folder",
    # --- batch import ---
    "Alege fișierul cu răspunsuri (.xlsx exportat din Google Forms)":
        "Choose the responses file (.xlsx exported from Google Forms)",
    "Fișiere Excel (*.xlsx *.xlsm)": "Excel files (*.xlsx *.xlsm)",
    "Import eșuat": "Import failed",
    "Import eșuat.": "Import failed.",
    "Import": "Import",
    "Nu s-au găsit elevi în fișier.": "No students found in the file.",
    "Confirmare import": "Confirm import",
    "Alege folderul unde se salvează rapoartele PDF":
        "Choose the folder where the PDF reports are saved",
    "Se importă și se evaluează elevii…": "Importing and assessing students…",
    "Import finalizat": "Import complete",
    # --- model menu ---
    "Reantrenare": "Retrain",
    "Reantrenezi modelul pe date sintetice? (câteva secunde)":
        "Retrain the model on synthetic data? (a few seconds)",
    "Se reantrenează modelul…": "Retraining the model…",
    "Metrici model": "Model metrics",
    "Metrici indisponibile.": "Metrics unavailable.",
    "Se încarcă metricile modelului…": "Loading the model metrics…",
    "Alege folderul unde se salvează graficele metricilor":
        "Choose the folder where the metric charts are saved",
    "Grafice metrici": "Metric charts",
    "Grafice metrici salvate.": "Metric charts saved.",
    # --- model metrics box ---
    "— Metrici pe setul de test (echilibru real) —":
        "— Test-set metrics (real class balance) —",
    "Acuratețe": "Accuracy",
    "Acuratețe echilibrată": "Balanced accuracy",
    "Precizie (abandon)": "Precision (dropout)",
    "Recall/sensibilitate": "Recall / sensitivity",
    "Specificitate": "Specificity",
    "F1 (abandon)": "F1 (dropout)",
    "Matrice confuzie [[TN, FP], [FN, TP]]":
        "Confusion matrix [[TN, FP], [FN, TP]]",
    "Versiune": "Version",
    "Echilibrare înainte SMOTE-NC": "Balance before SMOTE-NC",
    "Echilibrare după SMOTE-NC": "Balance after SMOTE-NC",
    # --- about ---
    # --- model feature labels (config.FEATURES) ---
    # Shown as SHAP row labels and inside LIME rule text.
    "Vârsta (ani)": "Age (years)",
    "Sexul": "Sex",
    "Media modulului anterior": "Previous module average",
    "Mediu rezidențial": "Residential environment",
    "Situație familială": "Family situation",
    "Educația mamei": "Mother's education",
    "Educația tatălui": "Father's education",
    "Absențe nemotivate (3 luni)": "Unexcused absences (3 months)",
    "Absențe motivate (3 luni)": "Excused absences (3 months)",
    "Participare extrașcolară": "Extracurricular participation",
    "Număr note sub 5": "Number of grades below 5",
    "Scor Studentship (implicare 0–10)": "Studentship score (engagement 0–10)",
    "Atitudinea față de școală": "Attitude towards school",
    "Sancțiuni / avertismente": "Sanctions / warnings",
    "Cum se simte la școală": "How they feel at school",
    "Școala ajută obiectivele personale": "School supports personal goals",
    "Stres emoțional (NLP, 0–2)": "Emotional stress (NLP, 0–2)",
    # Attribution / condition direction (models.FeatureAttribution.direction).
    "crește riscul": "raises the risk",
    "reduce riscul": "lowers the risk",
    # --- report: titles and chrome ---
    "Analiză xAI și plan personalizat de intervenție":
        "xAI Analysis and Personalized Intervention Plan",
    "Raport de evaluare a riscului de abandon școlar":
        "School dropout risk assessment report",
    "CONFIDENȚIAL": "CONFIDENTIAL",
    "Datele introduse": "Input data",
    "Evaluarea riscului": "Risk assessment",
    "Planul de recomandare": "Recommendation plan",
    "Date de identificare și răspunsurile din chestionar, așa cum au fost "
    "introduse pentru această evaluare.":
        "Identification data and questionnaire answers, exactly as entered for "
        "this assessment.",
    "Nu au fost introduse date de chestionar.":
        "No questionnaire data was entered.",
    # meta strip
    "Elev": "Student",
    "Clasa": "Class",
    "Școala": "School",
    "Mediu": "Environment",
    "Generat": "Generated",
    "Model": "Model",
    # --- report: risk profile ---
    "Profilul de risc al elevului": "Student risk profile",
    "NIVEL DE RISC": "RISK LEVEL",
    "Urgență intervenție:": "Intervention urgency:",
    "Indicatori principali de risc:": "Main risk indicators:",
    "Niciun factor major de risc identificat.":
        "No major risk factor identified.",
    "Scor Studentship (implicare)": "Studentship score (engagement)",
    # --- report: xAI ---
    "Explicație xAI (de ce acest nivel de risc)":
        "xAI explanation (why this risk level)",
    "Factorii care justifică nivelul de risc, în ordinea importanței pentru "
    "model:":
        "The factors behind this risk level, in order of importance to the "
        "model:",
    "Profilul elevului este preponderent protectiv; niciun factor nu crește "
    "semnificativ riscul.":
        "This student's profile is predominantly protective; no factor raises "
        "the risk significantly.",
    "Contribuția factorilor (SHAP)": "Factor contributions (SHAP)",
    "Contribuția în puncte procentuale (roșu = crește riscul, verde = reduce "
    "riscul).":
        "Contribution in percentage points (red = raises risk, green = lowers "
        "risk).",
    "Sub-scoruri pe domenii": "Sub-scores by domain",
    "Ponderea fiecărui domeniu în explicația totală (%).":
        "Each domain's share of the total explanation (%).",
    "valoare": "value",
    # --- report: LIME ---
    "Profil individual de risc (LIME)": "Individual risk profile (LIME)",
    "Regulile care descriu situația <i>acestui</i> elev, așa cum le-a "
    "identificat un model local aproximativ (LIME), antrenat în jurul cazului "
    "său. Spre deosebire de analiza SHAP, ponderile de mai jos <b>nu se "
    "adună</b> la scorul final — ele arată ce anume diferențiază local acest "
    "elev, nu din ce se compune procentul.":
        "The rules describing <i>this</i> student's situation, as identified by "
        "a local approximate model (LIME) fitted around their case. Unlike the "
        "SHAP analysis, the weights below <b>do not add up</b> to the final "
        "score — they show what locally distinguishes this student, not what "
        "the percentage is composed of.",
    # --- report: NLP ---
    "Analiză text (NLP)": "Text analysis (NLP)",
    "Termeni negativi:": "Negative terms:",
    "Termeni pozitivi:": "Positive terms:",
    "Ton preponderent negativ / semne de stres":
        "Predominantly negative tone / signs of stress",
    "Ton preponderent pozitiv": "Predominantly positive tone",
    "Ton neutru / mixt": "Neutral / mixed tone",
    # --- report: LIME fidelity bands (models.LimeExplanation.fidelity_label) ---
    "bună": "good",
    "moderată": "moderate",
    "slabă": "weak",
    # --- report: sub-score domains (explainability.FEATURE_DOMAINS) ---
    "Frecvență": "Attendance",
    "Performanță academică": "Academic performance",
    "Context familial": "Family context",
    "Climat școlar": "School climate",
    "Stare emoțională (NLP)": "Emotional state (NLP)",
    "Profil demografic": "Demographic profile",
    "Altele": "Other",
    # --- report: critical indicators without numbers ---
    "Stare emoțională vulnerabilă (stres / izolare)":
        "Vulnerable emotional state (stress / isolation)",
    "Sancțiuni disciplinare active": "Active disciplinary sanctions",
    "Atitudine negativă față de școală": "Negative attitude towards school",
    "Nu percepe sprijin pentru obiectivele personale":
        "Does not perceive support for personal goals",
    "Fără participare extrașcolară": "No extracurricular participation",
    "Sprijin familial redus (situație monoparentală / tutore)":
        "Reduced family support (single-parent / guardian situation)",
    # --- report: plan + success indicators ---
    "Plan personalizat de intervenție": "Personalized intervention plan",
    "Contract de implicare al elevului: „Proiectul Podul”.":
        "Student engagement contract: “The Bridge Project”.",
    "Planul de intervenție nu a fost generat.":
        "The intervention plan was not generated.",
    "Indicatori de succes (4 săptămâni)": "Success indicators (4 weeks)",
    "Absențe: sub 2 absențe nemotivate pe săptămână.":
        "Absences: fewer than 2 unexcused absences per week.",
    "Participare: cel puțin un moment / o activitate școlară activă pe "
    "săptămână.":
        "Participation: at least one active school moment / activity per week.",
    "Atitudine: trecere spre „neutru / pozitiv” față de școală.":
        "Attitude: shift towards “neutral / positive” about school.",
    "Reevaluarea scorului de risc la finalul celor 4 săptămâni.":
        "Re-assessment of the risk score at the end of the 4 weeks.",
    # --- report: placeholder view ---
    "Niciun rezultat încă": "No result yet",
    "Scorul este calculat local de un model XGBoost real, iar explicația "
    "provine din valori SHAP autentice — nu dintr-o simulare a unui LLM.":
        "The score is computed locally by a real XGBoost model, and the "
        "explanation comes from genuine SHAP values — not from an LLM "
        "simulation.",
    # --- summary (group) report ---
    "Raport general — prioritizarea intervențiilor":
        "Summary report — intervention prioritization",
    "Evaluarea riscului de abandon școlar la nivel de grup":
        "Group-level school dropout risk assessment",
    "Elevi evaluați": "Students assessed",
    "Distribuția pe niveluri de risc": "Distribution by risk level",
    "Prioritizarea intervențiilor": "Intervention prioritization",
    "Elevii sunt ordonați după urgența intervenției: mai întâi nivelul de risc "
    "(Critic → Scăzut), apoi scorul modelului; la risc egal, o implicare "
    "(Studentship) mai scăzută urcă în prioritate.":
        "Students are ordered by intervention urgency: risk level first "
        "(Critical → Low), then the model score; at equal risk, lower "
        "engagement (Studentship) moves up the list.",
    "Nivel de risc": "Risk level",
    "Scor": "Score",
    "Urgență": "Urgency",
    "Implicare": "Engagement",
    "Factori principali": "Main factors",
    "Scorurile și explicațiile provin dintr-un model XGBoost real cu atribuiri "
    "SHAP autentice. Acest raport sintetizează evaluările individuale; pentru "
    "fiecare elev există un raport detaliat separat.":
        "Scores and explanations come from a real XGBoost model with genuine "
        "SHAP attributions. This report summarizes the individual assessments; "
        "a separate detailed report exists for each student.",
    # --- performance metrics dialog ---
    "Metrici de performanță": "Performance metrics",
    "Metrici de performanță pe rulare": "Per-run performance metrics",
    "Exportă CSV…": "Export CSV…",
    "Golește istoricul": "Clear history",
    "Exportă metricile ca CSV": "Export the metrics as CSV",
    "Fișier CSV (*.csv)": "CSV file (*.csv)",
    "Export eșuat": "Export failed",
    "Export finalizat": "Export complete",
    "Ștergi toate metricile de performanță înregistrate? "
    "Acțiunea nu poate fi anulată.":
        "Delete all recorded performance metrics? This cannot be undone.",
    "Rulare individuală (un elev)": "Single run (one student)",
    "Import fișier (rapoarte multiple)": "File import (multiple reports)",
    "Etapă": "Stage",
    "Medie": "Mean",
    "Ab. std.": "Std. dev.",
    "Apel LLM (plan de intervenție)": "LLM call (intervention plan)",
    "Defalcarea latenței pe etape": "Latency breakdown by stage",
    "(medie per raport)": "(mean per report)",
    "Nucleu xAI local (NLP + predicție + SHAP + LIME)":
        "Local xAI core (NLP + prediction + SHAP + LIME)",
    "Total local / raport": "Local total / report",
    "Total end-to-end / raport": "End-to-end total / report",
    "Încărcare model (o singură dată)": "Model load (once)",
    "Rapoarte generate": "Reports generated",
    "Apeluri cloud / local": "Cloud / local calls",
    "Tokeni (intrare / ieșire)": "Tokens (input / output)",
    "Latență totală": "Total latency",
    "Latență LLM / raport": "LLM latency / report",
    "Latență / apel cloud": "Latency / cloud call",
    "Latență min / max": "Min / max latency",
    "Cost total": "Total cost",
    "Cost / raport": "Cost / report",
    "Cost / apel cloud": "Cost / cloud call",
    "medie": "mean",
    "Sursă": "Source",
    "Latență": "Latency",
    "Tok. intr.": "In tok.",
    "Tok. ieș.": "Out tok.",
    "Cost": "Cost",
    "rulări": "runs",
    "rapoarte": "reports",
    "cost cumulat": "cumulative cost",
    "Nu există rulări înregistrate încă. Generează un plan sau importă un "
    "fișier pentru a colecta metrici.":
        "No runs recorded yet. Generate a plan or import a file to collect "
        "metrics.",
    # --- pipeline stage labels (app.timing.STAGE_LABELS) ---
    "NLP (text observațional)": "NLP (observation text)",
    "Predicție XGBoost": "XGBoost prediction",
    "Explicație SHAP": "SHAP explanation",
    "Explicație LIME": "LIME explanation",
    "Randare raport PDF": "PDF report rendering",
    "Construcție explainer SHAP (o singură dată)":
        "SHAP explainer construction (once)",
    "Construcție explainer LIME (o singură dată)":
        "LIME explainer construction (once)",
    # --- about ---
    "Asistent predictiv pentru identificarea timpurie („Ziua 14”) a "
    "elevilor cu risc de abandon școlar.\n\n"
    "Proof-of-concept: model XGBoost real + SMOTE-NC + explicații SHAP "
    "autentice, cu plan de intervenție generat local sau prin Claude.\n\n"
    "Datele elevilor rămân local; către cloud se trimite doar scorul "
    "anonimizat și explicația SHAP.\n\n"
    "© Ramona Richițeanu — concept și metodologie de cercetare.":
        "Predictive assistant for the early (“Day 14”) identification of students "
        "at risk of dropping out.\n\n"
        "Proof of concept: a real XGBoost model + SMOTE-NC + genuine SHAP "
        "explanations, with an intervention plan generated locally or via Claude."
        "\n\nStudent data stays local; only the anonymized score and the SHAP "
        "explanation are sent to the cloud.\n\n"
        "© Ramona Richițeanu — research concept and methodology.",
}

# Format templates. Kept apart from the plain strings because the placeholders
# must survive translation — a missing entry falls back to the Romanian template
# with the same field names, so ``.format()`` never raises on either branch.
_TEMPLATES: dict[str, str] = {
    "Stres emoțional (NLP): {score:.2f} / 2.0  — {label}":
        "Emotional stress (NLP): {score:.2f} / 2.0  — {label}",
    "Evaluare completă. Model: {version}":
        "Assessment complete. Model: {version}",
    "Sursă plan: {source}": "Plan source: {source}",
    "Sursă plan: se va folosi modul {mode}.{kb}":
        "Plan source: {mode} mode will be used.{kb}",
    "cloud ({provider}) — cheie din variabila de mediu":
        "cloud ({provider}) — key from environment variable",
    "cloud ({provider}) — cheie salvată": "cloud ({provider}) — stored key",
    " • bază de cunoștințe: {filename}": " • knowledge base: {filename}",
    "Salvat în baza de date locală (evaluare #{eval_id}).":
        "Saved to the local database (assessment #{eval_id}).",
    "Raport PDF salvat: {path}": "PDF report saved: {path}",
    "Raport salvat + {count} grafice metrici în „{target}”.":
        "Report saved + {count} metric charts in “{target}”.",
    "S-au salvat {count} grafice cu metrici în:\n{target}":
        "Saved {count} metric charts to:\n{target}",
    "Model reantrenat: {version}": "Model retrained: {version}",
    "Import complet: {count} rapoarte + raport general în {out_dir}":
        "Import complete: {count} reports + summary report in {out_dir}",
    "Au fost evaluați {count} elevi.\n\n"
    "S-au generat {count} rapoarte individuale și un raport general "
    "(raport_general.pdf) în:\n{out_dir}":
        "{count} students were assessed.\n\n"
        "{count} individual reports and a summary report "
        "(raport_general.pdf) were generated in:\n{out_dir}",
    "\n\nGraficele de evaluare a modelului (matrice de confuzie, ROC, "
    "precizie-recall, calibrare, SHAP, sumar) au fost salvate în:\n{metrics_dir}":
        "\n\nThe model evaluation charts (confusion matrix, ROC, "
        "precision-recall, calibration, SHAP, summary) were saved to:\n"
        "{metrics_dir}",
    "Se vor evalua {count} elevi, iar planurile de intervenție "
    "vor fi generate prin {provider} (cu revenire la planul "
    "local dacă un apel eșuează).\n\n"
    "Fiecare plan necesită un apel în cloud, deci procesul poate dura "
    "câteva minute pentru o clasă întreagă. Continuați?":
        "{count} students will be assessed and the intervention plans will be "
        "generated via {provider} (falling back to the local plan if a call "
        "fails).\n\nEach plan requires one cloud call, so the process may take "
        "several minutes for a whole class. Continue?",
    "Se vor evalua {count} elevi, iar planurile vor fi generate "
    "local (offline). Continuați?":
        "{count} students will be assessed and the plans will be generated "
        "locally (offline). Continue?",
    "— Validare încrucișată stratificată ({folds}-fold, SMOTE-NC în fold) —":
        "— Stratified cross-validation ({folds}-fold, SMOTE-NC inside fold) —",
    "Cheia API pentru {provider} a fost salvată securizat.":
        "The API key for {provider} has been stored securely.",
    "Ștergi cheia API salvată pentru {provider}?":
        "Delete the stored API key for {provider}?",
    "Cheia nu a putut fi salvată: {error}": "The key could not be saved: {error}",
    "Bază de cunoștințe actualizată: {filename} ({chars:,} caractere).":
        "Knowledge base updated: {filename} ({chars:,} characters).",
    "Caractere extrase: {chars:,}": "Extracted characters: {chars:,}",
    "Adăugat: {added}": "Added: {added}",
    "Activ: cheia din variabila de mediu {env_var} "
    "(are prioritate față de cheia salvată).":
        "Active: the key from environment variable {env_var} "
        "(takes precedence over the stored key).",
    "⚠️ Stocarea securizată nu este disponibilă pe acest sistem. "
    "Poți folosi în schimb variabila de mediu {env_var}.":
        "⚠️ Secure storage is unavailable on this system. You can use the "
        "{env_var} environment variable instead.",
    "{phase} ({done}/{total})": "{phase} ({done}/{total})",
    # --- report: SHAP / LIME / NLP body text ---
    "Scor model: {score:.0f}%": "Model score: {score:.0f}%",
    "Contribuția fiecărui factor la scorul final, în puncte procentuale "
    "(roșu = crește riscul, verde = reduce riscul). Suma contribuțiilor + "
    "valoarea de bază ({base:.3f}) = probabilitatea modelului.":
        "Each factor's contribution to the final score, in percentage points "
        "(red = raises risk, green = lowers risk). The sum of contributions + "
        "the base value ({base:.3f}) = the model's probability.",
    "<b>{label}</b> — crește riscul cu <b>+{points:.1f} pp</b> "
    "(valoare: {value})":
        "<b>{label}</b> — raises the risk by <b>+{points:.1f} pp</b> "
        "(value: {value})",
    "Scor stres emoțional (feature model): <b>{score:.2f}</b> / 2.0":
        "Emotional stress score (model feature): <b>{score:.2f}</b> / 2.0",
    "Valență: {valence:+.2f}": "Valence: {valence:+.2f}",
    "Probabilitate model: {probability:.3f} • valoare de referință "
    "(bază SHAP): {base:.3f} • model: {version}":
        "Model probability: {probability:.3f} • reference value (SHAP base): "
        "{base:.3f} • model: {version}",
    "Fidelitatea explicației locale: {label}":
        "Local explanation fidelity: {label}",
    "(R² = {r2:.2f} pe {samples} perturbări)":
        "(R² = {r2:.2f} over {samples} perturbations)",
    "Modelul local aproximează probabilitatea la <b>{local:.3f}</b>, față de "
    "<b>{actual:.3f}</b> cât indică modelul real (diferență: {gap:.3f}). Cu "
    "cât R² este mai mic și diferența mai mare, cu atât regulile de mai sus "
    "trebuie citite mai prudent — decizia rămâne a cadrului didactic.":
        "The local model approximates the probability at <b>{local:.3f}</b>, "
        "against <b>{actual:.3f}</b> from the real model (difference: "
        "{gap:.3f}). The lower the R² and the larger the difference, the more "
        "cautiously the rules above should be read — the decision remains the "
        "teacher's.",
    "Completează datele elevului în stânga și apasă <b>„Evaluează riscul”</b>.":
        "Fill in the student's data on the left and press "
        "<b>“Assess risk”</b>.",
    "Sursă: {source}": "Source: {source}",
    "Implicare: creșterea scorului Studentship de la <b>{current:g}/10</b> la "
    "<b>{target:g}/10</b>.":
        "Engagement: raise the Studentship score from <b>{current:g}/10</b> to "
        "<b>{target:g}/10</b>.",
    '<b>{count}</b> elevi <span style="color:#888;">({pct:.0f}%)</span>':
        '<b>{count}</b> students <span style="color:#888;">({pct:.0f}%)</span>',
    # --- report: critical indicators carrying numbers ---
    "Absenteism cronic ({count} absențe nemotivate / 3 luni)":
        "Chronic absenteeism ({count} unexcused absences / 3 months)",
    "Absențe nemotivate ridicate ({count} / 3 luni)":
        "High unexcused absences ({count} / 3 months)",
    "Implicare (Studentship) foarte scăzută ({score:g}/10)":
        "Very low engagement (Studentship) ({score:g}/10)",
    "Medie sub pragul de promovare ({average:g})":
        "Average below the pass threshold ({average:g})",
    "Note multiple sub 5 ({count})": "Multiple grades below 5 ({count})",
    "Latența, tokenii (raportați de furnizor) și costul derivat pentru fiecare "
    "apel de generare a planului, plus defalcarea pe etape a pipeline-ului "
    "local (NLP, predicție, SHAP, LIME, randare PDF). Costul este calculat "
    "din tokeni cu tabelul de prețuri din {as_of} — verifică ratele pentru "
    "model.":
        "Latency, tokens (as reported by the provider) and derived cost for each "
        "plan-generation call, plus the stage breakdown of the local pipeline "
        "(NLP, prediction, SHAP, LIME, PDF rendering). Cost is computed from "
        "tokens using the price table dated {as_of} — verify the rates for your "
        "model.",
    "Cost parțial: modelul nu are un preț în snapshot-ul {as_of}; "
    "completează MODEL_PRICING pentru cifre complete.":
        "Partial cost: the model has no price in the {as_of} snapshot; fill in "
        "MODEL_PRICING for complete figures.",
    "S-au exportat {rows} rânduri în:\n{path}":
        "Exported {rows} rows to:\n{path}",
    "<b>{runs}</b> rulări · <b>{reports}</b> rapoarte · cost cumulat <b>{cost}</b>":
        "<b>{runs}</b> runs · <b>{reports}</b> reports · cumulative cost "
        "<b>{cost}</b>",
    "{mean} medie": "{mean} mean",
    "\n\nPerformanță:\n"
    "• Apeluri cloud: {cloud_calls} din {report_count}\n"
    "• Latență totală: {total_latency} ({avg_latency} / raport)\n"
    "• Cost total: {cost}\n"
    "• Cost / raport (toate): {per_report}\n"
    "• Cost / apel cloud reușit: {per_cloud_call}\n"
    "Detalii complete în meniul „Performanță”.":
        "\n\nPerformance:\n"
        "• Cloud calls: {cloud_calls} of {report_count}\n"
        "• Total latency: {total_latency} ({avg_latency} / report)\n"
        "• Total cost: {cost}\n"
        "• Cost / report (all): {per_report}\n"
        "• Cost / successful cloud call: {per_cloud_call}\n"
        "Full details in the “Performance” menu.",
}


def trf(template: str, **kwargs) -> str:
    """Translate a format template and fill it. Romanian fallback keeps fields."""
    chosen = template if _current == "ro" else _TEMPLATES.get(template, template)
    return chosen.format(**kwargs)
