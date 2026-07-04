# Proof of Concept — how it works and how to run it

This is a working proof-of-concept of the app described in [README.md](README.md),
built around the **real machine-learning methodology** from the research
(XGBoost + SMOTE-NC + SHAP, with a BERT-style NLP signal and the "Day 14"
early-warning model). It is the *"Operationalizare Reală"* variant: the score
comes from a genuine trained model, and the explanation comes from genuine SHAP
values — **no LLM is used to simulate the model output**.

The one place an LLM is used is the final, optional step: turning the already
computed, **anonymized** score + SHAP explanation into a written intervention
plan ("Proiectul Podul"). The app is fully functional offline without it.

---

## What maps to what

### README architecture → this implementation

| README component        | Implementation | Notes |
|-------------------------|----------------|-------|
| Desktop GUI (PyQt/Tkinter) | `app/ui/` (PySide6) | Collects input, shows score/SHAP/plan |
| NLP engine (BERT/ONNX)  | `app/nlp_engine.py` | Transparent Romanian lexicon **stand-in** for BERT; same interface (`analyze()` → emotional-stress feature) so an ONNX model can replace it |
| Scoring engine          | `app/scoring_engine.py` | **Real XGBoost** + SMOTE-NC, the "Day 14" model |
| Explainability (SHAP/LIME) | `app/explainability.py` | **Genuine SHAP** — additive in probability space |
| Local database (SQLite) | `app/database.py` | Plain SQLite (a production build would add SQLCipher for at-rest encryption) |
| Cloud LLM API           | `app/llm_client.py` | Anthropic (Claude) when a key is set; local template otherwise |

### Research methodology → this implementation

- **XGBoost classifier**, **SMOTE-NC** balancing, **SHAP** explanations — exactly
  the stack from the methodology script.
- **"Day 14" non-linear risk rules** used to label the synthetic training data
  (`generate_synthetic_dataset`), so the model learns the same logic as the
  research (`test` accuracy ≈ 0.99–1.00 confirms it).
- **Synthetic SIIIR-like data** stands in for the real SIIIR export. To use real
  data, replace `generate_synthetic_dataset()` with `pd.read_csv(<anonymised>.csv)`
  keeping the same columns.
- **Global SHAP summary plot** (the research figure) is reproduced by
  `python train.py --plot shap_summary.png`.

### Feature schema ("Day 14")

| Feature (model column) | Meaning | Type |
|---|---|---|
| `Medie_Modul_Anterior` | Previous-module average (1–10) | numeric |
| `Note_Sub_7` | Number of grades below 7 | numeric |
| `Absente_Nemotivate_Zilele_1_13` | Unmotivated absences in the first 13 days | numeric |
| `Studentship_Score` | Engagement composite (cognitive + social presence, 0–10) | numeric |
| `Stres_Emotional_NLP` | Emotional-stress signal from the free-text observation (0–2) | numeric (derived by NLP) |
| `Mediu_Rezidential` | Urban / Rural | categorical |
| `Parinti_In_Strainatate` | Parents abroad (Nu / Da) | categorical |
| `Vulnerabilitate_Financiara` | Financial vulnerability (Scazuta / Medie / Ridicata) | categorical |

---

## Run it locally (developers)

Requires **Python 3.12** (the ML wheels — xgboost/shap/scikit-learn — and PySide6
support it; 3.14 is too new for some of them).

```bash
# 1. Create an isolated environment
py -3.12 -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) train + cache the model; otherwise it trains on first launch
python train.py

# 4. Launch the desktop app
python run.py
```

In the app: click **„Completează exemplu”**, then **„Evaluează riscul”**, then
**„Generează planul de intervenție”**.

### Optional: cloud-generated action plan

Set an Anthropic API key before launching to have Claude write the plan;
otherwise the local template is used automatically.

```bash
set ANTHROPIC_API_KEY=sk-ant-...   # Windows
# export ANTHROPIC_API_KEY=sk-ant-...
```

Only the anonymized score + SHAP explanation are sent — never names or the
free-text observation.

### Run the tests

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

---

## Build the Windows `.exe`

### Automatically, via GitHub Actions (recommended)

The workflow [`.github/workflows/build.yml`](.github/workflows/build.yml) runs on
every push and pull request:

1. installs dependencies,
2. runs the tests,
3. trains + bundles the model (`python train.py`),
4. compiles a single-file exe with PyInstaller,
5. uploads it as a build **artifact** (and attaches it to a **Release** when you
   push a `v*` tag).

**To get the exe:** open the repo's **Actions** tab → the latest run →
**Artifacts** → `RiskSolvingApp-windows`. Or push a tag (`git tag v0.1.0 &&
git push --tags`) to get a downloadable Release asset.

> This needs a git repository with a GitHub remote. If the folder isn't a repo
> yet: `git init`, commit, and push to GitHub.

### Manually, on a Windows machine

```bash
pip install -r requirements-dev.txt
python train.py                     # produce the bundled model
pyinstaller --noconfirm RiskSolvingApp.spec
# -> dist/RiskSolvingApp.exe
```

The exe bundles the GUI, the ML stack, and the pre-trained model, so it launches
without needing Python installed. The database and any freshly trained model are
written to `%LOCALAPPDATA%\RiskSolvingApp`.

---

## Privacy

- All student data is stored **locally** (SQLite).
- The only network call is the optional action-plan generation, and it receives
  **only** the aggregate score, risk band, sub-scores, and SHAP attributions —
  no names, no observations, no identifiers.
- If no API key is configured, nothing leaves the machine.

---

## Notes & honest limitations (it's a PoC)

- The **NLP engine is a lexicon stand-in** for BERT, not a transformer. The
  interface is designed so a real ONNX BERT model drops in without touching the
  rest of the app.
- Training data is **synthetic** (deterministic "Day 14" rules), so test metrics
  are near-perfect. Real SIIIR data will produce realistic, lower metrics — the
  pipeline stays the same.
- SQLite is **not encrypted at rest** here; a production build would layer
  SQLCipher and keep the API key in the OS credential store.

---

© Ramona Richițeanu — research concept and methodology.
