"""Optional cloud step: turn the score + SHAP explanation into an action plan.

This is the only network-dependent component. Per the README's privacy model,
**only the already-computed, de-identified result** is sent — the aggregate
score, the risk band, the sub-scores, and the SHAP feature attributions. No
student name, no free-text observation, no identifiers ever leave the machine.

Behaviour:
  * If the Anthropic SDK is installed *and* an API key is configured, the plan
    is written by Claude ("Planul de Intervenție: Proiectul Podul").
  * Otherwise the app stays fully functional and produces the same plan from a
    local template. The two paths are interchangeable from the UI's view.
"""

from __future__ import annotations

import os

from . import config
from .models import RiskEvaluation


SYSTEM_PROMPT = (
    "Ești un consilier educațional expert în prevenția abandonului școlar. "
    "Pe baza unui scor de risc calculat local și a unei explicații xAI (SHAP), "
    "redactezi un plan de intervenție numit «Proiectul Podul», concret și "
    "acționabil pentru un cadru didactic. NU primești date personale despre elev "
    "(nume, observații brute) — lucrezi exclusiv cu scorul și factorii numerici. "
    "Structura obligatorie a răspunsului (în limba română, fără preambul):\n"
    "1. Rezumatul riscului (2–3 propoziții).\n"
    "2. Contract educațional (angajamente reciproce elev–profesor–familie).\n"
    "3. Măsuri țintite pentru primii 2–3 factori de risc identificați.\n"
    "4. Indicatori de succes pe 4 săptămâni (măsurabili).\n"
    "Răspunde doar cu planul, fără explicații despre cum l-ai construit."
)


# Map each risk driver to a concrete, teacher-facing intervention.
_INTERVENTIONS: dict[str, str] = {
    "Absente_Nemotivate_Zilele_1_13":
        "Monitorizare zilnică a prezenței, alertă la 3 absențe, contactarea "
        "familiei și desemnarea unui mentor de prezență.",
    "Medie_Modul_Anterior":
        "Plan individualizat de recuperare la disciplinele-cheie și ore de "
        "sprijin (meditații / peer-tutoring).",
    "Note_Sub_7":
        "Program de remediere pe competențele deficitare, cu evaluări formative "
        "săptămânale și feedback pozitiv.",
    "Studentship_Score":
        "Activități de responsabilizare și implicare (roluri în clasă, proiecte "
        "de grup, sistem „buddy”) pentru creșterea sentimentului de apartenență.",
    "Stres_Emotional_NLP":
        "Sesiuni cu consilierul școlar, discuții individuale periodice și "
        "monitorizarea stării emoționale.",
    "Vulnerabilitate_Financiara":
        "Sprijin material (rechizite, bursă, program „masă caldă”) și orientare "
        "către serviciile sociale locale.",
    "Parinti_In_Strainatate":
        "Implicarea tutorelui legal / familiei extinse și canal de comunicare "
        "regulat cu părinții aflați la distanță.",
    "Mediu_Rezidential":
        "Facilitarea accesului (transport, resurse digitale) și activități de "
        "integrare pentru elevii navetiști.",
}


def anonymized_summary(evaluation: RiskEvaluation) -> str:
    """De-identified structured summary — the only thing sent to the cloud."""
    lines = [
        f"Scor agregat de risc: {evaluation.aggregate_score:.1f}% "
        f"(bandă: {evaluation.risk_band}).",
        f"Probabilitate model: {evaluation.probability:.3f} "
        f"(valoare de referință / bază: {evaluation.base_value:.3f}).",
        "",
        "Factori de risc (SHAP, în puncte procentuale față de referință):",
    ]
    for a in evaluation.top_drivers(6):
        sign = "+" if a.shap_value >= 0 else ""
        lines.append(f"  - {a.label}: {sign}{a.points} pp (valoare: {a.value_display})")
    lines.append("")
    lines.append("Ponderea domeniilor în explicație:")
    for s in evaluation.sub_scores:
        lines.append(f"  - {s.name}: {s.value:.0f}%")
    return "\n".join(lines)


def _local_plan(evaluation: RiskEvaluation) -> str:
    summary = anonymized_summary(evaluation)
    risky = [a for a in evaluation.top_drivers(4) if a.shap_value > 0][:3]

    parts: list[str] = []
    parts.append("PLANUL DE INTERVENȚIE: PROIECTUL PODUL")
    parts.append("=" * 42)
    parts.append("")
    parts.append("1. Rezumatul riscului")
    parts.append(
        f"   Elevul se află în banda de risc «{evaluation.risk_band}» "
        f"({evaluation.aggregate_score:.0f}%). Principalii factori care "
        f"cresc riscul sunt: "
        + (", ".join(a.label.lower() for a in risky) if risky else "niciun factor major")
        + "."
    )
    parts.append("")
    parts.append("2. Contract educațional")
    parts.append("   - Elev: prezență zilnică și participare la orele de sprijin.")
    parts.append("   - Profesor diriginte: monitorizare săptămânală și feedback pozitiv.")
    parts.append("   - Familie: comunicare bilunară privind progresul.")
    parts.append("")
    parts.append("3. Măsuri țintite")
    if risky:
        for a in risky:
            action = _INTERVENTIONS.get(
                a.feature_key, "Măsură de sprijin individualizată."
            )
            parts.append(f"   • {a.label} (+{a.points} pp): {action}")
    else:
        parts.append("   • Menținerea măsurilor curente și monitorizare de rutină.")
    parts.append("")
    parts.append("4. Indicatori de succes pe 4 săptămâni")
    parts.append("   - Săpt. 1: 0 absențe nemotivate noi; semnarea contractului.")
    parts.append("   - Săpt. 2: participare la ≥80% dintre orele de sprijin.")
    parts.append("   - Săpt. 3: ≥1 evaluare formativă promovată la disciplinele-țintă.")
    parts.append("   - Săpt. 4: reevaluarea scorului de risc și ajustarea planului.")
    parts.append("")
    parts.append("— Bază de calcul (date anonimizate) —")
    parts.append(summary)
    return "\n".join(parts)


def _cloud_plan(evaluation: RiskEvaluation, api_key: str) -> str:
    """Generate the plan with Claude. Raises on any SDK/API error."""
    import anthropic  # lazy: keeps the app usable when the SDK is absent

    client = anthropic.Anthropic(api_key=api_key)
    user_msg = (
        "Redactează planul «Proiectul Podul» pe baza următoarei evaluări "
        "anonimizate:\n\n" + anonymized_summary(evaluation)
    )
    response = client.messages.create(
        model=config.LLM_MODEL,
        max_tokens=config.LLM_MAX_TOKENS,
        thinking={"type": "adaptive"},
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )
    text = "".join(
        block.text for block in response.content if getattr(block, "type", "") == "text"
    ).strip()
    if not text:
        raise RuntimeError("Răspuns gol de la model.")
    return text


def generate_action_plan(
    evaluation: RiskEvaluation, api_key: str | None = None
) -> tuple[str, str]:
    """Return (plan_text, source). Never raises — falls back to the template."""
    key = api_key or os.environ.get(config.LLM_API_KEY_ENV, "").strip()
    if key:
        try:
            return _cloud_plan(evaluation, key), "cloud (Claude)"
        except Exception as exc:  # offline-first: degrade gracefully
            plan = _local_plan(evaluation)
            note = (
                "\n\n[Notă: generarea în cloud a eșuat "
                f"({type(exc).__name__}); s-a folosit planul local.]"
            )
            return plan + note, "local template (cloud indisponibil)"
    return _local_plan(evaluation), "local template"
