"""Optional cloud step: turn the score + SHAP explanation into an action plan.

This is the only network-dependent component. Per the README's privacy model,
**only the already-computed, de-identified result** is sent — the aggregate
score, the risk band, the sub-scores, and the SHAP feature attributions. No
student name, no free-text observation, no identifiers ever leave the machine.

Behaviour:
  * If the active provider's SDK is installed *and* an API key is configured,
    the plan is written in the cloud by Claude (Anthropic) or Gemini (Google) —
    whichever is selected in Settings ("Planul de Intervenție: Proiectul Podul").
  * Otherwise the app stays fully functional and produces the same plan from a
    local template. The two paths are interchangeable from the UI's view.
"""

from __future__ import annotations

from . import config, keystore
from .models import RiskEvaluation


SYSTEM_PROMPT = (
    "Ești un consilier educațional expert în prevenția abandonului școlar. "
    "Pe baza unui scor de risc calculat local și a unei explicații xAI (SHAP), "
    "redactezi un plan de intervenție numit «Proiectul Podul», concret și "
    "acționabil pentru un cadru didactic. Primești și un rezumat structurat al "
    "chestionarului, dar nu primești nume, școală sau data nașterii exacte. "
    "NU inventezi date lipsă — lucrezi cu scorul și factorii numerici deja calculați. "
    "Structura obligatorie a răspunsului (în limba română, fără preambul):\n"
    "1. Rezumatul riscului (2–3 propoziții).\n"
    "2. Contract educațional (angajamente reciproce elev–profesor–familie).\n"
    "3. Măsuri țintite pentru primii 2–3 factori de risc identificați.\n"
    "4. Indicatori de succes pe 4 săptămâni (măsurabili).\n"
    "FORMATARE (obligatorie): folosește exclusiv text simplu, titluri și liste "
    "(cu «-» sau numerotate). NU folosi tabele Markdown — nu folosi caracterul "
    "«|» pentru a alcătui coloane — și nu insera etichete HTML (de ex. «<br>»). "
    "Pentru contractul educațional, prezintă fiecare parte ca un subtitlu în "
    "bold (de ex. «**Angajamentul elevului:**», «**Angajamentul profesorului:**», "
    "«**Angajamentul familiei:**») urmat de o listă cu angajamentele.\n"
    "Răspunde doar cu planul, fără explicații despre cum l-ai construit."
)


# Map each risk driver to a concrete, teacher-facing intervention.
_INTERVENTIONS: dict[str, str] = {
    "Absente_Nemotivate_Zilele_1_13":
        "Monitorizare zilnică a prezenței, alertă la 3 absențe, contactarea "
        "familiei și desemnarea unui mentor de prezență.",
    "Absente_Motivate_3_Luni":
        "Revizuirea motivărilor și recuperarea conținutului ratat prin plan "
        "de sprijin și calendar scurt de follow-up.",
    "Medie_Modul_Anterior":
        "Plan individualizat de recuperare la disciplinele-cheie și ore de "
        "sprijin (meditații / peer-tutoring).",
    "Note_Sub_5":
        "Program de remediere pe disciplinele în care apar note sub 5, cu "
        "retestare scurtă și feedback imediat.",
    "Studentship_Score":
        "Activități de responsabilizare și implicare (roluri în clasă, proiecte "
        "de grup, sistem „buddy”) pentru creșterea sentimentului de apartenență.",
    "Stres_Emotional_NLP":
        "Sesiuni cu consilierul școlar, discuții individuale periodice și "
        "monitorizarea stării emoționale.",
    "Mediu_Rezidential":
        "Facilitarea accesului (transport, resurse digitale) și activități de "
        "integrare pentru elevii navetiști.",
    "Situatie_Familiala":
        "Implicarea tutorelui legal / familiei extinse și un canal de comunicare "
        "constant cu adultul de sprijin.",
    "Educatie_Mama":
        "Mesaje și materiale de sprijin adaptate nivelului de înțelegere al "
        "familiei, cu pași foarte concreți.",
    "Educatie_Tata":
        "Mesaje și materiale de sprijin adaptate nivelului de înțelegere al "
        "familiei, cu pași foarte concreți.",
    "Participare_Extrascolara":
        "Reactivarea interesului printr-o activitate extrașcolară scurtă, "
        "aleasă împreună cu elevul.",
    "Atitudine_Scoala":
        "Discuții de tip coaching, obiective scurte și întărire pozitivă la "
        "fiecare progres observabil.",
    "Sanctiuni_Avertismente":
        "Contract comportamental, reguli puține și clare și monitorizare "
        "săptămânală a respectării lor.",
    "Cum_te_Simti_La_Scoala":
        "Sprijin emoțional țintit și verificări scurte de stare, ideal în tandem "
        "cu dirigintele și consilierul.",
    "Scoala_Ajuta_Obiective":
        "Legarea directă a sarcinilor școlare de obiective personale concrete, "
        "ca elevul să vadă utilitatea muncii depuse.",
}


_SAFE_CONTEXT_KEYS = (
    ("Age_Years", "Vârsta aproximativă"),
    ("Sex", "Sex"),
    ("Mediu_Rezidential", "Mediul de proveniență"),
    ("Situatie_Familiala", "Situația familială"),
    ("Educatie_Mama", "Educația mamei"),
    ("Educatie_Tata", "Educația tatălui"),
    ("Absente_Nemotivate_Zilele_1_13", "Absențe nemotivate"),
    ("Absente_Motivate_3_Luni", "Absențe motivate"),
    ("Participare_Extrascolara", "Participare extrașcolară"),
    ("Medie_Modul_Anterior", "Media modulului anterior"),
    ("Note_Sub_5", "Număr note sub 5"),
    ("Studentship_Score", "Scor Studentship"),
    ("Atitudine_Scoala", "Atitudinea față de școală"),
    ("Sanctiuni_Avertismente", "Sancțiuni / avertismente"),
    ("Cum_te_Simti_La_Scoala", "Cum se simte la școală"),
    ("Scoala_Ajuta_Obiective", "Școala ajută obiectivele"),
)


def _trim_text(value: object, limit: int = 160) -> str:
    text = str(value).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _questionnaire_context_lines(questionnaire_answers: dict | None) -> list[str]:
    if not questionnaire_answers:
        return []
    from .scoring_engine import compose_model_features

    features = compose_model_features(questionnaire_answers)
    lines: list[str] = []
    for key, label in _SAFE_CONTEXT_KEYS:
        value = features.get(key)
        if value in (None, ""):
            continue
        if key == "Age_Years":
            lines.append(f"  - {label}: {int(round(float(value)))} ani")
        elif key == "Medie_Modul_Anterior":
            lines.append(f"  - {label}: {float(value):.1f}")
        elif key == "Studentship_Score":
            lines.append(f"  - {label}: {float(value):.1f} / 10")
        else:
            lines.append(f"  - {label}: {value}")
    return lines


def _local_context_lines(questionnaire_answers: dict | None, observation_text: str | None) -> list[str]:
    lines = _questionnaire_context_lines(questionnaire_answers)
    if questionnaire_answers:
        for key, label in (
            ("family_situation_other", "Situație familială - detalii"),
            ("low_grades_details", "Note sub 5 - detalii"),
            ("school_feeling_other", "Cum se simte la școală - detalii"),
            ("additional_notes", "Observații suplimentare"),
        ):
            value = questionnaire_answers.get(key)
            if value not in (None, ""):
                lines.append(f"  - {label}: {_trim_text(value)}")
    if observation_text:
        lines.append(f"  - Text observational: {_trim_text(observation_text)}")
    return lines


def anonymized_summary(
    evaluation: RiskEvaluation, questionnaire_answers: dict | None = None
) -> str:
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
    context_lines = _questionnaire_context_lines(questionnaire_answers)
    if context_lines:
        lines.append("")
        lines.append("Context chestionar (anonimizat):")
        lines.extend(context_lines)
    return "\n".join(lines)


def _local_plan(
    evaluation: RiskEvaluation,
    questionnaire_answers: dict | None = None,
    observation_text: str | None = None,
) -> str:
    summary = anonymized_summary(evaluation, questionnaire_answers)
    risky = [a for a in evaluation.top_drivers(4) if a.shap_value > 0][:3]
    context_lines = _local_context_lines(questionnaire_answers, observation_text)

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
    if context_lines:
        parts.append("")
        parts.append("   Context relevant din chestionar")
        for line in context_lines:
            parts.append(f"   {line}")
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


def local_action_plan(
    evaluation: RiskEvaluation,
    questionnaire_answers: dict | None = None,
    observation_text: str | None = None,
) -> tuple[str, str]:
    """The offline, rule-based plan (no network). Used for batch report runs."""
    return _local_plan(evaluation, questionnaire_answers, observation_text), "local template"


def _build_system_prompt(knowledge_text: str | None) -> str:
    """System prompt, optionally grounded in the knowledge-base document."""
    if not knowledge_text:
        return SYSTEM_PROMPT
    snippet = knowledge_text.strip()[: config.KNOWLEDGE_MAX_CHARS]
    return (
        SYSTEM_PROMPT
        + "\n\nMATERIAL METODOLOGIC DE REFERINȚĂ (bază de cunoștințe încărcată "
        "de utilizator). Folosește-l pentru a alinia planul la metodologia și "
        "terminologia din cercetare; nu îl cita textual:\n"
        "<<<\n" + snippet + "\n>>>"
    )


def _user_message(
    evaluation: RiskEvaluation, questionnaire_answers: dict | None = None
) -> str:
    """The de-identified prompt sent to whichever provider is active."""
    return (
        "Redactează planul «Proiectul Podul» pe baza următoarei evaluări "
        "anonimizate:\n\n" + anonymized_summary(evaluation, questionnaire_answers)
    )


def _claude_plan(
    evaluation: RiskEvaluation, model: str, api_key: str,
    knowledge_text: str | None,
    questionnaire_answers: dict | None = None,
) -> str:
    """Generate the plan with Claude (Anthropic). Raises on any SDK/API error."""
    import anthropic  # lazy: keeps the app usable when the SDK is absent

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        max_tokens=config.LLM_MAX_TOKENS,
        thinking={"type": "adaptive"},
        system=_build_system_prompt(knowledge_text),
        messages=[{"role": "user", "content": _user_message(evaluation, questionnaire_answers)}],
    )
    text = "".join(
        block.text for block in response.content if getattr(block, "type", "") == "text"
    ).strip()
    if not text:
        raise RuntimeError("Răspuns gol de la model.")
    return text


def _gemini_response_text(response) -> tuple[str, bool]:
    """Full visible answer text from a Gemini response + whether it was truncated.

    Concatenates every non-thought text part across the candidate's content so a
    multi-part answer is captured in full — ``response.text`` alone can miss
    parts or come back empty when the model spent its token budget on thinking.
    """
    chunks: list[str] = []
    truncated = False
    for candidate in (getattr(response, "candidates", None) or []):
        if str(getattr(candidate, "finish_reason", "")).endswith("MAX_TOKENS"):
            truncated = True
        content = getattr(candidate, "content", None)
        for part in (getattr(content, "parts", None) or []):
            if getattr(part, "thought", False):
                continue  # internal reasoning, not the plan itself
            piece = getattr(part, "text", None)
            if piece:
                chunks.append(piece)
    text = "".join(chunks).strip()
    if not text:  # last resort: the SDK convenience accessor
        text = (getattr(response, "text", None) or "").strip()
    return text, truncated


def _gemini_plan(
    evaluation: RiskEvaluation, model: str, api_key: str,
    knowledge_text: str | None,
    questionnaire_answers: dict | None = None,
) -> str:
    """Generate the plan with Gemini (Google).

    One independent request per case, reading the model's *complete* response.
    Thinking is bounded so it cannot consume the whole token budget and leave
    the written plan truncated or empty. Raises on any SDK/API error.
    """
    from google import genai  # lazy: keeps the app usable when the SDK is absent
    from google.genai import types

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=_user_message(evaluation, questionnaire_answers),
        config=types.GenerateContentConfig(
            system_instruction=_build_system_prompt(knowledge_text),
            max_output_tokens=config.LLM_MAX_TOKENS,
            thinking_config=types.ThinkingConfig(
                thinking_budget=config.GEMINI_THINKING_BUDGET
            ),
        ),
    )
    text, truncated = _gemini_response_text(response)
    if not text:
        raise RuntimeError(
            "Răspuns trunchiat de Gemini (limita de tokeni)." if truncated
            else "Răspuns gol de la model."
        )
    return text


# Dispatch table: one cloud backend per provider id (see config.LLM_PROVIDERS).
_CLOUD_BACKENDS = {
    "claude": _claude_plan,
    "gemini": _gemini_plan,
}


def _cloud_plan(
    evaluation: RiskEvaluation, provider: config.LLMProvider, api_key: str,
    knowledge_text: str | None = None,
    questionnaire_answers: dict | None = None,
) -> str:
    """Generate the plan via ``provider``. Raises on any SDK/API error."""
    backend = _CLOUD_BACKENDS[provider.id]
    return backend(evaluation, provider.default_model, api_key, knowledge_text, questionnaire_answers)


def generate_action_plan(
    evaluation: RiskEvaluation,
    provider_id: str | None = None,
    api_key: str | None = None,
    knowledge_text: str | None = None,
    questionnaire_answers: dict | None = None,
    observation_text: str | None = None,
) -> tuple[str, str]:
    """Return (plan_text, source). Never raises — falls back to the template.

    ``provider_id`` selects the cloud backend; when ``None`` the provider chosen
    in Settings is used. ``knowledge_text`` (optional) is the user's imported
    .docx knowledge base; when present it grounds the cloud generation. It is
    not used by the local template (which is rule-based).
    """
    from . import settings  # lazy: avoids an import cycle at module load

    pid = provider_id or settings.get_active_provider()
    provider = config.get_provider(pid)
    key = api_key or keystore.resolve_api_key(pid)
    if key:
        try:
            plan = _cloud_plan(evaluation, provider, key, knowledge_text, questionnaire_answers)
            source = f"cloud ({provider.label})"
            if knowledge_text:
                source += " + bază de cunoștințe"
            return plan, source
        except Exception as exc:  # offline-first: degrade gracefully
            plan = _local_plan(evaluation, questionnaire_answers, observation_text)
            note = (
                "\n\n[Notă: generarea în cloud a eșuat "
                f"({type(exc).__name__}); s-a folosit planul local.]"
            )
            return plan + note, "local template (cloud indisponibil)"
    return _local_plan(evaluation, questionnaire_answers, observation_text), "local template"
