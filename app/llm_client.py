"""Optional cloud step: turn the score + SHAP explanation into an action plan.

This is the only network-dependent component. Per the README's privacy model,
**only the already-computed, de-identified result** is sent — the aggregate
score, the risk band, the sub-scores, and the SHAP feature attributions. No
student name, no free-text observation, no identifiers ever leave the machine.

Direct identifiers were never in the payload; this module additionally reduces
the *quasi-identifiers*, since age + sex + family situation + both parents'
education could jointly re-identify a student in a small school even with no
name attached. Concretely, the cloud context drops the exact age and sex,
coarsens family structure to three buckets, and merges the two parental
education levels into one numeric index — collapsing 36 combinations to one
scalar. The local plan keeps the full detail; it never leaves the machine.

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
    "chestionarului, dar nu primești nume, școală, data nașterii, vârsta sau "
    "sexul elevului; situația familială îți este dată doar la nivel general. "
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


# Context sent to the cloud, in order. Deliberately excludes the quasi-
# identifiers: exact age and sex are dropped entirely, family situation is
# coarsened, and the two parental-education levels are merged into a single
# numeric index (see _deidentified_context_lines). Residential environment is
# kept — it has only two values, so it adds little identifiability, and it
# drives a real intervention (transport / digital access for commuting students).
_SAFE_CONTEXT_KEYS = (
    ("Mediu_Rezidential", "Mediul de proveniență"),
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

# The full-detail set used by the *local* plan, which never leaves the machine
# and is read by a teacher who already knows the student.
_LOCAL_CONTEXT_KEYS = (
    ("Age_Years", "Vârsta aproximativă"),
    ("Sex", "Sex"),
    ("Situatie_Familiala", "Situația familială"),
    ("Educatie_Mama", "Educația mamei"),
    ("Educatie_Tata", "Educația tatălui"),
) + _SAFE_CONTEXT_KEYS

# Family structure coarsened from 4 questionnaire categories to 3 buckets. The
# distinction that survives is the one that changes the intervention: involving
# a legal guardian is a different action from involving both parents.
_FAMILY_BUCKETS = {
    "Ambii părinți": "ambii părinți",
    "Monoparental": "un singur adult",
    "Părinți divortați/separați": "un singur adult",
    "Tutore / plasament": "tutore / altă situație",
    "Altă situație": "tutore / altă situație",
}
_FAMILY_BUCKET_DEFAULT = "tutore / altă situație"


def _trim_text(value: object, limit: int = 160) -> str:
    text = str(value).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _parental_education_index(features: dict) -> float | None:
    """Both parents' education levels merged into one risk scalar.

    Reuses the exact coefficients the scoring model uses internally, so the
    number the LLM sees is the same quantity that drove the prediction. Sending
    36 combinations of two 6-level categoricals is a strong quasi-identifier;
    one averaged scalar is not. Nothing actionable is lost because mother's and
    father's education already map to *identical* intervention text below.
    """
    from .scoring_engine import _EDUCATION_RISK

    levels = [features.get("Educatie_Mama"), features.get("Educatie_Tata")]
    scores = [
        _EDUCATION_RISK.get(str(level), 0.5)
        for level in levels
        if level not in (None, "")
    ]
    if not scores:
        return None
    return round(sum(scores) / len(scores), 2)


def _format_context_value(key: str, label: str, value: object) -> str:
    if key == "Age_Years":
        return f"  - {label}: {int(round(float(value)))} ani"
    if key == "Medie_Modul_Anterior":
        return f"  - {label}: {float(value):.1f}"
    if key == "Studentship_Score":
        return f"  - {label}: {float(value):.1f} / 10"
    codes = config.CATEGORY_CODES.get(key)
    if codes is not None and str(value) not in codes:
        # An unanswered categorical falls back to ``Feature.default`` (0.0), so
        # it would otherwise reach the model as a bare "0.0". Decoding that back
        # to the first category would assert an answer the teacher never gave;
        # saying so explicitly lets the plan account for the missing input.
        return f"  - {label}: nespecificat"
    return f"  - {label}: {value}"


def _questionnaire_context_lines(
    questionnaire_answers: dict | None, *, deidentified: bool = True
) -> list[str]:
    """Structured questionnaire context.

    ``deidentified=True`` (the default, and what the cloud path uses) drops the
    exact age and sex, coarsens family structure to three buckets, and replaces
    the two parental-education levels with a single numeric index. The default
    is the safe one on purpose: a new caller that forgets the flag gets the
    de-identified payload rather than leaking. ``deidentified=False`` is for the
    local plan, which never leaves the machine.
    """
    if not questionnaire_answers:
        return []
    from .scoring_engine import compose_model_features

    features = compose_model_features(questionnaire_answers)
    keys = _SAFE_CONTEXT_KEYS if deidentified else _LOCAL_CONTEXT_KEYS

    lines: list[str] = []
    if deidentified:
        family = features.get("Situatie_Familiala")
        if family not in (None, ""):
            bucket = _FAMILY_BUCKETS.get(str(family), _FAMILY_BUCKET_DEFAULT)
            lines.append(f"  - Structura familiei: {bucket}")
        index = _parental_education_index(features)
        if index is not None:
            lines.append(
                f"  - Indice educație parentală: {index:+.2f} "
                "(scală -0.30 … +0.50; valori mari = nivel educațional scăzut, "
                "risc mai mare)"
            )

    for key, label in keys:
        value = features.get(key)
        if value in (None, ""):
            continue
        lines.append(_format_context_value(key, label, value))
    return lines


def _local_context_lines(questionnaire_answers: dict | None, observation_text: str | None) -> list[str]:
    lines = _questionnaire_context_lines(questionnaire_answers, deidentified=False)
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


# Appended when the interface language is not Romanian. ``SYSTEM_PROMPT`` states
# its output language inline, so the override has to be explicit and last —
# including the section names, which are named in Romanian above.
_ENGLISH_OVERRIDE = (
    "\n\nLANGUAGE OVERRIDE — this instruction takes precedence over the "
    "Romanian-language requirement stated above: write the entire plan in "
    "English. Use these section headings instead of the Romanian ones:\n"
    "1. Risk summary (2–3 sentences).\n"
    "2. Educational contract (mutual commitments student–teacher–family).\n"
    "3. Targeted measures for the top 2–3 identified risk factors.\n"
    "4. Success indicators over 4 weeks (measurable).\n"
    "For the educational contract use bold sub-headings "
    "(«**Student commitment:**», «**Teacher commitment:**», "
    "«**Family commitment:**»). All other formatting rules still apply."
)


def _build_system_prompt(knowledge_text: str | None) -> str:
    """System prompt, optionally grounded in the knowledge-base document."""
    from .i18n import get_language

    prompt = SYSTEM_PROMPT
    if get_language() == "en":
        prompt += _ENGLISH_OVERRIDE
    if not knowledge_text:
        return prompt
    snippet = knowledge_text.strip()[: config.KNOWLEDGE_MAX_CHARS]
    return (
        prompt
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
) -> tuple[str, dict]:
    """Generate the plan with Claude (Anthropic). Raises on any SDK/API error.

    Returns ``(plan_text, usage)`` where ``usage`` is the provider-reported token
    accounting (``input_tokens`` / ``output_tokens`` / ``thinking_tokens``);
    Anthropic bills reasoning within ``output_tokens``, so ``thinking_tokens``
    stays 0 here.
    """
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
    usage = getattr(response, "usage", None)
    return text, {
        "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
        "thinking_tokens": 0,
    }


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


def _gemini_usage(response) -> dict:
    """Provider-reported token accounting from a Gemini response.

    ``candidates_token_count`` is the visible answer; ``thoughts_token_count`` is
    the reasoning, billed at the output rate. Any field may be absent/None.
    """
    meta = getattr(response, "usage_metadata", None)
    return {
        "input_tokens": int(getattr(meta, "prompt_token_count", 0) or 0),
        "output_tokens": int(getattr(meta, "candidates_token_count", 0) or 0),
        "thinking_tokens": int(getattr(meta, "thoughts_token_count", 0) or 0),
    }


def _gemini_plan(
    evaluation: RiskEvaluation, model: str, api_key: str,
    knowledge_text: str | None,
    questionnaire_answers: dict | None = None,
) -> tuple[str, dict]:
    """Generate the plan with Gemini (Google).

    One independent request per case, reading the model's *complete* response.
    Thinking is bounded so it cannot consume the whole token budget and leave
    the written plan truncated or empty. Raises on any SDK/API error. Returns
    ``(plan_text, usage)`` with the provider-reported token counts.
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
    return text, _gemini_usage(response)


# Dispatch table: one cloud backend per provider id (see config.LLM_PROVIDERS).
_CLOUD_BACKENDS = {
    "claude": _claude_plan,
    "gemini": _gemini_plan,
}


def _cloud_plan(
    evaluation: RiskEvaluation, provider: config.LLMProvider, api_key: str,
    knowledge_text: str | None = None,
    questionnaire_answers: dict | None = None,
) -> tuple[str, dict]:
    """Generate the plan via ``provider``. Raises on any SDK/API error.

    Returns ``(plan_text, usage)`` — the provider-reported token accounting.
    """
    backend = _CLOUD_BACKENDS[provider.id]
    return backend(evaluation, provider.default_model, api_key, knowledge_text, questionnaire_answers)


def _record_metric(
    metric_out: list | None, provider: config.LLMProvider, *,
    source: str, ok: bool, latency_s: float,
    usage: dict | None = None, error: str = "",
) -> None:
    """Append a performance metric for this call to ``metric_out`` (if given)."""
    if metric_out is None:
        return
    from . import metrics  # lazy: metrics is only pulled in when someone measures
    metric_out.append(
        metrics.CallMetric.build(
            provider_id=provider.id, model=provider.default_model,
            source=source, ok=ok, latency_s=latency_s, usage=usage, error=error,
        )
    )


def generate_action_plan(
    evaluation: RiskEvaluation,
    provider_id: str | None = None,
    api_key: str | None = None,
    knowledge_text: str | None = None,
    questionnaire_answers: dict | None = None,
    observation_text: str | None = None,
    metric_out: list | None = None,
) -> tuple[str, str]:
    """Return (plan_text, source). Never raises — falls back to the template.

    ``provider_id`` selects the cloud backend; when ``None`` the provider chosen
    in Settings is used. ``knowledge_text`` (optional) is the user's imported
    .docx knowledge base; when present it grounds the cloud generation. It is
    not used by the local template (which is rule-based).

    ``metric_out`` (optional): a list to which a single
    :class:`metrics.CallMetric` is appended, capturing the call's latency, token
    usage and derived cost. Leaving it ``None`` skips all measurement.
    """
    import time

    from . import settings  # lazy: avoids an import cycle at module load

    pid = provider_id or settings.get_active_provider()
    provider = config.get_provider(pid)
    key = api_key or keystore.resolve_api_key(pid)
    if key:
        start = time.perf_counter()
        try:
            plan, usage = _cloud_plan(
                evaluation, provider, key, knowledge_text, questionnaire_answers
            )
            latency = time.perf_counter() - start
            source = f"cloud ({provider.label})"
            if knowledge_text:
                source += " + bază de cunoștințe"
            _record_metric(
                metric_out, provider, source="cloud", ok=True,
                latency_s=latency, usage=usage,
            )
            return plan, source
        except Exception as exc:  # offline-first: degrade gracefully
            latency = time.perf_counter() - start
            plan = _local_plan(evaluation, questionnaire_answers, observation_text)
            note = (
                "\n\n[Notă: generarea în cloud a eșuat "
                f"({type(exc).__name__}); s-a folosit planul local.]"
            )
            _record_metric(
                metric_out, provider, source="local-fallback", ok=False,
                latency_s=latency, error=f"{type(exc).__name__}: {exc}",
            )
            return plan + note, "local template (cloud indisponibil)"
    start = time.perf_counter()
    plan = _local_plan(evaluation, questionnaire_answers, observation_text)
    _record_metric(
        metric_out, provider, source="local", ok=False,
        latency_s=time.perf_counter() - start,
    )
    return plan, "local template"
