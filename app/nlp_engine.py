"""Qualitative-text analysis (BERT-style emotional-stress signal).

The README describes a BERT model exported to ONNX that extracts emotional
valence from a teacher's free-text observations. Shipping a full transformer
inside a proof-of-concept exe is heavy, so this module implements a
**transparent, lexicon-based** Romanian valence analyzer as a drop-in stand-in.

It produces the same thing the downstream code needs: an emotional-stress
sub-score in ``[0, 2]`` (the ``Stres_Emotional_NLP`` model feature) plus a
signed valence and the matched terms (for explainability in the UI).

Swapping in a real ONNX BERT model later means replacing ``analyze()`` while
keeping its return contract — nothing else in the app changes.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field


def _strip_diacritics(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))


def _normalize(text: str) -> str:
    return _strip_diacritics(text.lower())


# Weights are on a 0..1 scale per term; summed then squashed into [0, 2].
# Terms are stored diacritic-free (matching _normalize output).
_NEGATIVE_TERMS: dict[str, float] = {
    "retras": 0.9, "izolat": 0.9, "singur": 0.7, "insingurat": 0.9,
    "obosit": 0.6, "epuizat": 0.8, "adormit": 0.4,
    "trist": 0.8, "deprimat": 1.0, "anxios": 0.9, "anxietate": 0.9,
    "speriat": 0.7, "frica": 0.7, "stres": 0.8, "stresat": 0.8,
    "demotivat": 0.9, "nemotivat": 0.9, "plictisit": 0.6, "apatic": 0.8,
    "absent": 0.7, "chiuleste": 0.8, "chiul": 0.8, "lipseste": 0.5,
    "conflict": 0.7, "agresiv": 0.8, "agitat": 0.6, "nervos": 0.5,
    "abandon": 1.0, "renunta": 1.0, "renuntat": 1.0, "abandoneaza": 1.0,
    "saracie": 0.7, "foame": 0.8, "flamand": 0.8, "bullying": 1.0, "hartuit": 1.0,
    "refuza": 0.7, "opozant": 0.6, "inchis": 0.5, "neincrezator": 0.6,
    "descurajat": 0.9, "coplesit": 0.8, "pierdut": 0.5,
    # Deprivation / behavioural cues from the teacher's-guide examples
    # ("vine obosit", "este neglijat", "somnoros").
    "neglijat": 0.8, "somnoros": 0.5, "murdar": 0.4, "vulnerabil": 0.6,
}

_POSITIVE_TERMS: dict[str, float] = {
    "motivat": 0.9, "implicat": 0.8, "entuziast": 0.9, "bucuros": 0.8,
    "vesel": 0.7, "fericit": 0.8, "increzator": 0.7, "calm": 0.5,
    "prieteni": 0.6, "sociabil": 0.7, "sprijin": 0.6, "sustinut": 0.6,
    "progres": 0.8, "imbunatatit": 0.8, "silitor": 0.8, "harnic": 0.8,
    "responsabil": 0.7, "atent": 0.6, "cooperant": 0.7, "optimist": 0.8,
    "ambitios": 0.8, "curios": 0.6, "activ": 0.6, "prezent": 0.5,
}

# Simple negation cues: "nu", "fara", "deloc" flip the polarity of the next
# few tokens (a lightweight approximation of contextual scoring).
_NEGATIONS = {"nu", "fara", "deloc", "niciodata", "nici"}
_NEGATION_WINDOW = 3

_TOKEN_RE = re.compile(r"[a-z]+")


@dataclass
class NlpResult:
    stress_score: float           # [0, 2] — the Stres_Emotional_NLP feature
    valence: float                # [-1, 1] — negative..positive
    negative_terms: list[str] = field(default_factory=list)
    positive_terms: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        from .i18n import tr

        if self.valence <= -0.4:
            return tr("Ton preponderent negativ / semne de stres")
        if self.valence >= 0.4:
            return tr("Ton preponderent pozitiv")
        return tr("Ton neutru / mixt")


def _squash(raw: float) -> float:
    """Map an unbounded non-negative accumulator into [0, 2]."""
    # Saturating curve: 0 -> 0, grows toward 2, never exceeds it.
    return round(2.0 * (1.0 - 1.0 / (1.0 + raw)), 3)


def analyze(text: str) -> NlpResult:
    """Analyze free-text observations and return the emotional-stress signal."""
    if not text or not text.strip():
        return NlpResult(stress_score=0.0, valence=0.0)

    tokens = _TOKEN_RE.findall(_normalize(text))

    neg_accum = 0.0
    pos_accum = 0.0
    neg_hits: list[str] = []
    pos_hits: list[str] = []

    negation_countdown = 0
    for tok in tokens:
        negated = negation_countdown > 0
        if negation_countdown > 0:
            negation_countdown -= 1
        if tok in _NEGATIONS:
            negation_countdown = _NEGATION_WINDOW
            continue

        if tok in _NEGATIVE_TERMS:
            w = _NEGATIVE_TERMS[tok]
            if negated:
                pos_accum += w
                pos_hits.append(f"nu {tok}")
            else:
                neg_accum += w
                neg_hits.append(tok)
        elif tok in _POSITIVE_TERMS:
            w = _POSITIVE_TERMS[tok]
            if negated:
                neg_accum += w
                neg_hits.append(f"nu {tok}")
            else:
                pos_accum += w
                pos_hits.append(tok)

    stress = _squash(neg_accum)
    total = neg_accum + pos_accum
    valence = 0.0 if total == 0 else round((pos_accum - neg_accum) / total, 3)

    return NlpResult(
        stress_score=stress,
        valence=valence,
        negative_terms=neg_hits,
        positive_terms=pos_hits,
    )
