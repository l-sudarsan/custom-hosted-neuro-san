"""Custom evaluator for the `custom-hosted-neuro-san` agent.

The agent is an "author of terse announcements": it must produce the *shortest
possible* message that conveys the requested sentiment, with no embellishment or
commentary. This evaluator rewards brevity and penalizes filler / multi-sentence
chatter, complementing the model-graded quality evaluators (relevance, coherence,
fluency).

The evaluator is callable and returns a dict of numeric scores so it can be used
directly with `azure.ai.evaluation.evaluate`.
"""

from __future__ import annotations

import re

# Words the front-man should never need for a terse announcement.
_FILLER = {
    "please",
    "kindly",
    "however",
    "therefore",
    "furthermore",
    "additionally",
    "basically",
    "actually",
    "just",
    "really",
    "very",
    "note",
    "regards",
    "sincerely",
}


def _word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9']+", text or ""))


def _sentence_count(text: str) -> int:
    parts = [p for p in re.split(r"[.!?]+", text or "") if p.strip()]
    return max(1, len(parts))


class TersenessEvaluator:
    """Scores how terse and on-brief an announcement is (0.0 - 1.0).

    Scoring blends three signals:
      * brevity     - fewer words is better; the agent aims for ~2 words.
      * single_line - one short sentence is better than a paragraph.
      * no_filler   - absence of conversational filler / sign-offs.
    """

    def __init__(self, ideal_words: int = 2, max_words: int = 8) -> None:
        self._ideal_words = ideal_words
        self._max_words = max_words

    def __call__(self, *, response: str, **kwargs) -> dict:
        text = (response or "").strip()
        words = _word_count(text)
        sentences = _sentence_count(text)

        # Brevity: 1.0 at or below ideal, decaying to 0.0 at max_words.
        if words <= self._ideal_words:
            brevity = 1.0
        elif words >= self._max_words:
            brevity = 0.0
        else:
            span = self._max_words - self._ideal_words
            brevity = max(0.0, 1.0 - (words - self._ideal_words) / span)

        single_line = 1.0 if sentences == 1 else max(0.0, 1.0 - 0.5 * (sentences - 1))

        lowered = text.lower()
        filler_hits = sum(1 for w in _FILLER if re.search(rf"\b{re.escape(w)}\b", lowered))
        no_filler = max(0.0, 1.0 - 0.34 * filler_hits)

        score = round(0.6 * brevity + 0.2 * single_line + 0.2 * no_filler, 4)

        return {
            "terseness_score": score,
            "terseness_word_count": words,
            "terseness_sentence_count": sentences,
            "terseness_filler_hits": filler_hits,
            "terseness_pass": score >= 0.6,
        }
