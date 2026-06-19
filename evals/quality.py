"""LLM-graded quality evaluators for the `custom-hosted-neuro-san` agent.

Why custom evaluators
---------------------
The built-in ``azure-ai-evaluation`` quality evaluators (Relevance, Coherence,
Fluency) send ``max_tokens`` to the judge model. The project's only deployed
judge model, ``gpt-5.4-mini``, is a reasoning-style model that rejects
``max_tokens`` and requires ``max_completion_tokens`` instead, so those built-in
evaluators fail with HTTP 400. These thin wrappers call the judge model directly
with the correct parameter and return the same 1-5 Likert scores, so the suite
runs end-to-end against the available deployment.

Each evaluator is callable and returns ``{"<metric>": <float 1-5>}`` so it plugs
straight into ``azure.ai.evaluation.evaluate``.
"""

from __future__ import annotations

import json
import os
import re

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import AzureOpenAI

JUDGE_ENDPOINT = os.environ.get(
    "FOUNDRY_OPENAI_ENDPOINT", "https://foundry-demo-su.services.ai.azure.com/"
)
JUDGE_DEPLOYMENT = os.environ.get("FOUNDRY_JUDGE_DEPLOYMENT", "gpt-5.4-mini")
JUDGE_API_VERSION = os.environ.get("FOUNDRY_JUDGE_API_VERSION", "2024-12-01-preview")

# Token scope for the Azure OpenAI / Cognitive Services data plane.
JUDGE_SCOPE = "https://cognitiveservices.azure.com/.default"


def _build_judge() -> AzureOpenAI:
    # When the parent process has pre-acquired a token (set in the env), reuse it.
    # The eval batch engine fans the judges out across concurrent workers; sharing
    # one token avoids every worker shelling out to `az` in parallel (which races
    # and intermittently fails). The token outlives a short eval run.
    static_token = os.environ.get("FOUNDRY_JUDGE_AAD_TOKEN")
    if static_token:
        return AzureOpenAI(
            azure_endpoint=JUDGE_ENDPOINT,
            azure_ad_token=static_token,
            api_version=JUDGE_API_VERSION,
        )

    token_provider = get_bearer_token_provider(DefaultAzureCredential(), JUDGE_SCOPE)
    return AzureOpenAI(
        azure_endpoint=JUDGE_ENDPOINT,
        azure_ad_token_provider=token_provider,
        api_version=JUDGE_API_VERSION,
    )


# Shared judge client (DefaultAzureCredential caches tokens internally).
_JUDGE = _build_judge()


def _grade(metric: str, rubric: str, query: str, response: str) -> float:
    """Ask the judge model for a 1-5 score and parse it robustly."""
    system = (
        "You are a strict evaluation judge. You rate a single assistant response "
        "on one quality dimension using an integer score from 1 (worst) to 5 (best). "
        "Respond ONLY with compact JSON: {\"score\": <int 1-5>, \"reason\": \"<short>\"}."
    )
    user = (
        f"Dimension: {metric}\n"
        f"Rubric: {rubric}\n\n"
        f"--- User request ---\n{query}\n\n"
        f"--- Assistant response ---\n{response}\n\n"
        "Return the JSON now."
    )

    completion = _JUDGE.chat.completions.create(
        model=JUDGE_DEPLOYMENT,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_completion_tokens=2000,
        response_format={"type": "json_object"},
    )
    content = (completion.choices[0].message.content or "").strip()

    try:
        score = float(json.loads(content)["score"])
    except (ValueError, KeyError, json.JSONDecodeError):
        match = re.search(r"[1-5]", content)
        score = float(match.group(0)) if match else 1.0

    return max(1.0, min(5.0, score))


class RelevanceJudge:
    """How well the response addresses the user's request (1-5)."""

    _RUBRIC = (
        "5 = fully conveys the requested sentiment to the right audience; "
        "3 = partially on-topic; 1 = unrelated or wrong sentiment. "
        "Brevity is expected and must NOT lower the score."
    )

    def __call__(self, *, query: str, response: str, **kwargs) -> dict:
        return {"relevance": _grade("Relevance", self._RUBRIC, query, response)}


class CoherenceJudge:
    """Logical clarity and readability of the response (1-5)."""

    _RUBRIC = (
        "5 = clear, well-formed and easy to understand; 3 = somewhat awkward; "
        "1 = confusing or contradictory. Short phrases can still score 5."
    )

    def __call__(self, *, query: str, response: str, **kwargs) -> dict:
        return {"coherence": _grade("Coherence", self._RUBRIC, query, response)}


class FluencyJudge:
    """Grammatical and natural-language quality of the response (1-5)."""

    _RUBRIC = (
        "5 = natural and grammatical; 3 = minor errors; 1 = broken or "
        "unnatural language. Deliberate short/stylized phrasing may score 4-5 "
        "if it reads naturally."
    )

    def __call__(self, *, query: str, response: str, **kwargs) -> dict:
        return {"fluency": _grade("Fluency", self._RUBRIC, query, response)}
