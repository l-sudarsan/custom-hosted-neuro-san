"""Evaluation suite for the deployed `custom-hosted-neuro-san` hosted agent.

What it does
------------
1. (generate) Invokes the *deployed* Foundry hosted agent for every query in
   ``dataset.jsonl`` and records the responses to ``results/agent_outputs.jsonl``.
2. (evaluate) Scores those responses with custom LLM-graded quality judges
   (Relevance, Coherence, Fluency in ``quality.py``) plus a custom
   ``TersenessEvaluator`` tailored to this agent's "shortest possible
   announcement" objective.

   The built-in ``azure-ai-evaluation`` quality evaluators are not used because
   they send ``max_tokens`` to the judge model, which the deployed
   ``gpt-5.4-mini`` reasoning model rejects (it requires
   ``max_completion_tokens``).

Usage
-----
    # Full run: invoke the live agent, then score the responses
    python evals/evaluate.py

    # Skip invocation and only re-score existing outputs
    python evals/evaluate.py --no-generate

    # Only invoke the agent and cache outputs (no scoring)
    python evals/evaluate.py --generate-only

Auth
----
Passwordless via ``DefaultAzureCredential`` (run ``az login`` locally):
  * Agent invocation needs the ``Foundry User`` role on the project.
  * The LLM-graded judges call the ``gpt-5.4-mini`` deployment and need
    ``Cognitive Services OpenAI User`` on the Foundry account.

Install
-------
    pip install -r requirements-dev.txt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Make the repo root importable so we can reuse the verified client helper.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from client import build_client  # noqa: E402  (reuses the working invocation path)

HERE = Path(__file__).resolve().parent
DATASET = HERE / "dataset.jsonl"
RESULTS_DIR = HERE / "results"
OUTPUTS = RESULTS_DIR / "agent_outputs.jsonl"


def _read_dataset() -> list[dict]:
    rows: list[dict] = []
    with DATASET.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def generate_outputs() -> Path:
    """Invoke the live agent for each query and cache {query, response} rows."""
    client = build_client()
    rows = _read_dataset()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Invoking deployed agent for {len(rows)} queries...\n")
    with OUTPUTS.open("w", encoding="utf-8") as out:
        for i, row in enumerate(rows, 1):
            query = row["query"]
            response = client.responses.create(input=query, store=True)
            text = (response.output_text or "").strip()
            print(f"[{i}/{len(rows)}] {query}\n        -> {text!r}")
            out.write(
                json.dumps(
                    {
                        "query": query,
                        "response": text,
                        "expected_intent": row.get("expected_intent", ""),
                    }
                )
                + "\n"
            )
    print(f"\nCached agent outputs -> {OUTPUTS}")
    return OUTPUTS


def run_evaluation() -> dict:
    """Score the cached outputs with quality + terseness evaluators."""
    from azure.ai.evaluation import evaluate

    from evals.quality import JUDGE_SCOPE, CoherenceJudge, FluencyJudge, RelevanceJudge
    from evals.terseness import TersenessEvaluator

    # Pre-acquire one AAD token in this parent process and hand it to the judge
    # workers via the environment. The eval batch engine runs evaluators
    # concurrently; without this, each worker shells out to `az` in parallel,
    # which races and intermittently fails to fetch a token.
    from azure.identity import DefaultAzureCredential

    token = DefaultAzureCredential().get_token(JUDGE_SCOPE).token
    os.environ["FOUNDRY_JUDGE_AAD_TOKEN"] = token

    # The deployed judge model (gpt-5.4-mini) rejects the `max_tokens` parameter
    # used by the built-in azure-ai-evaluation evaluators, so we use custom
    # LLM-graded judges that call the model with `max_completion_tokens`.
    evaluators = {
        "relevance": RelevanceJudge(),
        "coherence": CoherenceJudge(),
        "fluency": FluencyJudge(),
        "terseness": TersenessEvaluator(),
    }

    column_mapping = {
        "query": "${data.query}",
        "response": "${data.response}",
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = RESULTS_DIR / f"eval_results_{stamp}.json"

    result = evaluate(
        data=str(OUTPUTS),
        evaluators=evaluators,
        evaluator_config={name: {"column_mapping": column_mapping} for name in evaluators},
        output_path=str(output_path),
    )

    print("\n=== Aggregate metrics ===")
    for metric, value in sorted(result.get("metrics", {}).items()):
        print(f"  {metric}: {value}")
    print(f"\nFull results -> {output_path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the deployed hosted agent.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--no-generate",
        action="store_true",
        help="Skip live agent invocation; score the existing cached outputs.",
    )
    group.add_argument(
        "--generate-only",
        action="store_true",
        help="Only invoke the agent and cache outputs; do not score.",
    )
    args = parser.parse_args()

    if not args.no_generate:
        generate_outputs()

    if args.generate_only:
        return

    if not OUTPUTS.exists():
        raise SystemExit(
            f"No cached outputs at {OUTPUTS}. Run without --no-generate first."
        )

    run_evaluation()


if __name__ == "__main__":
    main()
