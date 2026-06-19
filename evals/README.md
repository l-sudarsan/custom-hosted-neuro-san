# Evaluation suite — `custom-hosted-neuro-san`

A runnable quality-evaluation harness for the **deployed** Foundry hosted agent.
It invokes the live agent for a set of representative prompts and scores the
responses with model-graded quality evaluators plus a custom terseness
evaluator that matches the agent's "shortest possible announcement" objective.

## Layout

| File | Purpose |
| --- | --- |
| [dataset.jsonl](dataset.jsonl) | Evaluation prompts (one JSON object per line). |
| [evaluate.py](evaluate.py) | Orchestrator: invokes the agent, then scores responses. |
| [quality.py](quality.py) | LLM-graded judges: relevance, coherence, fluency. |
| [terseness.py](terseness.py) | Custom rule-based terseness/brevity evaluator. |
| `results/` | Cached agent outputs + timestamped scoring runs. |

## Why custom judges instead of the built-in evaluators

The project's only judge deployment, `gpt-5.4-mini`, is a reasoning-style model
that **rejects the `max_tokens` parameter** and requires `max_completion_tokens`.
The built-in `azure-ai-evaluation` quality evaluators send `max_tokens` and fail
with HTTP 400, so `quality.py` provides thin LLM-graded judges that call the
model with the correct parameter and return the same 1-5 Likert scores.

> The Foundry-native auto-generated evaluation suite (MCP
> `evaluation_suite_*`) currently returns `500 "Unable to get resource
> information."` for this project even after linking a storage connection, so
> this SDK-based harness is the working path.

## Prerequisites

```powershell
# From the repo root, with the venv active
pip install -r requirements-dev.txt
az login
```

Passwordless auth via `DefaultAzureCredential`:

- **Agent invocation** needs the `Foundry User` role on the project.
- **Judge model calls** need `Cognitive Services OpenAI User` on the Foundry account.

## Run

```powershell
# Full run: invoke the live agent, then score the responses
python evals/evaluate.py

# Skip invocation; re-score the existing cached outputs (fast judge iteration)
python evals/evaluate.py --no-generate

# Only invoke the agent and cache outputs (no scoring; quick connectivity check)
python evals/evaluate.py --generate-only
```

> **Exit code:** `evaluate.py` exits with code `1` at interpreter teardown
> even on a fully successful run (an `azure-ai-evaluation`/promptflow quirk).
> The results are still written — trust the saved JSON and the printed
> `Run status: "Completed"` lines, not the exit code.

### Inspect the latest results

```powershell
$f = Get-ChildItem evals/results/eval_results_*.json | Sort-Object LastWriteTime | Select-Object -Last 1
$j = Get-Content $f.FullName -Raw | ConvertFrom-Json
$j.metrics
$j.rows | ForEach-Object {
  "{0} -> rel={1} terse={2} words={3}" -f `
    $_.'inputs.query', $_.'outputs.relevance.relevance', `
    $_.'outputs.terseness.terseness_score', $_.'outputs.terseness.terseness_word_count'
}
```

## Metrics

| Metric | Range | Meaning |
| --- | --- | --- |
| `relevance` | 1-5 | Conveys the requested sentiment to the right audience. |
| `coherence` | 1-5 | Clear, well-formed, easy to understand. |
| `fluency` | 1-5 | Natural, grammatical language. |
| `terseness_score` | 0-1 | Brevity + single-line + no-filler blend. |
| `terseness_word_count` | int | Words in the response (agent targets ~2). |
| `terseness_pass` | 0/1 | `terseness_score >= 0.6`. |

## Configuration (env overrides)

| Variable | Default | Used by |
| --- | --- | --- |
| `FOUNDRY_PROJECT_ENDPOINT` | `https://foundry-demo-su.services.ai.azure.com/api/projects/proj-default` | agent invocation |
| `FOUNDRY_AGENT_NAME` | `custom-hosted-neuro-san` | agent invocation |
| `FOUNDRY_API_VERSION` | `v1` | agent invocation |
| `FOUNDRY_OPENAI_ENDPOINT` | `https://foundry-demo-su.services.ai.azure.com/` | judge model |
| `FOUNDRY_JUDGE_DEPLOYMENT` | `gpt-5.4-mini` | judge model |
| `FOUNDRY_JUDGE_API_VERSION` | `2024-12-01-preview` | judge model |

> The harness pre-acquires one AAD token in the parent process and passes it to
> the concurrent judge workers via `FOUNDRY_JUDGE_AAD_TOKEN`. You don't set this
> yourself — it prevents every worker from shelling out to `az` in parallel
> (which races and intermittently fails to fetch a token).

## Latest baseline (8 prompts)

| Metric | Score |
| --- | --- |
| relevance | 3.75 / 5 |
| coherence | 3.875 / 5 |
| fluency | 3.375 / 5 |
| terseness_score | 0.60 / 1 |
| terseness_pass | 0.625 (5 / 8) |
| avg word count | 6.25 |

The agent excels at the canonical 2-word greeting (`"Hallo world"`, score 1.0)
and scores lower on prompts whose natural answer is longer, which is the
expected tension for a terse-announcement agent.
