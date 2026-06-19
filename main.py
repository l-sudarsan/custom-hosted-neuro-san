# Copyright (c) Microsoft. All rights reserved.
"""Foundry hosted-agent entrypoint.

Serves a neuro-san agent network over the Responses protocol. User input is
forwarded to the neuro-san network (running in-process) and its answer is
streamed back through the Responses protocol.
"""

import asyncio
import logging
import os
import sys

from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("custom-hosted-neuro-san")

# Load local .env for development; never override platform-injected values.
load_dotenv(override=False)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    # Ensure neuro-san can import the foundry_llm factory by dotted path.
    sys.path.insert(0, BASE_DIR)

# ---------------------------------------------------------------------------
# neuro-san configuration.
#
# neuro-san is configured exclusively through AGENT_* environment variables.
# Foundry RESERVES the AGENT_* prefix in agent.yaml / .env, so we MUST set these
# in-process here, BEFORE importing anything from neuro_san.
# ---------------------------------------------------------------------------
os.environ["AGENT_MANIFEST_FILE"] = os.path.join(BASE_DIR, "registries", "manifest.hocon")
os.environ["AGENT_TOOL_PATH"] = os.path.join(BASE_DIR, "coded_tools")
os.environ["AGENT_LLM_INFO_FILE"] = os.path.join(BASE_DIR, "llm_info", "foundry_llm_info.hocon")

# ---------------------------------------------------------------------------
# Azure OpenAI configuration normalization.
#
# Foundry injects AZURE_AI_MODEL_DEPLOYMENT_NAME. neuro-san's azure-openai policy
# reads AZURE_OPENAI_DEPLOYMENT_NAME / AZURE_OPENAI_ENDPOINT / OPENAI_API_VERSION,
# so map / default them here.
# ---------------------------------------------------------------------------
_deployment = os.environ.get("AZURE_AI_MODEL_DEPLOYMENT_NAME", "")
if _deployment:
    os.environ.setdefault("AZURE_OPENAI_DEPLOYMENT_NAME", _deployment)
os.environ.setdefault("OPENAI_API_VERSION", "2024-10-21")

# Name of the agent network (key in registries/manifest.hocon, without .hocon).
AGENT_NAME = os.environ.get("NEURO_SAN_AGENT_NAME", "hello_world")

# Configure tracing before importing/constructing the rest so auto-instrumentation
# captures startup work too.
from observability import setup_observability  # noqa: E402

_tracer = setup_observability()

# Imported AFTER AGENT_* env vars are set so neuro-san picks up configuration.
from neuro_san_runtime.session_runner import NeuroSanRunner  # noqa: E402

from azure.ai.agentserver.responses import (  # noqa: E402
    CreateResponse,
    ResponseContext,
    ResponsesAgentServerHost,
    ResponsesServerOptions,
    TextResponse,
)

_runner = NeuroSanRunner(AGENT_NAME)

# In-memory neuro-san chat_context cache keyed by a best-effort conversation id.
# The Responses platform manages user-visible history; this cache only preserves
# neuro-san's internal multi-agent state across turns of the SAME conversation.
# If no stable id can be derived from the request/context, the turn runs
# single-turn (still correct, just without carried-over neuro-san state).
_CONTEXT_CACHE: dict = {}

app = ResponsesAgentServerHost(
    options=ResponsesServerOptions(default_fetch_history_count=20),
)


def _conversation_id(request, context):
    """Best-effort extraction of a stable conversation id; None if unavailable."""
    for source in (request, context):
        for attr in ("conversation", "conversation_id", "previous_response_id", "thread_id"):
            value = getattr(source, attr, None)
            if isinstance(value, str) and value:
                return value
    return None


@app.response_handler
async def handler(
    request: CreateResponse,
    context: ResponseContext,
    _cancellation_signal: asyncio.Event,
):
    """Forward the user's input to the neuro-san network and return its answer."""
    user_input = await context.get_input_text() or "Hello!"
    convo_id = _conversation_id(request, context)
    previous_context = _CONTEXT_CACHE.get(convo_id) if convo_id else None

    loop = asyncio.get_running_loop()

    def _run():
        # neuro-san's Direct session is synchronous; run it off the event loop.
        if _tracer is not None:
            with _tracer.start_as_current_span("neuro_san.streaming_chat"):
                return _runner.run(user_input, previous_context)
        return _runner.run(user_input, previous_context)

    answer, new_context = await loop.run_in_executor(None, _run)

    if convo_id and new_context:
        _CONTEXT_CACHE[convo_id] = new_context

    return TextResponse(context, request, text=answer or "")


app.run()
