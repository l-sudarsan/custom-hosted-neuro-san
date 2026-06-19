"""Invoke the deployed `custom-hosted-neuro-san` hosted agent in Microsoft Foundry.

Usage:
    python client.py                       # runs a default sample prompt
    python client.py "your prompt here"    # runs a single prompt
    python client.py --stream "..."        # stream the response as it is generated

Auth:
    Uses DefaultAzureCredential. Locally, run `az login` first. The caller needs
    the `Foundry User` role on the project.

Install:
    pip install openai azure-identity

Note:
    This targets the hosted agent's dedicated Responses endpoint directly. The
    `AIProjectClient.get_openai_client()` helper currently fails for this project
    with "Workspace not found", so we point the OpenAI client at the verified
    agent endpoint and pass the Entra token as the bearer credential.
"""

from __future__ import annotations

import argparse
import os

from azure.identity import DefaultAzureCredential
from openai import OpenAI

# Format: https://<resource>.services.ai.azure.com/api/projects/<project>
PROJECT_ENDPOINT = os.environ.get(
    "FOUNDRY_PROJECT_ENDPOINT",
    "https://foundry-demo-su.services.ai.azure.com/api/projects/proj-default",
)
AGENT_NAME = os.environ.get("FOUNDRY_AGENT_NAME", "custom-hosted-neuro-san")
API_VERSION = os.environ.get("FOUNDRY_API_VERSION", "v1")

DEFAULT_PROMPT = (
    "From earth, I approach a new planet and wish to send a short 2-word "
    "greeting to the new orb."
)


def build_client() -> OpenAI:
    credential = DefaultAzureCredential()
    token = credential.get_token("https://ai.azure.com/.default").token

    # The hosted agent's dedicated Responses endpoint. The OpenAI client appends
    # "/responses" to this base URL.
    base_url = f"{PROJECT_ENDPOINT}/agents/{AGENT_NAME}/endpoint/protocols/openai"

    return OpenAI(
        base_url=base_url,
        api_key=token,  # sent as "Authorization: Bearer <token>"
        default_query={"api-version": API_VERSION},
        default_headers={"Foundry-Features": "HostedAgents=V1Preview"},
    )


def invoke(prompt: str, stream: bool = False) -> None:
    client = build_client()

    print(f"> {prompt}\n")

    if stream:
        events = client.responses.create(input=prompt, stream=True)
        for event in events:
            delta = getattr(event, "delta", None)
            if delta:
                print(delta, end="", flush=True)
        print()
    else:
        response = client.responses.create(input=prompt, store=True)
        print(response.output_text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Invoke the Foundry hosted agent.")
    parser.add_argument(
        "prompt",
        nargs="?",
        default=DEFAULT_PROMPT,
        help="Prompt to send to the agent.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream the response as server-sent events.",
    )
    args = parser.parse_args()

    invoke(args.prompt, stream=args.stream)


if __name__ == "__main__":
    main()
