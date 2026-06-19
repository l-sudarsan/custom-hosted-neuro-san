# Custom hosted neuro-san agent on Microsoft Foundry

A [neuro-san](https://github.com/cognizant-ai-lab/neuro-san) multi-agent network
packaged as a **Microsoft Foundry hosted agent** using the **Responses**
protocol. The agent network runs in-process; user input is forwarded to
neuro-san and its answer is returned through the Responses protocol.

Includes:

- **Passwordless model access** — neuro-san calls the Foundry project's Azure
  OpenAI deployment using the agent's managed identity (no API keys), with
  automatic token refresh.
- **Built-in observability** — OpenTelemetry traces export to the Application
  Insights instance that Foundry attaches to the agent.

## Project layout

| Path | Purpose |
| --- | --- |
| `main.py` | Responses-protocol entrypoint; wires neuro-san + tracing. |
| `observability.py` | Configures Azure Monitor OpenTelemetry when available. |
| `neuro_san_runtime/session_runner.py` | Runs the neuro-san network via an in-process Direct session. |
| `foundry_llm/foundry_azure_llm_policy.py` | Passwordless Azure OpenAI LLM policy (refreshing AAD token). |
| `foundry_llm/foundry_llm_factory.py` | Registers the passwordless policy with neuro-san. |
| `llm_info/foundry_llm_info.hocon` | Defines the `foundry-gpt` model + custom factory. |
| `registries/manifest.hocon` | Lists enabled agent networks. |
| `registries/hello_world.hocon` | The vendored sample agent network. |
| `coded_tools/` | CodedTool package (empty for the sample). |
| `Dockerfile` | Container image (port 8088, Linux AMD64). |
| `agent.yaml` | Foundry hosted-agent manifest. |

## How configuration flows

neuro-san is configured only through `AGENT_*` environment variables, but
Foundry reserves the `AGENT_*` prefix in `agent.yaml`/`.env`. So `main.py` sets
`AGENT_MANIFEST_FILE`, `AGENT_TOOL_PATH`, and `AGENT_LLM_INFO_FILE` **in-process
before importing neuro-san**. Only non-reserved variables go in `agent.yaml`.

## Prerequisites

- An Azure subscription with access to Microsoft Foundry (hosted agents, preview).
- A model deployed in your Foundry project (e.g. `gpt-4o`).
- Azure CLI, Docker, and an Azure Container Registry (ACR) the Foundry project can pull from.

## Local development

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

Copy-Item .env.example .env   # then edit values
# Sign in so DefaultAzureCredential can get tokens for passwordless auth:
az login

python main.py
```

The server listens on `http://localhost:8088`. Send a Responses request:

```powershell
curl -X POST http://localhost:8088/responses `
  -H "Content-Type: application/json" `
  -d '{"input":"Greet developers that wrote their very first program","stream":false}'
```

> Locally, ensure your signed-in identity has the **Cognitive Services OpenAI
> User** role on the Azure OpenAI / AI Services resource (see RBAC below), or set
> `AZURE_OPENAI_API_KEY` in `.env` to use key-based auth instead.

## Build & push the image

Foundry hosted agents run on Linux AMD64 — always build with `--platform`:

```powershell
$acr = "<your-acr-name>"
az acr login --name $acr
docker build --platform linux/amd64 -t "$acr.azurecr.io/custom-hosted-neuro-san:latest" .
docker push "$acr.azurecr.io/custom-hosted-neuro-san:latest"
```

## Deploy to Foundry

Provide `agent.yaml` and the pushed image to your Foundry project (portal or
CLI). At deploy time, set the environment variable values referenced by
`agent.yaml`:

- `AZURE_AI_MODEL_DEPLOYMENT_NAME` — your model deployment name.
- `AZURE_OPENAI_ENDPOINT` — your Azure OpenAI endpoint, e.g.
  `https://<resource>.openai.azure.com/`.
- `OPENAI_API_VERSION` — e.g. `2024-10-21`.

Foundry injects `APPLICATIONINSIGHTS_CONNECTION_STRING`, `FOUNDRY_PROJECT_ENDPOINT`,
the agent identity, and `PORT` automatically — do not set these yourself.

## RBAC for passwordless model access

The agent authenticates to Azure OpenAI with its **managed identity** via
`DefaultAzureCredential`. Grant that identity the **Cognitive Services OpenAI
User** role on the Azure OpenAI / AI Services account backing your deployment:

```powershell
az role assignment create `
  --assignee <agent-managed-identity-object-id> `
  --role "Cognitive Services OpenAI User" `
  --scope <ai-services-account-resource-id>
```

## Customize the agent network

1. Add or edit `.hocon` files under `registries/` and enable them in
   `registries/manifest.hocon`.
2. Point each network's `llm_config.model_name` at `foundry-gpt` (defined in
   `llm_info/foundry_llm_info.hocon`) for passwordless access.
3. Add any `CodedTool` classes under `coded_tools/`.
4. To serve a different network by default, set `NEURO_SAN_AGENT_NAME`.
