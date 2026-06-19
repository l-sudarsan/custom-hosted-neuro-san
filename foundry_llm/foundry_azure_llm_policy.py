# Copyright (c) Microsoft. All rights reserved.
"""Passwordless Azure OpenAI LlmPolicy for neuro-san.

The stock neuro-san ``azure-openai`` policy can only inject a *static*
``azure_ad_token`` from config (per its own source comment, an
``azure_ad_token_provider`` "is a complex object, and we can't set that through
config"). A static token expires, which breaks long-lived hosted agents.

This policy instead injects a refreshing ``azure_ad_token_provider`` backed by
``DefaultAzureCredential`` (the Foundry-injected agent managed identity; locally
your ``az login`` identity), so the underlying clients refresh tokens.

Two construction points need the provider:

1. ``create_client`` builds the ``AsyncAzureOpenAI`` reach-in that neuro-san
   hands to the chat model for async inference.
2. ``create_llm`` builds langchain's ``AzureChatOpenAI``, which ALSO constructs
   its own internal *sync* ``root_client``. Even when an async client is
   supplied, langchain validates that sync client at construction time, so it
   too needs credentials. The stock ``create_llm`` cannot pass a token provider,
   so we override it here.

If an explicit API key is present (e.g. for local development), both methods
defer to the stock key-based behavior.
"""

from typing import Any, Dict

from neuro_san.internals.run_context.langchain.llms.azure_llm_policy import AzureLlmPolicy

_COGNITIVE_SERVICES_SCOPE = "https://cognitiveservices.azure.com/.default"


class FoundryAzureLlmPolicy(AzureLlmPolicy):
    """Azure OpenAI policy that authenticates with a refreshing AAD token provider."""

    def __init__(self):
        super().__init__()
        self._cached_token_provider: Any = None

    def _token_provider(self) -> Any:
        """Lazily build and cache a refreshing bearer-token provider."""
        if self._cached_token_provider is None:
            from azure.identity import DefaultAzureCredential, get_bearer_token_provider

            self._cached_token_provider = get_bearer_token_provider(
                DefaultAzureCredential(),
                _COGNITIVE_SERVICES_SCOPE,
            )
        return self._cached_token_provider

    def _explicit_api_key(self, config: Dict[str, Any]) -> str | None:
        """Return an explicitly configured API key, if any."""
        key = self.get_value_or_env(config, "openai_api_key", "AZURE_OPENAI_API_KEY")
        if key is None:
            key = self.get_value_or_env(config, "openai_api_key", "OPENAI_API_KEY")
        return key

    def create_client(self, config: Dict[str, Any]) -> Any:
        """Create an AsyncAzureOpenAI client using managed-identity auth.

        :param config: The fully specified llm config.
        :return: The ``chat.completions`` reach-in handed to the BaseLanguageModel.
        """
        # If an API key is explicitly configured, fall back to stock behavior.
        if self._explicit_api_key(config) is not None:
            return super().create_client(config)

        # Lazy-resolve the OpenAI client class through neuro-san's resolver so the
        # same install-if-missing behavior applies as the stock policy.
        # pylint: disable=invalid-name
        AsyncAzureOpenAI = self.resolver.resolve_class_in_module(
            "AsyncAzureOpenAI",
            module_name="openai",
            install_if_missing="langchain-openai",
        )

        self.create_http_client(config)

        default_headers: Dict[str, str] = config.get("default_headers", {})
        default_headers.update({"User-Agent": "neuro-san-foundry-hosted-agent"})

        self.async_openai_client = AsyncAzureOpenAI(
            azure_endpoint=self.get_value_or_env(config, "azure_endpoint", "AZURE_OPENAI_ENDPOINT"),
            azure_deployment=self.get_value_or_env(config, "deployment_name", "AZURE_OPENAI_DEPLOYMENT_NAME"),
            api_version=self.get_value_or_env(config, "openai_api_version", "OPENAI_API_VERSION"),
            # Refreshing AAD token provider — this is what the stock policy cannot do.
            azure_ad_token_provider=self._token_provider(),
            organization=self.get_value_or_env(config, "openai_organization", "OPENAI_ORG_ID"),
            timeout=config.get("request_timeout"),
            max_retries=config.get("max_retries"),
            default_headers=default_headers,
            http_client=self.http_client,
        )

        # We retain async_openai_client for cleanup, but hand back this reach-in
        # to pass to the BaseLanguageModel constructor (mirrors the stock policy).
        return self.async_openai_client.chat.completions

    def create_llm(self, config: Dict[str, Any], model_name: str, client: Any) -> Any:
        """Build AzureChatOpenAI with a refreshing AAD token provider.

        Mirrors the stock ``AzureLlmPolicy.create_llm`` but adds
        ``azure_ad_token_provider`` so langchain's internal sync ``root_client``
        construction succeeds without an API key or static token.

        :param config: The fully specified llm config.
        :param model_name: The name of the model.
        :param client: The async web client created by ``create_client``.
        :return: A configured ``AzureChatOpenAI`` instance.
        """
        # If an API key is explicitly configured, defer to stock behavior.
        if self._explicit_api_key(config) is not None:
            return super().create_llm(config, model_name, client)

        # pylint: disable=invalid-name
        AzureChatOpenAI = self.resolver.resolve_class_in_module(
            "AzureChatOpenAI",
            module_name="langchain_openai.chat_models.azure",
            install_if_missing="langchain-openai",
        )

        llm = AzureChatOpenAI(
            async_client=client,
            # Refreshing AAD token provider — applied to both the sync root_client
            # (which langchain always validates at construction) and the async client.
            azure_ad_token_provider=self._token_provider(),
            model_name=model_name,
            temperature=config.get("temperature"),
            openai_api_base=self.get_value_or_env(config, "openai_api_base", "OPENAI_API_BASE", client),
            openai_organization=self.get_value_or_env(config, "openai_organization", "OPENAI_ORG_ID", client),
            openai_proxy=self.get_value_or_env(config, "openai_proxy", "OPENAI_PROXY", client),
            request_timeout=self.get_value_or_env(config, "request_timeout", None, client),
            max_retries=self.get_value_or_env(config, "max_retries", None, client),
            presence_penalty=config.get("presence_penalty"),
            frequency_penalty=config.get("frequency_penalty"),
            seed=config.get("seed"),
            logprobs=config.get("logprobs"),
            top_logprobs=config.get("top_logprobs"),
            logit_bias=config.get("logit_bias"),
            # See stock policy: neuro-san does not consume per-token chunks.
            streaming=False,
            n=1,
            top_p=config.get("top_p"),
            max_tokens=config.get("max_tokens"),
            tiktoken_model_name=config.get("tiktoken_model_name"),
            stop=config.get("stop"),
            verbose=False,
            azure_endpoint=self.get_value_or_env(config, "azure_endpoint", "AZURE_OPENAI_ENDPOINT", client),
            deployment_name=self.get_value_or_env(config, "deployment_name", "AZURE_OPENAI_DEPLOYMENT_NAME", client),
            openai_api_version=self.get_value_or_env(config, "openai_api_version", "OPENAI_API_VERSION", client),
            openai_api_type=self.get_value_or_env(config, "openai_api_type", "OPENAI_API_TYPE", client),
            model_version=config.get("model_version"),
        )

        return llm
