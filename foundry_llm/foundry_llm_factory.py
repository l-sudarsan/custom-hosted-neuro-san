# Copyright (c) Microsoft. All rights reserved.
"""Custom neuro-san LLM factory that registers the passwordless Azure policy.

Referenced from ``llm_info/foundry_llm_info.hocon`` via the ``classes.factories``
key. Must have a no-arg constructor and derive from StandardLangChainLlmFactory.
"""

from neuro_san.internals.run_context.langchain.llms.standard_langchain_llm_factory import (
    StandardLangChainLlmFactory,
)

from foundry_llm.foundry_azure_llm_policy import FoundryAzureLlmPolicy


class FoundryLlmFactory(StandardLangChainLlmFactory):
    """Standard factory that adds a passwordless ``azure-openai-passwordless`` policy.

    We register a NEW class name rather than overriding the stock ``azure-openai``
    entry. neuro-san's DefaultLlmFactory always runs the stock
    StandardLangChainLlmFactory FIRST and only falls through to this factory when
    the stock one raises a plain ValueError for an unrecognized class. If we
    reused ``azure-openai`` here, the stock factory would handle it first, fail
    on missing credentials with an API-key error, and re-raise immediately --
    never reaching this factory.
    """

    def __init__(self):
        # super() builds the full default class -> policy map, so every other
        # provider keeps working; we only add a new passwordless azure entry.
        super().__init__()
        self.class_to_llm_policy_type["azure-openai-passwordless"] = FoundryAzureLlmPolicy
