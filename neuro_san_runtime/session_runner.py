# Copyright (c) Microsoft. All rights reserved.
"""Thin wrapper that runs a neuro-san agent network in-process via a Direct session.

This module imports ``neuro_san``, so it MUST only be imported after the AGENT_*
environment variables have been set (see main.py).
"""

from typing import Any, Dict, Optional, Tuple

from neuro_san.client.agent_session_factory import AgentSessionFactory
from neuro_san.internals.messages.chat_message_type import ChatMessageType
from neuro_san.message_processing.basic_message_processor import BasicMessageProcessor


class NeuroSanRunner:
    """Runs a single neuro-san agent network using an in-process Direct session."""

    def __init__(self, agent_name: str):
        self._agent_name = agent_name
        # A Direct session runs the agent network in this same process — no
        # separate neuro-san server required.
        self._session = AgentSessionFactory().create_session(
            "direct", agent_name, use_direct=True
        )

    def run(
        self,
        user_text: str,
        chat_context: Optional[Dict[str, Any]] = None,
        sly_data: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Send ``user_text`` to the agent network.

        :param user_text: The user's message.
        :param chat_context: Prior neuro-san chat_context to continue a conversation.
        :param sly_data: Optional out-of-band data passed to the network.
        :return: A tuple of (answer, new_chat_context).
        """
        request: Dict[str, Any] = {"user_message": {"text": user_text}}
        if chat_context:
            request["chat_context"] = chat_context
        if sly_data:
            request["sly_data"] = sly_data

        processor = BasicMessageProcessor()
        for chat_response in self._session.streaming_chat(request):
            message = chat_response.get("response")
            if message is None:
                continue
            message_type = ChatMessageType.from_response_type(message.get("type"))
            processor.process_message(message, message_type)

        return processor.get_compiled_answer() or "", processor.get_chat_context()
