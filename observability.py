# Copyright (c) Microsoft. All rights reserved.
"""Optional OpenTelemetry wiring for Foundry's built-in Application Insights."""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


def setup_observability() -> Optional[object]:
    """Configure Azure Monitor (Application Insights) when running in Foundry.

    Foundry injects ``APPLICATIONINSIGHTS_CONNECTION_STRING`` into hosted agents.
    When present, this configures Azure Monitor's OpenTelemetry distro (which
    also auto-instruments the protocol libraries) and returns a tracer for
    manual spans. When absent (e.g. local runs without App Insights), tracing is
    disabled and None is returned.

    :return: An OpenTelemetry tracer, or None if tracing is not configured.
    """
    connection_string = os.environ.get("APPLICATIONINSIGHTS_CONNECTION_STRING")
    if not connection_string:
        logger.info("APPLICATIONINSIGHTS_CONNECTION_STRING not set; tracing disabled.")
        return None

    try:
        from azure.monitor.opentelemetry import configure_azure_monitor
        from opentelemetry import trace

        configure_azure_monitor(connection_string=connection_string)
        logger.info("Azure Monitor OpenTelemetry configured.")
        return trace.get_tracer("custom-hosted-neuro-san")
    except Exception:  # noqa: BLE001 - tracing must never break the agent.
        logger.exception("Failed to configure Azure Monitor; continuing without tracing.")
        return None
