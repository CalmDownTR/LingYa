"""OpenTelemetry observability — zero overhead when disabled.

Usage::

    from lingya.observability import init_observability
    tracer = init_observability(config)  # returns None if otel.enabled=false

    if tracer:
        with tracer.start_as_current_span("my_operation") as span:
            span.set_attribute("key", "value")
"""

from __future__ import annotations

import logging
from typing import Any

from lingya.config import Config

logger = logging.getLogger(__name__)


def init_observability(config: Config) -> Any | None:
    """Initialize OpenTelemetry if otel.enabled, return tracer or None.

    When disabled, returns None — zero overhead, no spans created.
    """

    if not config.otel.enabled:
        return None

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

        provider = TracerProvider()
        processor = BatchSpanProcessor(ConsoleSpanExporter())
        provider.add_span_processor(processor)
        trace.set_tracer_provider(provider)

        tracer = trace.get_tracer("lingya")
        logger.info("OpenTelemetry initialized (console exporter)")
        return tracer

    except ImportError:
        logger.warning("OpenTelemetry SDK not installed, tracing disabled")
        return None
    except Exception:
        logger.exception("Failed to initialize OpenTelemetry")
        return None
