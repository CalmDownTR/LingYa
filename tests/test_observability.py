"""Tests for lingya.observability."""

from __future__ import annotations


class TestInitObservability:
    def test_disabled_returns_none(self):
        """When otel.enabled=False, init_observability returns None."""
        from lingya.config import Config, OtelConfig
        from lingya.observability import init_observability

        config = Config(otel=OtelConfig(enabled=False))
        tracer = init_observability(config)
        assert tracer is None

    def test_no_otel_config_defaults_to_none(self):
        """Default Config (otel.enabled=False) returns None."""
        from lingya.config import Config
        from lingya.observability import init_observability

        config = Config()
        tracer = init_observability(config)
        assert tracer is None
