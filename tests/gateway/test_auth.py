"""Tests for lingya.gateway.auth — WebSocket authentication."""

from __future__ import annotations

import os

import pytest


class TestWSAuth:
    def test_enabled_false_skips_validation(self):
        """When auth_enabled is False, any message passes."""
        from lingya.gateway.auth import WSAuth

        auth = WSAuth(enabled=False)
        assert auth.validate({"type": "garbage"})

    def test_missing_key_env_var_rejects(self):
        """Missing LINGYA_API_KEY should reject auth message."""
        from lingya.gateway.auth import WSAuth

        # Ensure env var is unset
        old = os.environ.pop("LINGYA_API_KEY", None)
        try:
            auth = WSAuth(enabled=True)
            assert not auth.validate(
                {"type": "auth", "payload": {"key": "anything"}}
            )
        finally:
            if old is not None:
                os.environ["LINGYA_API_KEY"] = old

    def test_wrong_key_rejects(self):
        """Wrong API key should reject."""
        from lingya.gateway.auth import WSAuth

        os.environ["LINGYA_API_KEY"] = "correct-secret"
        try:
            auth = WSAuth(enabled=True)
            assert not auth.validate(
                {"type": "auth", "payload": {"key": "wrong-key"}}
            )
        finally:
            os.environ.pop("LINGYA_API_KEY", None)

    def test_correct_key_accepts(self):
        """Correct API key should accept."""
        from lingya.gateway.auth import WSAuth

        os.environ["LINGYA_API_KEY"] = "correct-secret"
        try:
            auth = WSAuth(enabled=True)
            assert auth.validate(
                {"type": "auth", "payload": {"key": "correct-secret"}}
            )
        finally:
            os.environ.pop("LINGYA_API_KEY", None)

    def test_non_auth_message_rejects(self):
        """A message that isn't type 'auth' should be rejected."""
        from lingya.gateway.auth import WSAuth

        os.environ["LINGYA_API_KEY"] = "secret"
        try:
            auth = WSAuth(enabled=True)
            assert not auth.validate(
                {"type": "chat", "payload": {"text": "hello"}}
            )
        finally:
            os.environ.pop("LINGYA_API_KEY", None)

    def test_missing_payload_rejects(self):
        """Message without payload should be rejected."""
        from lingya.gateway.auth import WSAuth

        os.environ["LINGYA_API_KEY"] = "secret"
        try:
            auth = WSAuth(enabled=True)
            assert not auth.validate({"type": "auth"})
        finally:
            os.environ.pop("LINGYA_API_KEY", None)
