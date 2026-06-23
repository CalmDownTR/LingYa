"""Tests for lingya.gateway.auth — FastAPI HTTPBearer authentication."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient


def _make_app(auth_enabled: bool):
    """Create a minimal FastAPI app with auth for testing."""
    from lingya.gateway.auth import create_auth_dependency

    app = FastAPI()
    auth = create_auth_dependency(auth_enabled=auth_enabled)

    @app.get("/protected")
    async def protected(_auth: bool = auth):
        return {"ok": True}

    return app


class TestAuthDisabled:
    """When auth_enabled=False, all requests pass."""

    def test_no_auth_header_passes(self):
        app = _make_app(auth_enabled=False)
        client = TestClient(app)
        resp = client.get("/protected")
        assert resp.status_code == 200

    def test_any_header_passes(self):
        app = _make_app(auth_enabled=False)
        client = TestClient(app)
        resp = client.get("/protected", headers={"Authorization": "Bearer garbage"})
        assert resp.status_code == 200


class TestAuthEnabled:
    """When auth_enabled=True, valid Bearer token required."""

    def test_missing_api_key_env_returns_500(self):
        """Server error when LINGYA_API_KEY env var is not set."""
        app = _make_app(auth_enabled=True)
        client = TestClient(app)

        old = os.environ.pop("LINGYA_API_KEY", None)
        try:
            resp = client.get("/protected", headers={"Authorization": "Bearer x"})
            assert resp.status_code == 500
        finally:
            if old is not None:
                os.environ["LINGYA_API_KEY"] = old

    def test_missing_auth_header_returns_401(self):
        """No Authorization header at all returns 401."""
        app = _make_app(auth_enabled=True)
        client = TestClient(app)

        os.environ["LINGYA_API_KEY"] = "secret"
        try:
            resp = client.get("/protected")
            assert resp.status_code == 401
        finally:
            os.environ.pop("LINGYA_API_KEY", None)

    def test_wrong_key_returns_401(self):
        """Wrong Bearer token returns 401."""
        app = _make_app(auth_enabled=True)
        client = TestClient(app)

        os.environ["LINGYA_API_KEY"] = "correct-secret"
        try:
            resp = client.get(
                "/protected",
                headers={"Authorization": "Bearer wrong-key"},
            )
            assert resp.status_code == 401
        finally:
            os.environ.pop("LINGYA_API_KEY", None)

    def test_correct_key_passes(self):
        """Correct Bearer token returns 200."""
        app = _make_app(auth_enabled=True)
        client = TestClient(app)

        os.environ["LINGYA_API_KEY"] = "correct-secret"
        try:
            resp = client.get(
                "/protected",
                headers={"Authorization": "Bearer correct-secret"},
            )
            assert resp.status_code == 200
            assert resp.json() == {"ok": True}
        finally:
            os.environ.pop("LINGYA_API_KEY", None)

    def test_invalid_header_format_returns_401(self):
        """Malformed Authorization header returns 401."""
        app = _make_app(auth_enabled=True)
        client = TestClient(app)

        os.environ["LINGYA_API_KEY"] = "secret"
        try:
            resp = client.get(
                "/protected",
                headers={"Authorization": "NotBearer xyz"},
            )
            assert resp.status_code == 401
        finally:
            os.environ.pop("LINGYA_API_KEY", None)
