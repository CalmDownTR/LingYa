"""FastAPI authentication — HTTP Bearer token validated against LINGYA_API_KEY.

Usage in FastAPI app::

    from fastapi import Depends
    from lingya.gateway.auth import create_auth_dependency

    # Enabled (production):
    auth = create_auth_dependency(auth_enabled=True)

    # Disabled (dev mode):
    auth = create_auth_dependency(auth_enabled=False)

    @app.get("/mind")
    async def get_mind(_auth: bool = auth):
        ...
"""

from __future__ import annotations

import logging
import os

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

logger = logging.getLogger(__name__)

_security = HTTPBearer(auto_error=False)


async def _verify_bearer(
    credentials: HTTPAuthorizationCredentials | None = Depends(_security),
) -> bool:
    """Verify the Bearer token against LINGYA_API_KEY. Returns True if valid."""
    expected = os.environ.get("LINGYA_API_KEY", "")

    if not expected:
        logger.warning("Auth enabled but LINGYA_API_KEY env var is not set")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server auth misconfigured",
        )

    if credentials is None:
        # No Authorization header at all
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if credentials.credentials != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return True


async def _no_auth() -> bool:
    """No-op auth — always passes (dev mode)."""
    return True


def create_auth_dependency(auth_enabled: bool = True):
    """Return a FastAPI Depends for Bearer auth, or a no-op for dev mode.

    Args:
        auth_enabled: When False, all requests pass without auth (dev mode).
    """
    if auth_enabled:
        return Depends(_verify_bearer)
    else:
        return Depends(_no_auth)