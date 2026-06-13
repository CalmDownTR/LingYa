"""WebSocket authentication — validates API key on connect.

Integrated into GatewayServer._handle_connection after the WebSocket
handshake and before the message loop.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class WSAuth:
    """Validates client auth messages against LINGYA_API_KEY.

    Usage in GatewayServer::

        auth = WSAuth(enabled=True)
        if not auth.validate(first_message):
            # send auth_failed response, close connection
    """

    enabled: bool = True
    timeout: float = 5.0  # seconds to send first message after connect

    def validate(self, message: dict[str, Any]) -> bool:
        """Return True if *message* is a valid auth message or auth is disabled."""
        if not self.enabled:
            return True

        if message.get("type") != "auth":
            return False

        payload = message.get("payload", {})
        key = payload.get("key", "")
        expected = os.environ.get("LINGYA_API_KEY", "")

        if not expected:
            logger.warning("WSAuth enabled but LINGYA_API_KEY env var is not set")
            return False

        return key == expected
