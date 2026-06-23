"""GatewayClient — HTTP + SSE client for LingYa Gateway.

Replaces the WebSocket client with standard HTTP:

- ``send_stream()`` — POST /chat, parses SSE event stream, yields events to callback
- ``send()`` — maps message types to REST endpoints (GET /mind, GET /memory, etc.)
- ``connect()`` — verifies server is reachable via /health
- ``close()`` — closes the httpx client

Uses ``httpx`` for async HTTP (already a dependency).
"""

from __future__ import annotations

import json
import os
from collections.abc import Awaitable, Callable

import httpx


class GatewayClient:
    """Async HTTP + SSE client for connecting to LingYa Gateway."""

    def __init__(self, host: str = "localhost", port: int = 8765) -> None:
        self._host = host
        self._port = port
        self._base_url = f"http://{host}:{port}"
        self._client: httpx.AsyncClient | None = None
        self._api_key: str = os.environ.get("LINGYA_API_KEY", "")

    @property
    def is_connected(self) -> bool:
        """Whether the HTTP client is initialized (not actually a persistent connection)."""
        return self._client is not None and not self._client.is_closed

    async def connect(self) -> None:
        """Initialize HTTP client and verify server is reachable via /health."""
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(120.0, connect=10.0),
        )
        headers = self._auth_headers()
        try:
            resp = await self._client.get("/health", headers=headers)
            resp.raise_for_status()
        except Exception as e:
            await self._client.aclose()
            self._client = None
            raise ConnectionError(f"Failed to connect to Gateway: {e}") from e

    async def send(self, message: dict) -> dict:
        """Send a JSON message and wait for the response.

        Maps message types to HTTP endpoints:

        - ``ping`` → GET /health
        - ``mind`` → GET /mind?query=...
        - ``memory`` → GET /memory?action=...&query=...
        - ``diary`` → GET /diary?action=...&index=...
        - ``session`` → POST /session?action=...
        - ``stats`` → GET /stats
        - ``chat`` → delegates to send_stream() (no on_event, returns final only)

        Raises:
            ConnectionError: If not connected.
        """
        if self._client is None:
            raise ConnectionError("Not connected to Gateway")

        msg_type = message.get("type", "")
        payload = message.get("payload", {})
        headers = self._auth_headers()

        try:
            if msg_type == "ping":
                resp = await self._client.get("/health", headers=headers)
                resp.raise_for_status()
                return resp.json()

            elif msg_type == "mind":
                query = payload.get("query", "state")
                resp = await self._client.get(
                    "/mind", params={"query": query}, headers=headers
                )
                resp.raise_for_status()
                return resp.json()

            elif msg_type == "memory":
                action = payload.get("action", "search")
                params: dict = {"action": action}
                if "query" in payload:
                    params["query"] = payload["query"]
                if "id" in payload:
                    params["id"] = payload["id"]
                resp = await self._client.get(
                    "/memory", params=params, headers=headers
                )
                resp.raise_for_status()
                return resp.json()

            elif msg_type == "diary":
                action = payload.get("action", "list")
                index = payload.get("index", 0)
                params = {"action": action, "index": index}
                resp = await self._client.get(
                    "/diary", params=params, headers=headers
                )
                resp.raise_for_status()
                return resp.json()

            elif msg_type == "session":
                action = payload.get("action", "new")
                resp = await self._client.post(
                    "/session", params={"action": action}, headers=headers
                )
                resp.raise_for_status()
                return resp.json()

            elif msg_type == "stats":
                resp = await self._client.get("/stats", headers=headers)
                resp.raise_for_status()
                return resp.json()

            elif msg_type == "chat":
                # Non-streaming chat — just return the final response
                return await self.send_stream(message, on_event=None)

            else:
                return {
                    "type": "error",
                    "payload": {"message": f"Unknown message type: {msg_type}"},
                }

        except httpx.HTTPStatusError as e:
            return {
                "type": "error",
                "payload": {"message": f"HTTP {e.response.status_code}"},
            }
        except httpx.RequestError as e:
            return {
                "type": "error",
                "payload": {"message": str(e)},
            }

    async def send_stream(
        self,
        message: dict,
        on_event: Callable[[dict], Awaitable[None]] | None = None,
    ) -> dict:
        """Send a chat message and receive streaming SSE responses.

        When *on_event* is provided, each ``{"type": "event", ...}`` frame
        received before the final response is passed to *on_event* immediately.
        The final ``{"type": "chat_response", ...}`` (or error) frame is
        returned.

        When *on_event* is None, only the final frame is returned (events are
        consumed silently).

        Raises:
            ConnectionError: If not connected.
        """
        if self._client is None:
            raise ConnectionError("Not connected to Gateway")

        text = message.get("payload", {}).get("text", "")
        headers = {
            **self._auth_headers(),
            "Accept": "text/event-stream",
        }

        try:
            async with self._client.stream(
                "POST",
                "/chat",
                json={"text": text},
                headers=headers,
                timeout=httpx.Timeout(300.0, connect=10.0),
            ) as response:
                response.raise_for_status()

                # If not SSE, return as plain JSON
                content_type = response.headers.get("content-type", "")
                if "text/event-stream" not in content_type:
                    # Non-streaming response (e.g., error)
                    return response.json()

                final: dict | None = None

                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue

                    data_str = line[6:]  # Strip "data: " prefix
                    try:
                        event_dict = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    event_type = event_dict.get("type", "")

                    if event_type in ("chat_response", "error"):
                        final = event_dict
                    elif event_type == "event":
                        if on_event is not None:
                            await on_event(event_dict)
                    # Ignore other types

                if final is not None:
                    return final

                # If we never got a final frame, return an error
                return {
                    "type": "error",
                    "payload": {"message": "Stream ended without final response"},
                }

        except httpx.HTTPStatusError as e:
            return {
                "type": "error",
                "payload": {"message": f"HTTP {e.response.status_code}"},
            }
        except httpx.RequestError as e:
            return {
                "type": "error",
                "payload": {"message": str(e)},
            }

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
        self._client = None

    def _auth_headers(self) -> dict[str, str]:
        """Return Authorization header if API key is set."""
        if self._api_key:
            return {"Authorization": f"Bearer {self._api_key}"}
        return {}