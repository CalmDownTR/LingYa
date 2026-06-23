"""GatewayServer — FastAPI HTTP + SSE server for LingYa Gateway.

Replaces the v0.8 WebSocket server with industry-standard HTTP + SSE:

- ``POST /chat`` — SSE stream (process events + chat_response)
- ``GET /mind`` / ``GET /memory`` / ``GET /diary`` — JSON REST
- ``POST /session`` — session management
- ``GET /health`` — health check
- ``GET /docs`` — auto-generated OpenAPI docs

Uses FastAPI (Starlette) + uvicorn. No more hand-rolled RFC 6455.
"""

from __future__ import annotations

import json
import time
from typing import Any

from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from lingya.gateway.auth import create_auth_dependency


# ── Request models ───────────────────────────────────────────────────


class ChatRequest(BaseModel):
    text: str


# ── App factory ───────────────────────────────────────────────────────


def create_app(
    router: Any = None,
    auth_enabled: bool = True,
    title: str = "LingYa Gateway",
    version: str = "0.8.2",
    shutdown_callback: Any = None,
) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        router: MessageRouter instance (injected by daemon).
        auth_enabled: Whether to require Bearer token auth.
        title: OpenAPI doc title.
        version: OpenAPI doc version.
        shutdown_callback: Optional callable invoked by POST /shutdown
            to trigger graceful daemon shutdown.
    """
    app = FastAPI(title=title, version=version)
    auth = create_auth_dependency(auth_enabled=auth_enabled)

    # CORS — allow all origins in dev, configurable later
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Health (no auth) ────────────────────────────────────────────

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    # ── Chat (SSE stream) ──────────────────────────────────────────

    @app.post("/chat")
    async def chat(body: ChatRequest, request: Request, _auth: bool = auth):
        """Process a chat message and stream the response via SSE.

        Returns ``text/event-stream`` with LingYa event frames:

        - ``{"type":"event","event":"process.phase","payload":{...}}``
        - ``{"type":"event","event":"chat.delta","payload":{"content":"..."}}``
        - ``{"type":"event","event":"mind.transition","payload":{...}}``
        - ``{"type":"chat_response","payload":{"text":"...","tone":{...}}}``
        """
        if router is None:
            return JSONResponse(
                {"type": "error", "payload": {"message": "Router not initialized"}},
                status_code=503,
            )

        if not body.text:
            return JSONResponse(
                {"type": "error", "payload": {"message": "Empty message"}},
                status_code=400,
            )

        if router._agent is None:
            return JSONResponse(
                {"type": "error", "payload": {"message": "Agent not initialized"}},
                status_code=503,
            )

        # Build messages (same logic as router._handle_chat)
        fragment = router._engine.get_prompt_fragment()
        messages: list = [HumanMessage(content=body.text)]
        if fragment:
            messages.insert(0, SystemMessage(content=fragment))
        config = {"configurable": {"thread_id": router._thread_id}}

        async def sse_generator():
            """Iterate router's async generator and emit SSE frames."""
            t_start = time.monotonic()
            async for event_dict in router._handle_chat_streaming(
                messages, config, body.text
            ):
                # Inject ws_hop_ms as sse_hop_ms for observability
                if event_dict.get("type") == "chat_response":
                    hop_ms = round((time.monotonic() - t_start) * 1000, 1)
                    event_dict.setdefault("payload", {})["meta"] = {
                        **(event_dict["payload"].get("meta", {})),
                        "sse_hop_ms": hop_ms,
                    }

                # Check for client disconnect
                if await request.is_disconnected():
                    break

                yield f"data: {json.dumps(event_dict, ensure_ascii=False, default=str)}\n\n"

        return StreamingResponse(
            sse_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Disable nginx buffering
            },
        )

    # ── Mind state ─────────────────────────────────────────────────

    @app.get("/mind")
    async def get_mind(
        query: str = Query("state", description="'state' or 'tone'"),
        _auth: bool = auth,
    ):
        if router is None:
            return JSONResponse(
                {"type": "error", "payload": {"message": "Router not initialized"}},
                status_code=503,
            )
        return await router._handle_mind({"query": query})

    # ── Memory ─────────────────────────────────────────────────────

    @app.get("/memory")
    async def get_memory(
        action: str = Query("search"),
        query: str = Query(""),
        id: str = Query(""),
        _auth: bool = auth,
    ):
        if router is None:
            return JSONResponse(
                {"type": "error", "payload": {"message": "Router not initialized"}},
                status_code=503,
            )
        payload: dict = {"action": action}
        if query:
            payload["query"] = query
        if id:
            payload["id"] = id
        return await router._handle_memory(payload)

    # ── Diary ──────────────────────────────────────────────────────

    @app.get("/diary")
    async def get_diary(
        action: str = Query("list"),
        index: int = Query(0, description="0-based diary index (0 = latest)"),
        _auth: bool = auth,
    ):
        if router is None:
            return JSONResponse(
                {"type": "error", "payload": {"message": "Router not initialized"}},
                status_code=503,
            )
        return await router._handle_diary({
            "action": action,
            "index": index,
        })

    # ── Session ────────────────────────────────────────────────────

    @app.post("/session")
    async def post_session(
        action: str = Query("new"),
        _auth: bool = auth,
    ):
        if router is None:
            return JSONResponse(
                {"type": "error", "payload": {"message": "Router not initialized"}},
                status_code=503,
            )
        return await router._handle_session({"action": action})

    # ── Stats (deprecated) ─────────────────────────────────────────

    @app.get("/stats")
    async def get_stats(_auth: bool = auth):
        if router is None:
            return JSONResponse(
                {"type": "error", "payload": {"message": "Router not initialized"}},
                status_code=503,
            )
        return await router._handle_stats({})

    # ── Shutdown ────────────────────────────────────────────────────

    @app.post("/shutdown")
    async def shutdown(_auth: bool = auth):
        """Trigger graceful daemon shutdown. Requires Bearer auth."""
        if shutdown_callback is not None:
            shutdown_callback()
        return {"status": "shutting_down"}

    return app