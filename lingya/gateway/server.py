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
from pydantic import BaseModel, Field

from lingya.gateway.auth import create_auth_dependency

try:
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
except ImportError:  # pragma: no cover — traceloop-sdk provides this
    FastAPIInstrumentor = None  # type: ignore[assignment,misc]


# ── Request models ───────────────────────────────────────────────────


class ChatRequest(BaseModel):
    text: str


class SessionRequest(BaseModel):
    action: str = "new"
    thread_id: str | None = None


class OceanUpdateRequest(BaseModel):
    O: float = Field(ge=0.0, le=1.0)
    C: float = Field(ge=0.0, le=1.0)
    E: float = Field(ge=0.0, le=1.0)
    A: float = Field(ge=0.0, le=1.0)
    N: float = Field(ge=0.0, le=1.0)


class IdentityUpdateRequest(BaseModel):
    identity: str | None = None
    core_belief: str | None = None


class ToneUpdateRequest(BaseModel):
    preset: str


# ── App factory ───────────────────────────────────────────────────────


def create_app(
    router: Any = None,
    auth_enabled: bool = True,
    title: str = "LingYa Gateway",
    version: str = "0.9.0",
    shutdown_callback: Any = None,
    session_service: Any = None,
    settings_service: Any = None,
    chat_handler: Any = None,
) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        router: MessageRouter instance (injected by daemon).
        auth_enabled: Whether to require Bearer token auth.
        title: OpenAPI doc title.
        version: OpenAPI doc version.
        shutdown_callback: Optional callable invoked by POST /shutdown
            to trigger graceful daemon shutdown.
        session_service: SessionService (v0.9.5 — replaces router._handle_session).
        settings_service: SettingsService (v0.9.5 — replaces router._handle_settings).
        chat_handler: ChatHandler (v0.9.5 — replaces router._handle_chat*).
    """
    app = FastAPI(title=title, version=version)
    auth = create_auth_dependency(auth_enabled=auth_enabled)

    # Auto-instrument HTTP requests with OTel spans (no-op if traceloop not installed).
    # FastAPIInstrumentor is provided by opentelemetry-instrumentation-fastapi
    # which is a transitive dependency of traceloop-sdk.
    if FastAPIInstrumentor is not None:
        FastAPIInstrumentor().instrument_app(app)

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

        # Backward compat: use chat_handler if provided, else fall back to
        # router delegation methods (tests inject mocks on router directly).
        _chat = chat_handler if chat_handler is not None else router
        _session = session_service if session_service is not None else router

        if _chat is None or _chat._agent is None:
            return JSONResponse(
                {"type": "error", "payload": {"message": "Agent not initialized"}},
                status_code=503,
            )

        # Build messages
        fragment = router._engine.get_prompt_fragment()
        messages: list = [HumanMessage(content=body.text)]
        if fragment:
            messages.insert(0, SystemMessage(content=fragment))
        # SessionService uses .thread_id, legacy router uses ._thread_id
        tid = _session.thread_id if session_service is not None else _session._thread_id
        config = {"configurable": {"thread_id": tid}}

        async def sse_generator():
            """Iterate the streaming generator and emit SSE frames."""
            t_start = time.monotonic()
            # ChatHandler uses _chat_streaming, legacy router uses _handle_chat_streaming
            if chat_handler is not None:
                streamer = chat_handler._chat_streaming
            else:
                streamer = router._handle_chat_streaming
            async for event_dict in streamer(
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
        query: str = Query("state", description="'state', 'tone', or 'health'"),
        _auth: bool = auth,
    ):
        if router is None:
            return JSONResponse(
                {"type": "error", "payload": {"message": "Router not initialized"}},
                status_code=503,
            )
        return await router._handle_mind({"query": query})

    @app.get("/mind/health")
    async def get_mind_health(_auth: bool = auth):
        """Return mind engine health metrics (v0.9.8).

        Includes importance scoring success rate, pre-score vs LLM-score
        averages, and recent failure reasons.
        """
        if router is None:
            return JSONResponse(
                {"type": "error", "payload": {"message": "Router not initialized"}},
                status_code=503,
            )
        return await router._handle_mind({"query": "health"})

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
    # Backward compat: use session_service if provided, else fall back
    # to router._handle_session (tests inject mocks on router directly).

    @app.post("/session")
    async def post_session(
        body: SessionRequest | None = None,
        action: str = Query("new"),
        _auth: bool = auth,
    ):
        if router is None and session_service is None:
            return JSONResponse(
                {"type": "error", "payload": {"message": "Session service not initialized"}},
                status_code=503,
            )
        effective_action = body.action if body is not None else action
        effective_thread_id = body.thread_id if body is not None else None

        payload: dict = {"action": effective_action}
        if effective_thread_id:
            payload["thread_id"] = effective_thread_id
        if session_service is not None:
            return await session_service.handle_session(payload)
        return await router._handle_session(payload)

    @app.get("/session/list")
    async def list_sessions(_auth: bool = auth):
        if router is None and session_service is None:
            return JSONResponse(
                {"type": "error", "payload": {"message": "Session service not initialized"}},
                status_code=503,
            )
        if session_service is not None:
            return await session_service.handle_session({"action": "list"})
        return await router._handle_session({"action": "list"})

    @app.get("/session/current")
    async def current_session(_auth: bool = auth):
        if router is None and session_service is None:
            return JSONResponse(
                {"type": "error", "payload": {"message": "Session service not initialized"}},
                status_code=503,
            )
        if session_service is not None:
            return await session_service.handle_session({"action": "current"})
        return await router._handle_session({"action": "current"})

    @app.get("/session/history")
    async def session_history(
        thread_id: str | None = Query(None, description="Thread ID (defaults to current)"),
        _auth: bool = auth,
    ):
        if router is None and session_service is None:
            return JSONResponse(
                {"type": "error", "payload": {"message": "Session service not initialized"}},
                status_code=503,
            )
        payload: dict = {"action": "history"}
        if thread_id:
            payload["thread_id"] = thread_id
        if session_service is not None:
            return await session_service.handle_session(payload)
        return await router._handle_session(payload)

    # ── Settings ────────────────────────────────────────────────────
    # Backward compat: use settings_service if provided, else fall back
    # to router._handle_settings (tests inject mocks on router directly).

    @app.get("/settings")
    async def get_settings(_auth: bool = auth):
        if router is None and settings_service is None:
            return JSONResponse(
                {"type": "error", "payload": {"message": "Settings service not initialized"}},
                status_code=503,
            )
        if settings_service is not None:
            return await settings_service.handle_settings({"action": "get"})
        return await router._handle_settings({"action": "get"})

    @app.put("/settings/ocean")
    async def update_ocean(body: OceanUpdateRequest, _auth: bool = auth):
        if router is None and settings_service is None:
            return JSONResponse(
                {"type": "error", "payload": {"message": "Settings service not initialized"}},
                status_code=503,
            )
        payload = {"action": "update_ocean", "ocean": body.model_dump()}
        if settings_service is not None:
            return await settings_service.handle_settings(payload)
        return await router._handle_settings(payload)

    @app.put("/settings/identity")
    async def update_identity(body: IdentityUpdateRequest, _auth: bool = auth):
        if router is None and settings_service is None:
            return JSONResponse(
                {"type": "error", "payload": {"message": "Settings service not initialized"}},
                status_code=503,
            )
        identity_data = body.model_dump(exclude_none=True)
        if not identity_data:
            return JSONResponse(
                {"type": "error", "payload": {"message": "No fields to update"}},
                status_code=400,
            )
        payload = {"action": "update_identity", "identity": identity_data}
        if settings_service is not None:
            return await settings_service.handle_settings(payload)
        return await router._handle_settings(payload)

    @app.put("/settings/tone")
    async def update_tone(body: ToneUpdateRequest, _auth: bool = auth):
        if router is None and settings_service is None:
            return JSONResponse(
                {"type": "error", "payload": {"message": "Settings service not initialized"}},
                status_code=503,
            )
        payload = {"action": "update_tone", "preset": body.preset}
        if settings_service is not None:
            return await settings_service.handle_settings(payload)
        return await router._handle_settings(payload)

    @app.post("/settings/reset")
    async def reset_settings(_auth: bool = auth):
        if router is None and settings_service is None:
            return JSONResponse(
                {"type": "error", "payload": {"message": "Settings service not initialized"}},
                status_code=503,
            )
        if settings_service is not None:
            return await settings_service.handle_settings({"action": "reset"})
        return await router._handle_settings({"action": "reset"})

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

    # ── Static Web UI (SPA fallback) ─────────────────────────────────
    # Mount MUST be after all API routes. FastAPI matches explicit routes
    # first, then falls through to StaticFiles. html=True enables SPA
    # fallback — unknown paths like /settings serve index.html.
    from pathlib import Path

    from fastapi.staticfiles import StaticFiles

    web_dist = Path("web/dist")
    if web_dist.exists() and web_dist.is_dir():
        app.mount("/", StaticFiles(directory=str(web_dist), html=True), name="web")

    return app