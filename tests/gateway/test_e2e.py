"""End-to-end tests — real daemon, real LLM, real HTTP, no mocks.

10-round multi-turn conversation tests exercise the full pipeline:
daemon → HTTP → SSE → agent → LLM → memory → mind engine → response.

Requires DEEPSEEK_API_KEY in environment.
Marked @pytest.mark.slow + @pytest.mark.e2e for optional CI filtering.
"""

from __future__ import annotations

import asyncio
import json
import os

import httpx
import pytest


# ── SSE helpers ──────────────────────────────────────────────────────────


def _parse_sse_frames(body: str) -> list[dict]:
    """Parse SSE text/event-stream body into list of JSON event dicts."""
    frames: list[dict] = []
    for line in body.split("\n"):
        if line.startswith("data: "):
            data_str = line[len("data: "):]
            try:
                frames.append(json.loads(data_str))
            except json.JSONDecodeError:
                pass
    return frames


def _extract_chat_text(body: str) -> str:
    """Extract the final chat_response text from an SSE stream body."""
    for frame in _parse_sse_frames(body):
        if frame.get("type") == "chat_response":
            return frame.get("payload", {}).get("text", "")
    return ""


# ── 10-Round E2E ────────────────────────────────────────────────────────


@pytest.mark.slow
@pytest.mark.e2e
class Test10RoundConversation:
    """10-round multi-turn conversation — full pipeline with real LLM.

    Covers: context retention, code generation, refinement,
    long-range recall, and summarization.
    """

    @pytest.mark.asyncio
    async def test_ten_round_context_retention(self, tmp_path, monkeypatch):
        """10回合对话 — 验证多轮上下文保持 + 代码生成 + 总结。

        Conversation flow:
        R1  建立上下文（名字 + 职业）
        R2  短距记忆：名字职业是否被记住
        R3  知识解释：观察者模式
        R4  代码生成：Python 观察者模式实现
        R5  代码迭代：取消订阅功能
        R6  推荐：三本程序员必读书
        R7  中距回忆：推荐的第二本书是什么
        R8  长距回忆：我之前说我用什么语言
        R9  总结：概括全部话题
        R10 收尾
        """
        if not os.environ.get("DEEPSEEK_API_KEY"):
            pytest.skip("DEEPSEEK_API_KEY not set — skipping E2E test")

        from lingya.config import Config
        from lingya.gateway.daemon import GatewayDaemon
        from lingya.mind.config import (
            BigFiveTraits,
            IdentityAnchor,
            MindConfig,
            PersonaMeta,
            ToneMatrix,
        )

        TEST_PORT = 18770
        db_path = str(tmp_path / "lingya.db")
        memory_path = str(tmp_path / "memory")
        data_dir = str(tmp_path / "data")
        pid_file = str(tmp_path / "lingya.pid")
        (tmp_path / "data").mkdir(exist_ok=True)

        config = Config(
            db_path=db_path,
            memory_path=memory_path,
            data_dir=data_dir,
        )
        mind_config = MindConfig(
            version="1.0",
            meta=PersonaMeta(agent_id="test-e2e", created_at="2025-01-01"),
            identity=IdentityAnchor(
                identity="你是 LingYa，一个友好、知识渊博的 AI 助手。请用中文回答问题，保持自然亲切的语气。",
                core_belief="帮助用户学习和成长是最重要的事。",
            ),
            ocean=BigFiveTraits(),
            tone_matrix=ToneMatrix(),
            behavior_guardrails=["保持诚实，不确定时坦诚说明。"],
        )

        daemon = GatewayDaemon(
            config=config,
            mind_config=mind_config,
            pid_file=pid_file,
            port=TEST_PORT,
        )

        # Start daemon — NO mocks, real LLM
        daemon_task = asyncio.create_task(daemon.start())

        try:
            # Wait for PID file (readiness signal); model init may take a moment
            for _ in range(200):  # 10 s timeout
                if os.path.exists(pid_file):
                    break
                await asyncio.sleep(0.05)
            else:
                pytest.fail("Daemon did not write PID file within 10 seconds")

            base_url = f"http://localhost:{TEST_PORT}"

            async with httpx.AsyncClient(base_url=base_url, timeout=httpx.Timeout(120.0)) as http:
                # Health check
                resp = await http.get("/health")
                assert resp.status_code == 200
                assert resp.json() == {"status": "ok"}

                # Create session
                resp = await http.post("/session", json={"action": "new"})
                assert resp.status_code == 200
                thread_id = resp.json()["payload"]["thread_id"]
                assert thread_id.startswith("ws-")

                # ── 10-round conversation ──────────────────────────────
                conversation = [
                    # R1: Establish context
                    "你好，我叫小明，今天想和你聊聊天。我是一名有5年经验的Python后端程序员。",
                    # R2: Short-range memory
                    "你还记得我的名字和职业吗？请直接说出来。",
                    # R3: Knowledge / explanation
                    "我最近在学设计模式，能用简单的例子解释一下观察者模式（Observer Pattern）吗？",
                    # R4: Code generation
                    "能给我写一个Python的观察者模式实现吗？要包含Subject和Observer类，以及一个简单的使用示例。",
                    # R5: Code refinement / iteration
                    "不错。如果要加一个功能——让观察者可以取消订阅（unsubscribe），你会怎么改？请给出修改后的代码。",
                    # R6: Recommendation
                    "换个轻松点的话题。如果让你推荐三本程序员必读的经典书籍，你会推荐什么？请列出书名并简要说明理由。",
                    # R7: Mid-range context recall
                    "你刚才推荐的第二本是什么书？为什么推荐它？",
                    # R8: Long-range context recall
                    "我之前说过我最喜欢什么编程语言？是什么时候说的？",
                    # R9: Summarization
                    "总结一下我们这次聊了哪些话题，用三到四句话概括。",
                    # R10: Graceful ending
                    "谢谢，今天就到这里。期待下次再聊。",
                ]

                responses: list[str] = []

                for i, msg in enumerate(conversation):
                    resp = await http.post("/chat", json={"text": msg})
                    assert resp.status_code == 200, (
                        f"Round {i + 1} failed: HTTP {resp.status_code}"
                    )

                    text = _extract_chat_text(resp.text)
                    assert text, f"Round {i + 1}: empty response from LLM"
                    responses.append(text)

                # ── Assertions ──────────────────────────────────────────

                # R2: Should remember name "小明" and profession
                r2 = responses[1]
                assert "小明" in r2, (
                    f"R2 should recall name '小明': {r2[:300]}"
                )

                # R4: Should contain Python class definitions
                r4 = responses[3]
                assert ("class" in r4 and ("Subject" in r4 or "Observer" in r4)), (
                    f"R4 should contain Observer pattern classes: {r4[:300]}"
                )

                # R5: Should mention unsubscribe
                r5 = responses[4]
                assert ("unsubscribe" in r5.lower() or "取消订阅" in r5 or "remove" in r5.lower()), (
                    f"R5 should add unsubscribe capability: {r5[:300]}"
                )

                # R6: Should list 3 books
                r6 = responses[5]
                assert len(r6) > 50, (
                    f"R6 should contain 3 book recommendations: {r6[:300]}"
                )

                # R8: Should mention Python
                r8 = responses[7]
                assert "Python" in r8, (
                    f"R8 should recall Python: {r8[:300]}"
                )

                # R9: Summary should be substantive (at least covers multiple topics)
                r9 = responses[8]
                assert len(r9) > 40, (
                    f"R9 summary should be substantive: {r9[:300]}"
                )

                # R10: Graceful ending, non-empty
                r10 = responses[9]
                assert len(r10) > 0, "R10 should have a graceful ending"

                # ── Session history integrity ──────────────────────────
                resp = await http.get(
                    "/session/history", params={"thread_id": thread_id}
                )
                assert resp.status_code == 200
                history = resp.json()
                msgs = history["payload"]["messages"]
                # At least 20 messages: 10 user + 10 assistant
                assert len(msgs) >= 20, (
                    f"Expected >= 20 messages in history, got {len(msgs)}"
                )

                # Verify alternating roles
                roles = [m["role"] for m in msgs]
                user_count = roles.count("user")
                her_count = roles.count("her")
                assert user_count >= 10, f"Expected >= 10 user messages, got {user_count}"
                assert her_count >= 10, f"Expected >= 10 assistant messages, got {her_count}"

        finally:
            # Graceful shutdown
            daemon._shutdown_event.set()
            try:
                await asyncio.wait_for(daemon_task, timeout=15.0)
            except asyncio.TimeoutError:
                daemon_task.cancel()
                try:
                    await daemon_task
                except asyncio.CancelledError:
                    pass
            try:
                await daemon.shutdown()
            except Exception:
                pass

        # PID file must be cleaned up
        assert not os.path.exists(pid_file), "PID file should be removed on shutdown"
