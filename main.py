#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from pydantic import SecretStr

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))

from deepagents import create_deep_agent
from deepagents.backends import StateBackend
from deepagents.middleware.summarization import create_summarization_tool_middleware

from langchain_core.tools import tool

from lingya.config import load_config
from lingya.cli import LingYaCLI
from lingya.memory import EnhancedMemoryStore
from lingya.storage.db import Database


async def daemon_main() -> None:
    """Run LingYa in daemon mode — long-running process with GatewayDaemon."""
    config = load_config()

    from lingya.mind import load_mind_config

    mind_config = load_mind_config(config.persona_config_path)

    from lingya.gateway import GatewayDaemon

    daemon = GatewayDaemon(config=config, mind_config=mind_config)

    try:
        await daemon.start()
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        await daemon.shutdown()


async def start_main() -> None:
    """lingya start — auto-launch daemon + attach CLI as WebSocket client.

    1. Check if daemon is already running (via PID file)
    2. If not, launch daemon as a subprocess and wait for it to be ready
    3. Connect CLI via WebSocket to the Gateway
    """
    from lingya.gateway.daemon import GatewayDaemon
    from lingya.gateway.client import GatewayClient

    PORT = 8765

    # 1. Check if daemon is already running
    if not GatewayDaemon.is_running():
        import subprocess

        # Launch daemon as a subprocess using the same Python + main.py
        daemon_proc = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--daemon"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Wait for daemon to be ready (poll the PID file)
        for _ in range(50):  # 5 second timeout (50 * 0.1s)
            if GatewayDaemon.is_running():
                break
            await asyncio.sleep(0.1)
        else:
            print("Failed to start daemon after 5 seconds", file=sys.stderr)
            return

    # 2. Connect CLI via WebSocket
    client = GatewayClient(port=PORT)
    try:
        await client.connect()

        cli = LingYaCLI(
            agent=None,     # Not used in WS mode
            db=None,        # Not used in WS mode
            model=None,     # Not used in WS mode
            engine=None,    # Not used in WS mode
            memory=None,    # Not used in WS mode
            ws_client=client,
        )
        try:
            await cli.run_ws()
        except (KeyboardInterrupt, EOFError):
            pass
    except ConnectionError as e:
        print(f"Failed to connect to Gateway: {e}", file=sys.stderr)
    finally:
        await client.close()


async def main() -> None:
    config = load_config()

    # ── Model ──
    model = ChatOpenAI(
        model=config.llm.model,
        api_key=SecretStr(os.environ[config.llm.api_key_env]),
        base_url=config.llm.api_base_url,
        temperature=config.llm.temperature,
    )
    # Tell deepagents the actual context window so summarization triggers
    # at 85% (on-time) instead of falling back to a fixed 170k default.
    model.profile = {"max_input_tokens": config.llm.max_input_tokens}

    # ── Database ──
    db = Database(config.db_path)
    await db.initialize()

    # ── Memory ──
    memory = EnhancedMemoryStore(persist_path=config.memory_path)
    memory.warmup()  # download the embedding model before entering the chat loop

    @tool
    def memory_store(text: str) -> str:
        """Remember important information about the user.

        Use this tool when the user shares personal preferences, identity,
        emotional states, or context useful for future interactions.
        Examples: "I like rainy days", "I'm afraid of loneliness",
        "I'm a freelancer".

        Do NOT use for transient information (e.g. "I'm running late"),
        one-time tasks, small talk, or credentials/API keys.
        """
        return memory.store(text)

    @tool
    def memory_search(query: str) -> str:
        """Search for prior memories about the user.

        Use this tool when the user asks "do you remember...", or when you
        need to recall context from past conversations to answer accurately.
        Returns matching memories with their text content.
        """
        results = memory.search(query)
        if not results:
            return "(No matching memories found)"
        lines = []
        for r in results:
            lines.append(f"[{r['id']}] {r['text']}")
        return "\n".join(lines)

    # ── LLM call wrapper for MindEngine ──
    async def llm_call(prompt: str) -> str:
        """Simple LLM call for MindEngine's cognitive appraisal, IPC, etc."""
        from langchain_core.messages import HumanMessage
        result = await model.ainvoke([HumanMessage(content=prompt)])
        return str(result.content) if hasattr(result, "content") else str(result)

    # ── Checkpointer (persists conversation state across restarts) ──
    async with AsyncSqliteSaver.from_conn_string(config.db_path) as checkpointer:
        await checkpointer.setup()

        # ── MCP tools (optional) ──
        mcp_tools: list = []
        try:
            from langchain_mcp_adapters.client import MultiServerMCPClient
            # If MCP servers are configured, connect and discover tools
            # mcp_client = MultiServerMCPClient({...})  # TODO: config-driven
            # mcp_tools = await mcp_client.get_tools()
            pass
        except Exception:
            pass

        # ── Mind Engine ──
        from lingya.mind import MindEngine, load_mind_config, build_static_prompt

        mind_config = load_mind_config(config.persona_config_path)
        engine = MindEngine(
            config=mind_config,
            memory_store=memory,
            llm_call=llm_call,
        )
        engine.set_db(db)
        await engine.load_state(db)  # Restore from SQLite if available

        system_prompt = build_static_prompt(mind_config)

        # ── Agent ──
        backend = StateBackend()

        agent = create_deep_agent(
            model=model,
            tools=[*mcp_tools, memory_store, memory_search],
            middleware=[
                create_summarization_tool_middleware(model, backend=backend),
            ],
            system_prompt=system_prompt,
            backend=backend,
            checkpointer=checkpointer,
        )

        # ── CLI ──
        cli = LingYaCLI(
            agent, db, model, engine, memory,
            data_dir=config.data_dir,
            diary_period_days=config.diary_period_days,
        )
        try:
            await cli.run()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
        finally:
            await db.close()


if __name__ == "__main__":
    if "--daemon" in sys.argv:
        asyncio.run(daemon_main())
    elif "start" in sys.argv:
        asyncio.run(start_main())
    else:
        asyncio.run(main())
