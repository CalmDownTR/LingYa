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
    else:
        asyncio.run(main())
