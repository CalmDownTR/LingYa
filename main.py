#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import os
import sqlite3
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import SecretStr

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))

from deepagents import create_deep_agent
from deepagents.backends import StateBackend

from lingya.config import Config, load_config
from lingya.cli import LingYaCLI
from lingya.memory.long_term import LongTermMemory
from lingya.memory.tools import create_memory_tools
from lingya.middleware import PersonalityMiddleware
from lingya.personality.engine import PersonalityEngine
from lingya.storage.db import Database


async def main() -> None:
    config = load_config()

    # ── Model ──
    model = ChatOpenAI(
        model=config.llm.model,
        api_key=SecretStr(os.environ[config.llm.api_key_env]),
        base_url=config.llm.api_base_url,
        temperature=config.llm.temperature,
    )

    # ── Personality (LingYa's unique module, kept as-is) ──
    db = Database(config.db_path)
    await db.initialize()
    personality_engine = PersonalityEngine(config.personality, db, llm=model)
    await personality_engine.load()

    # ── Checkpointer (persists conversation state across restarts) ──
    checkpoint_conn = sqlite3.connect(config.db_path, check_same_thread=False)
    checkpointer = SqliteSaver(checkpoint_conn)
    checkpointer.setup()

    personality = PersonalityMiddleware(personality_engine)

    # ── Memory tools (ChromaDB) ──
    long_term = LongTermMemory(
        persist_dir=config.chroma_persist_dir,
        embedding_model_name=config.embedding_model,
    )
    memory_tools = create_memory_tools(long_term)

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

    # ── Agent ──
    base_prompt = (
        "You are LingYa, an AI companion with personality and memory.\n"
        "You have a long-term memory system (search_memory, save_memory) and "
        "a virtual filesystem (ls, read_file, write_file, edit_file) for "
        "managing context. Use them when appropriate.\n"
        "If you can answer directly without tools, just respond naturally."
    )

    agent = create_deep_agent(
        model=model,
        tools=[*memory_tools, *mcp_tools],
        middleware=[personality],
        system_prompt=base_prompt,
        backend=StateBackend(),
        checkpointer=checkpointer,
    )

    # ── CLI ──
    cli = LingYaCLI(agent, personality_engine, db)
    try:
        await cli.run()
    except (KeyboardInterrupt, EOFError):
        print("\nGoodbye.")
    finally:
        checkpoint_conn.close()
        await db.close()


if __name__ == "__main__":
    asyncio.run(main())
