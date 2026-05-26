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

from lingya.config import Config, load_config
from lingya.cli import LingYaCLI
from lingya.memory.long_term import LongTermMemory
from lingya.memory.tools import create_memory_tools
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
    # Tell deepagents the actual context window so summarization triggers
    # at 85% (on-time) instead of falling back to a fixed 170k default.
    model.profile = {"max_input_tokens": config.llm.max_input_tokens}

    # ── Database ──
    db = Database(config.db_path)
    await db.initialize()

    # ── Checkpointer (persists conversation state across restarts) ──
    async with AsyncSqliteSaver.from_conn_string(config.db_path) as checkpointer:
        await checkpointer.setup()

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

        # ── Persona ──
        from lingya.persona import PromptAssembler, load_persona_config

        persona_config = load_persona_config(config.persona_config_path)
        assembler = PromptAssembler(persona_config)
        system_prompt = assembler.assemble()

        # ── Agent ──
        backend = StateBackend()

        agent = create_deep_agent(
            model=model,
            tools=[*memory_tools, *mcp_tools],
            middleware=[
                create_summarization_tool_middleware(model, backend=backend),
            ],
            system_prompt=system_prompt,
            backend=backend,
            checkpointer=checkpointer,
        )

        # ── CLI ──
        cli = LingYaCLI(agent, db)
        try:
            await cli.run()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
        finally:
            await db.close()


if __name__ == "__main__":
    asyncio.run(main())
