# Architecture Spec (2026-05-24)

## File Tree

```
main.py                    # Entry point: loads .env, wires agent + CLI
config.yaml                # Runtime config (safe to commit)

lingya/
├── cli.py                 # LingYaCLI — Rich interactive terminal loop
├── config.py              # Pydantic models: Config, LLMConfig
├── embedder.py            # SentenceTransformer wrapper, LRU-cached, async
├── memory/
│   ├── long_term.py       # ChromaDB PersistentClient, BGE embeddings, cosine search
│   └── tools.py           # search_memory / save_memory langchain tools
├── ingestion/
│   └── chunker.py         # Recursive token-based text splitter (tiktoken cl100k_base)
└── storage/
    ├── db.py              # SQLite via aiosqlite, 2 tables, CRUD methods
    └── migrations.py      # 2 ordered SQL migration statements
```

## Dependency Graph

```
main.py → config, cli, memory/long_term, memory/tools, storage/db

cli.py → storage/db

memory/tools.py → memory/long_term, ingestion/chunker
memory/long_term.py → embedder

storage/db.py → storage/migrations
```

No circular dependencies.

## Agent Architecture

LingYa uses **deepagents** (`create_deep_agent`) as its agent harness, built on LangGraph. The agent is assembled in `main.py`:

```
create_deep_agent()
  ├── model: ChatOpenAI (DeepSeek API)
  ├── tools: memory tools (ChromaDB) + MCP tools (optional)
  ├── middleware: [SummarizationToolMiddleware]
  ├── system_prompt: base agent prompt
  └── backend: StateBackend (shared with summarization middleware)
```

### Request Lifecycle (deepagents)

```
1. CLI calls agent.ainvoke({"messages": [user_msg]}, config)
2. LangGraph checkpoint loads conversation state by thread_id
3. Middleware pipeline executes:
   a. TodoListMiddleware — task planning
   b. FilesystemMiddleware — virtual filesystem tools (ls, read, write, edit)
   c. SubAgentMiddleware — sub-agent spawning
   d. SummarizationMiddleware — auto context compression at 85% token threshold
   e. SummarizationToolMiddleware — optional `compact_conversation` tool for manual summarization
4. LLM called with tools available
5. Tool calls executed (memory tools, filesystem, MCP tools)
6. Response extracted, displayed
7. State checkpointed by LangGraph
```

## Long-Term Memory

ChromaDB with BGE embeddings, exposed as two langchain tools:
- `search_memory(query)` — semantic search across stored memories
- `save_memory(content, source)` — chunk and store new memories

Independent from deepagents' LangGraph Memory Store. They serve different purposes:
- ChromaDB: semantic search over ingested content and saved facts
- LangGraph Store: cross-session agent state (future use)

## Known Gaps

| Gap | Detail |
|-----|--------|
| MCP tools | Config-driven MCP server connection not yet wired (TODO in main.py) |
| Persistent checkpointing | ✅ Resolved — SqliteSaver checkpointer added (a62efaf) |
