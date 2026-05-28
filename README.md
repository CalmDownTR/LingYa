# LingYa

An autonomous AI agent with persistent memory and an evolving personality. As LingYa consumes more information through conversations, it builds up useful context and its internal state (OCEAN traits, PAD mood, IPC stance) drifts over time.

## Quick Start

```bash
git clone <repo> && cd LingYa

# Set up secrets
cp .env.example .env
# Edit .env and fill in your DEEPSEEK_API_KEY

# Set up mind config
cp agent_config.example.yaml agent_config.yaml
# Edit agent_config.yaml to customize your agent's personality

# Install dependencies
uv sync

# Run
uv run python main.py
```

Supports any OpenAI-compatible API — edit `config.yaml` to switch between DeepSeek, OpenAI, or Ollama.

## Commands

| Command | Description |
|---------|-------------|
| `/sessions` | List all sessions |
| `/new` | Start a new session |
| `/switch <id>` | Switch to a session by ID |
| `/memories` | List all stored memories |
| `/forget <index>` | Delete a memory by its index |
| `/remember <text>` | Manually add a memory |
| `/diary` | Read the latest diary entry |
| `/diary list` | List all diary entries |
| `/diary <n>` | Read the Nth diary entry |
| `/help` | Show available commands |
| `/exit` | Quit |

## Architecture

```
User Input → CLI (Rich)
    ↓
Agent (deepagents + LangGraph)
    ├── MindEngine     (dynamic personality: OCC → PAD → tone → OCEAN drift)
    ├── Memory         (ChromaDB vector store, importance-weighted retrieval)
    ├── LLM Backend    (OpenAI-compatible, ChatOpenAI)
    ├── SQLite         (conversations, turns, mind_state persistence)
    └── Diary          (Markdown diaries in LingYa's own voice)
    ↓
Rich Markdown Response
```

### Key Modules

- **Mind Engine** — Dynamic personality with OCC 22-emotion classification, PAD (Pleasure-Arousal-Dominance) mood evolution, IPC (Interpersonal Circumplex) stance, and long-term OCEAN trait drift. Pure computation layer with zero framework dependency.
- **Memory** — ChromaDB vector store with importance scoring (rule-based pre-score + LLM refinement in background), weighted retrieval (recency × importance × similarity), and reflection tree for self-notion extraction.
- **CLI** — Rich-based terminal UI with session management, memory management, and diary browsing.
- **Diary** — LingYa writes private diary entries in her own voice, reflecting on conversations rather than summarizing them.
- **LLM Abstraction** — Single backend serves DeepSeek, OpenAI, and Ollama through the same interface.

## Configuration

- `config.yaml` — LLM, memory, and data path settings (committed, safe to share)
- `agent_config.yaml` — Mind config: identity, OCEAN personality traits, tone matrix, behavior guardrails
- `.env` — Secrets like API keys (never committed, use `.env.example` as template)

## Roadmap

- **v0.1.0** — CLI chat, memory system, web ingestion
- **v0.2.0** — Semantic memory (store + search), agent tool integration, memory management CLI
- **v0.3.0** (current) — Dynamic personality engine (OCC + PAD + IPC + OCEAN drift), diary, reflection
- **v0.4.0** — Proactive interaction, belief anchoring, identity guard
- **v1.0.0** — Quality closed-loop, documentation, community-ready

## Requirements

- Python >= 3.12
- [uv](https://docs.astral.sh/uv/) for dependency management
