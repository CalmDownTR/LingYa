# LingYa

An autonomous AI agent with persistent memory and an evolving personality — she remembers you, writes private diary entries, and her internal state drifts with every conversation.

## Quick Start

```bash
git clone <repo> && cd LingYa

# Configure
cp .env.example .env            # Fill in DEEPSEEK_API_KEY + LINGYA_API_KEY
cp agent_config.example.yaml agent_config.yaml

# Build Web UI (one-time)
cd web && npm install && npm run build && cd ..

# Install dependencies
uv sync

# Start the daemon
uv run python main.py
```

Then open <http://localhost:8765> in your browser.

Supports any OpenAI-compatible API via [LiteLLM](https://docs.litellm.ai/) — edit `config.yaml` to switch providers (DeepSeek, OpenAI, Ollama, 100+ supported).

## Operations

| Command | Description |
|---------|-------------|
| `python main.py` | Start daemon in foreground (Ctrl+C to stop) |
| `python main.py --stop` | Gracefully stop a running daemon |
| `python main.py --status` | Show daemon running status |
| `python main.py --diary` | Show the latest diary entry |
| `lingya start` | Start via console entry point |
| `lingya stop` | Stop via console entry point |
| `lingya status` | Status via console entry point |

## Web UI

The Web UI is the primary interaction interface — chat with LingYa, adjust her personality (OCEAN traits, tone presets), manage sessions, and browse her diary (upcoming v0.10), all through the browser.

- **Chat**: SSE streaming with markdown rendering, phase indicators (recalling → thinking → generating), memory recall counts
- **Settings**: OCEAN five-dimension sliders, identity editor, 5 tone presets (warm / neutral / cool / passionate / gentle)
- **Sessions**: Create, switch, list, delete conversation sessions

## Configuration

| File | Purpose |
|------|---------|
| `config.yaml` | Runtime settings (LLM provider/model, db_path, memory_path, otel) |
| `agent_config.yaml` | Mind config (identity, OCEAN baseline, tone matrix, guardrails) |
| `.env` | Secrets: `DEEPSEEK_API_KEY`, `LINGYA_API_KEY` |

### LiteLLM Model Configuration

Edit `config.yaml` to use any LiteLLM-supported provider:

```yaml
llm:
  model: "deepseek/deepseek-chat"    # or: openai/gpt-4o, ollama/llama3, ...
  temperature: 0.7
  max_tokens: 4096
  max_input_tokens: 64000
```

## Docker

```bash
docker build -t lingya .
docker run -p 8765:8765 --env-file .env lingya
```

## Requirements

- Python ≥ 3.12
- [uv](https://docs.astral.sh/uv/) for dependency management
- Node.js ≥ 18 (for Web UI build only)
