# LingYa

An autonomous AI agent with persistent memory and an evolving personality — she remembers you, writes private diary entries, and her internal state drifts with every conversation.

## Quick Start

```bash
git clone <repo> && cd LingYa

cp .env.example .env            # Fill in your DEEPSEEK_API_KEY
cp agent_config.example.yaml agent_config.yaml

uv sync
uv run python main.py
```

Supports any OpenAI-compatible API — edit `config.yaml` to switch providers.

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

## Requirements

- Python >= 3.12
- [uv](https://docs.astral.sh/uv/) for dependency management
