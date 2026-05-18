#!/usr/bin/env python3
import asyncio
import sys
from pathlib import Path

# Load .env before anything else
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))

from lingya.config import load_config
from lingya.agent import LingYaAgent
from lingya.cli import LingYaCLI


async def main() -> None:
    config = load_config()
    agent = LingYaAgent(config)
    await agent.initialize()

    cli = LingYaCLI(agent)
    try:
        await cli.run()
    except (KeyboardInterrupt, EOFError):
        print("\nGoodbye.")
    finally:
        await agent.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
