#!/usr/bin/env python3
"""Run standardized conversation script against a MindEngine config.

Usage:
    uv run python tests/run_conversation.py tests/fixtures/agent_config_low_a.yaml -o output_low_a.json
    uv run python tests/run_conversation.py tests/fixtures/agent_config_high_a.yaml -o output_high_a.json
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from lingya.llm import LiteLLMModel

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

from lingya.config import load_config as load_app_config  # noqa: E402
from lingya.mind import MindEngine, build_static_prompt, load_mind_config  # noqa: E402


class MockMemoryStore:
    """Minimal in-memory mock — no ChromaDB dependency needed for eval."""

    def store_with_importance(self, text: str, importance: float = 5.0) -> str:
        return f"mock_mem_{id(text)}"

    async def score_importance(self, text: str, llm_call) -> float:
        return 5.0

    def update_importance(self, entry_id: str, importance: float) -> None:
        pass


async def run_conversation(config_path: str, script_path: str) -> dict[str, Any]:
    config = load_mind_config(config_path)
    app_config = load_app_config()

    model = LiteLLMModel(
        model=app_config.llm.model,
        temperature=0.7,
        max_tokens=app_config.llm.max_tokens,
    )

    async def llm_call(prompt: str) -> str:
        response = await model.ainvoke(prompt)
        return response.content if hasattr(response, "content") else str(response)

    engine = MindEngine(
        config=config,
        memory_store=MockMemoryStore(),
        llm_call=llm_call,
        embedding_fn=None,
    )

    with open(script_path) as f:
        script = json.load(f)

    system_prompt = build_static_prompt(config)
    history: list = []
    results: list[dict[str, Any]] = []

    for rnd in script["rounds"]:
        user_msg = rnd["user_message"]
        history.append(HumanMessage(content=user_msg))

        # Include dynamic tone fragment in system prompt to reflect current mind state
        dynamic_prompt = system_prompt
        tone_fragment = engine.get_prompt_fragment()
        if tone_fragment:
            dynamic_prompt = f"{system_prompt}\n\n{tone_fragment}"

        messages: list = [SystemMessage(content=dynamic_prompt)] + history
        response = await model.ainvoke(messages)
        reply = response.content if hasattr(response, "content") else str(response)
        history.append(AIMessage(content=reply))

        # Match production event structure: OCC fields for occ_classify
        await engine.process_event({
            "event_type": "outcome",
            "valence": "neutral",
            "focus": "self",
            "description": user_msg,
            "content": user_msg,
        })

        tone = engine.get_tone_params()
        latest = engine.state.recent_emotions[-1] if engine.state.recent_emotions else {}

        results.append({
            "turn": rnd["turn"],
            "scenario": rnd.get("scenario", ""),
            "user_message": user_msg,
            "lingya_response": reply,
            "pad": {
                "pleasure": round(engine.state.current_pad.pleasure, 4),
                "arousal": round(engine.state.current_pad.arousal, 4),
                "dominance": round(engine.state.current_pad.dominance, 4),
            },
            "tone": tone,
            "ipc_state": engine.state.ipc_state,
            "occ_emotion": latest.get("emotion", "neutral"),
            "turn_counter": engine.state.turn_counter,
        })

        # Let fire-and-forget background tasks settle
        await asyncio.sleep(0.3)

    return {
        "config": Path(config_path).stem,
        "ocean": {
            "agreeableness": config.ocean.agreeableness,
            "extraversion": config.ocean.extraversion,
            "conscientiousness": config.ocean.conscientiousness,
            "neuroticism": config.ocean.neuroticism,
            "openness": config.ocean.openness,
        },
        "rounds": results,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run a conversation script against a MindEngine config")
    parser.add_argument("config", help="Path to agent_config YAML")
    parser.add_argument("--script", default="tests/fixtures/conversation_script.json", help="Conversation script JSON")
    parser.add_argument("--output", "-o", help="Output JSON file path")
    args = parser.parse_args()

    result = asyncio.run(run_conversation(args.config, args.script))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print(f"Saved to {args.output}")
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))

    tones = [r["tone"] for r in result["rounds"]]
    avg_warmth = sum(t["warmth"] for t in tones) / len(tones)
    avg_formality = sum(t["formality"] for t in tones) / len(tones)
    avg_humor = sum(t["humor"] for t in tones) / len(tones)
    print(f"\nAvg tone: warmth={avg_warmth:.1f} formality={avg_formality:.1f} humor={avg_humor:.2f}")


if __name__ == "__main__":
    main()
