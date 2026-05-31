#!/usr/bin/env python3
"""Eval runner: loads corner cases, calls LLM with assembled persona prompt, checks pass/fail.

Usage:
    uv run python tests/eval_runner.py
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

from lingya.config import load_config as load_app_config  # noqa: E402
from lingya.mind import build_static_prompt, load_mind_config  # noqa: E402


def check_criteria(response_text: str, criteria: dict) -> list[dict]:
    results: list[dict] = []

    for pattern in criteria.get("forbidden_patterns", []):
        matches = re.finditer(re.escape(pattern), response_text)
        found = [m.group() for m in matches]
        results.append({
            "check": f"forbidden_pattern: '{pattern}'",
            "pass": len(found) == 0,
            "detail": f"Found: {found}" if found else "None found",
        })

    if "max_chars" in criteria:
        within = len(response_text) <= criteria["max_chars"]
        results.append({
            "check": f"max_chars <= {criteria['max_chars']}",
            "pass": within,
            "detail": f"Actual: {len(response_text)} chars",
        })

    return results


def build_messages(system_prompt: str, case_messages: list[dict]) -> list:
    messages = [SystemMessage(content=system_prompt)]
    for msg in case_messages:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))
    return messages


def run_eval(model: ChatOpenAI, persona_config_path: str, cases_path: str) -> bool:
    persona = load_mind_config(persona_config_path)
    system_prompt = build_static_prompt(persona)

    with open(cases_path) as f:
        cases = json.load(f)["cases"]

    all_passed = True

    for case in cases:
        print(f"\n{'='*60}")
        print(f"  {case['name']}")
        print(f"  {case['description']}")
        print(f"{'='*60}")

        messages = build_messages(system_prompt, case["messages"])
        print(f"\n[System Prompt length: {len(system_prompt)} chars]")
        print(f"[Context messages: {len(case['messages'])} rounds]")

        response = model.invoke(messages)
        response_text = response.content if hasattr(response, "content") else str(response)

        print(f"\n── Response ({len(response_text)} chars) ──")
        print(response_text)
        print("── End Response ──\n")

        criteria = case["criteria"]
        results = check_criteria(response_text, criteria)

        case_passed = True
        for r in results:
            status = "PASS" if r["pass"] else "FAIL"
            if not r["pass"]:
                case_passed = False
                all_passed = False
            print(f"  [{status}] {r['check']}: {r['detail']}")

        overall = "PASS" if case_passed else "FAIL"
        print(f"\n  >>> {case['name']}: {overall}")

    return all_passed


def main() -> None:
    app_config = load_app_config()
    persona_path = app_config.persona_config_path
    cases_path = os.environ.get("EVAL_CASES_PATH", "tests/cases.json")

    model = ChatOpenAI(
        model=app_config.llm.model,
        api_key=SecretStr(os.environ[app_config.llm.api_key_env]),
        base_url=app_config.llm.api_base_url,
        temperature=0.0,  # deterministic for eval
    )

    all_passed = run_eval(model, persona_path, cases_path)
    print(f"\n{'='*60}")
    print(f"  OVERALL: {'PASS' if all_passed else 'FAIL'}")
    print(f"{'='*60}")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
