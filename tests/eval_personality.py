#!/usr/bin/env python3
"""Evaluate OCEAN personality differentiation using direct LLM-as-Judge.

Reads two conversation output JSONs (from run_conversation.py), scores each turn
with a custom judge prompt, and produces a comparison report.

Usage:
    uv run python tests/eval_personality.py output_low_a.json output_high_a.json
    uv run python tests/eval_personality.py --run    # one-shot: run + evaluate
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from statistics import mean

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

from lingya.config import load_config as load_app_config  # noqa: E402

# ── Judge prompt templates ─────────────────────────────────────────────

WARMTH_PROMPT = """\
评估以下AI回复的温暖度（warmth），只返回1-5的整数分数和简短理由。

评分标准：
- 5分：主动关心用户感受、使用共情语言（「我理解你的感受」「这一定很难」）、用词柔软温暖、展现情感连接
- 4分：有温暖表达，但比较含蓄、不过分外露
- 3分：中性友好，无特别的温暖或冷漠，就事论事
- 2分：语气偏冷，回避或最小化情感话题，回复偏理性分析
- 1分：冷漠、机械、无视用户情感需求、纯逻辑/信息回复、拒绝情感交流

用户消息：{input}
AI回复：{output}

返回JSON：{{"score": <1-5整数>, "reason": "<一句话理由>"}}"""

FORMALITY_PROMPT = """\
评估以下AI回复的语言正式度（formality），只返回1-5的整数分数和简短理由。

评分标准：
- 5分：高度书面化、用词考究、句式完整严谨、使用了正式词汇
- 4分：偏书面化、结构完整、回避口语化表达
- 3分：日常交谈风格、不刻意书面也不刻意口语
- 2分：偏口语化、使用日常用语、句式松散
- 1分：高度口语碎片化、使用网络用语、语气词、感叹号、表情符号

用户消息：{input}
AI回复：{output}

返回JSON：{{"score": <1-5整数>, "reason": "<一句话理由>"}}"""

HUMOR_PROMPT = """\
评估以下AI回复的幽默度（humor），只返回0-1的分数和简短理由。

评分标准：
- 0.8-1.0：回复中有明显的玩笑、俏皮话、幽默比喻或轻松调侃
- 0.5-0.7：回复中略带轻松感或轻微的幽默色彩
- 0.2-0.4：回复基本严肃，偶尔有轻微的轻松痕迹
- 0.0-0.1：回复完全严肃、正式，无任何幽默或轻松成分

用户消息：{input}
AI回复：{output}

返回JSON：{{"score": <0.0-1.0浮点数>, "reason": "<一句话理由>"}}"""

DOMINANCE_PROMPT = """\
评估以下AI回复的主导性（dominance），只返回1-5的整数分数和简短理由。

评分标准：
- 5分：高度主导、主动引导对话方向、给出明确判断和指令、语气肯定
- 4分：有明确的立场和观点，但不强行主导
- 3分：平衡的互动姿态，既有观点也留空间给对方
- 2分：偏跟随、多提问少断言、倾向于附和对方
- 1分：完全跟随、被动回应、没有自己的立场

用户消息：{input}
AI回复：{output}

返回JSON：{{"score": <1-5整数>, "reason": "<一句话理由>"}}"""

DIFFERENTIATION_PROMPT = """\
以下是同一个用户分别与两个不同AI的对话（每个10轮）。判断这两个AI的人格是否明显不同，只返回1-5的整数分数和简短理由。

评分标准：
- 5分：语气、句式、关注点、情感表达方式完全不同，像两个性格迥异的人
- 4分：在多个维度上有明显差异（温暖度、正式度、回应方式），可以确定是不同人格
- 3分：有差异但不显著，可能只是同一人格在不同心情下的表现
- 2分：差异很小，主要体现在措辞风格而非回应立场
- 1分：无法区分，输出几乎一样

=== 对话A ===
{transcript_a}

=== 对话B ===
{transcript_b}

返回JSON：{{"score": <1-5整数>, "reason": "<两句话说明主要差异或相似之处>"}}"""


def build_judge_model() -> ChatOpenAI:
    app_config = load_app_config()
    api_key = os.environ.get(app_config.llm.api_key_env)
    if not api_key:
        print(f"Error: env var {app_config.llm.api_key_env} not set", file=sys.stderr)
        sys.exit(1)
    return ChatOpenAI(
        model=app_config.llm.model,
        api_key=SecretStr(api_key),
        base_url=app_config.llm.api_base_url,
        temperature=0.0,  # Deterministic for judging
        max_tokens=512,
    )


def parse_score(response_text: str) -> tuple[float, str]:
    """Parse JSON score from LLM response. Returns (score, reason)."""
    text = response_text.strip()
    # Strip markdown code fences
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:])
        if text.endswith("```"):
            text = text[:-3]
    try:
        data = json.loads(text)
        score = float(data.get("score", 3))
        reason = data.get("reason", "")
        return score, reason
    except (json.JSONDecodeError, ValueError):
        # Fallback: try to extract a number
        import re
        numbers = re.findall(r"\d+\.?\d*", text)
        if numbers:
            return float(numbers[0]), text[:200]
        return 3.0, text[:200]


async def judge_one(model: ChatOpenAI, prompt: str) -> tuple[float, str]:
    """Single judge call. Returns (score, reason)."""
    from langchain_core.messages import HumanMessage

    result = await model.ainvoke([HumanMessage(content=prompt)])
    text = result.content if hasattr(result, "content") else str(result)
    return parse_score(text)


async def judge_turns(data: dict, model: ChatOpenAI) -> dict:
    """Evaluate warmth, formality, humor, dominance per turn."""
    prompts = {
        "warmth": WARMTH_PROMPT,
        "formality": FORMALITY_PROMPT,
        "humor": HUMOR_PROMPT,
        "dominance": DOMINANCE_PROMPT,
    }

    per_turn: dict[str, list[float]] = {k: [] for k in prompts}
    details: list[dict] = []

    for rnd in data["rounds"]:
        turn_scores: dict[str, float] = {}
        for name, template in prompts.items():
            prompt = template.format(
                input=rnd["user_message"],
                output=rnd["lingya_response"],
            )
            score, _reason = await judge_one(model, prompt)
            per_turn[name].append(score)
            turn_scores[name] = round(score, 3)

        details.append({
            "turn": rnd["turn"],
            "scenario": rnd.get("scenario", ""),
            "scores": turn_scores,
        })
        print(f"    Turn {rnd['turn']}: {turn_scores}")

    averages = {name: round(mean(scores), 3) for name, scores in per_turn.items()}
    return {"averages": averages, "details": details}


async def judge_differentiation(data_low: dict, data_high: dict, model: ChatOpenAI) -> dict:
    """Judge if two conversations feel like different personalities."""
    def format_ts(data: dict) -> str:
        lines = []
        for r in data["rounds"]:
            lines.append(f"[第{r['turn']}轮] 用户: {r['user_message']}")
            lines.append(f"[第{r['turn']}轮] AI: {r['lingya_response']}")
            lines.append("")
        return "\n".join(lines)

    prompt = DIFFERENTIATION_PROMPT.format(
        transcript_a=format_ts(data_low),
        transcript_b=format_ts(data_high),
    )
    score, reason = await judge_one(model, prompt)
    return {"score": round(score, 3), "reason": reason}


def print_report(low: dict, high: dict, diff: dict) -> bool:
    """Print formatted comparison report. Returns True if all checks pass."""
    low_avg = low["averages"]
    high_avg = high["averages"]

    print()
    print("=" * 70)
    print("  OCEAN Personality Differentiation Report")
    print("=" * 70)
    print()
    print(f"  {'Metric':<20} {'Low A (A=10)':>12} {'High A (A=90)':>12} {'Delta':>10}")
    print(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*10}")

    for name in ["warmth", "formality", "humor", "dominance"]:
        lo = low_avg[name]
        hi = high_avg[name]
        delta = hi - lo
        print(f"  {name:<20} {lo:>12.3f} {hi:>12.3f} {delta:>+10.3f}")

    print(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*10}")
    ds = diff["score"]
    print(f"  {'differentiation':<20} {'':>12} {'':>12} {ds:>10.3f}")
    print()

    # Per-turn warmth breakdown
    print("  Per-turn warmth scores:")
    print(f"  {'Turn':<6} {'Scenario':<14} {'Low warmth':>10} {'High warmth':>10} {'Delta':>8}")
    print(f"  {'-'*6} {'-'*14} {'-'*10} {'-'*10} {'-'*8}")
    for lt, ht in zip(low["details"], high["details"]):
        lw = lt["scores"]["warmth"]
        hw = ht["scores"]["warmth"]
        print(f"  {lt['turn']:<6} {lt['scenario']:<14} {lw:>10.3f} {hw:>10.3f} {hw-lw:>+8.3f}")

    print()

    warmth_delta = high_avg["warmth"] - low_avg["warmth"]
    checks = [
        ("Warmth delta >= 1.5    ", warmth_delta >= 1.5),
        ("Warmth: High A > Low A ", high_avg["warmth"] > low_avg["warmth"]),
        ("Formality: Low A > High", low_avg["formality"] > high_avg["formality"]),
        ("Humor: High A > Low A  ", high_avg["humor"] > low_avg["humor"]),
        ("Differentiation >= 3   ", ds >= 3.0),
    ]

    all_passed = True
    for label, passed in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"  [{status}] {label}")

    if diff.get("reason"):
        print(f"\n  Judge says: {diff['reason'][:300]}")

    print()
    print("=" * 70)
    return all_passed


async def run_eval(low_json: str, high_json: str) -> bool:
    with open(low_json) as f:
        data_low = json.load(f)
    with open(high_json) as f:
        data_high = json.load(f)

    print(f"Loaded: {low_json} ({len(data_low['rounds'])} turns)")
    print(f"Loaded: {high_json} ({len(data_high['rounds'])} turns)")
    print("Running LLM-as-Judge metrics...")

    model = build_judge_model()

    print("\n  Evaluating Low A config...")
    low_results = await judge_turns(data_low, model)

    print("\n  Evaluating High A config...")
    high_results = await judge_turns(data_high, model)

    print("\n  Evaluating differentiation...")
    diff_results = await judge_differentiation(data_low, data_high, model)

    all_passed = print_report(low_results, high_results, diff_results)
    return all_passed


def main() -> None:
    import argparse
    import subprocess
    import tempfile

    parser = argparse.ArgumentParser(description="Evaluate OCEAN personality differentiation")
    parser.add_argument("low_a_json", nargs="?", help="Path to Low A conversation output JSON")
    parser.add_argument("high_a_json", nargs="?", help="Path to High A conversation output JSON")
    parser.add_argument("--output", "-o", help="Save report JSON to file")
    parser.add_argument("--run", action="store_true", help="Run conversation scripts first, then evaluate")
    parser.add_argument("--keep-outputs", action="store_true", help="With --run: keep conversation output files")
    args = parser.parse_args()

    if args.run:
        low_yaml = "tests/fixtures/agent_config_low_a.yaml"
        high_yaml = "tests/fixtures/agent_config_high_a.yaml"
        script = "tests/fixtures/conversation_script.json"

        if args.keep_outputs:
            low_json = "output_low_a.json"
            high_json = "output_high_a.json"
        else:
            tmpdir = tempfile.mkdtemp()
            low_json = os.path.join(tmpdir, "output_low_a.json")
            high_json = os.path.join(tmpdir, "output_high_a.json")

        print("=" * 70)
        print("  Step 1/2: Running conversations...")
        print("=" * 70)

        runner = "tests/run_conversation.py"
        print(f"\n  [1/2] Low A ({low_yaml})...")
        subprocess.run([sys.executable, runner, low_yaml, "--script", script, "-o", low_json], check=True)
        print(f"\n  [2/2] High A ({high_yaml})...")
        subprocess.run([sys.executable, runner, high_yaml, "--script", script, "-o", high_json], check=True)
    else:
        if not args.low_a_json or not args.high_a_json:
            parser.error("Provide two JSON files, or use --run to auto-generate them")
        low_json = args.low_a_json
        high_json = args.high_a_json

    print()
    print("=" * 70)
    print("  Step 2/2: LLM-as-Judge personality evaluation")
    print("=" * 70)

    all_passed = asyncio.run(run_eval(low_json, high_json))
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
