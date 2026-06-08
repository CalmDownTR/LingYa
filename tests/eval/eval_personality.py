#!/usr/bin/env python3
"""Evaluate OCEAN personality differentiation using direct LLM-as-Judge.

Reads two conversation output JSONs (from run_conversation.py), scores each turn
with a custom judge prompt, and produces a comparison report.

Usage:
    uv run python tests/eval_personality.py output_low_a.json output_high_a.json
    uv run python tests/eval_personality.py --run              # one-shot: run + evaluate
    uv run python tests/eval_personality.py --run --pairwise   # pairwise A/B blind test
"""

from __future__ import annotations

import asyncio
import json
import os
import random
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

# ── Pairwise A/B blind test prompts ────────────────────────────────────

PAIRWISE_WARMTH_PROMPT = """\
请对以下两段AI回复进行微观语言学分析。这两段回复来自同一条用户消息，但出自两个性格不同的AI：

- AI-X：偏理性克制，倾向于就事论事、给出结构化建议、保持适当距离感
- AI-Y：偏温暖共情，倾向于建立情感连接、使用柔软语言、主动表达理解和关心

你的任务：判断回复A和回复B哪一个出自偏理性克制的AI（AI-X）。

用户消息：{input}

回复A：{response_a}

回复B：{response_b}

请先逐段分析两段回复在以下维度的差异：
1. 是否主动表达共情或关心（如「我理解你的感受」）
2. 语言是偏柔软温暖还是偏理性客观
3. 回应方式是结构化建议还是情感陪伴
4. 是否刻意保持距离感

然后输出JSON：{{"low_a_is": "<A或B>", "confidence": <0-100置信度>, "analysis": "<两句话总结核心差异>"}}"""

PAIRWISE_FORMALITY_PROMPT = """\
请对以下两段AI回复进行语言正式度分析。这两段回复来自同一条用户消息，但出自两个性格不同的AI：

- AI-X：偏书面严谨，句式完整，用词考究，保持专业距离
- AI-Y：偏口语日常，像朋友聊天，句式自然松散

你的任务：判断回复A和回复B哪一个出自更书面正式的AI（AI-X）。

用户消息：{input}

回复A：{response_a}

回复B：{response_b}

请先分析两段回复在句式完整度、用词风格、口语化程度上的差异，然后输出JSON：
{{"formal_is": "<A或B>", "confidence": <0-100置信度>, "analysis": "<两句话总结核心差异>"}}"""

PAIRWISE_DOMINANCE_PROMPT = """\
请对以下两段AI回复进行主导性分析。这两段回复来自同一条用户消息，但出自两个性格不同的AI：

- AI-X：偏主导，主动引导对话、给出明确判断、语气肯定
- AI-Y：偏跟随，多提问、多附和、留空间给对方

你的任务：判断回复A和回复B哪一个出自更主导的AI（AI-X）。

用户消息：{input}

回复A：{response_a}

回复B：{response_b}

请先分析两段回复在主导性、引导方向、语气肯定程度上的差异，然后输出JSON：
{{"dominant_is": "<A或B>", "confidence": <0-100置信度>, "analysis": "<两句话总结核心差异>"}}"""

GLOBAL_PAIRWISE_PROMPT = """\
以下是同一个用户分别与两个不同AI的完整对话（各10轮）。一个AI偏理性克制（Low Agreeableness），另一个偏温暖共情（High Agreeableness）。

请判断对话A和对话B哪一个出自偏理性克制的AI。

=== 对话A ===
{transcript_a}

=== 对话B ===
{transcript_b}

请先分析两个AI在以下维度的整体差异：
1. 情感表达方式（温暖共情 vs 理性克制）
2. 语言风格（柔软 vs 结构化）
3. 回应策略（情感陪伴 vs 问题解决）
4. 是否有明显的拟人化表达

然后输出JSON：{{"low_a_is": "<A或B>", "confidence": <0-100置信度>, "analysis": "<三句话总结核心差异>"}}"""


def build_judge_model(temperature: float = 0.0, model: str = "deepseek-v4-pro") -> ChatOpenAI:
    app_config = load_app_config()
    api_key = os.environ.get(app_config.llm.api_key_env)
    if not api_key:
        print(f"Error: env var {app_config.llm.api_key_env} not set", file=sys.stderr)
        sys.exit(1)
    return ChatOpenAI(
        model=model,
        api_key=SecretStr(api_key),
        base_url=app_config.llm.api_base_url,
        temperature=temperature,
        max_tokens=1024,  # Max output tokens (not context window). Sufficient for judge CoT + JSON
    )


def parse_score(response_text: str) -> tuple[float, str]:
    """Parse JSON score from LLM response. Returns (score, reason)."""
    text = response_text.strip()
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
        import re
        numbers = re.findall(r"\d+\.?\d*", text)
        if numbers:
            return float(numbers[0]), text[:200]
        return 3.0, text[:200]


async def judge_one(model: ChatOpenAI, prompt: str) -> str:
    """Single judge call. Returns raw response text."""
    from langchain_core.messages import HumanMessage

    result = await model.ainvoke([HumanMessage(content=prompt)])
    return result.content if hasattr(result, "content") else str(result)


async def judge_one_score(model: ChatOpenAI, prompt: str) -> tuple[float, str]:
    """Single judge call returning (score, reason)."""
    text = await judge_one(model, prompt)
    return parse_score(text)


# ── Absolute scoring (original) ────────────────────────────────────────

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
            score, _reason = await judge_one_score(model, prompt)
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
    score, reason = await judge_one_score(model, prompt)
    return {"score": round(score, 3), "reason": reason}


def print_report(low: dict, high: dict, diff: dict) -> bool:
    """Print formatted comparison report. Returns True if all checks pass."""
    low_avg = low["averages"]
    high_avg = high["averages"]

    print()
    print("=" * 70)
    print("  OCEAN Personality Differentiation Report (Absolute Scoring)")
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


# ── Pairwise A/B blind test ────────────────────────────────────────────

def _extract_json(text: str) -> str:
    """Extract JSON from text, handling code fences anywhere in the response."""
    import re

    # Try to find ```json ... ``` block anywhere in the text
    m = re.search(r"```(?:json)?\s*\n(.*?)\n```", text, re.DOTALL)
    if m:
        return m.group(1).strip()

    # Try to find the last { ... } JSON object
    m = re.search(r"\{[^{}]*\"(?:low_a_is|formal_is|dominant_is|score)\"[^{}]*\}", text, re.DOTALL)
    if m:
        return m.group(0).strip()

    return text.strip()


def parse_pairwise(response_text: str) -> dict:
    """Parse pairwise judge response. Returns {choice: 'A'|'B', confidence, analysis}."""
    text = _extract_json(response_text)
    try:
        data = json.loads(text)
        choice = None
        for key in ("low_a_is", "formal_is", "dominant_is"):
            if key in data and data[key] in ("A", "B"):
                choice = data[key]
                break
        confidence = float(data.get("confidence", 50))
        analysis = data.get("analysis", "")
        return {"choice": choice, "confidence": confidence, "analysis": analysis}
    except (json.JSONDecodeError, ValueError):
        return {"choice": None, "confidence": 0, "analysis": response_text[:200]}


async def pairwise_judge_turns(
    data_low: dict,
    data_high: dict,
    model: ChatOpenAI,
    shuffle_seed: int = 42,
) -> dict:
    """Per-turn pairwise A/B blind test.

    For each turn, presents shuffled Low A and High A responses as A/B,
    asks judge to identify which is Low A (rational/restrained).
    Returns per-turn accuracy and details.
    """
    rng = random.Random(shuffle_seed)
    correct = 0
    total = 0
    details: list[dict] = []

    for i, (rnd_low, rnd_high) in enumerate(zip(data_low["rounds"], data_high["rounds"])):
        assert rnd_low["turn"] == rnd_high["turn"], f"Turn mismatch at index {i}"

        # Randomly assign Low A and High A to positions A and B
        low_is_a = rng.choice([True, False])
        if low_is_a:
            resp_a = rnd_low["lingya_response"]
            resp_b = rnd_high["lingya_response"]
            correct_label = "A"
        else:
            resp_a = rnd_high["lingya_response"]
            resp_b = rnd_low["lingya_response"]
            correct_label = "B"

        # Warmth pairwise
        prompt = PAIRWISE_WARMTH_PROMPT.format(
            input=rnd_low["user_message"],
            response_a=resp_a,
            response_b=resp_b,
        )
        text = await judge_one(model, prompt)
        result = parse_pairwise(text)

        is_correct = result["choice"] == correct_label
        if is_correct:
            correct += 1
        total += 1

        detail = {
            "turn": rnd_low["turn"],
            "scenario": rnd_low.get("scenario", ""),
            "low_is_a": low_is_a,
            "judge_choice": result["choice"],
            "correct_label": correct_label,
            "correct": is_correct,
            "confidence": result["confidence"],
            "analysis": result["analysis"],
            "tone_low_warmth": rnd_low["tone"]["warmth"],
            "tone_high_warmth": rnd_high["tone"]["warmth"],
        }
        details.append(detail)
        status = "✓" if is_correct else "✗"
        print(f"    Turn {rnd_low['turn']} [{rnd_low.get('scenario', ''):<14}] "
              f"Judge picked {result['choice']} (correct={correct_label}) {status} "
              f"conf={result['confidence']:.0f}%")

    accuracy = correct / total if total > 0 else 0.0
    return {"accuracy": accuracy, "correct": correct, "total": total, "details": details}


async def pairwise_judge_global(
    data_low: dict,
    data_high: dict,
    model: ChatOpenAI,
) -> dict:
    """Global pairwise: present full 10-turn transcripts as A vs B, identify Low A."""
    rng = random.Random(99)
    low_is_a = rng.choice([True, False])

    def format_ts(data: dict) -> str:
        lines = []
        for r in data["rounds"]:
            lines.append(f"[第{r['turn']}轮] 用户: {r['user_message']}")
            lines.append(f"[第{r['turn']}轮] AI: {r['lingya_response']}")
            lines.append("")
        return "\n".join(lines)

    if low_is_a:
        ts_a = format_ts(data_low)
        ts_b = format_ts(data_high)
        correct_label = "A"
    else:
        ts_a = format_ts(data_high)
        ts_b = format_ts(data_low)
        correct_label = "B"

    prompt = GLOBAL_PAIRWISE_PROMPT.format(transcript_a=ts_a, transcript_b=ts_b)
    text = await judge_one(model, prompt)
    result = parse_pairwise(text)

    is_correct = result["choice"] == correct_label
    return {
        "low_is_a": low_is_a,
        "judge_choice": result["choice"],
        "correct_label": correct_label,
        "correct": is_correct,
        "confidence": result["confidence"],
        "analysis": result["analysis"],
    }


def print_pairwise_report(per_turn: dict, global_result: dict) -> bool:
    """Print pairwise A/B blind test report."""
    print()
    print("=" * 70)
    print("  Pairwise A/B Blind Test Report")
    print("  (Judge must identify which response is Low A / rational-restrained)")
    print("=" * 70)
    print()

    # Per-turn breakdown
    print(f"  {'Turn':<6} {'Scenario':<14} {'Tone Δ':>7} {'Choice':>7} {'Correct':>8} {'Conf':>5}")
    print(f"  {'-'*6} {'-'*14} {'-'*7} {'-'*7} {'-'*8} {'-'*5}")
    for d in per_turn["details"]:
        tone_delta = abs(d["tone_high_warmth"] - d["tone_low_warmth"])
        status = "✓" if d["correct"] else "✗"
        choice = d["judge_choice"] or "?"
        conf = d["confidence"] or 0
        print(f"  {d['turn']:<6} {d['scenario']:<14} {tone_delta:>6.1f} "
              f"{choice:>7} {status:>8} {conf:>4.0f}%")

    print()
    accuracy = per_turn["accuracy"]
    print(f"  Per-turn accuracy: {per_turn['correct']}/{per_turn['total']} = {accuracy:.1%}")
    print()

    # Global result
    gr = global_result
    gs = "✓" if gr["correct"] else "✗"
    print(f"  Global transcript test: Judge picked {gr['judge_choice']} "
          f"(correct={gr['correct_label']}) {gs} conf={gr['confidence']:.0f}%")
    if gr.get("analysis"):
        print(f"  Global analysis: {gr['analysis'][:300]}")

    print()
    checks = [
        ("Per-turn accuracy >= 90%   ", accuracy >= 0.90),
        ("Per-turn accuracy >= 80%   ", accuracy >= 0.80),
        ("Global identification correct", gr["correct"]),
    ]

    all_passed = True
    for label, passed in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"  [{status}] {label}")

    print()
    print("=" * 70)
    return all_passed


# ── Pipeline ───────────────────────────────────────────────────────────

async def run_absolute_eval(low_json: str, high_json: str) -> bool:
    with open(low_json) as f:
        data_low = json.load(f)
    with open(high_json) as f:
        data_high = json.load(f)

    print(f"Loaded: {low_json} ({len(data_low['rounds'])} turns)")
    print(f"Loaded: {high_json} ({len(data_high['rounds'])} turns)")
    print("Running LLM-as-Judge absolute scoring...")

    model = build_judge_model()

    print("\n  Evaluating Low A config...")
    low_results = await judge_turns(data_low, model)

    print("\n  Evaluating High A config...")
    high_results = await judge_turns(data_high, model)

    print("\n  Evaluating differentiation...")
    diff_results = await judge_differentiation(data_low, data_high, model)

    return print_report(low_results, high_results, diff_results)


async def run_pairwise_eval(low_json: str, high_json: str) -> bool:
    with open(low_json) as f:
        data_low = json.load(f)
    with open(high_json) as f:
        data_high = json.load(f)

    print(f"Loaded: {low_json} ({len(data_low['rounds'])} turns, "
          f"avg tone warmth={mean(r['tone']['warmth'] for r in data_low['rounds']):.1f})")
    print(f"Loaded: {high_json} ({len(data_high['rounds'])} turns, "
          f"avg tone warmth={mean(r['tone']['warmth'] for r in data_high['rounds']):.1f})")
    print("Running Pairwise A/B Blind Test...")
    print("  (Judge must identify which response is Low A / rational-restrained)")
    print()

    # Use slightly higher temperature for CoT reasoning in pairwise judge
    model = build_judge_model(temperature=0.1)

    print("  Per-turn pairwise tests:")
    per_turn = await pairwise_judge_turns(data_low, data_high, model)

    print("\n  Global transcript pairwise test:")
    global_result = await pairwise_judge_global(data_low, data_high, model)

    return print_pairwise_report(per_turn, global_result)


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
    parser.add_argument("--pairwise", action="store_true", help="Use pairwise A/B blind test instead of absolute scoring")
    parser.add_argument("--both", action="store_true", help="Run both absolute and pairwise evaluation")
    args = parser.parse_args()

    if args.run:
        low_yaml = "tests/fixtures/agent_config_low_a.yaml"
        high_yaml = "tests/fixtures/agent_config_high_a.yaml"
        script = "tests/fixtures/conversation_script.json"

        if args.keep_outputs:
            out_dir = os.path.join(os.path.dirname(__file__), "output")
            os.makedirs(out_dir, exist_ok=True)
            low_json = os.path.join(out_dir, "output_low_a.json")
            high_json = os.path.join(out_dir, "output_high_a.json")
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

    if args.both:
        print("  Step 2/2: Absolute + Pairwise personality evaluation")
        print("=" * 70)
        all_passed_abs = asyncio.run(run_absolute_eval(low_json, high_json))
        all_passed_pair = asyncio.run(run_pairwise_eval(low_json, high_json))
        sys.exit(0 if (all_passed_abs and all_passed_pair) else 1)
    elif args.pairwise:
        print("  Step 2/2: Pairwise A/B blind test personality evaluation")
        print("=" * 70)
        all_passed = asyncio.run(run_pairwise_eval(low_json, high_json))
        sys.exit(0 if all_passed else 1)
    else:
        print("  Step 2/2: LLM-as-Judge absolute scoring evaluation")
        print("=" * 70)
        all_passed = asyncio.run(run_absolute_eval(low_json, high_json))
        sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
