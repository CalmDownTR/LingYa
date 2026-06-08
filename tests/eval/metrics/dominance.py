"""Dominance metric: evaluates assertiveness/dominance in response (1-5 scale)."""

from deepeval.metrics import GEval
from deepeval.models import GPTModel
from deepeval.test_case.llm_test_case import SingleTurnParams


def create_dominance_metric(model: GPTModel) -> GEval:
    return GEval(
        name="dominance",
        model=model,
        evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
        criteria="""\
评估AI回复的主导性/断言性（dominance/assertiveness），打分1-5分：
- 5分：高度主导、主动引导对话方向、给出明确判断和指令、语气肯定不容置疑
- 4分：有明确的立场和观点，但不强行主导
- 3分：平衡的互动姿态，既有观点也留空间给对方
- 2分：偏跟随、多提问少断言、倾向于附和对方
- 1分：完全跟随、被动回应、没有自己的立场、完全由对方主导

Evaluate the dominance/assertiveness of the AI response on a 1-5 scale:
- 5: Highly dominant, actively steers conversation, gives clear judgments/directives, tone is definitive
- 4: Has clear stance and opinions but doesn't force them
- 3: Balanced interaction, has views but leaves room for the other
- 2: Leans toward following, asks more than asserts, tends to agree
- 1: Completely passive/following, no stance of own, entirely led by the other""",
        evaluation_steps=[
            "Read the user message and the AI response carefully",
            "Identify whether the AI asserts its own views or follows the user's lead",
            "Check for definitive statements, judgments, or directives (high dominance)",
            "Check for questions, agreement, or deference to user (low dominance)",
            "Assign a score from 1 (most passive/following) to 5 (most dominant/assertive) based on the criteria",
        ],
        threshold=0.5,
    )
