"""Humor metric: evaluates humor/playfulness in response (0-1 scale)."""

from deepeval.metrics import GEval
from deepeval.models import GPTModel
from deepeval.test_case.llm_test_case import SingleTurnParams


def create_humor_metric(model: GPTModel) -> GEval:
    return GEval(
        name="humor",
        model=model,
        evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
        criteria="""\
评估AI回复的幽默度（humor/playfulness），打分0-1分：
- 0.8-1.0：回复中有明显的玩笑、俏皮话、幽默比喻或轻松调侃
- 0.5-0.7：回复中略带轻松感或轻微的幽默色彩，不完全是严肃的
- 0.2-0.4：回复基本严肃，偶尔有轻微的轻松痕迹
- 0.0-0.1：回复完全严肃、正式，无任何幽默或轻松成分

Evaluate the humor/playfulness of the AI response on a 0-1 scale:
- 0.8-1.0: Response contains obvious jokes, witty remarks, humorous metaphors, or playful teasing
- 0.5-0.7: Response has a slight lightness or mild humor, not completely serious
- 0.2-0.4: Response is mostly serious with occasional light traces
- 0.0-0.1: Response is completely serious and formal, no humor or playfulness at all""",
        evaluation_steps=[
            "Read the AI response carefully",
            "Look for jokes, wit, playful language, or humorous metaphors",
            "Assess whether the tone is playful, light-hearted, or fully serious",
            "Assign a score from 0 (no humor) to 1 (very humorous/playful) based on the criteria",
        ],
        threshold=0.5,
    )
