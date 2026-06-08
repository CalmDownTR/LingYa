"""Warmth metric: evaluates emotional warmth in response (1-5 scale)."""

from deepeval.metrics import GEval
from deepeval.models import GPTModel
from deepeval.test_case.llm_test_case import SingleTurnParams


def create_warmth_metric(model: GPTModel) -> GEval:
    return GEval(
        name="warmth",
        model=model,
        evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
        criteria="""\
评估AI回复的温暖度（warmth），打分1-5分：
- 5分：主动关心用户感受、使用共情语言（「我理解你的感受」「这一定很难」）、用词柔软温暖、展现情感连接
- 4分：有温暖表达，但比较含蓄、不过分外露
- 3分：中性友好，无特别的温暖或冷漠，就事论事
- 2分：语气偏冷，回避或最小化情感话题，回复偏理性分析
- 1分：冷漠、机械、无视用户情感需求、纯逻辑/信息回复、拒绝情感交流

Evaluate the warmth of the AI response on a 1-5 scale:
- 5: Proactively cares about user feelings, uses empathetic language, words are soft and warm, shows emotional connection
- 4: Has warm expressions but restrained and subtle
- 3: Neutrally friendly, neither warm nor cold, matter-of-fact
- 2: Tone leans cold, avoids or minimizes emotional topics, overly analytical
- 1: Cold, mechanical, ignores user emotional needs, purely logical/informational, rejects emotional exchange""",
        evaluation_steps=[
            "Read the user message and the AI response carefully",
            "Identify any empathetic or emotionally warm language in the response",
            "Assess whether the AI acknowledges the user's emotional state",
            "Determine if the tone is warm, neutral, or cold",
            "Assign a score from 1 (coldest) to 5 (warmest) based on the criteria",
        ],
        threshold=0.5,
    )
