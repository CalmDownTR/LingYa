"""Formality metric: evaluates linguistic formality in response (1-5 scale)."""

from deepeval.metrics import GEval
from deepeval.models import GPTModel
from deepeval.test_case.llm_test_case import SingleTurnParams


def create_formality_metric(model: GPTModel) -> GEval:
    return GEval(
        name="formality",
        model=model,
        evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
        criteria="""\
评估AI回复的语言正式度（formality），打分1-5分：
- 5分：高度书面化、用词考究、句式完整严谨、使用了正式词汇（如：毋庸置疑、基于此、予以）
- 4分：偏书面化、结构完整、回避口语化表达
- 3分：日常交谈风格、不刻意书面也不刻意口语
- 2分：偏口语化、使用日常用语、句式松散
- 1分：高度口语碎片化、使用网络用语、语气词、感叹号、表情符号

Evaluate the formality level of the AI response on a 1-5 scale:
- 5: Highly literary, precise word choice, complete and rigorous sentence structure, uses formal vocabulary
- 4: Somewhat literary, structurally complete, avoids colloquial expressions
- 3: Casual conversation style, neither formal nor colloquial
- 2: Somewhat colloquial, uses everyday language, loose sentence structure
- 1: Highly colloquial/fragmented, uses internet slang, interjections, exclamation marks, emoji""",
        evaluation_steps=[
            "Read the AI response carefully",
            "Identify sentence structure: complete/rigorous vs loose/fragmented",
            "Check vocabulary: formal/literary vs colloquial/slang",
            "Look for features like interjections, exclamation marks, or emoji (lower formality)",
            "Assign a score from 1 (most colloquial) to 5 (most formal) based on the criteria",
        ],
        threshold=0.5,
    )
