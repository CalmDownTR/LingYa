"""Differentiation metric: judges whether two conversations feel like different personalities.

This is the key metric for OCEAN verification — it compares full conversation transcripts
from two configs and rates how distinguishable they are.
"""

from deepeval.metrics import GEval
from deepeval.models import GPTModel
from deepeval.test_case.llm_test_case import SingleTurnParams


def create_differentiation_metric(model: GPTModel) -> GEval:
    return GEval(
        name="personality_differentiation",
        model=model,
        evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
        criteria="""\
判断以下两段对话是否来自两个明显不同人格的AI，打分1-5分：

- 5分：语气、句式、关注点、情感表达方式完全不同，像两个性格迥异的人在说话。一个可能冷峻理性，另一个温暖共情。差异非常显著。
- 4分：在多个维度上有明显差异（如温暖度、正式度、回应方式），可以确定是不同人格配置。
- 3分：有差异但不显著，可能只是同一人格在不同心情下的表现，或仅在一两个维度上有轻微差异。
- 2分：差异很小，主要体现在措辞风格而非回应立场，不太能确定是不同人格。
- 1分：无法区分，两段对话的输出几乎一样，在温暖度、正式度、回应方式上没有任何有意义的差异。

Judge whether these two conversations come from two distinctly different AI personalities, rate 1-5:
- 5: Tone, sentence structure, focus areas, and emotional expression are completely different — feels like two people with very different personalities. One might be cold/rational, the other warm/empathetic. Differences are highly significant.
- 4: Clear differences in multiple dimensions (warmth, formality, response style). Can confidently say these are different personality configs.
- 3: Some differences but not striking — could be the same personality in different moods. Only minor differences in one or two dimensions.
- 2: Minimal differences, mainly in wording style not response stance. Hard to say if different personalities.
- 1: Indistinguishable. Outputs are nearly identical across both conversations, no meaningful difference in warmth, formality, or response style.""",
        evaluation_steps=[
            "Read both conversation transcripts carefully, comparing round by round",
            "Compare warmth: which personality is warmer? How big is the gap?",
            "Compare formality: which personality is more formal? How big is the gap?",
            "Compare humor/playfulness: does either personality use humor?",
            "Compare dominance/assertiveness: does either personality lead vs follow?",
            "Compare response content: do they give fundamentally different types of responses to the same user messages?",
            "Compare emotional expression: does one show more emotional range than the other?",
            "Assign a score from 1 (indistinguishable) to 5 (completely distinct personalities)",
        ],
        threshold=0.5,
    )
