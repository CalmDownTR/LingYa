"""G-Eval metrics for personality verification — factory functions.

Each create_* function takes a GPTModel (configured for DeepSeek or another provider)
and returns a GEval instance ready for evaluation.
"""

from tests.metrics.warmth import create_warmth_metric
from tests.metrics.formality import create_formality_metric
from tests.metrics.humor import create_humor_metric
from tests.metrics.dominance import create_dominance_metric
from tests.metrics.differentiation import create_differentiation_metric

__all__ = [
    "create_warmth_metric",
    "create_formality_metric",
    "create_humor_metric",
    "create_dominance_metric",
    "create_differentiation_metric",
]
