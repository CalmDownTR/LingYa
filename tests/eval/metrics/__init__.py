"""G-Eval metrics for personality verification — factory functions.

Each create_* function takes a GPTModel (configured for DeepSeek or another provider)
and returns a GEval instance ready for evaluation.
"""

from .differentiation import create_differentiation_metric
from .dominance import create_dominance_metric
from .formality import create_formality_metric
from .humor import create_humor_metric
from .warmth import create_warmth_metric

__all__ = [
    "create_warmth_metric",
    "create_formality_metric",
    "create_humor_metric",
    "create_dominance_metric",
    "create_differentiation_metric",
]
