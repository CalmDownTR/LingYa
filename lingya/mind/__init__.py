"""Mind module — dynamic personality engine.

Pure computational layer with zero dependency on agent framework.
The agent consumes its output (tone parameters + prompt fragments).
"""

from lingya.mind.config import (
    BigFiveTraits,
    IdentityAnchor,
    MindConfig,
    PADBaseline,
    ToneMatrix,
    load_mind_config,
)
from lingya.mind.state import MindState, PADPoint

__all__ = [
    "BigFiveTraits",
    "IdentityAnchor",
    "MindConfig",
    "MindState",
    "PADBaseline",
    "PADPoint",
    "ToneMatrix",
    "load_mind_config",
]
