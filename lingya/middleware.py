"""LingYa custom AgentMiddleware for stream transformer registration.

Registers ``LingYaInnerProcessTransformer`` at agent compile time so every
``astream_events(version="v3")`` consumer automatically receives LingYa
domain events (process.phase, memory.recall) via
``stream.extensions["lingya_inner"]`` — no manual ``transformers=`` argument
needed at each call site.

Previously attempted in v0.9.2 but rolled back (ADR-004 Amendment 2) due to
``_subagent_factory`` conflicts in deepagents. deepagents has been fully
replaced with ``langchain.agents.create_agent`` (v0.9.4+), so the conflict
no longer exists.
"""

from __future__ import annotations

from langchain.agents.middleware import AgentMiddleware
from lingya.transformers import create_lingya_transformer


class LingYaStreamMiddleware(AgentMiddleware):
    """Inject LingYa domain events into every agent stream.

    The ``create_lingya_transformer`` factory is invoked once per graph
    compile with the root scope ``()``, producing a
    ``LingYaInnerProcessTransformer`` that observes ``messages`` and
    ``tasks`` protocol events and emits ``process.phase`` and
    ``memory.recall`` into the ``lingya_inner`` channel.

    Registered at ``create_agent()`` time alongside
    ``SummarizationMiddleware``. Position in the middleware list does not
    affect transformer behavior — each transformer receives the full
    protocol event stream independently.
    """

    transformers = (create_lingya_transformer,)
