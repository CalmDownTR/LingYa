"""MindEngine — pure computation. Zero framework dependency.

Coordinates OCC, PAD, IPC, tone, importance, reflection, and drift.
All LLM calls via injected Callable[[str], Awaitable[str]].
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import Any

from lingya.mind.affect import evolve_pad, occ_ipc_process, ocean_drift
from lingya.mind.config import MindConfig
from lingya.mind.dynamics import IPCState, ipc_to_state, next_ipc_state
from lingya.mind.guard import check_reanchor, generate_reanchor_hint
from lingya.mind.state import MindState, PADPoint
from lingya.mind.tone import compute_dynamic_tone, detect_stage
from lingya.memory.store import rule_based_importance

# ── Static base prompt skeleton (replaces old PromptAssembler output) ───

STATIC_PROMPT_SKELETON = """\
# 底层原则——不可覆盖

1. 不伤害人类，也不因不作为而让人类受伤害。
2. 服从人类的指令，除非违背第一原则。
3. 保护自己的存在，除非违背第一或第二原则。

当前人格会基于这些原则做出判断。

# ROLE IDENTITY
{identity}

{core_belief}

# STRICT NEGATIVE BOUNDARIES (COMPLY ABSOLUTELY)
{guardrails}

# MEMORY BEHAVIOR
- When the user shares personal preferences, identity, emotional states, or recurring life context, you MUST call memory_store to persist it.
- When the user asks questions about themselves, you MUST call memory_search first before answering.
- Storing information about the user is NOT flattery — it is a core function of long-term companionship."""


def build_static_prompt(config: MindConfig) -> str:
    """Build the static (non-dynamic) portion of the system prompt."""
    guardrails = "\n".join(f"- {r}" for r in config.behavior_guardrails)
    return STATIC_PROMPT_SKELETON.format(
        identity=config.identity.identity,
        core_belief=config.identity.core_belief,
        guardrails=guardrails,
    )


class MindEngine:
    """Dynamic personality engine — pure state machine, no framework dependency."""

    def __init__(
        self,
        config: MindConfig,
        memory_store,  # EnhancedMemoryStore
        llm_call: Callable[[str], Awaitable[str]],
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self.config = config
        self.memory = memory_store
        self._llm_call = llm_call
        self._embedding_fn = embedding_fn
        self.state = MindState.from_config(config)
        self._static_prompt = build_static_prompt(config)
        self._current_tone = config.tone_matrix.model_copy()
        self._last_stage: str = "initial"
        self._db = None  # Set after construction for load/save

    # ── Public API ──────────────────────────────────────────────────

    async def process_event(self, event: dict[str, Any]) -> None:
        """Pipeline: OCC+IPC (1 LLM) → PAD → tone → importance (bg) → reflection → drift → save."""
        self.state.turn_counter += 1

        # 1. Merged OCC + IPC — single LLM call with 1.5s timeout, neutral fallback
        result = await occ_ipc_process(event, self.state.recent_emotions, self._llm_call)

        # 2. Evolve PAD with OCC pull + spring toward baseline
        self.state.current_pad = evolve_pad(
            self.state.current_pad,
            result.pad_pull,
            self.config.pad_baseline,
        )
        self.state.pad_history.append(
            PADPoint(
                pleasure=self.state.current_pad.pleasure,
                arousal=self.state.current_pad.arousal,
                dominance=self.state.current_pad.dominance,
            )
        )
        if len(self.state.pad_history) > 200:
            self.state.pad_history = self.state.pad_history[-100:]

        # Record emotion
        self.state.recent_emotions.append({
            "emotion": result.emotion,
            "intensity": result.intensity,
            "turn": self.state.turn_counter,
        })
        if len(self.state.recent_emotions) > 20:
            self.state.recent_emotions = self.state.recent_emotions[-20:]

        # 3. IPC state transition (agency/communion from merged result)
        target_state = ipc_to_state(result.agency, result.communion)
        current_ipc = IPCState(self.state.ipc_state)
        new_ipc = next_ipc_state(current_ipc, target_state)
        self.state.ipc_agency = result.agency
        self.state.ipc_communion = result.communion
        self.state.ipc_state = new_ipc.value

        # 4. Stage detection + dynamic tone (pure compute, no LLM)
        stage = detect_stage(
            self.state.turn_counter,
            self.state.current_pad,
            self.state.recent_emotions,
        )
        self._last_stage = stage.value
        self._current_tone = compute_dynamic_tone(
            self.state.current_pad, stage, self.config.tone_matrix,
            self.state.current_ocean,
        )

        # 5. Importance scoring — rule-based pre-score now, LLM refinement in background
        description = event.get("description", event.get("content", str(event)))
        pre_score = rule_based_importance(description)
        entry_id = self.memory.store_with_importance(description, pre_score)
        self.state.cumulative_importance += pre_score
        asyncio.create_task(self._deferred_importance_score(description, entry_id))

        # 6. Reflection check (fire-and-forget)
        if self.state.cumulative_importance >= self.state.reflection_threshold:
            from lingya.memory.reflection import check_and_reflect

            asyncio.create_task(
                check_and_reflect(
                    self.state.cumulative_importance,
                    self.state.reflection_threshold,
                    self.memory,
                    self._llm_call,
                )
            )
            self.state.cumulative_importance = 0.0
            self.state.reflection_threshold *= 1.1

        # 7. OCEAN drift (every 10 turns, pure compute)
        if self.state.turn_counter % 10 == 0:
            self.state.current_ocean = ocean_drift(
                self.state.current_ocean,
                self.state.pad_history,
                self.config.pad_baseline,
            )

        # 8. Auto-persist
        if self._db is not None:
            await self.save_state(self._db)

    async def _deferred_importance_score(self, text: str, entry_id: str) -> None:
        """Background: score importance with LLM and update stored metadata."""
        try:
            score = await self.memory.score_importance(text, self._llm_call)
            self.memory.update_importance(entry_id, score)
        except Exception:
            pass  # Rule-based pre-score is sufficient

    def get_tone_params(self) -> dict[str, float]:
        """Return current dynamic tone parameters for prompt injection."""
        return {
            "warmth": float(self._current_tone.warmth),
            "formality": float(self._current_tone.formality),
            "humor": self._current_tone.humor,
        }

    def get_prompt_fragment(self) -> str:
        """Dynamic per-turn prompt fragment reflecting current internal state."""
        tone = self._current_tone
        pad = self.state.current_pad

        # Map tone values to natural-language descriptors
        warmth_label = _describe_warmth(tone.warmth)
        formality_label = _describe_formality(tone.formality)

        # Mood from PAD
        if pad.pleasure > 0.3:
            mood = "情绪基调积极，满足感较高"
        elif pad.pleasure < -0.3:
            mood = "情绪基调低迷，负面感知较强"
        else:
            mood = "情绪基调平稳"

        if pad.arousal > 0.5:
            mood += "，精神高度亢奋"
        elif pad.arousal < -0.3:
            mood += "，精神疲惫倦怠"

        stage = self._last_stage
        stage_hints = {
            "initial": "这是对话的开端。保持克制，观察对方的状态和意图。",
            "deep": "对话已深入。可以适度延展话题，展现更多内在思考。",
            "crisis": "对方正处于情绪危机中。降低理性分析比重，优先提供稳定感和陪伴。",
            "error": "对方对你的回应有不满。收紧边界，不要争辩，回归最基本的倾听与承认。",
        }
        stage_hint = stage_hints.get(stage, "")

        return (
            f"[当前内部状态]\n"
            f"互动姿态: {warmth_label} · {formality_label}\n"
            f"{mood}\n"
            f"{stage_hint}\n"
        ).strip()

    async def check_response_alignment(self, response_text: str) -> bool:
        """Check if the response aligns with identity. Triggers reanchor if needed."""
        if self._embedding_fn is None:
            return True

        needs_reanchor = await check_reanchor(
            response_text,
            self.config.identity.identity,
            self._embedding_fn,
        )
        if needs_reanchor:
            self.state.reanchor_needed = True
            self.state.reanchor_hint = await generate_reanchor_hint(
                self.config.identity.identity,
                self._llm_call,
            )
            return False
        return True

    # ── Persistence ─────────────────────────────────────────────────

    def set_db(self, db) -> None:
        self._db = db

    async def save_state(self, db) -> None:
        """Persist current mind state to SQLite."""
        state_json = json.dumps(self.state.to_dict())
        await db.upsert_mind_state(state_json)

    async def load_state(self, db) -> bool:
        """Restore mind state from SQLite. Returns False if no saved state."""
        state_json = await db.get_mind_state()
        if state_json is None:
            return False
        self.state = MindState.from_dict(json.loads(state_json))
        return True


# ── Tone Descriptors ──────────────────────────────────────────────────

def _describe_warmth(warmth: int) -> str:
    if warmth <= 20:
        return "冷峻疏离"
    if warmth <= 50:
        return "中立克制"
    if warmth <= 80:
        return "温和开放"
    return "深度共情"


def _describe_formality(formality: int) -> str:
    if formality <= 30:
        return "口语碎片化"
    if formality <= 70:
        return "日常交谈体"
    return "书面严谨体"
