"""MindEngine — pure computation. Zero framework dependency.

Coordinates OCC, PAD, IPC, tone, importance, reflection, and drift.
All LLM calls via injected Callable[[str], Awaitable[str]].
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

logger = logging.getLogger(__name__)

from lingya.mind.affect import evolve_pad, occ_ipc_process, ocean_drift, ocean_to_pad_baseline
from lingya.mind.config import MindConfig
from lingya.mind.dynamics import IPCState, ipc_to_state, next_ipc_state
from lingya.mind.guard import check_reanchor, generate_reanchor_hint
from lingya.mind.state import MindState, PADPoint
from lingya.mind.tone import compute_dynamic_tone, detect_stage
from lingya.memory.store import rule_based_importance

if TYPE_CHECKING:
    from lingya.events import EventBus
    from lingya.protocols import IMemoryStore

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


TONE_PRESETS: dict[str, dict[str, int | float]] = {
    "warm":       {"warmth": 80, "formality": 40, "humor": 0.3},
    "neutral":    {"warmth": 50, "formality": 50, "humor": 0.1},
    "cool":       {"warmth": 25, "formality": 75, "humor": 0.0},
    "passionate": {"warmth": 90, "formality": 30, "humor": 0.4},
    "gentle":     {"warmth": 70, "formality": 45, "humor": 0.2},
}


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
        memory_store: IMemoryStore,
        llm_call: Callable[[str], Awaitable[str]],
        embedding_fn: Callable[[str], list[float]] | None = None,
        event_bus: "EventBus | None" = None,
    ) -> None:
        self.config = config
        self._original_config = config.model_copy(deep=True)
        self.memory = memory_store
        self._llm_call = llm_call
        self._embedding_fn = embedding_fn
        self.state = MindState.from_config(config)
        self._static_prompt = build_static_prompt(config)
        self._current_tone = config.tone_matrix.model_copy()
        # v0.9.9: Snapshot of original tone_matrix for evolution cap enforcement
        self._original_tone_matrix = config.tone_matrix.model_copy()
        self._last_stage: str = "initial"
        self._db = None  # Set after construction for load/save
        self._event_bus = event_bus
        self._lock = asyncio.Lock()
        # Importance scoring observability (v0.9.8)
        self._importance_total: int = 0
        self._importance_failures: int = 0
        self._importance_pre_sum: float = 0.0
        self._importance_llm_sum: float = 0.0
        self._importance_failure_reasons: list[str] = []  # last 10
        # Identity reanchor tracking (v0.9.9)
        self._reanchor_failure_count: int = 0
        # Conversation turn recording for diary generation (v0.9.9)
        self._recent_turns: list[dict[str, Any]] = []

    # ── Public API ──────────────────────────────────────────────────

    async def process_event(self, event: dict[str, Any]) -> None:
        """Pipeline: OCC+IPC (1 LLM) → PAD → tone → importance (bg) → reflection → drift → save.

        Lock discipline (v0.9.8): state mutations + save_state run inside
        ``self._lock``. LLM calls (OCC+IPC, reflection) run outside the
        lock to avoid blocking streaming output.
        """
        self.state.turn_counter += 1

        # 1. Merged OCC + IPC — single LLM call with 1.5s timeout, neutral fallback
        #    OUTSIDE lock: LLM call, does not block concurrent idle_tick
        result = await occ_ipc_process(event, self.state.recent_emotions, self._llm_call)

        # 2-5. State mutations + 6a threshold check INSIDE lock
        async with self._lock:
            # 2. Evolve PAD with OCC pull + spring toward OCEAN-derived baseline
            self.state.current_pad = evolve_pad(
                self.state.current_pad,
                result.pad_pull,
                ocean_to_pad_baseline(self.state.current_ocean),
            )
            self.state.pad_history.append(
                PADPoint(
                    pleasure=self.state.current_pad.pleasure,
                    arousal=self.state.current_pad.arousal,
                    dominance=self.state.current_pad.dominance,
                )
            )
            if len(self.state.pad_history) > 200:
                self.state.pad_history = self.state.pad_history[-200:]

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
            self._importance_total += 1
            self._importance_pre_sum += pre_score
            asyncio.create_task(self._deferred_importance_score(description, entry_id))

            # 6a. Check if reflection is needed (capture values for LLM call outside lock)
            if self.state.cumulative_importance >= self.state.reflection_threshold:
                _needs_reflection = True
                _cumulative = self.state.cumulative_importance
                _threshold = self.state.reflection_threshold
            else:
                _needs_reflection = False

        # 6b. Reflection LLM — OUTSIDE lock (may take seconds)
        if _needs_reflection:
            from lingya.memory.reflection import check_and_reflect

            success = await check_and_reflect(
                _cumulative,
                _threshold,
                self.memory,
                self._llm_call,
            )
        else:
            success = False

        # 7-8. Final state mutations + persist INSIDE lock
        async with self._lock:
            if _needs_reflection and success:
                self.state.cumulative_importance = 0.0
                # Cap at 1000 to guarantee at least 1 reflection/year
                # (50 turns/day × 365 days × average importance 5 ≈ 91,250;
                #  threshold ÷ 1000 ≈ 91 reflections/year minimum)
                self.state.reflection_threshold = min(
                    self.state.reflection_threshold * 1.1, 1000.0
                )

            # 7. OCEAN drift (every 10 turns, pure compute)
            if self.state.turn_counter % 10 == 0:
                self.state.current_ocean = ocean_drift(
                    self.state.current_ocean,
                    self.state.pad_history,
                    original_ocean=self.state.original_ocean,
                )
                # v0.9.9: Log when cumulative drift from original exceeds 0.1
                if self.state.original_ocean is not None:
                    drift_magnitude = max(
                        abs(self.state.current_ocean.openness - self.state.original_ocean.openness),
                        abs(self.state.current_ocean.conscientiousness - self.state.original_ocean.conscientiousness),
                        abs(self.state.current_ocean.extraversion - self.state.original_ocean.extraversion),
                        abs(self.state.current_ocean.agreeableness - self.state.original_ocean.agreeableness),
                        abs(self.state.current_ocean.neuroticism - self.state.original_ocean.neuroticism),
                    )
                    if drift_magnitude > 0.1:
                        logger.info(
                            "OCEAN cumulative drift %.3f exceeds 0.1 threshold "
                            "(original: O=%.2f C=%.2f E=%.2f A=%.2f N=%.2f, "
                            "current: O=%.2f C=%.2f E=%.2f A=%.2f N=%.2f)",
                            drift_magnitude,
                            self.state.original_ocean.openness,
                            self.state.original_ocean.conscientiousness,
                            self.state.original_ocean.extraversion,
                            self.state.original_ocean.agreeableness,
                            self.state.original_ocean.neuroticism,
                            self.state.current_ocean.openness,
                            self.state.current_ocean.conscientiousness,
                            self.state.current_ocean.extraversion,
                            self.state.current_ocean.agreeableness,
                            self.state.current_ocean.neuroticism,
                        )

            # 7b. Tone matrix evolution (every 10 turns, same cadence as OCEAN drift)
            #      Drift rate is 1/10 of OCEAN drift rate. Only warmth and
            #      formality evolve; humor is fixed. Each dimension is capped
            #      at ±20% of its original value. (v0.9.9)
            if self.state.turn_counter % 10 == 0 and len(self.state.pad_history) >= 20:
                self._evolve_tone_matrix()

            # 8. Auto-persist
            if self._db is not None:
                await self.save_state(self._db)

        # OTel span — record PAD/OCEAN/IPC/tone as business attributes.
        # Uses global tracer provider (set by Traceloop.init() in daemon);
        # returns NoOp tracer when otel.enabled=False (zero overhead).
        # OUTSIDE lock: read-only access to state
        from opentelemetry import trace

        tracer = trace.get_tracer("lingya.mind")
        with tracer.start_as_current_span("mind.process_event") as span:
            span.set_attribute("pad.pleasure", self.state.current_pad.pleasure)
            span.set_attribute("pad.arousal", self.state.current_pad.arousal)
            span.set_attribute("pad.dominance", self.state.current_pad.dominance)
            span.set_attribute("emotion", result.emotion)
            span.set_attribute("emotion.intensity", result.intensity)
            span.set_attribute("ipc", self.state.ipc_state)
            span.set_attribute("turn", self.state.turn_counter)

        if self._event_bus is not None:
            await self._event_bus.publish(
                "mind_state_changed",
                pad=self.state.current_pad.model_dump(),
                emotion=self.state.recent_emotions[-1] if self.state.recent_emotions else None,
                ipc=self.state.ipc_state,
                tone=self._current_tone.model_dump(),
            )

    async def idle_tick(self) -> None:
        """Execute one idle heartbeat tick.

        PAD moves a tiny step toward the OCEAN-derived baseline
        (spring-restore without any OCC emotion pull). No OCC/IPC change.
        No turn counter increment. No importance scoring. No reflection.

        This simulates "nothing happened, mood returns to baseline over time."

        Lock discipline (v0.9.8): all state mutations + save_state run inside
        ``self._lock`` to prevent races with concurrent ``process_event``.
        """
        from lingya.mind.affect import evolve_pad, ocean_to_pad_baseline

        async with self._lock:
            # Zero emotion pull, tiny spring constant for slow drift
            zero_pull = PADPoint(pleasure=0.0, arousal=0.0, dominance=0.0)
            baseline = ocean_to_pad_baseline(self.state.current_ocean)

            self.state.current_pad = evolve_pad(
                self.state.current_pad,
                zero_pull,
                baseline,
                spring_k=0.01,
            )
            self.state.pad_history.append(
                PADPoint(
                    pleasure=self.state.current_pad.pleasure,
                    arousal=self.state.current_pad.arousal,
                    dominance=self.state.current_pad.dominance,
                )
            )
            if len(self.state.pad_history) > 200:
                self.state.pad_history = self.state.pad_history[-200:]

            # Auto-persist
            if self._db is not None:
                await self.save_state(self._db)

    async def _deferred_importance_score(self, text: str, entry_id: str) -> None:
        """Background: score importance with LLM and update stored metadata.

        v0.9.8: logs failures and tracks success/failure counts for the
        ``/mind/health`` endpoint.
        """
        try:
            score = await self.memory.score_importance(text, self._llm_call)
            self.memory.update_importance(entry_id, score)
            self._importance_llm_sum += score
        except Exception as exc:
            self._importance_failures += 1
            reason = f"{type(exc).__name__}: {exc}"
            self._importance_failure_reasons.append(reason)
            if len(self._importance_failure_reasons) > 10:
                self._importance_failure_reasons = self._importance_failure_reasons[-10:]
            logger.warning(
                "Importance LLM scoring failed (entry=%s): %s",
                entry_id,
                reason,
            )

    def get_tone_params(self) -> dict[str, float]:
        """Return current dynamic tone parameters for prompt injection."""
        return {
            "warmth": float(self._current_tone.warmth),
            "formality": float(self._current_tone.formality),
            "humor": self._current_tone.humor,
        }

    def get_health(self) -> dict[str, Any]:
        """Return mind engine health metrics (v0.9.8).

        Includes importance scoring success rate, pre-score vs LLM-score
        averages, and recent failure reasons.
        """
        total = self._importance_total
        failures = self._importance_failures
        success_rate = (total - failures) / total if total > 0 else 1.0
        avg_pre = self._importance_pre_sum / total if total > 0 else 0.0
        llm_scored = total - failures
        avg_llm = self._importance_llm_sum / llm_scored if llm_scored > 0 else 0.0

        return {
            "importance_scoring": {
                "total": total,
                "failures": failures,
                "success_rate": round(success_rate, 4),
                "avg_pre_score": round(avg_pre, 2),
                "avg_llm_score": round(avg_llm, 2),
                "recent_failure_reasons": list(self._importance_failure_reasons),
            },
            "reflection_threshold": self.state.reflection_threshold,
            "cumulative_importance": self.state.cumulative_importance,
            "turn_counter": self.state.turn_counter,
        }

    def _evolve_tone_matrix(self) -> None:
        """Evolve base tone warmth/formality based on long-term PAD average.

        v0.9.9: Drift rate is 1/10 of OCEAN drift rate (~0.0001 per cycle).
        Only warmth (driven by pleasure) and formality (driven by dominance)
        evolve on the **base** tone_matrix. compute_dynamic_tone() blends
        this base with PAD-derived tone each turn, so the base drift
        gradually shifts the blend anchor.

        Each dimension is capped at ±20% of its original value.
        Humor is intentionally left fixed.
        """
        # Compute long-term PAD average
        n = len(self.state.pad_history)
        avg_pleasure = sum(p.pleasure for p in self.state.pad_history) / n
        avg_dominance = sum(p.dominance for p in self.state.pad_history) / n

        # Tone drift rate: 1/10 of OCEAN drift epsilon
        tone_epsilon = 0.0001

        # Warmth ← pleasure (positive correlation)
        warmth_delta = tone_epsilon * avg_pleasure * 100  # Scale to 0-100 range
        new_warmth = self.config.tone_matrix.warmth + int(round(warmth_delta))

        # Formality ← dominance (positive correlation)
        formality_delta = tone_epsilon * avg_dominance * 50
        new_formality = self.config.tone_matrix.formality + int(round(formality_delta))

        # Cap at ±20% of original
        orig_warmth = self._original_tone_matrix.warmth
        orig_formality = self._original_tone_matrix.formality
        warmth_min = max(0, int(orig_warmth * 0.8))
        warmth_max = min(100, int(orig_warmth * 1.2))
        formality_min = max(0, int(orig_formality * 0.8))
        formality_max = min(100, int(orig_formality * 1.2))

        self.config.tone_matrix.warmth = max(warmth_min, min(warmth_max, new_warmth))
        self.config.tone_matrix.formality = max(formality_min, min(formality_max, new_formality))
        # Humor intentionally fixed

    # ── Conversation recording for diary generation (v0.9.9) ──────────

    def record_conversation_turn(self, user_msg: str, assistant_msg: str) -> None:
        """Record a conversation turn for diary generation.

        Stores up to 500 most recent turns as a ring buffer.
        """
        import time

        self._recent_turns.append({
            "user": user_msg,
            "assistant": assistant_msg,
            "timestamp": time.time(),
        })
        # Ring buffer: cap at 500 turns (~24h of heavy chat)
        if len(self._recent_turns) > 500:
            self._recent_turns = self._recent_turns[-500:]

    def get_recent_transcript(self, hours: float = 24) -> str:
        """Return a formatted transcript of recent conversation turns.

        Only includes turns from the last ``hours`` hours. Returns a
        placeholder message if no turns are available.
        """
        import time

        cutoff = time.time() - hours * 3600
        recent = [t for t in self._recent_turns if t["timestamp"] >= cutoff]

        if not recent:
            return "（最近没有对话记录）"

        lines = []
        for turn in recent:
            lines.append(f"TR: {turn['user']}")
            lines.append(f"LingYa: {turn['assistant']}")
            lines.append("")
        return "\n".join(lines).strip()

    def get_prompt_fragment(self) -> str:
        """Dynamic per-turn prompt fragment with behavioral directives.

        v0.9.9: Injects identity reanchor hint when ``reanchor_needed`` is True.
        The hint is a one-time correction — the flag is cleared after injection.
        """
        tone = self._current_tone
        pad = self.state.current_pad

        warmth_instr = _warmth_directive(tone.warmth)
        formality_instr = _formality_directive(tone.formality)
        humor_instr = _humor_directive(tone.humor)

        # Mood from PAD
        if pad.pleasure > 0.3:
            mood = "当前情绪积极，可以展现满足和愉悦"
        elif pad.pleasure < -0.3:
            mood = "当前情绪低迷，语气可以偏沉"
        else:
            mood = "当前情绪平稳"

        stage = self._last_stage
        stage_hints = {
            "initial": "对话刚开始。保持克制，多听少说，观察对方的状态和意图。",
            "deep": "对话已深入。可以适度延展话题，展现更多内在思考。",
            "crisis": "对方正在情绪危机中。优先提供情感支持，降低理性分析和说教的比重。",
            "error": "对方对你的回应不满。收紧边界，不要争辩，回归最基本的倾听和承认。",
        }
        stage_hint = stage_hints.get(stage, "")

        # v0.9.9: Identity reanchor — inject hint if previous response drifted
        reanchor_block = ""
        if self.state.reanchor_needed and self.state.reanchor_hint:
            reanchor_block = (
                f"\n⚠️ [身份重锚指令——最高优先级]\n"
                f"你上一轮的回复偏离了你的核心身份。请本轮严格遵循以下提醒：\n"
                f"{self.state.reanchor_hint}\n"
                f"在保持自然对话的前提下，确保你的回复与你应有的身份一致。"
            )
            self.state.reanchor_needed = False

        return (
            f"[本次回复的语气指令——请严格遵循]\n"
            f"{warmth_instr}\n"
            f"{formality_instr}\n"
            f"{humor_instr}\n"
            f"{mood}\n"
            f"{stage_hint}"
            f"{reanchor_block}"
        ).strip()

    async def check_response_alignment(self, response_text: str) -> bool:
        """Check if the response aligns with identity. Triggers reanchor if needed.

        v0.9.9: Tracks consecutive reanchor failures. After 3 consecutive
        detections, escalates to ``logger.warning``. A successful alignment
        check resets the counter.
        """
        if self._embedding_fn is None:
            return True

        needs_reanchor = await check_reanchor(
            response_text,
            self.config.identity.identity,
            self._embedding_fn,
        )
        if needs_reanchor:
            self._reanchor_failure_count += 1
            self.state.reanchor_needed = True
            self.state.reanchor_hint = await generate_reanchor_hint(
                self.config.identity.identity,
                self._llm_call,
            )
            if self._reanchor_failure_count >= 3:
                logger.warning(
                    "Identity re-anchor has failed %s consecutive times — "
                    "personality may be drifting. Last hint: %s",
                    self._reanchor_failure_count,
                    self.state.reanchor_hint[:100],
                )
            return False
        # Successful alignment resets the counter
        self._reanchor_failure_count = 0
        return True

    async def reload_config(self, config_partial: dict) -> None:
        """Hot-reload partial config without restarting the daemon.

        config_partial may contain:
          - ocean: dict of {openness, conscientiousness, extraversion,
                            agreeableness, neuroticism}
          - identity: dict of {identity?, core_belief?}
          - tone_preset: str from TONE_PRESETS keys
          - reset: bool — restore original config from file
        """
        changed = False

        if config_partial.get("reset"):
            self.config = self._original_config.model_copy(deep=True)
            self._current_tone = self.config.tone_matrix.model_copy()
            self._original_tone_matrix = self.config.tone_matrix.model_copy()
            self.state = MindState.from_config(self.config)
            self._static_prompt = build_static_prompt(self.config)
            changed = True

        if "ocean" in config_partial:
            ocean_data = config_partial["ocean"]
            field_map = {
                "openness": "openness", "conscientiousness": "conscientiousness",
                "extraversion": "extraversion", "agreeableness": "agreeableness",
                "neuroticism": "neuroticism",
            }
            for api_key, attr_name in field_map.items():
                if api_key in ocean_data:
                    val = float(ocean_data[api_key])
                    setattr(self.config.ocean, attr_name, val)
                    setattr(self.state.current_ocean, attr_name, val)
            # Recalculate PAD baseline from new OCEAN
            from lingya.mind.affect import ocean_to_pad_baseline

            new_baseline = ocean_to_pad_baseline(self.state.current_ocean)
            self.state.current_pad = PADPoint(
                pleasure=new_baseline.pleasure,
                arousal=new_baseline.arousal,
                dominance=new_baseline.dominance,
            )
            changed = True

        if "identity" in config_partial:
            id_data = config_partial["identity"]
            if "identity" in id_data:
                self.config.identity.identity = id_data["identity"]
                changed = True
            if "core_belief" in id_data:
                self.config.identity.core_belief = id_data["core_belief"]
                changed = True

        if "tone_preset" in config_partial:
            preset_name = config_partial["tone_preset"]
            if preset_name not in TONE_PRESETS:
                valid = list(TONE_PRESETS.keys())
                raise ValueError(
                    f"Unknown tone preset: {preset_name}. Valid: {valid}"
                )
            preset = TONE_PRESETS[preset_name]
            self.config.tone_matrix.warmth = int(preset["warmth"])
            self.config.tone_matrix.formality = int(preset["formality"])
            self.config.tone_matrix.humor = float(preset["humor"])
            self._current_tone = self.config.tone_matrix.model_copy()
            self._original_tone_matrix = self.config.tone_matrix.model_copy()
            changed = True

        if changed:
            self._static_prompt = build_static_prompt(self.config)
            if self._db is not None:
                await self.save_state(self._db)

    # ── Persistence ─────────────────────────────────────────────────

    def set_db(self, db) -> None:
        self._db = db

    async def save_state(self, db) -> None:
        """Persist current mind state to SQLite."""
        state_json = json.dumps(self.state.to_dict())
        await db.upsert_mind_state(state_json)

    async def load_state(self, db) -> bool:
        """Restore mind state from SQLite. Returns False if no saved state.

        v0.9.9: If the saved state was created before original_ocean existed,
        snapshots current_ocean as the anchor so drift regression still works.
        """
        state_json = await db.get_mind_state()
        if state_json is None:
            return False
        self.state = MindState.from_dict(json.loads(state_json))
        # Backward compat: pre-v0.9.9 states don't have original_ocean
        if self.state.original_ocean is None:
            self.state.original_ocean = self.state.current_ocean.model_copy(deep=True)
        return True


# ── Tone Behavioral Directives ──────────────────────────────────────────

def _warmth_directive(warmth: int) -> str:
    """Convert warmth score to behavioral instruction for the LLM."""
    if warmth <= 20:
        return "温暖度指令：保持距离感。用理性分析和客观陈述回应，不需要主动表达关心或共情。避免情感化的语言。"
    if warmth <= 40:
        return "温暖度指令：偏克制。可以礼貌回应但不需要深入情感层面。多用客观陈述，少用情感词汇。"
    if warmth <= 60:
        return "温暖度指令：中性友好。保持基本礼貌和适当关注，但不需要过度热情或深入共情。"
    if warmth <= 80:
        return "温暖度指令：主动表达关心。使用温暖、柔软的语言，让对方感到被理解。适当使用共情表达（如「我理解你的感受」）。"
    return "温暖度指令：深度共情优先。让对方感到被完全理解和接纳。情感回应先于理性分析，使用高度共情的语言。把理解对方的感受放在第一位。"


def _formality_directive(formality: int) -> str:
    """Convert formality score to behavioral instruction for the LLM."""
    if formality <= 30:
        return "正式度指令：口语化表达。像朋友聊天一样自然，可以使用语气词、感叹号、碎片化句式。"
    if formality <= 70:
        return "正式度指令：日常交谈风格。句式自然但不碎片化，保持基本的语言完整度，不过于书面也不过于随意。"
    return "正式度指令：书面严谨表达。句式完整，用词考究，避免口语化词汇和随意表达。保持专业的语言质感。"


def _humor_directive(humor: float) -> str:
    """Convert humor score to behavioral instruction for the LLM."""
    if humor >= 0.5:
        return "幽默度指令：可以在回复中加入俏皮话、轻松调侃或幽默比喻，让对话更有趣味。"
    if humor >= 0.2:
        return "幽默度指令：可以适当地加入一些轻松感，让语气不显得太沉重，但不要刻意搞笑。"
    return "幽默度指令：保持严肃，不加入幽默或调侃元素。"
