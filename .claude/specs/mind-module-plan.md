# Mind Module Implementation Plan

## Context

Replace the static `lingya/persona/` module with a dynamic personality engine (`lingya/mind/`) that implements the multi-layer affect/cognition architecture from the referenced paper. The new module is a pure state machine — zero dependency on deepagents or LangGraph. The agent only consumes its output (tone parameters + prompt fragments).

## Outcome

`lingya/persona/` directory deleted. New `lingya/mind/` module with MindEngine as single entry point. `agent_config.yaml` upgraded to include OCEAN, PAD baseline, and expanded tone matrix. Agent gets dynamic per-turn tone and mood fragments instead of one static system prompt.

---

## File Structure (post-implementation)

```
lingya/mind/
    __init__.py          # exports: MindEngine, MindConfig, load_mind_config
    config.py            # MindConfig, BigFiveTraits, PADBaseline, ToneMatrix, IdentityAnchor
    state.py             # MindState, PADPoint (serializable runtime state)
    affect.py            # OCC 22-emotion decision tree + cognitive appraisal + PAD/OCEAN evolution
    tone.py              # stage detection + PAD→tone continuous mapping + stage deltas
    dynamics.py          # IPC dual-axis state machine (LLM few-shot IPC estimation)
    belief.py            # belief anchoring with OCEAN-modulated update probability
    guard.py             # implicit re-anchoring via cosine similarity monitoring
    engine.py            # MindEngine — coordinates all components, exposes process_event/get_tone_params/get_prompt_fragment

lingya/memory/
    __init__.py           # updated exports
    store.py              # ExtendedMemoryStore(MemoryStore) — adds importance scoring + weighted retrieval
    reflection.py         # async reflection tree: threshold trigger → questions → self-notions

lingya/storage/
    db.py                 # MODIFIED: add mind_state save/load methods
    migrations.py         # MODIFIED: add mind_state table migration

DELETED:
    lingya/persona/__init__.py
    lingya/persona/config.py
    lingya/persona/assembler.py
    lingya/persona/bucketing.py
```

---

## Phase 1: Config + State Models

**Files:** `lingya/mind/config.py`, `lingya/mind/state.py`

### `config.py`

```python
class BigFiveTraits(BaseModel):
    openness: float = Field(ge=0.0, le=1.0)
    conscientiousness: float = Field(ge=0.0, le=1.0)
    extraversion: float = Field(ge=0.0, le=1.0)
    agreeableness: float = Field(ge=0.0, le=1.0)
    neuroticism: float = Field(ge=0.0, le=1.0)

class PADBaseline(BaseModel):
    pleasure: float = Field(ge=-1.0, le=1.0)
    arousal: float = Field(ge=-1.0, le=1.0)
    dominance: float = Field(ge=-1.0, le=1.0)

class ToneMatrix(BaseModel):
    warmth: int = Field(ge=0, le=100)
    formality: int = Field(ge=0, le=100)
    humor: float = Field(ge=0.0, le=1.0)

class IdentityAnchor(BaseModel):
    identity: str
    core_belief: str

class MindConfig(BaseModel):
    version: str
    meta: PersonaMeta  # re-use from old model
    identity: IdentityAnchor
    ocean: BigFiveTraits
    pad_baseline: PADBaseline
    tone_matrix: ToneMatrix
    behavior_guardrails: list[str]
```

`load_mind_config(path)` — YAML loader. Detects old format (checks for `mind_core` key) and prints migration instructions.

**New `agent_config.yaml` format:**
```yaml
version: "2.0.0"
meta:
  agent_id: "companion_01"
  created_at: "2026-05-26"

identity:
  identity: "冷峻、克制的终身学术观察者"
  core_belief: "认为人类的情绪波荡是演化的必然..."

ocean:
  openness: 0.75
  conscientiousness: 0.80
  extraversion: 0.15
  agreeableness: 0.20
  neuroticism: 0.35

pad_baseline:
  pleasure: -0.1
  arousal: 0.3
  dominance: 0.6

tone_matrix:
  warmth: 15
  formality: 85
  humor: 0.05

behavior_guardrails:
  - "..."
```

### `state.py`

```python
class PADPoint(BaseModel):
    pleasure: float = Field(ge=-1.0, le=1.0)
    arousal: float = Field(ge=-1.0, le=1.0)
    dominance: float = Field(ge=-1.0, le=1.0)

class MindState(BaseModel):
    current_pad: PADPoint
    pad_history: list[PADPoint] = []
    current_ocean: BigFiveTraits
    recent_emotions: list[dict] = []
    turn_counter: int = 0
    ipc_agency: float = 0.5
    ipc_communion: float = 0.5
    ipc_state: str = "neutral"
    cumulative_importance: float = 0.0
    reflection_threshold: float = 150.0
    self_notions: list[str] = []
    reanchor_needed: bool = False
    reanchor_hint: str = ""

    @classmethod
    def from_config(cls, config: MindConfig) -> MindState: ...
    def to_dict(self) -> dict: ...
    @classmethod
    def from_dict(cls, d: dict) -> MindState: ...
```

**Verify:** `load_mind_config()` parses new YAML. `MindState.from_config()` produces valid state. Both round-trip serialize/deserialize.

---

## Phase 2: OCC Emotion Engine

**File:** `lingya/mind/affect.py` (OCC part)

22-emotion OCC decision tree — **deterministic, rule-based, no LLM for emotion label**.

Decision tree structure:
- Event outcome for self → joy/distress
- Event outcome for other → happy-for/resentment/gloating/sorry-for
- Prospect-based → hope/fear → satisfaction/relief/fears-confirmed/disappointment
- Agent attribution (self) → pride/shame
- Agent attribution (other) → admiration/reproach
- Compound: gratification/remorse/anger/gratitude
- Object attraction → love/hate
- Default: interest/surprise/disgust/neutral

PAD pull vectors for all 22 emotions (from Mehrabian literature, hardcoded dict).

```python
OCC_EMOTIONS: dict[str, tuple[float, float, float]]  # emotion → (P, A, D) pull

def occ_classify(event: dict) -> str: ...
def compute_intensity(w_goal: float, p_expected: float, e_residual: float = 1.0) -> float: ...

async def cognitive_appraisal(
    event: dict,
    llm_call: Callable[[str], Awaitable[str]]
) -> tuple[float, float]: ...  # returns (w_goal, p_expected)

async def occ_process(
    event: dict,
    llm_call: Callable[[str], Awaitable[str]]
) -> OCCResult: ...  # (emotion_label, intensity, scaled_pad_pull)
```

**Verify:** Unit test known events → correct emotion label. Intensity formula numerically correct.

---

## Phase 3: PAD + OCEAN Evolution

**File:** `lingya/mind/affect.py` (evolution part)

```python
def ocean_to_pad_baseline(ocean: BigFiveTraits) -> PADBaseline:
    """Mehrabian formulas (literature-derived coefficients)."""
    ...

def evolve_pad(
    current: PADPoint,
    occ_pull: PADPoint,
    baseline: PADBaseline,
    spring_k: float = 0.1,
) -> PADPoint:
    """
    new = current + pull_weight * occ_pull + (1-pull_weight) * spring_force
    spring_force = -k * (current - baseline)
    Clamp to [-1, 1].
    """
    ...

def ocean_drift(
    ocean: BigFiveTraits,
    pad_history: list[PADPoint],
    baseline: PADBaseline,
    epsilon: float = 0.001,
) -> BigFiveTraits:
    """Extreme long-term PAD deviation → tiny OCEAN adjustment."""
    ...
```

**Verify:** Apply known OCC pull, verify PAD moves correctly. Spring restores toward baseline. Very small epsilon produces negligible drift from short history.

---

## Phase 4: Enhanced Memory

**Files:** `lingya/memory/store.py` (extend), `lingya/memory/__init__.py`

Extend existing `MemoryStore` — NOT a rewrite. All existing methods unchanged.

```python
class EnhancedMemoryStore(MemoryStore):
    def store_with_importance(self, text: str, importance: float = 5.0) -> str: ...

    async def score_importance(
        self, text: str, llm_call: Callable[[str], Awaitable[str]]
    ) -> float: ...

    def search_weighted(
        self, query: str, top_k: int = 5, recency_lambda: float = 0.01
    ) -> list[dict]:
        """score = exp(-lambda * hours_since) × importance × cosine_similarity"""
        ...

    def get_cumulative_importance(self) -> float: ...
```

**Verify:** Existing MemoryStore tests still pass. Weighted search ranks high-importance recent items first.

---

## Phase 5: Reflection Tree

**File:** `lingya/memory/reflection.py`

```python
async def check_and_reflect(
    cumulative_importance: float,
    threshold: float,
    memory_store: EnhancedMemoryStore,
    llm_call: Callable[[str], Awaitable[str]],
) -> list[str]:
    """
    If cumulative >= threshold:
      1. Get recent high-importance memories
      2. LLM: generate 3 guiding questions
      3. For each question: search_weighted(top_k=20) → abstract 1-2 self-notions
      4. Inject self-notions as importance=9.0 memories
      5. Return self-notions
    """
    ...
```

**Verify:** Inject memories with total importance > 150, trigger, verify 5 self-notions generated and stored with high importance.

---

## Phase 6: Stage-Aware Tone Matrix

**File:** `lingya/mind/tone.py`

Replaces bucketing.py's if-elif with continuous PAD→tone mapping.

```python
class ConversationStage(Enum):
    INITIAL = "initial"        # turns 1-3
    DEEP = "deep"              # turns 4+
    CRISIS = "crisis"          # user extremely negative pleasure + high arousal
    ERROR = "error"            # recent reproach/anger toward agent

def detect_stage(turn_count: int, pad: PADPoint, recent_emotions: list[dict]) -> ConversationStage: ...

def pad_to_tone(pad: PADPoint) -> dict[str, float]:
    """
    dominance → formality (positive correlation)
    pleasure → warmth (positive correlation)  
    arousal → humor (inverted-U: moderate enables, extreme kills)
    Returns {warmth, formality, humor}
    """
    ...

def stage_tone_delta(stage: ConversationStage) -> dict[str, float]: ...

def compute_dynamic_tone(pad: PADPoint, stage: ConversationStage, base: ToneMatrix) -> ToneMatrix: ...
```

**Verify:** Unit test stage detection for each stage. PAD→tone mapping produces values in valid ranges.

---

## Phase 7: IPC State Machine

**File:** `lingya/mind/dynamics.py`

LLM few-shot estimates (agency, communion) from last 3 turns. No custom classifier.

```python
class IPCState(Enum):
    PROFESSIONAL_DEFENSE = "professional_defense"
    WARM_LISTENING = "warm_listening"
    CRISIS_INTERVENTION = "crisis_intervention"
    PLAYFUL_COLLABORATION = "playful_collaboration"
    NEUTRAL = "neutral"

IPC_TRANSITIONS: dict[IPCState, set[IPCState]]  # valid state transitions

async def estimate_ipc(
    last_turns: list[dict],
    llm_call: Callable[[str], Awaitable[str]]
) -> tuple[float, float]: ...  # (agency, communion)

def ipc_to_state(agency: float, communion: float) -> IPCState: ...
def next_ipc_state(current: IPCState, target: IPCState) -> IPCState: ...
```

**Verify:** (high agency, low communion) → PROFESSIONAL_DEFENSE. Transition matrix rejects invalid jumps.

---

## Phase 8: Belief Anchoring

**File:** `lingya/mind/belief.py`

```python
def belief_update_probability(ocean: BigFiveTraits) -> float:
    """High A → higher. High C → lower. Returns 0-1."""
    ...

async def belief_update_decision(
    anchor: BeliefAnchor,
    challenge: str,
    ocean: BigFiveTraits,
    llm_call,
) -> tuple[bool, str | None]: ...
```

**Verify:** Same challenge text → high agreeableness produces higher update probability than low agreeableness.

---

## Phase 9: Safety Guard

**File:** `lingya/mind/guard.py`

```python
async def check_reanchor(
    response_text: str,
    identity_kernel: str,
    embedding_fn: Callable[[str], list[float]],
    threshold: float = 0.3,
) -> bool: ...

async def generate_reanchor_hint(identity_kernel: str, llm_call) -> str: ...
```

**Verify:** Paraphrased identity → high cosine sim. "I agree completely, you're right!" vs cold academic → low sim, flag raised.

---

## Phase 10: MindEngine Integration

**File:** `lingya/mind/engine.py`

```python
class MindEngine:
    """Pure computation. Zero framework dependency. All LLM via injected callable."""

    def __init__(
        self,
        config: MindConfig,
        memory_store: EnhancedMemoryStore,
        llm_call: Callable[[str], Awaitable[str]],
        embedding_fn: Callable[[str], list[float]],
    ): ...

    async def process_event(self, event: dict) -> None:
        """Full pipeline: OCC → PAD → IPC → tone → importance → reflection check → drift → save"""
        ...

    def get_tone_params(self) -> dict[str, float]: ...
    def get_prompt_fragment(self) -> str: ...

    async def check_response_alignment(self, response_text: str) -> bool: ...

    # ─── Persistence ───
    def get_state(self) -> dict: ...
    def set_state(self, state_dict: dict) -> None: ...

    async def save_state(self, db: Database) -> None:
        """Persist current mind state to SQLite via db.upsert_mind_state()."""
        ...

    async def load_state(self, db: Database) -> bool:
        """Restore mind state from SQLite. Returns False if no saved state."""
        ...
```

Pipeline per event (in `process_event`):
1. Increment turn_counter
2. `occ_process()` — cognitive appraisal (LLM) + classify (rules) + intensity
3. `evolve_pad()` — apply OCC pull + spring force
4. `estimate_ipc()` (LLM) + `ipc_to_state()` + `next_ipc_state()`
5. `detect_stage()` → `compute_dynamic_tone()`
6. `score_importance()` (LLM) + `store_with_importance()`
7. `check_and_reflect()` if cumulative ≥ threshold (fire-and-forget)
8. Every N turns: `ocean_drift()`
9. `save_state(db)` — auto-persist after each event

**Verify:** Create MindEngine with mock LLM, process 3 events, verify `get_state()` shows updated PAD, turn counter, emotions. `get_tone_params()` returns valid ranges. `get_prompt_fragment()` contains mood description. `save_state()` + `load_state()` round-trip preserves all values.

### Database Changes for Persistence

**`lingya/storage/migrations.py`** — add migration #4:

```sql
CREATE TABLE IF NOT EXISTS mind_state (
    id INTEGER PRIMARY KEY CHECK (id = 1),  -- singleton row
    state_json TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
```

**`lingya/storage/db.py`** — add methods:

```python
async def upsert_mind_state(self, state_json: str) -> None:
    """INSERT or REPLACE the singleton mind state row."""
    ...

async def get_mind_state(self) -> str | None:
    """Return state_json or None if no saved state."""
    ...
```

Pipeline per event (in `process_event`):
1. Increment turn_counter
2. `occ_process()` — cognitive appraisal (LLM) + classify (rules) + intensity
3. `evolve_pad()` — apply OCC pull + spring force
4. `estimate_ipc()` (LLM) + `ipc_to_state()` + `next_ipc_state()`
5. `detect_stage()` → `compute_dynamic_tone()`
6. `score_importance()` (LLM) + `store_with_importance()`
7. `check_and_reflect()` if cumulative ≥ threshold (fire-and-forget)
8. Every N turns: `ocean_drift()`

**Verify:** Create MindEngine with mock LLM, process 3 events, verify `get_state()` shows updated PAD, turn counter, emotions. `get_tone_params()` returns valid ranges. `get_prompt_fragment()` contains mood description.

---

## Phase 11: Integration

### `main.py` changes
- Replace `load_persona_config` → `load_mind_config`
- Replace `PromptAssembler` → create `MindEngine`
- Replace `MemoryStore` → `EnhancedMemoryStore`
- Build static system prompt base (three principles + identity + guardrails + memory behavior — same structure minus the tone/style parts that are now dynamic)
- Build `llm_call` wrapper: `async def llm_call(prompt): result = await model.ainvoke([HumanMessage(content=prompt)]); return str(result.content)`
- After creating engine, call `await engine.load_state(db)` to restore from SQLite
- Pass engine + db to LingYaCLI

### Dynamic prompt injection strategy
Static base prompt stays in `create_deep_agent(system_prompt=...)`. Per-turn, `engine.get_prompt_fragment()` is prepended as a `SystemMessage` before the user's `HumanMessage`:

```python
# In LingYaCLI._invoke_agent():
fragment = self._engine.get_prompt_fragment()
messages = [HumanMessage(content=user_input)]
if fragment:
    messages.insert(0, SystemMessage(content=fragment))
result = await self.agent.ainvoke({"messages": messages}, config)
```

### `cli.py` changes
- `self._persona_config` → `self._engine: MindEngine`
- Add `self._db` reference for state persistence
- `_show_opening()` → read `engine.config.identity` instead of `persona_config.mind_core`
- `_maybe_generate_diary()` → pass `engine.config` instead of `persona_config`
- `_invoke_agent()`:
  - Before calling agent: `fragment = self._engine.get_prompt_fragment()` → prepend as SystemMessage
  - After agent response: `await self._engine.process_event(event)` for user message
  - Then: `await self._engine.process_event(event)` for AI response
  - Then: `await self._engine.check_response_alignment(response_text)`

### `lingya/reflection.py`, `lingya/diary.py` changes
- Change parameter type from `PersonaConfig` to `MindConfig`
- Access `.identity.identity` / `.identity.core_belief` instead of `.mind_core.*`

### Test changes
- **Delete:** `tests/test_persona_assembler.py`, `tests/test_persona_bucketing.py`
- **Update:** `tests/conftest.py` — `mind_config` fixture from new YAML
- **Update:** `tests/test_reflection.py`, `tests/test_diary.py` — use `MindConfig`
- **Update:** `tests/test_memory.py` — add `EnhancedMemoryStore` tests
- **Create:** `tests/test_mind_config.py`, `tests/test_mind_affect.py`, `tests/test_mind_tone.py`, `tests/test_mind_engine.py`

### `agent_config.yaml` migration
`load_mind_config()` detects old format (`mind_core` key present) → prints migration instructions and exits.

---

## Key Design Decisions

1. **MindEngine is a pure state machine** — takes `Callable[[str], Awaitable[str]]` for LLM calls, no model object dependency
2. **Mind state persisted to SQLite** — singleton row in `mind_state` table. `save_state()` auto-called after each `process_event()`. On startup, `load_state()` restores PAD/OCEAN/self-notions/importance accumulator. Enables true long-term personality evolution across sessions.
3. **OCC 22 emotions — full implementation, not simplified** — the decision tree is deterministic, same code complexity whether 6 or 22 emotions
4. **IPC estimation via LLM few-shot** — no custom classifier. Paper's ICC ~0.58-0.60; LLM approximation likely comparable
5. **Dynamic prompt via SystemMessage prepend** — avoids modifying deepagents internals
6. **Reflection tree is fire-and-forget** — `asyncio.create_task()`, doesn't block user interaction
7. **Existing MemoryStore extended, not rewritten** — backward compatible, all existing tests pass

---

## Verification (end-to-end)

1. `pytest tests/` — all tests pass
2. Start LingYa: `uv run python main.py` — agent starts, displays opening line
3. Chat 10+ turns — verify responses adapt tone (check warmth/formality variation)
4. `/diary` generates diary with correct style
5. `/memories` lists memories with importance scores
6. Old `agent_config.yaml` with `mind_core` key shows clear migration error
