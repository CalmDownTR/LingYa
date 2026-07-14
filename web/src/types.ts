/** OCEAN personality dimensions */
export interface OceanValues {
  openness: number
  conscientiousness: number
  extraversion: number
  agreeableness: number
  neuroticism: number
}

/** Short-key OCEAN for API */
export interface OceanRequest {
  O: number
  C: number
  E: number
  A: number
  N: number
}

/** Tone matrix */
export interface ToneValues {
  warmth: number
  formality: number
  humor: number
}

/** Identity */
export interface IdentityValues {
  identity: string
  core_belief: string
}

/** Settings response from GET /settings */
export interface SettingsResponse {
  type: string
  payload: {
    ocean: OceanValues
    tone: ToneValues
    identity: IdentityValues
    available_presets: string[]
  }
}

/** Session info */
export interface SessionInfo {
  thread_id: string
  label: string
  message_count: number
  /** LangGraph checkpoint_id of the last activity — time-ordered UUID (v1) */
  last_activity?: string
  is_current: boolean
}

/** Chat message.
 *
 * `timestamp` is optional because the backend `/session/history` payload
 * does not include timestamps, and the UI currently does not render them.
 * Callers that actually know the time (e.g. an event handler) may still
 * populate it; render-phase code must not call Date.now() to fabricate one.
 */
export interface ChatMessage {
  id: string
  role: 'her' | 'user'
  content: string
  timestamp?: number
}

/** SSE event from the server */
export interface LingYaEvent {
  type: 'event'
  event: string
  payload: Record<string, unknown>
}

/** Process phase from LingYaInnerProcessTransformer */
export type ProcessPhase = 'recalling' | 'thinking' | 'generating'

/** process.phase event payload */
export interface ProcessPhasePayload {
  phase: ProcessPhase
}

/** memory.recall event payload */
export interface MemoryRecallPayload {
  count: number
  top_match: string
}

/** Chat response from SSE */
export interface ChatResponsePayload {
  type: 'chat_response'
  payload: {
    text: string
    tone: { warmth: number; formality: number; humor: number }
  }
}
