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
  is_current: boolean
}

/** Chat message */
export interface ChatMessage {
  id: string
  role: 'her' | 'user'
  content: string
  timestamp: number
}

/** SSE event from the server */
export interface LingYaEvent {
  type: 'event'
  event: string
  payload: Record<string, unknown>
}

/** Chat response from SSE */
export interface ChatResponsePayload {
  type: 'chat_response'
  payload: {
    text: string
    tone: { warmth: number; formality: number; humor: number }
    meta: { engine_ms: number }
  }
}
