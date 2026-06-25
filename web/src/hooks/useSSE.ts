import { useState, useCallback, useRef } from 'react'
import { getToken } from './useApi'

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8765'

interface SSECallbacks {
  onDelta?: (content: string) => void
  onEvent?: (event: { type: string; event: string; payload: Record<string, unknown> }) => void
  onComplete?: (response: { type: string; payload: { text: string; tone: Record<string, number> } }) => void
  onError?: (error: Error) => void
}

export function useSSE() {
  const [isStreaming, setIsStreaming] = useState(false)
  const [error, setError] = useState<Error | null>(null)
  const controllerRef = useRef<AbortController | null>(null)

  const startStream = useCallback(async (text: string, callbacks: SSECallbacks) => {
    setIsStreaming(true)
    setError(null)

    const controller = new AbortController()
    controllerRef.current = controller

    try {
      const token = getToken()
      const response = await fetch(`${API_BASE}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(token ? { 'Authorization': `Bearer ${token}` } : {}),
        },
        body: JSON.stringify({ text }),
        signal: controller.signal,
      })

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}))
        throw new Error(errData.payload?.message || `HTTP ${response.status}`)
      }

      const reader = response.body?.getReader()
      if (!reader) throw new Error('No response body')

      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6))
              if (data.type === 'event' && data.event === 'chat.delta') {
                callbacks.onDelta?.(data.payload.content)
              } else if (data.type === 'event') {
                callbacks.onEvent?.(data)
              } else if (data.type === 'chat_response') {
                callbacks.onComplete?.(data)
              } else if (data.type === 'error') {
                throw new Error(data.payload?.message || 'Unknown error')
              }
            } catch (e) {
              if (e instanceof SyntaxError) continue // skip parse errors
              throw e
            }
          }
        }
      }
    } catch (e) {
      if ((e as Error).name !== 'AbortError') {
        const err = e as Error
        setError(err)
        callbacks.onError?.(err)
      }
    } finally {
      setIsStreaming(false)
      controllerRef.current = null
    }
  }, [])

  const abort = useCallback(() => {
    controllerRef.current?.abort()
  }, [])

  return { startStream, abort, isStreaming, error }
}
