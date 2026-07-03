import { useState, useRef, useEffect, useCallback, useMemo } from 'react'
import { Menu, Settings, AlertCircle, RotateCw } from 'lucide-react'
import { useSSE } from '../../hooks/useSSE'
import { useCurrentSession, useSessionHistory } from '../../lib/api'
import { MessageBubble } from './MessageBubble'
import { ChatInput } from './ChatInput'
import { SessionDrawer } from '../sessions/SessionDrawer'
import { SettingsPanel } from '../settings/SettingsPanel'
import { PhaseIndicator } from './PhaseIndicator'
import type { ChatMessage } from '../../types'
import type { ProcessPhase, ProcessPhasePayload, MemoryRecallPayload } from '../../types'

export function ChatWindow() {
  const [streamingContent, setStreamingContent] = useState('')
  const [sentMessages, setSentMessages] = useState<ChatMessage[]>([])
  const [currentThreadId, setCurrentThreadId] = useState<string | null>(null)
  const [drawerOpen, setDrawerOpen] = useState(false)
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [currentPhase, setCurrentPhase] = useState<ProcessPhase | null>(null)
  const [memoryRecall, setMemoryRecall] = useState<MemoryRecallPayload | null>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const { startStream, isStreaming, abort } = useSSE()
  // Track the last thread_id we synced FROM the server, so we can detect
  // external session changes (e.g. another tab switched, daemon restarted
  // and restored a different session) without clobbering the user's own
  // active switch before the server has caught up.
  const lastSyncedServerTidRef = useRef<string | null>(null)

  // Fetch current session on mount to discover initial thread_id
  const { data: currentData } = useCurrentSession()

  // Fetch history whenever threadId changes (staleTime:0 → refetch on every switch)
  const { data: historyData, isError: historyError, refetch: refetchHistory } =
    useSessionHistory(currentThreadId)

  // Sync currentThreadId from server current session.
  // - First load: always sync (currentThreadId === null)
  // - Server-side change: only sync if the user hasn't actively switched
  //   since the last server sync (currentThreadId === lastSyncedServerTidRef).
  //   This prevents stale server responses from clobbering an in-flight
  //   optimistic switch while still catching external changes.
  useEffect(() => {
    const tid = currentData?.payload?.session?.thread_id
    if (!tid) return

    const firstLoad = currentThreadId === null
    const localMatchesLastSync = currentThreadId === lastSyncedServerTidRef.current
    const serverChanged = lastSyncedServerTidRef.current !== null &&
                          lastSyncedServerTidRef.current !== tid

    if (firstLoad || (serverChanged && localMatchesLastSync)) {
      setCurrentThreadId(tid)
      if (serverChanged) {
        // External change — clear any locally-sent messages that belong
        // to the previous session so they don't bleed into the new one.
        setSentMessages([])
      }
      lastSyncedServerTidRef.current = tid
    }
  }, [currentData, currentThreadId])

  // Derive history messages from query data — let refetch results flow through
  // naturally. No ref guard; React Query's queryKey already scopes per thread.
  // NOTE: the backend /session/history payload has no timestamp field, and the
  // UI does not render timestamps, so we intentionally omit it here. Do not
  // call Date.now() during render to fabricate one (react-hooks/purity).
  const historyMessages = useMemo<ChatMessage[]>(() => {
    const msgs = historyData?.payload?.messages
    if (!msgs || !currentThreadId) return []
    return msgs.map((msg, i) => ({
      id: `hist-${currentThreadId}-${i}`,
      role: msg.role,
      content: msg.content,
    }))
  }, [historyData, currentThreadId])

  // Combine history + newly sent messages in the current session.
  // Filter out empty/non-string messages so they don't render as tiny colored pills.
  // The API can return content as null/non-string, so guard with typeof.
  const messages = useMemo<ChatMessage[]>(
    () =>
      [...historyMessages, ...sentMessages].filter(
        (msg) => typeof msg.content === 'string' && msg.content.trim().length > 0,
      ),
    [historyMessages, sentMessages],
  )

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [])

  useEffect(() => {
    scrollToBottom()
  }, [messages, streamingContent, scrollToBottom])

  const handleSessionChange = useCallback((threadId: string) => {
    abort()
    setCurrentThreadId(threadId)
    setSentMessages([])
    setStreamingContent('')
    // Mark this as a user-initiated switch — don't treat the next server
    // refetch (which may briefly return the old tid) as an external change.
    lastSyncedServerTidRef.current = threadId
  }, [abort])

  const handleSend = useCallback(
    async (text: string) => {
      const userMsg: ChatMessage = {
        id: `user-${Date.now()}`,
        role: 'user',
        content: text,
        timestamp: Date.now(),
      }
      setSentMessages((prev) => [...prev, userMsg])
      setStreamingContent('')
      setCurrentPhase(null)
      setMemoryRecall(null)

      let herContent = ''

      await startStream(text, {
        onDelta: (content) => {
          herContent += content
          setStreamingContent(herContent)
        },
        onEvent: (event) => {
          if (event.event === 'process.phase') {
            const payload = event.payload as unknown as ProcessPhasePayload
            setCurrentPhase(payload.phase)
          } else if (event.event === 'memory.recall') {
            const payload = event.payload as unknown as MemoryRecallPayload
            setMemoryRecall(payload)
          }
        },
        onComplete: (response) => {
          const herMsg: ChatMessage = {
            id: `her-${Date.now()}`,
            role: 'her',
            content: response.payload.text || herContent,
            timestamp: Date.now(),
          }
          setSentMessages((prev) => [...prev, herMsg])
          setStreamingContent('')
          setCurrentPhase(null)
          setMemoryRecall(null)
        },
        onError: (err) => {
          console.error('Chat error:', err)
          if (herContent) {
            const herMsg: ChatMessage = {
              id: `her-${Date.now()}`,
              role: 'her',
              content: herContent,
              timestamp: Date.now(),
            }
            setSentMessages((prev) => [...prev, herMsg])
          }
          setStreamingContent('')
          setCurrentPhase(null)
          setMemoryRecall(null)
        },
      })
    },
    [startStream],
  )

  return (
    <div className="h-full flex flex-col bg-canvas">
      {/* Top bar */}
      <header
        className="flex-shrink-0 flex items-center justify-between px-4 py-3
                   bg-canvas border-b border-hairline-soft"
      >
        <button
          onClick={() => setDrawerOpen(true)}
          className="text-ink-muted hover:text-ink transition-colors p-1"
          aria-label="会话列表"
        >
          <Menu size={20} />
        </button>
        <span
          className="text-ink-secondary text-[14px]"
          style={{ fontVariationSettings: "'wght' 540" }}
        >
          LingYa{currentThreadId ? ` · ${currentThreadId.slice(-8)}` : ''}
        </span>
        <button
          onClick={() => setSettingsOpen(true)}
          className="text-ink-muted hover:text-ink transition-colors p-1"
          aria-label="设置"
        >
          <Settings size={20} />
        </button>
      </header>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto bg-surface px-4 py-6">
        <div className="max-w-2xl mx-auto">
          {/* History load error — explicit feedback so users don't see blank */}
          {historyError && (
            <div className="flex flex-col items-center justify-center min-h-[200px] gap-3">
              <AlertCircle size={28} className="text-error" />
              <p
                className="text-error text-[14px]"
                style={{ fontVariationSettings: "'wght' 500" }}
              >
                聊天记录加载失败
              </p>
              <button
                onClick={() => refetchHistory()}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-[8px]
                           border border-hairline text-ink-secondary text-[13px]
                           hover:bg-surface-input transition-colors"
                style={{ fontVariationSettings: "'wght' 460" }}
              >
                <RotateCw size={14} />
                重试
              </button>
            </div>
          )}

          {!historyError && messages.length === 0 && !streamingContent && (
            <div className="flex items-center justify-center h-full min-h-[200px]">
              <p
                className="text-ink-muted text-[16px]"
                style={{ fontVariationSettings: "'wght' 460" }}
              >
                开始对话...
              </p>
            </div>
          )}

          {messages.map((msg) => (
            <MessageBubble key={msg.id} message={msg} />
          ))}

          {/* Process phase indicator — shows what LingYa is doing before/during streaming */}
          {isStreaming && currentPhase && (
            <PhaseIndicator
              phase={currentPhase}
              memoryRecall={memoryRecall}
            />
          )}

          {streamingContent && (
            <MessageBubble
              message={{
                id: 'streaming',
                role: 'her',
                content: streamingContent,
              }}
              isStreaming
            />
          )}

          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Input */}
      <ChatInput onSend={handleSend} disabled={isStreaming} />

      {/* Drawers */}
      <SessionDrawer
        open={drawerOpen}
        onClose={() => setDrawerOpen(false)}
        onSessionChange={handleSessionChange}
      />
      <SettingsPanel open={settingsOpen} onClose={() => setSettingsOpen(false)} />
    </div>
  )
}
