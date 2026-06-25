import { useState, useRef, useEffect, useCallback } from 'react'
import { Menu, Settings } from 'lucide-react'
import { useSSE } from '../../hooks/useSSE'
import { useCurrentSession, useSessionHistory } from '../../lib/api'
import { MessageBubble } from './MessageBubble'
import { ChatInput } from './ChatInput'
import { SessionDrawer } from '../sessions/SessionDrawer'
import { SettingsPanel } from '../settings/SettingsPanel'
import type { ChatMessage } from '../../types'

export function ChatWindow() {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [streamingContent, setStreamingContent] = useState('')
  const [currentThreadId, setCurrentThreadId] = useState<string | null>(null)
  const [drawerOpen, setDrawerOpen] = useState(false)
  const [settingsOpen, setSettingsOpen] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const { startStream, isStreaming, abort } = useSSE()
  const historyLoadedRef = useRef<string | null>(null)

  // Fetch current session on mount to discover initial thread_id
  const { data: currentData } = useCurrentSession()

  // Fetch history whenever threadId changes
  const { data: historyData } = useSessionHistory(currentThreadId)

  // On mount: set initial threadId from current session
  useEffect(() => {
    const tid = currentData?.payload?.session?.thread_id
    if (tid && currentThreadId === null) {
      setCurrentThreadId(tid)
    }
  }, [currentData, currentThreadId])

  // When history loads for a new threadId, populate messages
  useEffect(() => {
    const msgs = historyData?.payload?.messages
    if (msgs && currentThreadId && historyLoadedRef.current !== currentThreadId) {
      const chatMessages: ChatMessage[] = msgs.map((msg, i) => ({
        id: `hist-${currentThreadId.slice(-8)}-${i}`,
        role: msg.role,
        content: msg.content,
        timestamp: Date.now() - (msgs.length - i) * 1000,
      }))
      setMessages(chatMessages)
      setStreamingContent('')
      historyLoadedRef.current = currentThreadId
    }
  }, [historyData, currentThreadId])

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [])

  useEffect(() => {
    scrollToBottom()
  }, [messages, streamingContent, scrollToBottom])

  const handleSessionChange = useCallback((threadId: string) => {
    abort()
    setCurrentThreadId(threadId)
    setMessages([])
    setStreamingContent('')
    historyLoadedRef.current = null
  }, [abort])

  const handleSend = useCallback(
    async (text: string) => {
      const userMsg: ChatMessage = {
        id: `user-${Date.now()}`,
        role: 'user',
        content: text,
        timestamp: Date.now(),
      }
      setMessages((prev) => [...prev, userMsg])
      setStreamingContent('')

      let herContent = ''

      await startStream(text, {
        onDelta: (content) => {
          herContent += content
          setStreamingContent(herContent)
        },
        onComplete: (response) => {
          const herMsg: ChatMessage = {
            id: `her-${Date.now()}`,
            role: 'her',
            content: response.payload.text || herContent,
            timestamp: Date.now(),
          }
          setMessages((prev) => [...prev, herMsg])
          setStreamingContent('')
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
            setMessages((prev) => [...prev, herMsg])
          }
          setStreamingContent('')
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
          {messages.length === 0 && !streamingContent && (
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

          {streamingContent && (
            <MessageBubble
              message={{
                id: 'streaming',
                role: 'her',
                content: streamingContent,
                timestamp: Date.now(),
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
