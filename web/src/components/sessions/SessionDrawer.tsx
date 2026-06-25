import { X, Plus, Trash2 } from 'lucide-react'
import { useQueryClient } from '@tanstack/react-query'
import { useSessions, useNewSession, useSwitchSession, useDeleteSession } from '../../lib/api'
import type { SessionInfo } from '../../types'

interface Props {
  open: boolean
  onClose: () => void
  onSessionChange?: (threadId: string) => void
}

export function SessionDrawer({ open, onClose, onSessionChange }: Props) {
  const { data: sessionsData } = useSessions()
  const newSession = useNewSession()
  const switchSession = useSwitchSession()
  const deleteSession = useDeleteSession()
  const qc = useQueryClient()

  const sessions = sessionsData?.payload?.sessions ?? []

  if (!open) return null

  const handleNew = () => {
    newSession.mutate(undefined, {
      onSuccess: (data) => {
        if (data?.payload?.thread_id) {
          onSessionChange?.(data.payload.thread_id)
          onClose()
        }
      },
    })
  }

  const handleSwitch = (threadId: string) => {
    // Optimistic update — mark clicked session as current immediately
    qc.setQueryData(['sessions'], (old: unknown) => {
      if (!old || typeof old !== 'object') return old
      const typed = old as { type: string; payload: { action: string; sessions: SessionInfo[] } }
      return {
        ...typed,
        payload: {
          ...typed.payload,
          sessions: typed.payload.sessions.map((s: SessionInfo) => ({
            ...s,
            is_current: s.thread_id === threadId,
          })),
        },
      }
    })

    switchSession.mutate(threadId, {
      onSuccess: () => {
        onSessionChange?.(threadId)
        onClose()
      },
    })
  }

  return (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 z-40"
        style={{ backgroundColor: 'rgba(26, 24, 23, 0.6)' }}
        onClick={onClose}
      />

      {/* Drawer */}
      <div
        className="fixed left-0 top-0 bottom-0 z-50 w-[280px] bg-surface-elevated border-r border-hairline
                   flex flex-col shadow-lg"
      >
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-4 border-b border-hairline-soft">
          <span
            className="text-ink text-[14px]"
            style={{ fontVariationSettings: "'wght' 540" }}
          >
            会话
          </span>
          <button
            onClick={handleNew}
            className="bg-accent text-ink-on-accent rounded-[8px] px-3 py-1.5
                       hover:bg-accent-hover transition-colors flex items-center gap-1.5"
            style={{ fontVariationSettings: "'wght' 540", fontSize: 13 }}
          >
            <Plus size={14} />
            新建
          </button>
        </div>

        {/* Session list */}
        <div className="flex-1 overflow-y-auto py-1">
          {sessions.map((s: SessionInfo) => (
            <div
              key={s.thread_id}
              onClick={() => {
                if (!s.is_current) {
                  handleSwitch(s.thread_id)
                }
              }}
              className={`group flex items-center justify-between px-4 py-3 cursor-pointer
                         hover:bg-surface-input transition-colors border-l-[3px] ${
                           s.is_current ? 'border-accent bg-surface-input/50' : 'border-transparent'
                         }`}
            >
              <div className="min-w-0 flex-1">
                <div
                  className="text-ink text-[14px] truncate"
                  style={{ fontVariationSettings: "'wght' 460" }}
                >
                  {s.label}
                </div>
                <div
                  className="text-ink-muted text-[11px] mt-0.5"
                  style={{ fontVariationSettings: "'wght' 400" }}
                >
                  {s.message_count} 条消息
                </div>
              </div>
              {!s.is_current && (
                <button
                  onClick={(e) => {
                    e.stopPropagation()
                    deleteSession.mutate(s.thread_id)
                  }}
                  className="opacity-0 group-hover:opacity-100 text-ink-muted hover:text-error
                             transition-all p-1"
                >
                  <Trash2 size={14} />
                </button>
              )}
            </div>
          ))}
          {sessions.length === 0 && (
            <div
              className="text-ink-muted text-[13px] text-center py-8"
              style={{ fontVariationSettings: "'wght' 460" }}
            >
              还没有会话 — 开始对话后会自动创建
            </div>
          )}
        </div>

        {/* Close button */}
        <button
          onClick={onClose}
          className="absolute top-4 right-3 text-ink-muted hover:text-ink transition-colors p-1"
        >
          <X size={18} />
        </button>
      </div>
    </>
  )
}
