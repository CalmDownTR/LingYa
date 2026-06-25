import { useState, useRef, useEffect, type KeyboardEvent } from 'react'
import { Send } from 'lucide-react'

interface Props {
  onSend: (text: string) => void
  disabled?: boolean
}

export function ChatInput({ onSend, disabled }: Props) {
  const [text, setText] = useState('')
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  // Auto-resize
  useEffect(() => {
    const el = textareaRef.current
    if (el) {
      el.style.height = 'auto'
      el.style.height = `${Math.min(el.scrollHeight, 120)}px`
    }
  }, [text])

  const handleSend = () => {
    const trimmed = text.trim()
    if (!trimmed || disabled) return
    onSend(trimmed)
    setText('')
  }

  const handleKeyDown = (e: KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <div className="flex-shrink-0 px-4 pb-4 pt-2">
      <div
        className="flex items-end gap-3 bg-surface-input rounded-[12px] border border-hairline
                   focus-within:border-accent transition-colors px-4 py-3"
      >
        <textarea
          ref={textareaRef}
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="想说点什么..."
          disabled={disabled}
          rows={1}
          className="flex-1 bg-transparent text-ink placeholder:text-ink-muted resize-none
                     outline-none text-[16px] leading-relaxed min-h-[24px] max-h-[120px]"
          style={{ fontVariationSettings: "'wght' 460" }}
        />
        <button
          onClick={handleSend}
          disabled={disabled || !text.trim()}
          className="flex-shrink-0 bg-accent text-ink-on-accent rounded-[8px] px-4 py-2
                     hover:bg-accent-hover disabled:opacity-40 disabled:cursor-not-allowed
                     transition-colors flex items-center justify-center"
          style={{ fontVariationSettings: "'wght' 540", fontSize: 14 }}
        >
          <Send size={16} />
        </button>
      </div>
    </div>
  )
}
