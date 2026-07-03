import ReactMarkdown from 'react-markdown'
import type { ChatMessage } from '../../types'

interface Props {
  message: ChatMessage
  isStreaming?: boolean
}

export function MessageBubble({ message, isStreaming }: Props) {
  const isHer = message.role === 'her'

  // Don't render empty/non-string bubbles as tiny colored pills.
  // The API can return content as null/non-string, so guard with typeof.
  const content =
    typeof message.content === 'string' ? message.content.trim() : ''
  if (!content && !isStreaming) return null

  return (
    <div
      className={`flex ${isHer ? 'justify-start' : 'justify-end'} mb-2`}
    >
      <div
        className={`max-w-[80%] rounded-[min(18px,50%)] px-5 py-4 text-[16px] leading-relaxed ${
          isHer
            ? 'bg-bubble-her text-ink-on-accent rounded-bl-[4px]'
            : 'bg-bubble-user text-ink rounded-br-[4px]'
        }`}
        style={{ fontVariationSettings: "'wght' 460" }}
      >
        <div className="prose prose-sm prose-invert max-w-none">
          <ReactMarkdown
            components={{
              p: ({ children }) => <p className="m-0 mb-2.5 last:mb-0">{children}</p>,
              code: ({ children }) => (
                <code className="text-[13px] bg-white/10 rounded px-1 py-0.5">
                  {children}
                </code>
              ),
              pre: ({ children }) => (
                <pre className="text-[13px] bg-white/10 rounded-lg p-3 overflow-x-auto my-2">
                  {children}
                </pre>
              ),
            }}
          >
            {content || (isStreaming ? '...' : '')}
          </ReactMarkdown>
        </div>
        {isStreaming && isHer && (
          <span className="inline-block w-1.5 h-4 bg-ink-on-accent/60 rounded-sm ml-0.5 animate-pulse align-middle" />
        )}
      </div>
    </div>
  )
}
