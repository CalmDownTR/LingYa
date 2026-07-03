import { Database, Brain, Pencil } from 'lucide-react'
import type { ProcessPhase, MemoryRecallPayload } from '../../types'

interface Props {
  phase: ProcessPhase
  memoryRecall: MemoryRecallPayload | null
}

const PHASE_CONFIG: Record<
  ProcessPhase,
  { icon: typeof Database; label: string; color: string }
> = {
  recalling: {
    icon: Database,
    label: '正在回忆...',
    color: 'text-amber-400',
  },
  thinking: {
    icon: Brain,
    label: '正在思考...',
    color: 'text-violet-400',
  },
  generating: {
    icon: Pencil,
    label: '正在生成...',
    color: 'text-emerald-400',
  },
}

export function PhaseIndicator({ phase, memoryRecall }: Props) {
  const config = PHASE_CONFIG[phase]
  const Icon = config.icon

  return (
    <div className="flex justify-start mb-2">
      <div
        className="flex items-center gap-2 px-4 py-2 rounded-[12px]
                   bg-surface-input/60 text-[13px] animate-in fade-in
                   slide-in-from-top-1 duration-300"
        style={{ fontVariationSettings: "'wght' 460" }}
      >
        <Icon size={14} className={`${config.color} animate-pulse`} />
        <span className="text-ink-secondary">{config.label}</span>
        {phase === 'recalling' && memoryRecall && memoryRecall.count > 0 && (
          <span className="text-ink-muted">
            · {memoryRecall.count} 条记忆
          </span>
        )}
      </div>
    </div>
  )
}
