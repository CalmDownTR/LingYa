interface Props {
  identity: string
  coreBelief: string
  onChange: (field: 'identity' | 'core_belief', value: string) => void
}

export function IdentityEditor({ identity, coreBelief, onChange }: Props) {
  const inputClass =
    'w-full bg-surface-input text-ink rounded-[12px] px-4 py-3 border border-hairline ' +
    'focus:border-accent outline-none resize-none text-[14px] leading-relaxed ' +
    'placeholder:text-ink-muted transition-colors'

  return (
    <div className="space-y-4">
      <div>
        <label
          className="block text-ink-secondary text-[13px] mb-1.5"
          style={{ fontVariationSettings: "'wght' 460" }}
        >
          她是谁
        </label>
        <textarea
          value={identity}
          onChange={(e) => onChange('identity', e.target.value)}
          className={inputClass}
          rows={3}
          style={{ fontVariationSettings: "'wght' 460" }}
        />
      </div>
      <div>
        <label
          className="block text-ink-secondary text-[13px] mb-1.5"
          style={{ fontVariationSettings: "'wght' 460" }}
        >
          核心信念
        </label>
        <textarea
          value={coreBelief}
          onChange={(e) => onChange('core_belief', e.target.value)}
          className={inputClass}
          rows={3}
          style={{ fontVariationSettings: "'wght' 460" }}
        />
      </div>
    </div>
  )
}
