const PRESETS: { key: string; label: string; desc: string }[] = [
  { key: 'warm', label: '温暖', desc: '主动关心，柔软共情' },
  { key: 'neutral', label: '中性', desc: '礼貌克制，不冷不热' },
  { key: 'cool', label: '清冷', desc: '保持距离，理性客观' },
  { key: 'passionate', label: '热烈', desc: '高度温暖，深度共情' },
  { key: 'gentle', label: '温柔', desc: '温和柔软，适度关心' },
]

interface Props {
  selected: string
  onSelect: (preset: string) => void
}

export function TonePresetPicker({ selected, onSelect }: Props) {
  return (
    <div className="grid grid-cols-2 gap-2">
      {PRESETS.map(({ key, label, desc }) => (
        <button
          key={key}
          onClick={() => onSelect(key)}
          className={`text-left p-3 rounded-[8px] transition-colors ${
            selected === key
              ? 'border-2 border-accent bg-accent/10'
              : 'border-2 border-transparent bg-surface-input hover:bg-surface-input/70'
          }`}
        >
          <div
            className="text-ink text-[12px] mb-0.5"
            style={{ fontVariationSettings: "'wght' 540" }}
          >
            {label}
          </div>
          <div
            className="text-ink-muted text-[11px]"
            style={{ fontVariationSettings: "'wght' 400" }}
          >
            {desc}
          </div>
        </button>
      ))}
    </div>
  )
}
