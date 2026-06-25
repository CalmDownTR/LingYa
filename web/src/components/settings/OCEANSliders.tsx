import type { OceanValues } from '../../types'

const TRAITS: { key: keyof OceanValues; label: string }[] = [
  { key: 'openness', label: '开放性' },
  { key: 'conscientiousness', label: '尽责性' },
  { key: 'extraversion', label: '外向性' },
  { key: 'agreeableness', label: '宜人性' },
  { key: 'neuroticism', label: '神经质' },
]

interface Props {
  values: OceanValues
  onChange: (values: OceanValues) => void
}

export function OCEANSliders({ values, onChange }: Props) {
  const update = (key: keyof OceanValues, val: number) => {
    onChange({ ...values, [key]: val })
  }

  return (
    <div className="space-y-4">
      {TRAITS.map(({ key, label }) => (
        <div key={key} className="flex items-center gap-3">
          <span
            className="w-16 text-ink-secondary text-[13px] flex-shrink-0"
            style={{ fontVariationSettings: "'wght' 460" }}
          >
            {label}
          </span>
          <input
            type="range"
            min="0"
            max="1"
            step="0.01"
            value={values[key]}
            onChange={(e) => update(key, parseFloat(e.target.value))}
            className="flex-1 h-1.5 rounded-full appearance-none cursor-pointer"
            style={{
              background: `linear-gradient(to right, #7F77DD ${values[key] * 100}%, #33302c ${values[key] * 100}%)`,
              accentColor: '#7F77DD',
            }}
          />
          <span
            className="w-10 text-right text-ink-muted text-[11px] flex-shrink-0 tabular-nums"
            style={{ fontVariationSettings: "'wght' 400" }}
          >
            {values[key].toFixed(2)}
          </span>
        </div>
      ))}
    </div>
  )
}
