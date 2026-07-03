import { useState } from 'react'
import { X } from 'lucide-react'
import { useSettings, useUpdateOcean, useUpdateIdentity, useUpdateTone, useResetSettings } from '../../lib/api'
import { OCEANSliders } from './OCEANSliders'
import { TonePresetPicker } from './TonePresetPicker'
import { IdentityEditor } from './IdentityEditor'
import type { OceanValues, OceanRequest } from '../../types'

interface Props {
  open: boolean
  onClose: () => void
}

export function SettingsPanel({ open, onClose }: Props) {
  const { data } = useSettings()
  const updateOcean = useUpdateOcean()
  const updateIdentity = useUpdateIdentity()
  const updateTone = useUpdateTone()
  const resetSettings = useResetSettings()

  const settings = data?.payload
  const [ocean, setOcean] = useState<OceanValues>({
    openness: 0.5, conscientiousness: 0.5, extraversion: 0.5,
    agreeableness: 0.5, neuroticism: 0.5,
  })
  const [identity, setIdentity] = useState({ identity: '', core_belief: '' })
  const [tonePreset, setTonePreset] = useState('warm')

  // Sync server settings into local form state using the "adjust state
  // during render" pattern (React docs: You Might Not Need an Effect).
  //
  // prevSettingsRef holds the last server payload we synced from. When the
  // query returns a new object (refetch, mutation invalidation, etc.) we
  // re-seed the local state. React Query keeps `data.payload` referentially
  // stable across renders when the data hasn't changed, so this comparison
  // is cheap and won't loop.
  const [prevSettings, setPrevSettings] = useState<typeof settings>(undefined)
  if (settings !== prevSettings) {
    setPrevSettings(settings)
    if (settings) {
      setOcean(settings.ocean)
      setIdentity(settings.identity)
      // Find matching preset from tone values.
      // NOTE: this reverse-inference is fragile (warm/passionate overlap) —
      // tracked as P1. Ideally the server returns the preset name directly.
      const t = settings.tone
      if (t.warmth >= 80 && t.formality <= 40) setTonePreset('warm')
      else if (t.warmth <= 25 && t.formality >= 75) setTonePreset('cool')
      else if (t.warmth >= 90) setTonePreset('passionate')
      else if (t.warmth >= 70) setTonePreset('gentle')
      else setTonePreset('neutral')
    }
  }

  if (!open) return null

  const handleSave = async () => {
    const oceanReq: OceanRequest = {
      O: ocean.openness, C: ocean.conscientiousness,
      E: ocean.extraversion, A: ocean.agreeableness, N: ocean.neuroticism,
    }
    // Run all three mutations in parallel — they are independent
    await Promise.all([
      updateOcean.mutateAsync(oceanReq),
      updateIdentity.mutateAsync(identity),
      updateTone.mutateAsync(tonePreset),
    ])
    onClose()
  }

  const handleReset = async () => {
    await resetSettings.mutateAsync()
  }

  return (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 z-40"
        style={{ backgroundColor: 'rgba(26, 24, 23, 0.6)' }}
        onClick={onClose}
      />

      {/* Panel */}
      <div className="fixed inset-0 z-50 flex items-start justify-center overflow-y-auto py-8">
        <div
          className="bg-surface-elevated rounded-[12px] w-full max-w-[480px] mx-4
                     border border-hairline shadow-2xl"
        >
          {/* Header */}
          <div className="flex items-center justify-between px-6 pt-6 pb-4">
            <h2
              className="text-ink m-0"
              style={{
                fontFamily: 'Inter Variable',
                fontSize: 24,
                fontVariationSettings: "'wght' 540",
                lineHeight: 1.15,
                letterSpacing: '-0.4px',
              }}
            >
              设置
            </h2>
            <button
              onClick={onClose}
              className="text-ink-muted hover:text-ink transition-colors p-1"
            >
              <X size={20} />
            </button>
          </div>

          {/* Content */}
          <div className="px-6 pb-6 space-y-6">
            {/* OCEAN */}
            <section>
              <h3
                className="text-ink-secondary text-[13px] mb-3"
                style={{ fontVariationSettings: "'wght' 540", letterSpacing: '0.02em' }}
              >
                人格维度
              </h3>
              <OCEANSliders values={ocean} onChange={setOcean} />
            </section>

            {/* Tone */}
            <section>
              <h3
                className="text-ink-secondary text-[13px] mb-3"
                style={{ fontVariationSettings: "'wght' 540", letterSpacing: '0.02em' }}
              >
                语气预设
              </h3>
              <TonePresetPicker selected={tonePreset} onSelect={setTonePreset} />
            </section>

            {/* Identity */}
            <section>
              <h3
                className="text-ink-secondary text-[13px] mb-3"
                style={{ fontVariationSettings: "'wght' 540", letterSpacing: '0.02em' }}
              >
                身份
              </h3>
              <IdentityEditor
                identity={identity.identity}
                coreBelief={identity.core_belief}
                onChange={(field, value) =>
                  setIdentity((prev) => ({ ...prev, [field]: value }))
                }
              />
            </section>
          </div>

          {/* Footer */}
          <div className="flex items-center justify-end gap-3 px-6 pb-6">
            <button
              onClick={handleReset}
              className="text-ink rounded-[8px] px-5 py-2.5 border border-hairline
                         hover:bg-surface-input transition-colors"
              style={{ fontVariationSettings: "'wght' 540", fontSize: 14 }}
            >
              重置默认
            </button>
            <button
              onClick={handleSave}
              disabled={updateOcean.isPending || updateIdentity.isPending}
              className="bg-accent text-ink-on-accent rounded-[8px] px-5 py-2.5
                         hover:bg-accent-hover disabled:opacity-40 transition-colors"
              style={{ fontVariationSettings: "'wght' 540", fontSize: 14 }}
            >
              保存
            </button>
          </div>
        </div>
      </div>
    </>
  )
}
