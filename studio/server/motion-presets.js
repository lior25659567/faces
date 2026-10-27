import { createHash } from 'node:crypto'
import { readFileSync } from 'node:fs'
import path from 'node:path'
import { REPO } from './paths.js'

const PRESETS = JSON.parse(readFileSync(path.join(REPO, 'studio/measure/motion-presets.json'), 'utf8')).presets

export const presetDefinition = (exercise, id) => {
  if (id == null) return null
  const preset = typeof id === 'string' && Object.hasOwn(PRESETS, id) ? PRESETS[id] : null
  if (!preset || preset.exercise !== exercise) throw new Error('That movement preset is not available for this exercise.')
  const file = path.join(REPO, preset.clip)
  let sha
  try { sha = createHash('sha256').update(readFileSync(file)).digest('hex') } catch { throw new Error('The pinned movement clip is missing.') }
  if (sha !== preset.sha256) throw new Error('The movement clip changed. Restore the pinned clip before using this preset.')
  return { id, ...structuredClone(preset) }
}

export const presetsFor = (exercise) => Object.entries(PRESETS).filter(([, preset]) => preset.exercise === exercise)
  .map(([id, preset]) => ({ id, label: preset.label, source: preset.source }))

export const defaultPreset = (exercise) => presetsFor(exercise)[0]?.id ?? null
