import { afterEach, expect, test } from 'vitest'
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { createStore } from './store.js'
import { presetDefinition } from './motion-presets.js'
import { commandFor, requiredOutputs } from './steps.js'

const roots = []
afterEach(() => { for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true }) })
const temp = () => { const root = mkdtempSync(path.join(os.tmpdir(), 'studio-motion-source-')); roots.push(root); return root }

test('the explicit pinned squat source is independent from uploaded photo views', () => {
  const store = createStore(temp())
  const image = store.addReference('barbell-back-squat', Buffer.from('context photo'), 'photo.jpg', 'jpg', { view: 'three-quarter' })
  const before = store.listReferences('barbell-back-squat')
  const run = store.createRun('barbell-back-squat', { motionPreset: 'squat-v1' })
  const snapshot = JSON.parse(readFileSync(path.join(store.runDir(run.slug, run.id), 'references.json')))
  expect(run.motionPreset).toBe('squat-v1')
  expect(snapshot.motionPreset).toBe('squat-v1')
  expect(snapshot.images[0]).toMatchObject({ id: image.id, view: 'three-quarter', contextOnly: false })
  expect(store.listReferences(run.slug)).toEqual(before)
})

test('unknown and cross-exercise presets cannot select an arbitrary clip', () => {
  const store = createStore(temp())
  expect(() => store.createRun('barbell-back-squat', { motionPreset: '../other.fbx' })).toThrow(/not available/)
  expect(() => store.createRun('pull-up', { motionPreset: 'squat-v1' })).toThrow(/not available/)
  expect(presetDefinition('Barbell Back Squat', 'squat-v1').sha256).toMatch(/^[a-f0-9]{64}$/)
})

test('preset execution never passes stale photograph-correction targets to Blender', () => {
  const store = createStore(temp())
  const run = store.createRun('barbell-back-squat', { motionPreset: 'squat-v1' })
  const dir = store.runDir(run.slug, run.id)
  writeFileSync(path.join(dir, 'preflight.json'), JSON.stringify({ ready: true, motionPreset: 'squat-v1' }))
  writeFileSync(path.join(dir, 'pipeline-targets.json'), JSON.stringify({ stale: true }))
  const command = commandFor('animate', run, dir, store.getExercise(run.slug))
  expect(command.args).toContain('--motion-preset')
  expect(command.args).toContain('squat-v1')
  expect(command.args).toContain('--reference-run')
  expect(command.args).not.toContain('--form-targets')
  expect(command.args).not.toContain('--keyposes')
  expect(requiredOutputs('stills', dir, { motionPreset: 'squat-v1', images: [{ id: 'context' }] })).toEqual(['ours.json'])
})

test('old photo-mode runs remain distinct and blocked preflight still stops a preset', () => {
  const store = createStore(temp())
  const photo = store.createRun('barbell-back-squat')
  expect(photo.motionPreset).toBeNull()
  const preset = store.createRun('barbell-back-squat', { motionPreset: 'squat-v1' })
  const dir = store.runDir(preset.slug, preset.id)
  writeFileSync(path.join(dir, 'preflight.json'), JSON.stringify({ ready: false, blockers: ['Pinned source unavailable.'] }))
  expect(commandFor('animate', preset, dir, store.getExercise(preset.slug))).toEqual({ error: 'Pinned source unavailable.' })
})
