import { expect, test } from 'vitest'
import { referenceCheck } from './reference-check.js'

const refs = [{ id: 'start', file: 'start.png', order: 0, phase: 'start', view: 'side', viewSetBy: 'user' }]
const run = { id: 'check-1', status: 'done', readOnly: true }
const bundle = { runId: run.id, snapshot: { images: refs }, preflight: { ready: true, blockers: [], warnings: ['The middle position will be estimated.'] } }

test('a successful reference check enables building while preserving warnings', () => {
  expect(referenceCheck(refs, run, bundle)).toEqual({ state: 'ready', canBuild: true, blockers: [], warnings: bundle.preflight.warnings })
})

test('editing a phase invalidates an earlier successful reference check', () => {
  expect(referenceCheck([{ ...refs[0], phase: 'end' }], run, bundle)).toEqual({ state: 'stale', canBuild: false })
})

test('switching between photo reconstruction and a movement preset requires its own check', () => {
  const preset = { ...bundle, snapshot: { images: refs, motionPreset: 'squat-v1' } }
  expect(referenceCheck(refs, run, bundle, 'squat-v1')).toEqual({ state: 'stale', canBuild: false })
  expect(referenceCheck(refs, run, preset, null)).toEqual({ state: 'stale', canBuild: false })
  expect(referenceCheck(refs, run, preset, 'squat-v1').canBuild).toBe(true)
})

test('an explicitly checked preset can build without photographs', () => {
  const preset = { ...bundle, snapshot: { images: [], motionPreset: 'squat-v1' } }
  expect(referenceCheck([], run, preset, 'squat-v1').canBuild).toBe(true)
  expect(referenceCheck([], run, { ...preset, preflight: { ready: false, blockers: ['The preset clip changed.'] } }, 'squat-v1').canBuild).toBe(false)
})

test('a movement-sequence check stays tied to its exact reviewed bundle revision', () => {
  const selected = { id: 'walking-lunge-one-step', commit: 'revision-a' }
  const sequence = { ...bundle, snapshot: { images: refs, referenceBundle: selected } }
  expect(referenceCheck(refs, run, sequence, null, selected).canBuild).toBe(true)
  expect(referenceCheck(refs, run, sequence, null, null).state).toBe('stale')
  expect(referenceCheck(refs, run, sequence, null, { ...selected, commit: 'revision-b' }).state).toBe('stale')
})

test('a detector-filled view does not make the checked references stale', () => {
  const images = [{ ...refs[0], view: null, viewSetBy: null }]
  const current = [{ ...images[0], view: 'side', viewSetBy: 'detector' }]
  expect(referenceCheck(current, run, { ...bundle, snapshot: { images } }).canBuild).toBe(true)
})

test('missing, mismatched, and snapshotless checks never enable building', () => {
  expect(referenceCheck(refs, run, null).canBuild).toBe(false)
  expect(referenceCheck(refs, run, { ...bundle, runId: 'older' }).canBuild).toBe(false)
  expect(referenceCheck(refs, run, { ...bundle, snapshot: null }).canBuild).toBe(false)
})

test('a failed read still exposes its actionable preflight blockers', () => {
  const preflight = { ready: false, blockers: ['Add a front or behind view.'], warnings: [] }
  expect(referenceCheck(refs, { ...run, status: 'failed' }, { ...bundle, preflight })).toEqual({ state: 'blocked', canBuild: false, blockers: preflight.blockers, warnings: [] })
})

test('blocked or unfinished checks never enable building', () => {
  expect(referenceCheck(refs, run, { ...bundle, preflight: { ready: true, blockers: ['Missing input.'] } }).canBuild).toBe(false)
  expect(referenceCheck(refs, { ...run, status: 'queued' }, bundle).state).toBe('reading')
  expect(referenceCheck(refs, { ...run, status: 'running' }, { ...bundle, preflight: null }).state).toBe('reading')
  expect(referenceCheck(refs, { ...run, status: 'failed', error: { message: 'Python is unavailable.' } }, { ...bundle, preflight: null })).toEqual({ state: 'failed', canBuild: false, message: 'Python is unavailable.' })
})
