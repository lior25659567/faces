import { afterEach, beforeEach, expect, test } from 'vitest'
import { mkdtemp, rm } from 'node:fs/promises'
import { mkdirSync, writeFileSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { createStore } from '../server/store.js'
import { RecheckError, pickRun, resolveSlug } from './recheck.mjs'

let root, store
beforeEach(async () => {
  root = await mkdtemp(path.join(os.tmpdir(), 'studio-recheck-'))
  store = createStore(root)
})
afterEach(() => rm(root, { recursive: true, force: true }))

// A run with a chosen id; `rendered` writes the files the check reads (so the run can be re-checked).
const mkRun = (slug, id, { rendered }) => {
  store.writeRun({ id, slug, status: 'done', steps: [] })
  if (!rendered) return
  const dir = store.runDir(slug, id)
  writeFileSync(path.join(dir, 'targets.json'), '{}')
  writeFileSync(path.join(dir, 'ours.json'), '{}')
  mkdirSync(path.join(dir, 'frames'), { recursive: true })
  writeFileSync(path.join(dir, 'frames', 'retarget.json'), '{}')
}

test('resolveSlug accepts a slug or a name, and lists the exercises for an unknown one', () => {
  expect(resolveSlug(store, 'barbell-back-squat')).toBe('barbell-back-squat')
  expect(resolveSlug(store, 'Barbell Back Squat')).toBe('barbell-back-squat')
  expect(() => resolveSlug(store, 'nope')).toThrow(RecheckError)
})

test('pickRun chooses the newest run that was actually rendered', () => {
  mkRun('barbell-back-squat', '20260101-000001-aaaa', { rendered: true })
  mkRun('barbell-back-squat', '20260101-000002-bbbb', { rendered: false }) // newer, but not rendered
  expect(pickRun(store, 'barbell-back-squat').id).toBe('20260101-000001-aaaa')
  mkRun('barbell-back-squat', '20260101-000003-cccc', { rendered: true })
  expect(pickRun(store, 'barbell-back-squat').id).toBe('20260101-000003-cccc')
})

test('pickRun takes a chosen run, and errors clearly on no runs, an unknown run, or an unrendered one', () => {
  expect(() => pickRun(store, 'barbell-back-squat')).toThrow(/No runs/)
  mkRun('barbell-back-squat', '20260101-000001-aaaa', { rendered: false })
  expect(() => pickRun(store, 'barbell-back-squat')).toThrow(/No rendered run/)
  expect(() => pickRun(store, 'barbell-back-squat', '20260101-000001-aaaa')).toThrow(/hasn't been rendered/)
  expect(() => pickRun(store, 'barbell-back-squat', 'nope')).toThrow(/no run "nope"/)
  mkRun('barbell-back-squat', '20260101-000002-bbbb', { rendered: true })
  expect(pickRun(store, 'barbell-back-squat', '20260101-000002-bbbb').id).toBe('20260101-000002-bbbb')
})
