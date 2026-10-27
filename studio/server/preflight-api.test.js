import { afterEach, beforeEach, expect, test } from 'vitest'
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import http from 'node:http'
import { createStore } from './store.js'
import { studioApi } from './api.js'
import { getOneStepBundledReferenceSpec } from './bundled-references.js'

let root, store, server, base, calls
const slug = 'barbell-back-squat'
beforeEach(async () => {
  root = mkdtempSync(path.join(os.tmpdir(), 'studio-preflight-api-'))
  store = createStore(root)
  calls = []
  const queue = {
    state: () => ({ running: null, queued: [] }),
    enqueueRead: (s, options) => { calls.push(s); return store.createRun(s, { ...options, readOnly: true }) },
    enqueue: (s, options) => { calls.push(s); return store.createRun(s, options) },
  }
  let middleware
  studioApi({ store, queue }).configureServer({ middlewares: { use: (f) => { middleware = f } } })
  server = http.createServer((req, res) => middleware(req, res, () => { res.statusCode = 404; res.end() }))
  await new Promise((r) => server.listen(0, '127.0.0.1', r))
  base = `http://127.0.0.1:${server.address().port}`
})
afterEach(async () => {
  await new Promise((r) => server.close(r))
  rmSync(root, { recursive: true, force: true })
})

test('reference read endpoint creates only a reference run and exposes its findings', async () => {
  const response = await fetch(`${base}/api/exercises/${slug}/read`, { method: 'POST' })
  expect(response.status).toBe(201)
  const run = await response.json()
  expect(run.readOnly).toBe(true)
  expect(calls).toEqual([slug])
  const preflight = { ready: false, blockers: ['Add a rear view.'], warnings: [], mode: 'Reference check' }
  writeFileSync(path.join(store.runDir(slug, run.id), 'preflight.json'), JSON.stringify(preflight))
  const detail = await (await fetch(`${base}/api/exercises/${slug}`)).json()
  expect(detail.latest.preflight).toEqual(preflight)
  expect(detail.latest.snapshot.images).toEqual([])
})

test('reference check cannot replace a completed animation score', async () => {
  store.addReference(slug, Buffer.from('fixture'), 'test.png', 'png')
  const full = store.createRun(slug)
  store.writeRun({ ...full, status: 'done', summary: { pass: 3, fail: 0 } })
  const preview = store.createRun(slug, { readOnly: true })
  store.writeRun({ ...preview, status: 'failed' })
  const exercises = await (await fetch(`${base}/api/exercises`)).json()
  expect(exercises.find((e) => e.slug === slug).status).toBe('passing')
})

test('reference endpoint preserves existence and same-origin guards', async () => {
  expect((await fetch(`${base}/api/exercises/missing/read`, { method: 'POST' })).status).toBe(404)
  expect((await fetch(`${base}/api/exercises/${slug}/read`, { method: 'POST', headers: { origin: 'http://unrelated.example' } })).status).toBe(403)
  expect(calls).toEqual([])
})

test('an explicit movement check snapshots the allowed preset without requiring or retagging photos', async () => {
  const before = store.listReferences(slug)
  const response = await fetch(`${base}/api/exercises/${slug}/read`, {
    method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ motionPreset: 'squat-v1' }),
  })
  expect(response.status).toBe(201)
  const run = await response.json()
  expect(run).toMatchObject({ motionPreset: 'squat-v1', readOnly: true })
  expect(store.listReferences(slug)).toEqual(before)
})

test('the API rejects unknown or cross-exercise movement sources before queueing', async () => {
  for (const [exercise, preset] of [[slug, '../other.fbx'], ['pull-up', 'squat-v1']]) {
    const response = await fetch(`${base}/api/exercises/${exercise}/read`, {
      method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ motionPreset: preset }),
    })
    expect(response.status).toBe(400)
  }
  expect(calls).toEqual([])
})

test('manual checks and builds preserve the exact selected observed sequence', async () => {
  const lunge = 'walking-lunge-dumbbells-optional'
  const source = getOneStepBundledReferenceSpec(store.getExercise(lunge).name)
  const referenceBundle = { id: source.id, commit: source.commit }
  for (const route of ['read', 'runs']) {
    const response = await fetch(`${base}/api/exercises/${lunge}/${route}`, {
      method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ referenceBundle }),
    })
    expect(response.status).toBe(201)
    expect((await response.json()).referenceBundle).toEqual(referenceBundle)
  }
  const rejected = await fetch(`${base}/api/exercises/${lunge}/read`, {
    method: 'POST', headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ referenceBundle: { ...referenceBundle, commit: '0'.repeat(64) } }),
  })
  expect(rejected.status).toBe(400)
  expect(calls).toEqual([lunge, lunge])
})


test('fast reference checks are ordered by creation time, not their random ID suffix', () => {
  const first = store.createRun(slug, { readOnly: true })
  const second = store.createRun(slug, { readOnly: true })
  const byId = [first, second].sort((a, b) => a.id.localeCompare(b.id))
  store.writeRun({ ...byId[0], createdAt: '2026-10-24T13:00:00.200Z' })
  store.writeRun({ ...byId[1], createdAt: '2026-10-24T13:00:00.100Z' })
  expect(store.listRuns(slug)[0].id).toBe(byId[0].id)
})
