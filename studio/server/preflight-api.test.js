import { afterEach, beforeEach, expect, test } from 'vitest'
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import http from 'node:http'
import { createStore } from './store.js'
import { studioApi } from './api.js'

let root, store, server, base, calls
const slug = 'barbell-back-squat'
beforeEach(async () => {
  root = mkdtempSync(path.join(os.tmpdir(), 'studio-preflight-api-'))
  store = createStore(root)
  calls = []
  const queue = {
    state: () => ({ running: null, queued: [] }),
    enqueueRead: (s) => { calls.push(s); return store.createRun(s, { readOnly: true }) },
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


test('fast reference checks are ordered by creation time, not their random ID suffix', () => {
  const first = store.createRun(slug, { readOnly: true })
  const second = store.createRun(slug, { readOnly: true })
  const byId = [first, second].sort((a, b) => a.id.localeCompare(b.id))
  store.writeRun({ ...byId[0], createdAt: '2026-10-24T13:00:00.200Z' })
  store.writeRun({ ...byId[1], createdAt: '2026-10-24T13:00:00.100Z' })
  expect(store.listRuns(slug)[0].id).toBe(byId[0].id)
})
