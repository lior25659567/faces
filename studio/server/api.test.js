import { afterEach, beforeEach, expect, test } from 'vitest'
import http from 'node:http'
import { mkdtemp, rm, readFile } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { EventEmitter } from 'node:events'
import { createStore } from './store.js'
import { studioApi } from './api.js'
import { createWork } from './work.js'

const SAMPLE = new URL('../../3d-character/reference-squat/15.png', import.meta.url)
const PDF = new URL('./fixtures/minimal.pdf', import.meta.url)
let root, server, base, store, queued, work
beforeEach(async () => {
  root = await mkdtemp(path.join(os.tmpdir(), 'studio-api-'))
  store = createStore(root)
  work = createWork(root)
  queued = []
  const queue = {
    enqueue: (slug, extra) => { queued.push(slug); return store.createRun(slug, extra) },
    state: () => ({ running: null, queued: queued.map((slug) => ({ slug, runId: 'x' })) }),
  }
  let handler
  studioApi({ store, queue, work }).configureServer({ middlewares: { use: (fn) => (handler = fn) } })
  server = http.createServer((req, res) => handler(req, res, () => { res.statusCode = 404; res.end('next') }))
  await new Promise((r) => server.listen(0, '127.0.0.1', r))
  base = `http://127.0.0.1:${server.address().port}`
})
afterEach(async () => {
  server.closeAllConnections()
  await new Promise((r) => server.close(r))
  await rm(root, { recursive: true, force: true })
})

const send = (p, init) => fetch(base + p, init)
const post = async () =>
  (await send('/api/exercises/barbell-back-squat/references', {
    method: 'POST',
    headers: { 'x-filename': '15.png' },
    body: await readFile(SAMPLE),
  })).json()

test('lists exercises', async () => {
  const rows = await (await send('/api/exercises')).json()
  expect(rows).toHaveLength(14)
})

const raw = (rawPath, headers = {}) =>
  new Promise((resolve, reject) => {
    const url = new URL(base)
    const req = http.request({ host: url.hostname, port: url.port, path: rawPath, headers, agent: false }, (res) => {
      let body = ''
      res.on('data', (c) => (body += c))
      res.on('end', () => resolve({ status: res.statusCode, type: res.headers['content-type'], body }))
    })
    req.on('error', reject)
    req.end()
  })

test('exercise list carries a status', async () => {
  const status = async () => (await (await send('/api/exercises')).json()).find((e) => e.slug === 'barbell-back-squat').status
  expect(await status()).toBe('needs-reference')
  await post()
  expect(await status()).toBe('not-run')
  const run = store.createRun('barbell-back-squat')
  run.status = 'done'; run.summary = { pass: 4, fail: 0, limited: 0, uncertain: 0, info: 0 }
  store.writeRun(run)
  expect(await status()).toBe('passing')
  // a check that would be judged but wasn't measured on the character keeps it from passing (final fix 1)
  run.summary.unmeasured = 2; store.writeRun(run)
  expect(await status()).toBe('unmeasured')
  run.summary.fail = 1; store.writeRun(run)
  expect(await status()).toBe('failing')
  run.status = 'failed'; run.summary = null; store.writeRun(run)
  expect(await status()).toBe('failing')
  queued.push('barbell-back-squat')
  expect(await status()).toBe('running')
})

test('serves files inside the data folder', async () => {
  const img = await post()
  const ok = await send(`/files/references/barbell-back-squat/${img.file}`)
  expect(ok.status).toBe(200)
  expect(ok.headers.get('content-type')).toBe('image/png')
})

test.each([
  ['encoded slash', '/files/..%2f..%2fpackage.json'],
  ['encoded dots', '/files/%2e%2e/%2e%2e/package.json'],
  ['raw dots', '/files/../../package.json'],
  ['absolute path', '/files/%2fetc%2fpasswd'],
  ['absolute path after folder', '/files/references/%2f..%2f..%2f..%2fetc%2fpasswd'],
])('the file guard refuses %s', async (_, p) => {
  const res = await raw(p)
  expect(res.status).toBe(404)
  expect(JSON.parse(res.body)).toEqual({ error: 'Not found.' }) // the guard's body, not next()'s
})

test('uploads, edits, reorders, deletes and shows the exercise', async () => {
  const a = await post(), b = await post()
  const j = { 'content-type': 'application/json' }
  const patched = await (await send(`/api/exercises/barbell-back-squat/references/${a.id}`, { method: 'PATCH', headers: j, body: JSON.stringify({ phase: 'end' }) })).json()
  expect(patched.phase).toBe('end')
  await send('/api/exercises/barbell-back-squat/references/order', { method: 'PUT', headers: j, body: JSON.stringify({ ids: [b.id, a.id] }) })
  let ex = await (await send('/api/exercises/barbell-back-squat')).json()
  expect(ex.exercise.name).toBe('Barbell Back Squat')
  expect(ex.references.map((i) => i.id)).toEqual([b.id, a.id])
  expect(ex.latest).toBeNull()
  await send(`/api/exercises/barbell-back-squat/references/${a.id}`, { method: 'DELETE' })
  ex = await (await send('/api/exercises/barbell-back-squat')).json()
  expect(ex.references).toHaveLength(1)
})

test('starts a run through the queue and reads it back', async () => {
  await post()
  const run = await (await send('/api/exercises/barbell-back-squat/runs', { method: 'POST' })).json()
  expect(queued).toEqual(['barbell-back-squat'])
  const back = await (await send(`/api/runs/barbell-back-squat/${run.id}`)).json()
  expect(back.id).toBe(run.id)
  const state = await (await send('/api/state')).json()
  expect(state.queued).toHaveLength(1)
})

test('run API validates character proportions and passes only the chosen option', async () => {
  const sendMode = (body) => send('/api/exercises/barbell-back-squat/runs', {
    method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body),
  })
  const response = await sendMode({ characterFit: 'fitness', status: 'done', automationJobId: 'injected' })
  expect(response.status).toBe(201)
  const run = await response.json()
  expect(run).toMatchObject({ characterFit: 'fitness', status: 'queued' })
  expect(run.automationJobId).toBeUndefined()
  const bad = await sendMode({ characterFit: 'longer' })
  expect(bad.status).toBe(400)
  expect((await bad.json()).error).toMatch(/Character proportions/)
  expect(queued).toHaveLength(1)
})

test('refuses a video with its plain message and a 4xx', async () => {
  const res = await send('/api/exercises/barbell-back-squat/references', {
    method: 'POST',
    headers: { 'x-filename': 'a.mp4' },
    body: Buffer.concat([Buffer.from([0, 0, 0, 24]), Buffer.from('ftypmp42')]),
  })
  expect(res.status).toBe(415)
  expect((await res.json()).error).toBe("Videos aren't supported yet. Upload still images of the key positions.")
})

test('unknown exercise is a 404', async () => {
  expect((await send('/api/exercises/nope')).status).toBe(404)
})

const VIDEO = "Videos aren't supported yet. Upload still images of the key positions."
const big = (head = Buffer.alloc(0), mb = 30) => Buffer.concat([head, Buffer.alloc(mb * 1024 * 1024)])

test('a body over 25 MB gets the plain message, delivered', async () => {
  const res = await send('/api/exercises/barbell-back-squat/references', {
    method: 'POST', headers: { 'x-filename': 'huge.png' }, body: big(Buffer.from('89504e470d0a1a0a', 'hex')),
  })
  expect(res.status).toBe(413)
  expect((await res.json()).error).toBe('That picture is over 25 MB.')
  expect(store.listReferences('barbell-back-squat')).toHaveLength(0)
})

test('a large .mp4 gets the video message, delivered', async () => {
  const res = await send('/api/exercises/barbell-back-squat/references', {
    method: 'POST', headers: { 'x-filename': 'workout.mp4' }, body: big(),
  })
  expect(res.status).toBe(415)
  expect((await res.json()).error).toBe(VIDEO)
})

test('a large video with a neutral name is caught by its first bytes', async () => {
  const res = await send('/api/exercises/barbell-back-squat/references', {
    method: 'POST', headers: { 'x-filename': 'clip' }, body: big(Buffer.concat([Buffer.from([0, 0, 0, 24]), Buffer.from('ftypmp42')])),
  })
  expect(res.status).toBe(415)
  expect((await res.json()).error).toBe(VIDEO)
})

test('a PDF is refused as not a picture', async () => {
  const res = await send('/api/exercises/barbell-back-squat/references', {
    method: 'POST', headers: { 'x-filename': 'a.pdf' }, body: await readFile(PDF),
  })
  expect(res.status).toBe(415)
  expect((await res.json()).error).toBe("That file isn't a picture this Mac can open.")
})

test.each([
  [{ phase: 'later' }, /Phase/],
  [{ view: 'upside-down' }, /View/],
  [{ contextOnly: 'yes' }, /Context/],
  [null, /JSON object/],
  [[1], /JSON object/],
])('PATCH %j is a 400 and changes nothing', async (body, message) => {
  const a = await post()
  const res = await send(`/api/exercises/barbell-back-squat/references/${a.id}`, {
    method: 'PATCH', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body),
  })
  expect(res.status).toBe(400)
  expect((await res.json()).error).toMatch(message)
  expect(store.listReferences('barbell-back-squat')[0]).toMatchObject({ phase: 'start', phaseSetBy: 'guess', view: null })
})

test('errors map by kind: unknown picture 404, bad exercise 400, disk trouble 500', async () => {
  const j = { 'content-type': 'application/json' }
  expect((await send('/api/exercises/barbell-back-squat/references/img-00000000', { method: 'PATCH', headers: j, body: '{"phase":"end"}' })).status).toBe(404)
  expect((await send('/api/exercises', { method: 'POST', headers: j, body: '{"name":"  "}' })).status).toBe(400)
  const a = await post()
  const boom = Object.assign(Object.create(store), { updateReference: () => { throw new Error('disk full') } })
  store.updateReference = boom.updateReference
  const res = await send(`/api/exercises/barbell-back-squat/references/${a.id}`, { method: 'PATCH', headers: j, body: '{"phase":"end"}' })
  expect(res.status).toBe(500)
  expect((await res.json()).error).toBe('disk full')
})

test('GET /api/work lists the work items, running first, then newest', async () => {
  expect(await (await send('/api/work')).json()).toEqual([])
  const a = work.start({ title: 'Old fix', exercises: ['barbell-back-squat'] })
  const b = work.start({ title: 'Running fix', exercises: [] })
  const c = work.start({ title: 'Newest finished', exercises: [] })
  work.finish(a.id, 'done', 'ok')
  work.finish(c.id, 'failed', 'no')
  // Same-second ids: make the order of starts unambiguous.
  for (const [id, at] of [[a.id, '2026-10-22T10:00:00.000Z'], [b.id, '2026-10-22T09:00:00.000Z'], [c.id, '2026-10-22T11:00:00.000Z']]) {
    work.write({ ...work.read(id), startedAt: at })
  }
  const items = await (await send('/api/work')).json()
  expect(items.map((w) => w.title)).toEqual(['Running fix', 'Newest finished', 'Old fix'])
  expect(items[2]).toMatchObject({ status: 'done', summary: 'ok', exercises: ['barbell-back-squat'] })
})

test('the work watcher starts only once the server really listens, and stops when it closes', () => {
  const calls = []
  const httpServer = new EventEmitter()
  const watcher = { start: () => calls.push('start'), stop: () => calls.push('stop') }
  const queue = {
    recoverInterrupted: () => calls.push('recover'), restartInterrupted: () => calls.push('restart'), stop: () => calls.push('queue-stop'),
  }
  studioApi({ store, queue, work, watcher, onListening: () => calls.push('listening') })
    .configureServer({ httpServer, middlewares: { use: () => {} } })
  expect(calls).toEqual([]) // a Studio that loses the port never gets here
  httpServer.emit('listening')
  expect(calls).toEqual(['recover', 'restart', 'start', 'listening'])
  httpServer.emit('close')
  expect(calls).toEqual(['recover', 'restart', 'start', 'listening', 'stop', 'queue-stop'])
})

test('GET /api/work reads through the mtime cache: unchanged finished items are not read again', async () => {
  const a = work.start({ title: 'Finished', exercises: [] })
  work.finish(a.id, 'done', 'ok')
  work.start({ title: 'Running', exercises: [] })
  let reads = 0
  const read = work.read
  work.read = (id) => { reads += 1; return read(id) }
  expect(await (await send('/api/work')).json()).toHaveLength(2)
  expect(reads).toBe(2)
  await send('/api/work')
  expect(reads).toBe(3) // only the running item
})

test('GET /api/work returns every running item and the newest 50 others', async () => {
  for (let i = 0; i < 2; i++) work.start({ title: `Running ${i}`, exercises: [] })
  for (let i = 0; i < 52; i++) {
    const item = work.start({ title: `Old ${i}`, exercises: [] })
    work.write({ ...work.read(item.id), status: 'done', startedAt: `2026-10-01T10:${String(i).padStart(2, '0')}:00.000Z` })
  }
  const items = await (await send('/api/work')).json()
  expect(items).toHaveLength(52)
  expect(items.filter((w) => w.status === 'running')).toHaveLength(2)
  expect(items.at(-1).title).toBe('Old 2')
})

// ---- only the Studio's own page may change things (final fix 5) ----

const FOREIGN = 'The Studio only accepts changes from its own page (http://localhost:5199).'

test('a change sent from another web page is refused, in plain words', async () => {
  const a = await post()
  const j = { 'content-type': 'application/json' }
  for (const [method, p, body] of [
    ['POST', '/api/exercises/barbell-back-squat/runs', undefined],
    ['PATCH', `/api/exercises/barbell-back-squat/references/${a.id}`, '{"phase":"end"}'],
    ['PUT', '/api/exercises/barbell-back-squat/references/order', '{"ids":[]}'],
    ['DELETE', `/api/exercises/barbell-back-squat/references/${a.id}`, undefined],
    ['POST', '/api/exercises', '{"name":"Evil Lift"}'],
    ['POST', '/api/exercises/barbell-back-squat/references', 'not a picture'],
  ]) {
    for (const origin of ['http://evil.example', 'null', 'http://localhost:5173']) {
      const res = await send(p, { method, headers: { ...j, origin }, body })
      expect(res.status, `${method} ${p} from ${origin}`).toBe(403)
      expect((await res.json()).error).toBe(FOREIGN)
    }
  }
  expect(queued).toEqual([])
  expect(store.listReferences('barbell-back-squat')).toMatchObject([{ id: a.id, phase: 'start' }])
  // reading is fine from anywhere the host check allows
  expect((await send('/api/exercises', { headers: { origin: 'http://evil.example' } })).status).toBe(200)
})

test('changes from the Studio page, its own origin, or a script without an Origin go through', async () => {
  const a = await post() // a script: no Origin header
  const j = { 'content-type': 'application/json' }
  const patch = (origin, phase) => send(`/api/exercises/barbell-back-squat/references/${a.id}`, {
    method: 'PATCH', headers: { ...j, origin }, body: JSON.stringify({ phase }),
  })
  expect((await patch('http://localhost:5199', 'end')).status).toBe(200)
  expect((await patch(base, 'middle')).status).toBe(200) // the server's own origin (a throwaway port)
  expect(store.listReferences('barbell-back-squat')[0].phase).toBe('middle')
})

// ---- a failed read's own reasons reach the page (final fix 2) ----

test('the exercise carries the newest run\'s read problems', async () => {
  await post()
  const run = store.createRun('barbell-back-squat')
  const problems = { exercise: 'Barbell Back Squat', message: 'missing: end', images: { 'img-1': { used: false, why: 'No body found in this image', view: null } } }
  const { writeFile } = await import('node:fs/promises')
  await writeFile(path.join(store.runDir('barbell-back-squat', run.id), 'read-problems.json'), JSON.stringify(problems))
  const ex = await (await send('/api/exercises/barbell-back-squat')).json()
  expect(ex.latest.readProblems).toEqual(problems)
  expect(ex.latest.targets).toBeNull()
})
