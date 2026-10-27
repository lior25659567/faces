import { afterEach, beforeEach, expect, test } from 'vitest'
import http from 'node:http'
import { mkdtemp, readdir, rm } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { createStore } from './store.js'
import { studioApi } from './api.js'

let root, server, base, store
beforeEach(async () => {
  root = await mkdtemp(path.join(os.tmpdir(), 'studio-preview-api-'))
  store = createStore(root)
  let handler
  studioApi({ store, queue: { state: () => ({ running: null, queued: [] }) } })
    .configureServer({ middlewares: { use: (fn) => { handler = fn } } })
  server = http.createServer((req, res) => handler(req, res, () => { res.statusCode = 404; res.end() }))
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve))
  base = `http://127.0.0.1:${server.address().port}`
})
afterEach(async () => {
  server.closeAllConnections()
  await new Promise((resolve) => server.close(resolve))
  await rm(root, { recursive: true, force: true })
})
const raw = (url) => new Promise((resolve, reject) => {
  const address = new URL(base)
  const request = http.get({ host: address.hostname, port: address.port, path: url, agent: false }, (res) => {
    res.resume(); res.on('end', () => resolve(res.statusCode))
  })
  request.on('error', reject)
})

test('all previews appear on a fresh Studio without creating references or run records', async () => {
  const response = await fetch(base + '/api/previews')
  expect(response.status).toBe(200)
  const rows = await response.json()
  expect(rows).toHaveLength(14)
  expect(rows.filter((row) => row.available)).toHaveLength(14)
  expect(rows.find((row) => row.slug === 'barbell-bent-over-row').summary.fail).toBe(2)
  expect(await readdir(root)).toEqual([])
})

test('serves the actual whitelisted GIF, PNG and exact Checks with matching content types', async () => {
  for (const [file, type] of [['animation.gif', 'image/gif'], ['start.png', 'image/png'], ['end.png', 'image/png'], ['checks.json', 'application/json']]) {
    const response = await fetch(base + '/api/previews/barbell-bent-over-row/' + file)
    expect(response.status).toBe(200)
    expect(response.headers.get('content-type')).toBe(type)
    expect(response.headers.get('etag')).toMatch(/^"[a-f0-9]{64}"$/)
    if (file === 'checks.json') expect((await response.json()).summary.fail).toBe(2)
    else expect((await response.arrayBuffer()).byteLength).toBeGreaterThan(100)
  }
  const head = await fetch(base + '/api/previews/barbell-bent-over-row/animation.gif', { method: 'HEAD' })
  expect(head.status).toBe(200)
  expect(await head.text()).toBe('')
})

test.each(['/api/previews/barbell-back-squat/source.fbx', '/api/previews/missing/animation.gif',
  '/api/previews/barbell-bent-over-row/scene.blend', '/api/previews/barbell-bent-over-row/../../package.json',
  '/api/previews/%2e%2e/%2e%2e/package.json', '/api/previews/barbell-bent-over-row/%2fetc%2fpasswd',
  '/api/previews/barbell-bent-over-row/%zz'])('unavailable and untrusted assets are rejected: %s', async (url) => {
  expect(await raw(url)).toBe(404)
})

test('preview endpoints are read-only', async () => {
  const response = await fetch(base + '/api/previews', { method: 'POST' })
  expect(response.status).toBe(404)
  expect(store.listRuns('barbell-bent-over-row')).toEqual([])
  expect(await readdir(root)).toEqual([])
})
