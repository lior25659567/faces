import { afterEach, expect, test } from 'vitest'
import net from 'node:net'
import { mkdtempSync, rmSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { createServer } from 'vite'
import { REPO } from './paths.js'
import { studioPlugin } from './plugin.js'
import { liveDataRoot } from './live-data-root.js'
import { createWork } from './work.js'

const HOST = '127.0.0.1'
const cleanup = []
afterEach(async () => {
  for (const fn of cleanup.splice(0).reverse()) await fn()
})

const occupy = () =>
  new Promise((resolve) => {
    const blocker = net.createServer().listen(0, HOST, () => resolve(blocker))
  })

const freePort = async () => {
  const s = await occupy()
  const { port } = s.address()
  await new Promise((r) => s.close(r))
  return port
}

// A Studio dev server like `npm run dev`, but on a temp data root, with the shutdown hook recorded.
const studio = (dataRoot, port, installed) => {
  const cacheDir = mkdtempSync(path.join(os.tmpdir(), 'studio-vite-cache-'))
  cleanup.push(() => rmSync(cacheDir, { recursive: true, force: true }))
  return createServer({
    configFile: false,
    root: path.join(REPO, 'studio'),
    cacheDir,
    logLevel: 'silent',
    plugins: [studioPlugin({ dataRoot, onListening: (queue) => installed.push(queue) })],
    server: { host: HOST, port, strictPort: true, watch: null },
    optimizeDeps: { noDiscovery: true, include: [] },
  })
}

test('a second Studio that loses the port recovers nothing and installs no shutdown hook', async () => {
  const live = liveDataRoot()
  cleanup.push(() => rmSync(live.root, { recursive: true, force: true }))
  const blocker = await occupy()
  cleanup.push(() => new Promise((r) => blocker.close(r)))
  const installed = []
  const server = await studio(live.root, blocker.address().port, installed)
  cleanup.push(() => server.close())

  await expect(server.listen()).rejects.toThrow(/in use/i)
  expect(live.status()).toBe('running')
  expect(installed).toHaveLength(0)
})

test('a Studio that does listen marks left-over runs interrupted, then installs its shutdown hook', async () => {
  const live = liveDataRoot()
  cleanup.push(() => rmSync(live.root, { recursive: true, force: true }))
  const installed = []
  const server = await studio(live.root, await freePort(), installed)
  cleanup.push(() => server.close())

  expect(live.status()).toBe('running') // nothing happens before the server listens
  await server.listen()
  expect(live.status()).toBe('interrupted')
  expect(installed).toHaveLength(1)
  expect(typeof installed[0].stop).toBe('function')
})

// A work item finished while no Studio watched: only a Studio that really listens looks at it.
const finishedWork = (root) => {
  const work = createWork(root)
  const item = work.start({ title: 'Fix', exercises: ['barbell-back-squat'] })
  work.finish(item.id, 'done', 'ok')
  return () => work.read(item.id)
}

// (The temp root has no pictures, so the catch-up records "nothing to re-run" and no process starts.)
test('the work watcher runs only in a Studio that listens, and catches up on work finished while it was closed', async () => {
  const lost = liveDataRoot()
  cleanup.push(() => rmSync(lost.root, { recursive: true, force: true }))
  const lostItem = finishedWork(lost.root)
  const blocker = await occupy()
  cleanup.push(() => new Promise((r) => blocker.close(r)))
  const loser = await studio(lost.root, blocker.address().port, [])
  cleanup.push(() => loser.close())
  await expect(loser.listen()).rejects.toThrow(/in use/i)
  expect(lostItem().rerunQueuedAt).toBeUndefined()

  const live = liveDataRoot()
  cleanup.push(() => rmSync(live.root, { recursive: true, force: true }))
  const liveItem = finishedWork(live.root)
  const server = await studio(live.root, await freePort(), [])
  cleanup.push(() => server.close())
  await server.listen()
  expect(liveItem().rerunQueuedAt).toBeTruthy()
  expect(liveItem().rerun).toEqual([])
  expect(liveItem().rerunNote).toMatch(/none of its exercises have reference pictures/i)
})
