import { afterEach, beforeEach, expect, test } from 'vitest'
import { EventEmitter } from 'node:events'
import { PassThrough } from 'node:stream'
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { createStore } from './store.js'
import { createQueue } from './queue.js'
import { commandFor } from './steps.js'

let root, store, queue
const slug = 'barbell-back-squat'
beforeEach(() => {
  root = mkdtempSync(path.join(os.tmpdir(), 'studio-preflight-'))
  store = createStore(root)
})
afterEach(() => { queue?.stop(); rmSync(root, { recursive: true, force: true }) })
const until = async (fn) => {
  for (let n = 0; n < 100; n++) {
    if (fn()) return
    await new Promise((r) => setTimeout(r, 10))
  }
  throw new Error('preflight did not finish')
}
const reader = (record, ready = true, hold = false) => (cmd, args) => {
  record.push({ cmd, args })
  const child = new EventEmitter()
  child.stdout = new PassThrough(); child.stderr = new PassThrough()
  child.kill = () => child.emit('close', null)
  if (!hold) setImmediate(() => {
    const dir = args.at(-1)
    writeFileSync(path.join(dir, 'measurements.json'), '{}')
    writeFileSync(path.join(dir, 'targets.json'), '{}')
    writeFileSync(path.join(dir, 'preflight.json'), JSON.stringify({ ready, blockers: ready ? [] : ['Add a rear reference.'] }))
    if (!ready) child.stderr.write('Add a rear reference.\n')
    child.emit('close', ready ? 0 : 3)
  })
  return child
}

test('reference check runs only the reader and is not a passing animation', async () => {
  const calls = []
  queue = createQueue({ store, spawnProcess: reader(calls), otherBlenderRunning: () => { throw Error('Must not need Blender') } })
  const run = queue.enqueueRead(slug)
  await until(() => store.readRun(slug, run.id).status === 'done')
  expect(calls).toHaveLength(1)
  expect(calls[0].args[0]).toMatch(/run_read.py$/)
  expect(store.readRun(slug, run.id)).toMatchObject({ readOnly: true, summary: null, steps: [{ id: 'read', status: 'done' }] })
})

test('failed reference check prevents every Blender process during a full build', async () => {
  const calls = []
  queue = createQueue({ store, spawnProcess: reader(calls, false), otherBlenderRunning: () => false })
  const run = queue.enqueue(slug)
  await until(() => store.readRun(slug, run.id).status === 'failed')
  expect(calls).toHaveLength(1)
  expect(store.readRun(slug, run.id).error.message).toBe('Add a rear reference.')
})

test('queued preview and build cannot overwrite each other', () => {
  queue = createQueue({ store, spawnProcess: reader([], true, true), otherBlenderRunning: () => false })
  const preview = queue.enqueueRead(slug)
  const full = queue.enqueue(slug)
  expect(full.id).not.toBe(preview.id)
  expect(store.readRun(slug, full.id).readOnly).toBeUndefined()
  expect(store.readRun(slug, preview.id).steps).toHaveLength(1)
  expect(queue.state().queued).toHaveLength(2)
})

test('command selection rejects a blocked preflight even if stale targets exist', () => {
  writeFileSync(path.join(root, 'preflight.json'), JSON.stringify({ ready: false, blockers: ['Add a rear reference.'] }))
  writeFileSync(path.join(root, 'pipeline-targets.json'), '{}')
  expect(commandFor('animate', {}, root, { name: 'Barbell Back Squat', clip: 'squat.fbx' })).toEqual({ error: 'Add a rear reference.' })
})

test('checking changed pictures retires a stale queued build and still needs an explicit rebuild', async () => {
  const calls = []
  queue = createQueue({ store, spawnProcess: reader(calls, true, true), otherBlenderRunning: () => false })
  const other = queue.enqueueRead('barbell-bent-over-row')
  await until(() => store.readRun(other.slug, other.id).status === 'running')
  const picture = store.addReference(slug, Buffer.from('fixture'), 'pose.png')
  const stale = queue.enqueue(slug)
  store.updateReference(slug, picture.id, { view: 'side' })
  const check = queue.enqueueRead(slug)
  expect(store.readRun(slug, stale.id)).toMatchObject({ status: 'interrupted', replacedBy: check.id })
  expect(queue.state().queued.map((q) => q.runId)).toEqual([check.id])
  expect(store.readRun(slug, check.id)).toMatchObject({ readOnly: true, steps: [{ id: 'read' }] })
  expect(calls).toHaveLength(1)
  const explicitBuild = queue.enqueue(slug)
  expect(queue.state().queued.map((q) => q.runId)).toEqual([check.id, explicitBuild.id])
  expect(store.readRun(slug, explicitBuild.id).readOnly).toBeUndefined()
})

test('restarting an interrupted reference check never starts a full build', async () => {
  const calls = []
  store.addReference(slug, Buffer.from('fixture'), 'pose.png')
  const interrupted = store.createRun(slug, { readOnly: true })
  store.writeRun({ ...interrupted, status: 'interrupted' })
  queue = createQueue({ store, spawnProcess: reader(calls), otherBlenderRunning: () => { throw Error('Must not need Blender') } })
  const [restart] = queue.restartInterrupted()
  await until(() => store.readRun(slug, restart.runId).status === 'done')
  expect(store.readRun(slug, restart.runId)).toMatchObject({ readOnly: true, restartOf: interrupted.id })
  expect(calls).toHaveLength(1)
  expect(queue.restartInterrupted()).toEqual([])
})
