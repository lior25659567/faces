import { afterEach, beforeEach, expect, test } from 'vitest'
import { mkdtemp, rm } from 'node:fs/promises'
import { readFileSync, readdirSync, utimesSync, writeFileSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { createStore } from './store.js'
import { WorkError, createWork, createWorkWatcher, recentWork, resolveExercises, sortWork } from './work.js'

const PNG = Buffer.from('89504e470d0a1a0a0000000d49484452', 'hex')
let root, store, work, queued, queue

beforeEach(async () => {
  root = await mkdtemp(path.join(os.tmpdir(), 'studio-work-'))
  store = createStore(root)
  work = createWork(root)
  queued = []
  // The real queue coalesces; the watcher only needs enqueue(slug) -> run.
  queue = { enqueue: (slug) => { queued.push(slug); return store.createRun(slug) } }
})
afterEach(() => rm(root, { recursive: true, force: true }))

const withPicture = (slug) => store.addReference(slug, PNG, 'a.png')
const onDisk = (id) => JSON.parse(readFileSync(path.join(root, 'work', `${id}.json`), 'utf8'))

// ---- the work store ----

test('start writes a running item with a run-style id', () => {
  const item = work.start({ title: '  Fix skin folds at the shoulders ', exercises: ['barbell-back-squat'] })
  expect(item.id).toMatch(/^\d{8}-\d{6}-[0-9a-f]{4}$/)
  expect(onDisk(item.id)).toEqual({
    id: item.id, title: 'Fix skin folds at the shoulders', exercises: ['barbell-back-squat'], status: 'running',
    startedAt: item.startedAt, updatedAt: item.startedAt, finishedAt: null, log: [], summary: null, rerun: null,
  })
  expect(Number.isNaN(Date.parse(item.startedAt))).toBe(false)
})

test('log appends lines with times; done and fail finish with a summary', () => {
  const a = work.start({ title: 'A', exercises: [] })
  work.log(a.id, 'Measured the folds')
  work.log(a.id, ' Trying a corrective shape ')
  expect(onDisk(a.id).log.map((l) => l.text)).toEqual(['Measured the folds', 'Trying a corrective shape'])
  expect(onDisk(a.id).log.every((l) => !Number.isNaN(Date.parse(l.at)))).toBe(true)
  const done = work.finish(a.id, 'done', 'Shoulder folds are smaller')
  expect(done).toMatchObject({ status: 'done', summary: 'Shoulder folds are smaller' })
  expect(onDisk(a.id).finishedAt).toBeTruthy()

  const b = work.start({ title: 'B' })
  expect(b.exercises).toEqual([])
  expect(work.finish(b.id, 'failed', 'It made the elbows worse').status).toBe('failed')
})

test('writes are atomic: nothing but finished .json files is left in the folder', () => {
  const a = work.start({ title: 'A' })
  work.log(a.id, 'x')
  expect(readdirSync(path.join(root, 'work'))).toEqual([`${a.id}.json`])
})

test.each([
  ['an unknown id', () => work.log('20990101-000000-abcd', 'x'), /no work item "20990101-000000-abcd"/i],
  ['a made-up id', () => work.finish('../../etc/passwd', 'done', 'x'), /no work item/i],
  ['an empty title', () => work.start({ title: '  ' }), /title/i],
  ['an empty log line', () => work.log(work.start({ title: 'A' }).id, '  '), /what to log/i],
  ['a missing summary', () => work.finish(work.start({ title: 'A' }).id, 'done', ''), /summary/i],
])('plain errors: %s', (_, act, message) => {
  expect(act).toThrow(WorkError)
  expect(act).toThrow(message)
})

test('a finished item cannot be finished twice; a late log line is kept and changes nothing else', () => {
  const a = work.start({ title: 'A' })
  work.finish(a.id, 'done', 'ok')
  expect(() => work.finish(a.id, 'failed', 'no')).toThrow(/already marked done/)
  work.log(a.id, 'a note after the fact')
  expect(onDisk(a.id)).toMatchObject({ status: 'done', summary: 'ok' })
  expect(onDisk(a.id).log.map((l) => l.text)).toEqual(['a note after the fact'])
})

test('a writer waits for the item lock, and a lock left by a crashed writer is broken', () => {
  const a = work.start({ title: 'A' })
  const lock = path.join(root, 'work', `${a.id}.lock`)
  writeFileSync(lock, '')
  const old = new Date(Date.now() - 60_000)
  utimesSync(lock, old, old) // a writer that died a minute ago
  work.log(a.id, 'after the crash')
  expect(onDisk(a.id).log).toHaveLength(1)
  expect(readdirSync(path.join(root, 'work'))).toEqual([`${a.id}.json`]) // lock released
  writeFileSync(lock, '') // a live writer holding it
  expect(() => work.withLock(a.id, () => {}, { waitMs: 50 })).toThrow(/busy/)
})

test('resolveExercises: slugs or names, "all" = every exercise with usable pictures, unknown is a plain error', () => {
  withPicture('barbell-back-squat')
  withPicture('pull-up')
  store.updateReference('pull-up', store.listReferences('pull-up')[0].id, { contextOnly: true }) // not usable
  expect(resolveExercises(store, 'all')).toEqual(['barbell-back-squat'])
  expect(resolveExercises(store, 'Pull-Up, barbell-back-squat,pull-up')).toEqual(['pull-up', 'barbell-back-squat'])
  expect(resolveExercises(store, undefined)).toEqual([])
  expect(() => resolveExercises(store, 'squatt')).toThrow(/no exercise called "squatt"/i)
})

test('recentWork: every running item, then the newest 50 others', () => {
  const items = []
  for (let i = 0; i < 3; i++) items.push({ id: `r${i}`, status: 'running', startedAt: `2026-10-01T00:00:0${i}Z` })
  for (let i = 0; i < 55; i++) items.push({ id: `d${String(i).padStart(2, '0')}`, status: 'done', startedAt: `2026-10-22T10:${String(i).padStart(2, '0')}:00Z` })
  const out = recentWork(items)
  expect(out).toHaveLength(53)
  expect(out.slice(0, 3).map((w) => w.status)).toEqual(['running', 'running', 'running'])
  expect(out[3].id).toBe('d54')
  expect(out.at(-1).id).toBe('d05')
})

test('sortWork: running first, then newest', () => {
  const items = [
    { id: '1', status: 'done', startedAt: '2026-10-22T10:00:00Z' },
    { id: '2', status: 'running', startedAt: '2026-10-22T09:00:00Z' },
    { id: '3', status: 'failed', startedAt: '2026-10-22T11:00:00Z' },
    { id: '4', status: 'running', startedAt: '2026-10-22T12:00:00Z' },
  ]
  expect(sortWork(items).map((w) => w.id)).toEqual(['4', '2', '3', '1'])
})

// ---- the transition detector ----

const watch = () => createWorkWatcher({ work, store, queue })

test('running -> done re-runs each listed exercise with pictures, exactly once, and records the run ids', () => {
  withPicture('barbell-back-squat')
  const w = watch()
  const item = work.start({ title: 'Fix', exercises: ['barbell-back-squat', 'pull-up'] }) // pull-up has no pictures
  w.scan()
  expect(queued).toEqual([])
  work.log(item.id, 'working')
  w.scan()
  expect(queued).toEqual([])

  work.finish(item.id, 'done', 'Fixed the folds')
  w.scan()
  expect(queued).toEqual(['barbell-back-squat'])
  const after = onDisk(item.id)
  expect(after.rerun).toEqual([{ slug: 'barbell-back-squat', runId: store.listRuns('barbell-back-squat')[0].id }])
  expect(after.rerunQueuedAt).toBeTruthy()
  expect(after.rerunNote).toMatch(/Pull-Up/)

  w.scan()
  w.scan()
  expect(queued).toEqual(['barbell-back-squat'])

  // A second watcher (the server restarted) does not re-run it again.
  const again = watch()
  again.scan()
  again.scan()
  expect(queued).toEqual(['barbell-back-squat'])
})

test('an item started and finished between two looks still re-runs (it is new and done)', () => {
  withPicture('barbell-back-squat')
  const w = watch()
  w.scan()
  const item = work.start({ title: 'Quick fix', exercises: ['barbell-back-squat'] })
  work.finish(item.id, 'done', 'done fast')
  w.scan()
  expect(queued).toEqual(['barbell-back-squat'])
})

test('a failed item re-runs nothing', () => {
  withPicture('barbell-back-squat')
  const w = watch()
  const item = work.start({ title: 'Fix', exercises: ['barbell-back-squat'] })
  w.scan()
  work.finish(item.id, 'failed', 'Made it worse')
  w.scan()
  expect(queued).toEqual([])
  expect(onDisk(item.id).rerun).toBeNull()
})

test('catch-up: an item already done when the server starts (finished while it was closed) re-runs once', () => {
  withPicture('barbell-back-squat')
  const item = work.start({ title: 'Fixed while the Studio was closed', exercises: ['barbell-back-squat'] })
  work.finish(item.id, 'done', 'ok')
  const w = watch()
  w.scan()
  w.scan()
  expect(queued).toEqual(['barbell-back-squat'])
  expect(onDisk(item.id).rerun).toEqual([{ slug: 'barbell-back-squat', runId: store.listRuns('barbell-back-squat')[0].id }])
  watch().scan() // and the next start leaves it alone
  expect(queued).toEqual(['barbell-back-squat'])
})

test('catch-up skips an exercise already re-run by hand since the item finished, and records that run', () => {
  withPicture('barbell-back-squat')
  withPicture('pull-up')
  const item = work.start({ title: 'Fix', exercises: ['barbell-back-squat', 'pull-up'] })
  work.finish(item.id, 'done', 'ok')
  // The user pressed "Run again" on the squat after the fix finished, while the Studio was closed.
  const byHand = store.createRun('barbell-back-squat')
  byHand.createdAt = new Date(Date.parse(onDisk(item.id).finishedAt) + 1000).toISOString()
  store.writeRun(byHand)
  watch().scan()
  expect(queued).toEqual(['pull-up'])
  expect(onDisk(item.id).rerun).toEqual([
    { slug: 'barbell-back-squat', runId: byHand.id, byHand: true },
    { slug: 'pull-up', runId: store.listRuns('pull-up')[0].id },
  ])
})

test('two items finished together share the one automatic run (it is not "by hand")', () => {
  withPicture('barbell-back-squat')
  const a = work.start({ title: 'A', exercises: ['barbell-back-squat'] })
  const b = work.start({ title: 'B', exercises: ['barbell-back-squat'] })
  work.finish(b.id, 'done', 'ok')
  work.finish(a.id, 'done', 'ok')
  watch().scan()
  expect(queued).toEqual(['barbell-back-squat'])
  const runId = store.listRuns('barbell-back-squat')[0].id
  expect(onDisk(a.id).rerun).toEqual([{ slug: 'barbell-back-squat', runId }])
  expect(onDisk(b.id).rerun).toEqual([{ slug: 'barbell-back-squat', runId }])
})

test('a run that cannot start gets a plain note; the raw error goes to the server console', () => {
  withPicture('barbell-back-squat')
  const errors = []
  const was = console.error
  console.error = (...a) => errors.push(a.join(' '))
  try {
    queue.enqueue = () => { throw new Error("EACCES: permission denied, open '/x/run.json'") }
    const item = work.start({ title: 'Fix', exercises: ['barbell-back-squat'] })
    work.finish(item.id, 'done', 'ok')
    watch().scan()
    expect(onDisk(item.id).rerunNote).toBe('Could not start a run for Barbell Back Squat.')
    expect(onDisk(item.id).rerunNote).not.toMatch(/EACCES/)
    expect(errors.join('\n')).toMatch(/EACCES/)
  } finally {
    console.error = was
  }
})

test('nothing listed with pictures: nothing re-runs and the note says so', () => {
  const w = watch()
  w.scan()
  const item = work.start({ title: 'Fix', exercises: ['pull-up'] })
  work.finish(item.id, 'done', 'ok')
  w.scan()
  expect(queued).toEqual([])
  expect(onDisk(item.id)).toMatchObject({ rerun: [] })
  expect(onDisk(item.id).rerunNote).toMatch(/none of its exercises have reference pictures/i)
})

const counted = () => {
  let reads = 0
  const read = work.read
  work.read = (id) => { reads += 1; return read(id) }
  return () => reads
}

test('finished items are read again only when they change; running ones every time', () => {
  const w = watch()
  const done = work.start({ title: 'Done', exercises: [] })
  work.finish(done.id, 'failed', 'x')
  const running = work.start({ title: 'Running', exercises: [] })
  const reads = counted()
  w.scan()
  expect(reads()).toBe(2)
  w.scan()
  w.scan()
  expect(reads()).toBe(4) // only the running one, twice
  const later = new Date(Date.now() + 5000)
  utimesSync(path.join(root, 'work', `${done.id}.json`), later, later)
  w.scan()
  expect(reads()).toBe(6)
  expect(running.status).toBe('running')
})

// A rewrite inside one mtime tick must not hide running -> done: the file's mtime is pinned to the
// same whole second before the first look and after the rewrite.
const TICK = new Date('2026-10-22T12:00:00Z')
const pin = (id) => utimesSync(path.join(root, 'work', `${id}.json`), TICK, TICK)
const sameTick = (id, rewrite) => {
  rewrite(path.join(root, 'work', `${id}.json`))
  pin(id)
}

test('done in the same mtime tick is still seen (the size changed)', () => {
  withPicture('barbell-back-squat')
  const w = watch()
  const item = work.start({ title: 'Fix', exercises: ['barbell-back-squat'] })
  pin(item.id)
  w.scan()
  sameTick(item.id, () => work.finish(item.id, 'done', 'ok'))
  w.scan()
  expect(queued).toEqual(['barbell-back-squat'])
})

test('done in the same mtime tick and the same size is still seen (running items are always re-read)', () => {
  withPicture('barbell-back-squat')
  const w = watch()
  const item = work.start({ title: 'Fix', exercises: ['barbell-back-squat'] })
  pin(item.id)
  w.scan()
  sameTick(item.id, (file) => {
    const before = readFileSync(file, 'utf8')
    const after = before.replace('"status": "running"', '"status":    "done"')
    expect(after.length).toBe(before.length)
    writeFileSync(file, after)
  })
  w.scan()
  expect(queued).toEqual(['barbell-back-squat'])
})

test('the cached list is what the page reads: same items as a fresh read', () => {
  const a = work.start({ title: 'A', exercises: [] })
  work.log(a.id, 'one')
  expect(work.cached().map((w) => w.log.length)).toEqual([1])
  work.log(a.id, 'two')
  expect(work.cached().map((w) => w.log.length)).toEqual([2])
  expect(work.cached()).toEqual(work.list())
})

test('a broken file is skipped without stopping the others', () => {
  withPicture('barbell-back-squat')
  const w = watch()
  w.scan()
  const good = work.start({ title: 'Fix', exercises: ['barbell-back-squat'] })
  work.finish(good.id, 'done', 'ok')
  // A half-written file from something other than the CLI.
  const bad = path.join(root, 'work', '20990101-000000-dead.json')
  writeFileSync(bad, '{"id": ')
  expect(() => w.scan()).not.toThrow()
  expect(queued).toEqual(['barbell-back-squat'])
})

test('start() looks at once and then every few seconds; stop() ends it', async () => {
  withPicture('barbell-back-squat')
  const item = work.start({ title: 'Fix', exercises: ['barbell-back-squat'] })
  const w = createWorkWatcher({ work, store, queue, intervalMs: 10 })
  w.start()
  work.finish(item.id, 'done', 'ok')
  const end = Date.now() + 2000
  while (!queued.length && Date.now() < end) await new Promise((r) => setTimeout(r, 5))
  expect(queued).toEqual(['barbell-back-squat'])
  w.stop()
})

// ---- check-only re-runs: a fix that only changes the rules re-judges without re-rendering ----

test('done with recheck marks the item as a check-only re-run; a plain done does not', () => {
  const a = work.start({ title: 'Tighten the depth rule', exercises: ['barbell-back-squat'] })
  work.finish(a.id, 'done', 'depth judged on the thigh angle', { recheck: true })
  expect(onDisk(a.id)).toMatchObject({ status: 'done', summary: 'depth judged on the thigh angle', recheck: true })

  const b = work.start({ title: 'Rebuild the arms', exercises: [] })
  work.finish(b.id, 'done', 'arms rebuilt')
  expect(onDisk(b.id).recheck).toBeFalsy()
})

test('the watcher re-checks a done recheck item instead of re-rendering it', () => {
  withPicture('barbell-back-squat')
  const rechecked = []
  queue.enqueueRecheck = (slug) => { rechecked.push(slug); return store.createRun(slug) }
  const item = work.start({ title: 'tighten depth', exercises: ['barbell-back-squat'] })
  work.finish(item.id, 'done', 'depth rule tightened', { recheck: true })
  watch().scan()
  expect(rechecked).toEqual(['barbell-back-squat'])
  expect(queued).toEqual([]) // the full-render path was not used
  expect(onDisk(item.id).rerun).toEqual([{ slug: 'barbell-back-squat', runId: store.listRuns('barbell-back-squat')[0].id }])
})
