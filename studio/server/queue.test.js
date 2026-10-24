import { afterEach, beforeEach, expect, test } from 'vitest'
import { EventEmitter } from 'node:events'
import { PassThrough } from 'node:stream'
import { mkdtemp, mkdir, rm, writeFile, readFile } from 'node:fs/promises'
import { existsSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { createStore } from './store.js'
import { createQueue, installShutdown, realSpawn } from './queue.js'

const SAMPLE = new URL('../../3d-character/reference-squat/15.png', import.meta.url)
let root, store, queue

beforeEach(async () => {
  root = await mkdtemp(path.join(os.tmpdir(), 'studio-queue-'))
  store = createStore(root)
})
afterEach(async () => {
  queue?.stop()
  await rm(root, { recursive: true, force: true })
})

const sleep = (ms) => new Promise((r) => setTimeout(r, ms))
const until = async (fn, ms = 3000) => {
  const end = Date.now() + ms
  while (Date.now() < end) {
    if (await fn()) return
    await sleep(5)
  }
  throw new Error('timed out waiting')
}
const finished = (slug, id) => async () => ['done', 'failed', 'interrupted'].includes(store.readRun(slug, id)?.status)

const write = (dir, rel, text = '{"x":1}') =>
  mkdir(path.dirname(path.join(dir, rel)), { recursive: true }).then(() => writeFile(path.join(dir, rel), text))

// Which of the four processes a command is.
const kindOf = ({ cmd, args }) =>
  args.some((a) => a.endsWith('run_read.py')) ? 'read'
    : args.some((a) => a.endsWith('run_check.py')) ? 'check'
      : args.some((a) => a.endsWith('inspect_run.py')) ? 'stills'
        : 'animate'

// A fake child process; `script(call, child, dir)` decides what it does.
const fakeSpawn = (script, record = {}) => (cmd, args, opts) => {
  const child = new EventEmitter()
  child.stdout = new PassThrough()
  child.stderr = new PassThrough()
  child.killed = false
  child.kill = () => { child.killed = true; record.killed = (record.killed || 0) + 1; child.emit('close', null) }
  child.out = (s) => child.stdout.write(s + '\n')
  child.err = (s) => child.stderr.write(s + '\n')
  child.exit = (code) => setTimeout(() => child.emit('close', code), 5)
  const call = { cmd, args, cwd: opts?.cwd }
  record.calls = [...(record.calls || []), call]
  record.live = (record.live || 0) + 1
  record.max = Math.max(record.max || 0, record.live)
  child.on('close', () => { record.live -= 1 })
  Promise.resolve().then(() => script(call, child, findRunDir(call)))
  return child
}
const findRunDir = (call) => {
  const i = call.args.findIndex((a) => /\/runs\/[^/]+\/[^/]+(\/|$)/.test(a))
  const m = call.args[i].match(/^(.*\/runs\/[^/]+\/[^/]+)/)
  return m[1]
}

const goodScript = async (call, child, dir) => {
  const kind = kindOf(call)
  if (kind === 'read') {
    await write(dir, 'measurements.json'); await write(dir, 'targets.json')
  } else if (kind === 'animate') {
    child.out('retarget: read squat.fbx: 2 frames at 30 fps')
    child.out('bake_to_skin: done')
    child.out('retarget report -> x')
    await write(dir, 'animation.gif', 'GIF'); await write(dir, 'frames/retarget.json'); await write(dir, 'scene.blend', 'B')
  } else if (kind === 'stills') {
    const snap = JSON.parse(await readFile(path.join(dir, 'references.json'), 'utf8'))
    await write(dir, 'ours.json', JSON.stringify({ stills: snap.images.map((i) => ({ file: `stills/${i.id}.png` })) }))
    for (const i of snap.images) await write(dir, `stills/${i.id}.png`, 'P')
  } else {
    await write(dir, 'checks.json', JSON.stringify({ summary: { pass: 3, fail: 0, limited: 0, uncertain: 1, info: 0 } }))
  }
  child.exit(0)
}

const make = (script, extra = {}) => {
  const record = extra.record || {}
  queue = createQueue({
    store, spawnProcess: fakeSpawn(script, record), otherBlenderRunning: () => false, pollMs: 10, waitMs: 10, ...extra.opts,
  })
  return record
}
const addImage = async (slug = 'barbell-back-squat') => {
  const { readFile } = await import('node:fs/promises')
  return store.addReference(slug, await readFile(SAMPLE), '15.png', 'png')
}

test('runs one process at a time', async () => {
  const record = make(async (call, child, dir) => { await sleep(15); await goodScript(call, child, dir) })
  await addImage('barbell-back-squat')
  const a = queue.enqueue('barbell-back-squat')
  const b = queue.enqueue('pull-up')
  await until(finished('barbell-back-squat', a.id), 8000)
  await until(finished('pull-up', b.id), 8000)
  expect(record.max).toBe(1)
})

test('coalesces queued runs for the same exercise', async () => {
  let release
  const gate = new Promise((r) => (release = r))
  make(async (call, child, dir) => { await gate; await goodScript(call, child, dir) })
  await addImage('barbell-back-squat')
  const a = queue.enqueue('barbell-back-squat')
  await until(() => queue.state().running)
  const b1 = queue.enqueue('pull-up')
  await addImage('pull-up')
  const b2 = queue.enqueue('pull-up')
  expect(b2.id).toBe(b1.id)
  expect(queue.state().queued).toEqual([{ slug: 'pull-up', runId: b1.id }])
  const snap = JSON.parse(await readFile(path.join(store.runDir('pull-up', b1.id), 'references.json'), 'utf8'))
  expect(snap.images).toHaveLength(1)
  expect(store.listRuns('pull-up')).toHaveLength(1)
  release()
  await until(finished('barbell-back-squat', a.id), 8000)
})

test('fails a step that exits 0 without its outputs', async () => {
  make(async (call, child) => { child.exit(0) })
  await addImage()
  const r = queue.enqueue('barbell-back-squat')
  await until(finished('barbell-back-squat', r.id))
  const run = store.readRun('barbell-back-squat', r.id)
  expect(run.status).toBe('failed')
  expect(run.steps.find((s) => s.id === 'read').status).toBe('failed')
  expect(run.steps.find((s) => s.id === 'build').status).toBe('waiting')
  expect(run.error.step).toBe('read')
  expect(run.error.message).toBe('This step finished without producing measurements.json, targets.json')
  expect(run.summary).toBeNull()
})

test('treats an empty output file as missing', async () => {
  make(async (call, child, dir) => { await write(dir, 'measurements.json', ''); await write(dir, 'targets.json'); child.exit(0) })
  await addImage()
  const r = queue.enqueue('barbell-back-squat')
  await until(finished('barbell-back-squat', r.id))
  expect(store.readRun('barbell-back-squat', r.id).error.message).toContain('measurements.json')
})

test('fails with the log tail on a non-zero exit', async () => {
  make(async (call, child) => {
    for (let i = 1; i <= 30; i++) child.out(`line ${i}`)
    child.err('Traceback: boom')
    child.err('Error: the script broke')
    child.exit(1)
  })
  await addImage()
  const r = queue.enqueue('barbell-back-squat')
  await until(finished('barbell-back-squat', r.id))
  const run = store.readRun('barbell-back-squat', r.id)
  expect(run.status).toBe('failed')
  expect(run.error.message).toBe('Error: the script broke')
  expect(run.error.logTail).toHaveLength(20)
  expect(run.error.logTail.at(-1)).toBe('Error: the script broke')
  const log = await readFile(path.join(store.runDir('barbell-back-squat', r.id), 'logs', 'read.log'), 'utf8')
  expect(log).toContain('line 1')
})

test('waits while another Blender runs', async () => {
  let polls = 0
  const record = make(goodScript, { opts: { otherBlenderRunning: () => ++polls <= 2 } })
  await addImage()
  const r = queue.enqueue('barbell-back-squat')
  let note = null
  await until(() => {
    const s = queue.state().running
    if (s?.note) note = s.note
    return finished('barbell-back-squat', r.id)()
  })
  expect(note).toBe('Waiting for Blender to close (another Blender is open or running)')
  expect(polls).toBeGreaterThan(2)
  const run = store.readRun('barbell-back-squat', r.id)
  expect(run.status).toBe('done')
  expect(run.steps.every((s) => s.status === 'done')).toBe(true)
  expect(run.steps.find((s) => s.id === 'build').note).toBeNull()
  expect(record.calls.map(kindOf)).toEqual(['read', 'animate', 'stills', 'check'])
})

test('marks runs left running as interrupted', async () => {
  make(goodScript)
  await addImage()
  const run = store.createRun('barbell-back-squat')
  run.status = 'running'
  run.steps[0].status = 'done'
  run.steps[1].status = 'running'
  store.writeRun(run)
  queue.recoverInterrupted()
  const back = store.readRun('barbell-back-squat', run.id)
  expect(back.status).toBe('interrupted')
  expect(back.error.message).toBe('The Studio was stopped while this ran.')
  expect(back.error.step).toBe('build')
  expect(back.steps[1].status).toBe('failed')
  expect(back.steps[0].status).toBe('done')
})

test('stop kills the running child and marks the run interrupted', async () => {
  let child
  const record = make(async (call, c) => { child = c })
  await addImage()
  const r = queue.enqueue('barbell-back-squat')
  await until(() => child)
  queue.stop()
  expect(record.killed).toBe(1)
  await until(() => store.readRun('barbell-back-squat', r.id).status === 'interrupted')
  expect(store.readRun('barbell-back-squat', r.id).error.message).toBe('The Studio was stopped while this ran.')
})

test('copies the summary from checks.json and applies detected views', async () => {
  let applied
  store.applyDetectedViews = (slug, m) => { applied = [slug, m] }
  make(async (call, child, dir) => {
    if (kindOf(call) === 'read') {
      await write(dir, 'measurements.json', '{"img-1":{"view":"side"}}'); await write(dir, 'targets.json'); child.exit(0)
    } else goodScript(call, child, dir)
  })
  await addImage()
  const r = queue.enqueue('barbell-back-squat')
  await until(finished('barbell-back-squat', r.id))
  const run = store.readRun('barbell-back-squat', r.id)
  expect(run.status).toBe('done')
  expect(run.summary).toEqual({ pass: 3, fail: 0, limited: 0, uncertain: 1, info: 0 })
  expect(applied).toEqual(['barbell-back-squat', { 'img-1': { view: 'side' } }])
  expect(run.steps.map((s) => s.status)).toEqual(Array(6).fill('done'))
  expect(typeof run.steps[0].seconds).toBe('number')
})

test('a missing motion clip fails the build step in plain words', async () => {
  // (no key positions written by this read either: task 9's picture path has nothing to move between)
  make(goodScript)
  const slug = 'conventional-barbell-deadlift'
  store.getExercise = (s) => ({ ...createStore(root).getExercise(s), clip: null, expectedClip: 'deadlift.fbx' })
  const r = queue.enqueue(slug)
  await until(finished(slug, r.id))
  const run = store.readRun(slug, r.id)
  expect(run.status).toBe('failed')
  expect(run.error.step).toBe('build')
  expect(run.error.message).toContain('named deadlift.fbx')
  expect(run.steps.find((x) => x.id === 'read').status).toBe('done')
})

test('an exercise the picture path can\'t build a support for fails the build step with that refusal (task 9 review)', async () => {
  make(goodScript)
  store.getExercise = (slug) => ({ ...createStore(root).getExercise(slug), clip: null, expectedClip: 'pull-up.fbx' })
  const r = queue.enqueue('pull-up')
  await until(finished('pull-up', r.id))
  const run = store.readRun('pull-up', r.id)
  expect(run.status).toBe('failed')
  expect(run.error.step).toBe('build')
  expect(run.error.message).toBe('No motion for this exercise: it needs a pull-up bar the picture path doesn\'t build yet.')
})

test('state reports step, label, elapsed and time left', async () => {
  let release
  const gate = new Promise((r) => (release = r))
  make(async (call, child, dir) => { await gate; await goodScript(call, child, dir) })
  await addImage()
  const r = queue.enqueue('barbell-back-squat')
  await until(() => queue.state().running)
  const s = queue.state().running
  expect(s).toMatchObject({ slug: 'barbell-back-squat', runId: r.id, step: 'read', label: 'Reading the reference' })
  expect(s.left).toBeGreaterThan(0)
  expect(typeof s.overdue).toBe('boolean')
  // The page shows the current step's time left beside the whole run's.
  expect(s.stepLeft).toBeGreaterThan(0)
  expect(s.stepLeft).toBeLessThan(s.left)
  release()
  await until(finished('barbell-back-squat', r.id))
  expect(queue.state().running).toBeNull()
})

test('render progress comes from the frames on disk', async () => {
  let release
  const gate = new Promise((r) => (release = r))
  make(async (call, child, dir) => {
    if (kindOf(call) !== 'animate') return goodScript(call, child, dir)
    child.out('retarget: read squat.fbx: 2 frames at 30 fps')
    child.out('retarget report -> x')
    await write(dir, 'frames/frame_0001.png', 'x'); await write(dir, 'frames/side/frame_0001.png', 'x')
    await gate
    await goodScript(call, child, dir)
  })
  await addImage()
  const r = queue.enqueue('barbell-back-squat')
  await until(() => queue.state().running?.step === 'render' && queue.state().running.fraction === 0.5)
  release()
  await until(finished('barbell-back-squat', r.id))
})

test('time spent waiting for another Blender is not counted in the step seconds', async () => {
  let clock = 1_000_000
  let calls = 0
  make(async (call, child, dir) => {
    if (kindOf(call) === 'animate') {
      child.out('retarget: read squat.fbx: 2 frames at 30 fps')
      clock += 4000
      child.out('bake_to_skin: done')
      await sleep(10)
      child.out('retarget report -> x')
      await write(dir, 'animation.gif', 'GIF'); await write(dir, 'frames/retarget.json'); await write(dir, 'scene.blend', 'B')
      return child.exit(0)
    }
    return goodScript(call, child, dir)
  }, { opts: { now: () => clock, otherBlenderRunning: () => { calls += 1; if (calls <= 2) clock += 50_000; return calls <= 2 } } })
  await addImage()
  const r = queue.enqueue('barbell-back-squat')
  await until(finished('barbell-back-squat', r.id))
  const run = store.readRun('barbell-back-squat', r.id)
  expect(run.status).toBe('done')
  expect(run.steps.find((s) => s.id === 'build').seconds).toBe(4)
})

test('a checks.json without a summary fails the check step', async () => {
  make(async (call, child, dir) => {
    if (kindOf(call) === 'check') { await write(dir, 'checks.json', '{}'); child.exit(0) } else goodScript(call, child, dir)
  })
  await addImage()
  const r = queue.enqueue('barbell-back-squat')
  await until(finished('barbell-back-squat', r.id))
  const run = store.readRun('barbell-back-squat', r.id)
  expect(run.status).toBe('failed')
  expect(run.error.step).toBe('check')
  expect(run.steps.find((s) => s.id === 'check').status).toBe('failed')
  expect(run.summary).toBeNull()
})

test('a process that cannot start fails in plain words', async () => {
  make(async (call, child) => { child.emit('error', Object.assign(new Error('spawn x ENOENT'), { code: 'ENOENT' })) })
  await addImage()
  const r = queue.enqueue('barbell-back-squat')
  await until(finished('barbell-back-squat', r.id))
  const run = store.readRun('barbell-back-squat', r.id)
  expect(run.status).toBe('failed')
  expect(run.error.message).toBe("The Studio's Python isn't set up. Run: cd studio && npm run setup")
})

test('a missing Blender is named in plain words', async () => {
  make(async (call, child) => {
    if (kindOf(call) === 'animate') child.emit('error', Object.assign(new Error('x'), { code: 'ENOENT' }))
    else goodScript(call, child, findRunDir(call))
  })
  await addImage()
  const r = queue.enqueue('barbell-back-squat')
  await until(finished('barbell-back-squat', r.id))
  expect(store.readRun('barbell-back-squat', r.id).error.message).toBe("Blender isn't where the Studio expects it (/Applications/Blender.app).")
})

test('applies detected views even when read exits 3', async () => {
  let applied
  store.applyDetectedViews = (slug, m) => { applied = m }
  make(async (call, child, dir) => {
    await write(dir, 'measurements.json', '{"img-1":{"view":"front"}}')
    child.err('Only one position found.')
    child.exit(3)
  })
  await addImage()
  const r = queue.enqueue('barbell-back-squat')
  await until(finished('barbell-back-squat', r.id))
  expect(applied).toEqual({ 'img-1': { view: 'front' } })
  expect(store.readRun('barbell-back-squat', r.id).error.message).toBe('Only one position found.')
})

test('steps never seen starting record no seconds', async () => {
  make(async (call, child, dir) => {
    if (kindOf(call) === 'animate') {
      await write(dir, 'animation.gif', 'GIF'); await write(dir, 'frames/retarget.json'); await write(dir, 'scene.blend', 'B'); child.exit(0)
    } else goodScript(call, child, dir)
  })
  await addImage()
  const r = queue.enqueue('barbell-back-squat')
  await until(finished('barbell-back-squat', r.id))
  const run = store.readRun('barbell-back-squat', r.id)
  expect(run.status).toBe('done')
  expect(run.steps.find((s) => s.id === 'animate').seconds).toBeNull()
  expect(run.steps.find((s) => s.id === 'render').seconds).toBeNull()
  expect(store.stepHistory('animate')).toEqual([])
})

test('an error inside the queue fails that run and the queue keeps going', async () => {
  let n = 0
  make(goodScript, { opts: { otherBlenderRunning: () => { if (++n === 1) throw new Error('pgrep exploded'); return false } } })
  await addImage('barbell-back-squat')
  const a = queue.enqueue('barbell-back-squat')
  await until(finished('barbell-back-squat', a.id))
  const run = store.readRun('barbell-back-squat', a.id)
  expect(run.status).toBe('failed')
  expect(run.error.step).toBe('internal')
  expect(run.error.message).toContain('pgrep exploded')
  expect(run.steps.find((s) => s.id === 'read').status).toBe('done')
  expect(queue.state().running).toBeNull()
  const b = queue.enqueue('barbell-back-squat')
  await until(finished('barbell-back-squat', b.id))
  expect(store.readRun('barbell-back-squat', b.id).status).toBe('done')
})

test('the real spawn kills the whole process group', async () => {
  const child = realSpawn('sh', ['-c', 'sleep 60 & echo $!; wait'], {})
  const pid = await new Promise((resolve) => child.stdout.once('data', (d) => resolve(Number(d.toString().trim()))))
  const alive = (p) => { try { process.kill(p, 0); return true } catch { return false } }
  expect(alive(pid)).toBe(true)
  child.kill('SIGTERM')
  await until(() => !alive(pid), 3000)
  expect(alive(pid)).toBe(false)
})

test('shutdown handlers are installed once and stop the queue on SIGINT/SIGHUP', async () => {
  const proc = new EventEmitter()
  const calls = []
  const mk = (name) => ({ stop: () => calls.push(`${name}:stop`), forceKill: () => calls.push(`${name}:kill`) })
  const exits = []
  installShutdown(mk('a'), { proc, graceMs: 1, exit: (c) => exits.push(c) })
  installShutdown(mk('b'), { proc, graceMs: 1, exit: (c) => exits.push(c) })
  for (const sig of ['SIGINT', 'SIGHUP', 'SIGTERM']) expect(proc.listenerCount(sig)).toBe(1)
  expect(proc.listenerCount('exit')).toBe(1)
  expect(calls).toEqual(['a:stop'])
  proc.emit('SIGINT')
  await until(() => exits.length)
  expect(calls).toEqual(['a:stop', 'b:stop', 'b:kill'])
  expect(exits).toEqual([130])
})

// ---- a run cut off by a Studio stop or restart runs again once on the next start ----

const RESTARTED = 'Restarted automatically after the Studio was stopped'
// A gate that never opens: the restarted run waits at its first step, nothing finishes.
const hold = () => make(() => new Promise(() => {}))
// An old run (its id sorts before anything created now) that a stop cut off.
const interruptedRun = (slug = 'barbell-back-squat') => {
  const run = {
    id: '20000101-000000-0001', slug, name: slug, status: 'interrupted', createdAt: '2000-01-01T00:00:00.000Z',
    startedAt: null, finishedAt: null, steps: [], summary: null,
    error: { step: 'animate', message: 'The Studio was stopped while this ran.', logTail: [] },
  }
  store.writeRun(run)
  return run
}

test('the newest run, interrupted by a stop, is queued again once with a note, and the old run records it', async () => {
  await addImage()
  const cut = interruptedRun()
  hold()
  const restarted = queue.restartInterrupted()
  expect(restarted).toHaveLength(1)
  const fresh = store.readRun('barbell-back-squat', restarted[0].runId)
  expect(fresh).toMatchObject({ note: RESTARTED, restartOf: cut.id })
  expect(store.readRun('barbell-back-squat', cut.id).restartedAs).toBe(fresh.id)
  expect(queue.restartInterrupted()).toEqual([]) // once
})

test('a run left running by a crash is marked interrupted, then restarted on the same start', async () => {
  await addImage()
  const left = store.createRun('barbell-back-squat')
  left.status = 'running'
  store.writeRun(left)
  hold()
  queue.recoverInterrupted()
  const [again] = queue.restartInterrupted()
  expect(store.readRun('barbell-back-squat', left.id)).toMatchObject({ status: 'interrupted', restartedAs: again.runId })
})

test('nothing loops: a restart that is itself cut off is not restarted again', async () => {
  await addImage()
  const cut = interruptedRun()
  hold()
  const [first] = queue.restartInterrupted()
  queue.stop() // the Studio stops again while the restart waits
  expect(store.readRun('barbell-back-squat', first.runId)).toMatchObject({ status: 'interrupted', restartOf: cut.id })
  hold()
  expect(queue.restartInterrupted()).toEqual([])
})

test('only the newest run counts, and only with usable pictures', async () => {
  await addImage()
  interruptedRun()
  const later = store.createRun('barbell-back-squat')
  later.status = 'done'
  store.writeRun(later)
  interruptedRun('pull-up') // no pictures: a run could only fail
  hold()
  expect(queue.restartInterrupted()).toEqual([])
})

// ---- pictures changed while this exercise's run is going (final fix 4) ----

const REPLACED = 'Replaced: the pictures changed'
const SQUAT = 'barbell-back-squat'

test('pictures changed mid-run: the live run is stopped, marked replaced, and the new run goes next', async () => {
  let first = null
  const record = make(async (call, child, dir) => {
    first ??= dir
    if (dir === first) return // the first run's read never ends on its own
    await goodScript(call, child, dir)
  })
  const img = await addImage()
  const a = queue.enqueue(SQUAT)
  await until(() => queue.state().running?.runId === a.id && record.calls?.length)
  store.updateReference(SQUAT, img.id, { phase: 'end' })
  const b = queue.enqueue(SQUAT)
  expect(record.killed).toBe(1)
  const old = store.readRun(SQUAT, a.id)
  expect(old).toMatchObject({ status: 'interrupted', replacedBy: b.id, error: { message: REPLACED } })
  expect(queue.state().running?.runId).not.toBe(a.id) // never shown running once replaced
  await until(finished(SQUAT, b.id))
  expect(store.readRun(SQUAT, b.id).status).toBe('done')
  expect(store.readRun(SQUAT, a.id)).toMatchObject({ status: 'interrupted', error: { message: REPLACED } })
  // a replaced run never restarts on its own, even when it is the newest left
  await rm(store.runDir(SQUAT, b.id), { recursive: true, force: true })
  expect(queue.restartInterrupted()).toEqual([])
})

test('a run replaced while it waits for Blender stops waiting at once', async () => {
  const record = make(goodScript, { opts: { otherBlenderRunning: () => true, waitMs: 60_000 } })
  const img = await addImage()
  const a = queue.enqueue(SQUAT)
  await until(() => queue.state().running?.note)
  store.updateReference(SQUAT, img.id, { contextOnly: true })
  await addImage()
  const b = queue.enqueue(SQUAT)
  expect(store.readRun(SQUAT, a.id)).toMatchObject({ status: 'interrupted', replacedBy: b.id })
  await until(() => queue.state().running?.runId === b.id)
  expect(record.calls.map(kindOf)).toEqual(['read', 'read'])
})

test('the same pictures mid-run (Run again, a work re-run): nothing is stopped, the new run waits its turn', async () => {
  let first = null
  let release
  const gate = new Promise((r) => (release = r))
  const record = make(async (call, child, dir) => {
    first ??= dir
    if (dir === first) await gate
    await goodScript(call, child, dir)
  })
  const img = await addImage()
  const a = queue.enqueue(SQUAT)
  await until(() => queue.state().running?.runId === a.id)
  // the detector filling in a view is not a change the user made
  store.applyDetectedViews(SQUAT, { [img.id]: { view: 'three-quarter' } })
  const b = queue.enqueue(SQUAT)
  expect(record.killed).toBeUndefined()
  expect(queue.state()).toMatchObject({ running: { runId: a.id }, queued: [{ slug: SQUAT, runId: b.id }] })
  release()
  await until(finished(SQUAT, b.id))
  expect(store.readRun(SQUAT, a.id).status).toBe('done')
})

test('a queued run whose folder was deleted by hand is recreated cleanly, never stuck waiting', async () => {
  let release
  const gate = new Promise((r) => (release = r))
  make(async (call, child, dir) => {
    if (dir.includes('/pull-up/')) await gate
    await goodScript(call, child, dir)
  })
  await addImage('pull-up')
  await addImage()
  queue.enqueue('pull-up')
  await until(() => queue.state().running)
  const b = queue.enqueue(SQUAT)
  await rm(store.runDir(SQUAT, b.id), { recursive: true, force: true })
  const c = queue.enqueue(SQUAT)
  expect(c.id).not.toBe(b.id)
  expect(store.readRun(SQUAT, c.id).status).toBe('queued')
  expect(queue.state().queued).toEqual([{ slug: SQUAT, runId: c.id }])
  release()
  await until(finished(SQUAT, c.id), 8000)
  expect(store.readRun(SQUAT, c.id).status).toBe('done')
})

// ---- a disk error while a step reports progress fails the run, never the dev server (final fix 8) ----

const SAVE_FAILED = "The Studio could not save this run's progress: disk full"

test('a disk error on a step change from the log fails the run plainly', async () => {
  const realWrite = store.writeRun
  let thrown = 0
  store.writeRun = (run) => {
    if (!thrown && run.steps.find((s) => s.id === 'animate')?.status === 'running') {
      thrown += 1
      throw new Error('disk full')
    }
    return realWrite(run)
  }
  const record = make(async (call, child, dir) => {
    if (kindOf(call) !== 'animate') return goodScript(call, child, dir)
    child.out('retarget: read squat.fbx: 2 frames at 30 fps')
    child.out('bake_to_skin: done') // the step change saves run.json, and that save fails
  })
  await addImage()
  const r = queue.enqueue(SQUAT)
  await until(finished(SQUAT, r.id))
  const run = store.readRun(SQUAT, r.id)
  expect(run).toMatchObject({ status: 'failed', error: { step: 'animate', message: SAVE_FAILED } })
  expect(record.killed).toBe(1)
  expect(queue.state().running).toBeNull()
})

test('a disk error while counting rendered frames fails the run plainly', async () => {
  const realWrite = store.writeRun
  let thrown = 0
  store.writeRun = (run) => {
    if (!thrown && typeof run.steps.find((s) => s.id === 'render')?.fraction === 'number') {
      thrown += 1
      throw new Error('disk full')
    }
    return realWrite(run)
  }
  const record = make(async (call, child, dir) => {
    if (kindOf(call) !== 'animate') return goodScript(call, child, dir)
    child.out('retarget: read squat.fbx: 2 frames at 30 fps')
    child.out('retarget report -> x')
    await write(dir, 'frames/frame_0001.png', 'x')
  })
  await addImage()
  const r = queue.enqueue(SQUAT)
  await until(finished(SQUAT, r.id))
  expect(store.readRun(SQUAT, r.id)).toMatchObject({ status: 'failed', error: { step: 'render', message: SAVE_FAILED } })
  expect(record.killed).toBe(1)
})

test('a note from the animate log shows on the running step, and clears with the next step (task 19 review)', async () => {
  let release
  const gate = new Promise((r) => { release = r })
  make(async (call, child, dir) => {
    if (kindOf(call) === 'animate') {
      child.out('retarget: read squat.fbx: 2 frames at 30 fps')
      child.out('bake_to_skin: done')
      child.out('form: picture round 0 (frames start 1, middle 19, end 38): ...')
      await gate
    }
    await goodScript(call, child, dir)
  })
  await addImage()
  const r = queue.enqueue('barbell-back-squat')
  await until(() => queue.state().running?.note === 'Matching the coach in our picture: reading 1')
  expect(queue.state().running.step).toBe('animate')
  release()
  await until(finished('barbell-back-squat', r.id))
  const run = store.readRun('barbell-back-squat', r.id)
  expect(run.status).toBe('done')
  expect(run.steps.every((s) => s.note === null || s.note === undefined)).toBe(true)
})

// ---- re-checking a run without re-rendering (the Studio's check-only re-run) ----

test('enqueueRecheck re-judges the newest rendered run with only the check step', async () => {
  const record = make(goodScript)
  await addImage()
  const full = queue.enqueue('barbell-back-squat')
  await until(finished('barbell-back-squat', full.id), 8000)
  record.calls = [] // forget the full run's spawns; only the re-check's should follow
  const re = queue.enqueueRecheck('barbell-back-squat')
  await until(finished('barbell-back-squat', re.id), 8000)
  const run = store.readRun('barbell-back-squat', re.id)
  expect(run.status).toBe('done')
  expect(run.checkOnly).toBe(true)
  expect(run.recheckOf).toBe(full.id)
  expect(record.calls.map(kindOf)).toEqual(['check']) // nothing was re-rendered
  expect(run.summary).toMatchObject({ pass: 3 })
  expect(store.listRuns('barbell-back-squat').map((r) => r.id)).toContain(re.id)
})

test('enqueueRecheck falls back to a full run when nothing has been rendered yet', async () => {
  const record = make(goodScript)
  await addImage()
  const re = queue.enqueueRecheck('barbell-back-squat')
  await until(finished('barbell-back-squat', re.id), 8000)
  const run = store.readRun('barbell-back-squat', re.id)
  expect(run.status).toBe('done')
  expect(run.checkOnly).toBeFalsy()
  expect(record.calls.map(kindOf)).toEqual(['read', 'animate', 'stills', 'check'])
})
