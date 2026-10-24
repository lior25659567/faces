import { spawn, execFileSync } from 'node:child_process'
import { appendFileSync, existsSync, mkdirSync, readFileSync, readdirSync, rmSync, statSync, writeFileSync } from 'node:fs'
import path from 'node:path'
import { DEFAULT_SECONDS, estimateStep, timeLeft } from './eta.js'
import { BLENDER, PYTHON } from './paths.js'
import { PROCESSES, STEPS, commandFor, readLine, requiredOutputs } from './steps.js'

const WAIT_NOTE = 'Waiting for Blender to close (another Blender is open or running)'
const STOPPED = 'The Studio was stopped while this ran.'
const RESTARTED = 'Restarted automatically after the Studio was stopped'
const REPLACED = 'Replaced: the pictures changed'
const REPLACES = 'The pictures changed while the run before this one was going, so that run was stopped.'
const SAVE_FAILED = "The Studio could not save this run's progress"
const BLENDER_PROCESSES = new Set(['animate', 'stills'])

// Own process group so one signal reaches Blender's helpers (ffmpeg...) too.
export const realSpawn = (cmd, args, opts) => {
  const child = spawn(cmd, args, {
    cwd: opts?.cwd, stdio: ['ignore', 'pipe', 'pipe'], detached: true, env: { ...process.env, PYTHONUNBUFFERED: '1' },
  })
  const own = child.kill.bind(child)
  child.kill = (signal = 'SIGTERM') => {
    try {
      if (child.pid) return process.kill(-child.pid, signal)
    } catch { /* group already gone */ }
    return own(signal)
  }
  return child
}

const spawnFailure = (cmd, e) => {
  if (e?.code === 'ENOENT') {
    if (cmd === BLENDER) return "Blender isn't where the Studio expects it (/Applications/Blender.app)."
    if (cmd === PYTHON) return "The Studio's Python isn't set up. Run: cd studio && npm run setup"
  }
  return `The Studio could not start this step: ${e?.message || e}`
}

// Ctrl-C, closing the terminal and kill all stop the child too. One registration per process,
// however many times Vite reloads its config; later calls just swap in the newest queue.
const SHUTDOWN = Symbol.for('exercise-studio.shutdown')
export const installShutdown = (queue, { proc = process, graceMs = 1500, exit = (c) => process.exit(c) } = {}) => {
  if (proc[SHUTDOWN]) {
    proc[SHUTDOWN].queue.stop()
    proc[SHUTDOWN].queue = queue
    return
  }
  const box = (proc[SHUTDOWN] = { queue })
  const codes = { SIGINT: 130, SIGHUP: 129, SIGTERM: 143 }
  for (const [sig, code] of Object.entries(codes)) {
    proc.on(sig, () => {
      box.queue.stop()
      setTimeout(() => { box.queue.forceKill(); exit(code) }, graceMs)
    })
  }
  proc.on('exit', () => box.queue.forceKill())
}

const pgrepBlender = () => {
  try {
    execFileSync('pgrep', ['-f', 'Blender.app/Contents/MacOS/Blender'], { stdio: 'ignore' })
    return true
  } catch {
    return false // pgrep exits 1 when nothing matches
  }
}

const nonEmpty = (file) => {
  try {
    return statSync(file).isFile() && statSync(file).size > 0
  } catch {
    return false
  }
}

// What a run measures from the pictures: which ones, in what order, with which tags. A view the
// detector filled in is not a change the user made (src/lib/run-view.js picturesKey reads them the same way).
const picturesKey = (snapshot) =>
  [...(snapshot?.images ?? [])]
    .sort((a, b) => a.order - b.order)
    .map((i) => `${i.id}:${i.file}:${i.phase}:${i.viewSetBy === 'user' ? i.view : ''}:${i.contextOnly ? 1 : 0}`)
    .join('|')

const countFrames = (dir) => {
  try {
    return readdirSync(dir).filter((f) => /^frame_.*\.png$/.test(f)).length
  } catch {
    return 0
  }
}

export const createQueue = ({
  store, spawnProcess = realSpawn, otherBlenderRunning = pgrepBlender, now = Date.now, pollMs = 2000, waitMs = 5000,
}) => {
  const waiting = [] // [{slug, runId}] not started yet
  let live = null // the run being worked on
  let child = null
  let stopped = false
  let pumping = false
  const timers = new Set()
  const wakers = new Set()

  const sleep = (ms) =>
    new Promise((resolve) => {
      const t = setTimeout(() => { timers.delete(t); wakers.delete(resolve); resolve() }, ms)
      timers.add(t)
      wakers.add(resolve)
    })

  const iso = () => new Date(now()).toISOString()

  // TERM to the child's process group (Blender and its helpers), KILL two seconds later if it lingers.
  const killChild = () => {
    if (!child) return
    const c = child
    try {
      c.kill('SIGTERM')
    } catch { /* already gone */ }
    const t = setTimeout(() => { try { c.kill('SIGKILL') } catch { /* already gone */ } }, 2000)
    t.unref?.()
  }

  // ---- step bookkeeping (all on `live.run`, saved to run.json at each change) ----
  const save = () => store.writeRun(live.run)
  const stepOf = (id) => live.run.steps.find((s) => s.id === id)

  const startStep = (id) => {
    const s = stepOf(id)
    s.status = 'running'
    s.startedAt = iso()
    s.note = null
    s.fraction = null
    live.step = id
    live.t0 = now()
    live.fraction = null
    live.note = null
    save()
  }

  const finishStep = (id, status) => {
    const s = stepOf(id)
    if (s.status === 'done' || s.status === 'failed') return
    s.status = status
    s.finishedAt = iso()
    // A step never seen starting has no time of its own: null, so history and estimates ignore it.
    s.seconds = live.step === id && live.t0 !== null ? Math.round(((now() - live.t0) / 1000) * 10) / 10 : null
    s.note = null
    save()
  }

  const fail = (stepId, message, logTail = []) => {
    try {
      finishStep(stepId, 'failed')
    } catch { /* still record the failure below */ }
    live.run.status = 'failed'
    live.run.finishedAt = iso()
    live.run.error = { step: stepId, message, logTail }
    try {
      save()
    } catch { /* disk trouble: nothing more the queue can do */ }
  }

  // Something broke in the queue itself. File it under the step that was really running.
  const failInternal = (error) => {
    const message = `The Studio hit an error: ${error?.message || error}`
    if (live?.replaced) return true // already marked: replaced by a run with newer pictures
    if (live?.run) {
      const running = live.run.steps.find((s) => s.status === 'running')
      if (running) {
        live.step = running.id
        fail(running.id, message)
      } else {
        fail('internal', message)
      }
      return true
    }
    return false
  }

  // ---- running one process ----
  // `onError` hears of anything that fails while a line is handled (the log or run.json can't be written):
  // thrown inside the child's 'data' event it would take the whole dev server down.
  const spawnAndWatch = (cmd, args, cwd, logFile, onLine, onError) =>
    new Promise((resolve) => {
      mkdirSync(path.dirname(logFile), { recursive: true })
      writeFileSync(logFile, '')
      const tail = []
      let lastErr = ''
      let settled = false
      const done = (code, spawnError) => {
        if (settled) return
        settled = true
        try {
          for (const b of Object.values(bufs)) if (b.text) take(b, b.text)
        } catch (e) {
          onError(e)
        }
        resolve({ code, tail: tail.slice(-20), lastErr, spawnError })
      }
      const bufs = { out: { text: '', err: false }, err: { text: '', err: true } }
      const take = (b, line) => {
        b.text = ''
        if (!line.trim()) return
        appendFileSync(logFile, line + '\n')
        tail.push(line)
        if (tail.length > 40) tail.shift()
        if (b.err) lastErr = line
        onLine(line)
      }
      const feed = (b) => (chunk) => {
        try {
          b.text += chunk.toString()
          const parts = b.text.split(/\r?\n|\r/)
          b.text = parts.pop()
          for (const line of parts) take({ ...b, text: '' }, line)
        } catch (e) {
          onError(e)
        }
      }
      try {
        child = spawnProcess(cmd, args, { cwd })
      } catch (e) {
        return done(-1, spawnFailure(cmd, e))
      }
      child.stdout?.on('data', feed(bufs.out))
      child.stderr?.on('data', feed(bufs.err))
      child.on('error', (e) => done(-1, spawnFailure(cmd, e)))
      child.on('close', (code) => done(code))
    })

  // Stopped, or replaced by a run with newer pictures: this run must not touch anything any more.
  const cancelled = (current) => stopped || Boolean(current?.replaced)

  const waitForBlender = async (firstStepId) => {
    const s = stepOf(firstStepId)
    const mine = live
    while (!cancelled(mine) && otherBlenderRunning()) {
      s.status = 'running'
      s.note = WAIT_NOTE
      live.step = firstStepId
      live.t0 = null // time spent waiting is not the step's time
      live.note = WAIT_NOTE
      save()
      await sleep(waitMs)
    }
  }

  const runProcess = async (proc, runDir) => {
    const first = proc.steps[0]
    const cmd = commandFor(proc.id, live.run, runDir, store.getExercise?.(live.run.slug) || live.run)
    if (cmd.error) return fail(first, cmd.error)
    const mine = live
    if (BLENDER_PROCESSES.has(proc.id)) {
      await waitForBlender(first)
      if (cancelled(mine)) return
    }
    startStep(first)

    // A disk error while saving progress: stop the process and fail the run in plain words below.
    let broken = null
    const fault = (e) => {
      if (broken || cancelled(mine)) return
      broken = e
      killChild()
    }
    const state = {}
    let poll = null
    const startPoll = () => {
      if (poll || !state.total) return
      poll = setInterval(() => {
        if (broken || cancelled(mine) || live.step !== 'render' || !state.total) return
        try {
          const n = countFrames(path.join(runDir, 'frames')) + countFrames(path.join(runDir, 'frames', 'side'))
          live.fraction = Math.min(n / state.total, 1)
          stepOf('render').fraction = live.fraction
          save()
        } catch (e) {
          fault(e)
        }
      }, pollMs)
      timers.add(poll)
    }
    const stopPoll = () => { if (poll) { clearInterval(poll); timers.delete(poll); poll = null } }
    const switchTo = (id) => {
      if (cancelled(mine) || id === live.step || !proc.steps.includes(id)) return
      if (proc.steps.indexOf(id) < proc.steps.indexOf(live.step)) return
      for (const before of proc.steps.slice(0, proc.steps.indexOf(id))) finishStep(before, 'done')
      startStep(id)
      if (id === 'render') startPoll()
    }

    const result = await spawnAndWatch(cmd.cmd, cmd.args, cmd.cwd, path.join(runDir, 'logs', `${proc.id}.log`), (line) => {
      if (broken || cancelled(mine)) return
      const r = readLine(proc.id, line, state)
      if (r.step) switchTo(r.step)
      if (r.total) startPoll()
      if ('note' in r && live.step) {
        live.note = r.note
        stepOf(live.step).note = r.note
        save()
      }
    }, fault)
    stopPoll()
    child = null
    if (cancelled(mine)) return
    if (broken) return fail(live.step, `${SAVE_FAILED}: ${broken?.message || broken}`, result.tail)
    // The detector's views help even when too few positions were found (exit 3).
    if (proc.id === 'read') {
      try {
        const m = readJson(path.join(runDir, 'measurements.json'))
        if (m) store.applyDetectedViews(live.slug, m)
      } catch { /* views are a convenience */ }
    }

    const current = live.step
    if (result.code !== 0) {
      const message = result.spawnError || result.lastErr || (result.code === null ? 'The process was stopped.' : `The process exited with code ${result.code}.`)
      return fail(current, message, result.tail)
    }
    const snapshot = readJson(path.join(runDir, 'references.json'))
    const missing = requiredOutputs(proc.id, runDir, snapshot).filter((rel) => !nonEmpty(path.join(runDir, rel)))
    if (missing.length) return fail(current, `This step finished without producing ${missing.join(', ')}`, result.tail)
    for (const id of proc.steps) finishStep(id, 'done')
    return true
  }

  const readJson = (file) => {
    try {
      return JSON.parse(readFileSync(file, 'utf8'))
    } catch {
      return null
    }
  }

  const execute = async ({ slug, runId }) => {
    const run = store.readRun(slug, runId)
    if (!run) return
    const runDir = store.runDir(slug, runId)
    live = { slug, runId, run, step: null, t0: null, fraction: null, note: null, estimates: {} }
    run.status = 'running'
    run.startedAt = iso()
    live.estimates = Object.fromEntries(STEPS.map((s) => [s.id, estimateStep(s.id, store.stepHistory(s.id))]))
    save()
    const mine = live
    // A check-only re-run (store.createRecheckRun) reuses the rendered outputs and runs the check alone.
    const procs = run.checkOnly ? PROCESSES.filter((p) => p.id === 'check') : PROCESSES
    for (const proc of procs) {
      if (cancelled(mine)) return
      if (!(await runProcess(proc, runDir))) {
        if (!stopped) live = null
        return
      }
      if (proc.id === 'check') {
        const summary = readJson(path.join(runDir, 'checks.json'))?.summary
        if (!summary) {
          // Re-open the check step: a checks file without a summary proves nothing.
          const s = stepOf('check')
          s.status = 'running'
          live.step = 'check'
          return fail('check', 'This step finished without producing a summary in checks.json')
        }
        run.summary = summary
      }
    }
    run.status = 'done'
    run.finishedAt = iso()
    save()
  }

  const pump = async () => {
    if (pumping) return
    pumping = true
    try {
      while (!stopped && waiting.length) {
        const next = waiting.shift()
        try {
          await execute(next)
        } catch (e) {
          if (!stopped && !failInternal(e)) {
            // Failed before the run was picked up: mark it through the store so it cannot sit "queued".
            try {
              const r = store.readRun(next.slug, next.runId)
              if (r) {
                r.status = 'failed'
                r.finishedAt = iso()
                r.error = { step: 'internal', message: `The Studio hit an error: ${e?.message || e}`, logTail: [] }
                store.writeRun(r)
              }
            } catch { /* nothing left to try */ }
          }
        }
        if (!stopped) live = null
      }
    } finally {
      pumping = false
    }
  }

  // Stops the live run because its pictures changed: its process group is killed (as stop() does), it is
  // marked interrupted with the reason, and it is never restarted on its own.
  const replace = (current, byId) => {
    current.replaced = true
    current.run.replacedBy = byId
    for (const wake of wakers) wake() // a run waiting for Blender stops waiting
    wakers.clear()
    killChild()
    try {
      interrupt(current.run, REPLACED)
    } catch { /* disk trouble: the new run still goes ahead */ }
  }

  const enqueue = (slug, extra = {}) => {
    let fresh = store.createRun(slug, extra)
    const freshDir = store.runDir(slug, fresh.id)
    // This exercise's run is going with other pictures than these: the new run replaces it.
    const current = live && live.slug === slug && !live.replaced ? live : null
    if (current && picturesKey(readJson(path.join(store.runDir(slug, current.runId), 'references.json'))) !==
        picturesKey(readJson(path.join(freshDir, 'references.json')))) {
      fresh = { ...fresh, replaces: current.runId, note: fresh.note ?? REPLACES }
      store.writeRun(fresh)
      replace(current, fresh.id)
    }
    const at = waiting.findIndex((w) => w.slug === slug)
    const old = at < 0 ? null : store.readRun(slug, waiting[at].runId)
    let run = fresh
    if (old) {
      // Same exercise still waiting: keep its id, give it the newest references.
      const oldDir = store.runDir(slug, old.id)
      writeFileSync(path.join(oldDir, 'references.json'), readFileSync(path.join(freshDir, 'references.json')))
      rmSync(freshDir, { recursive: true, force: true })
      run = { ...old, ...extra }
      for (const k of ['name', 'clip', 'replaces', 'note']) if (fresh[k] !== undefined) run[k] = fresh[k]
      store.writeRun(run)
      if (fresh.replaces) {
        current.run.replacedBy = run.id
        try {
          store.writeRun(current.run)
        } catch { /* disk trouble: the new run still goes ahead */ }
      }
    } else {
      // A waiting run whose folder was deleted by hand can't be merged into: the new one takes its place.
      if (at >= 0) waiting.splice(at, 1)
      waiting.push({ slug, runId: fresh.id })
    }
    setImmediate(() => pump().catch(() => {}))
    return run
  }

  // The check reads these from a run's folder; a run that has them was rendered and can be re-checked.
  const RECHECK_INPUTS = ['targets.json', 'ours.json', path.join('frames', 'retarget.json')]
  const rendered = (slug, runId) => RECHECK_INPUTS.every((rel) => nonEmpty(path.join(store.runDir(slug, runId), rel)))

  // Re-judge the newest rendered run without re-rendering: a new run that reuses its outputs and runs only
  // the check step. Nothing rendered yet? A full run is the only way to get something to check.
  const enqueueRecheck = (slug) => {
    const source = store.listRuns(slug).find((r) => rendered(slug, r.id))
    if (!source) return enqueue(slug)
    const run = store.createRecheckRun(slug, source)
    waiting.push({ slug, runId: run.id })
    setImmediate(() => pump().catch(() => {}))
    return run
  }

  const state = () => {
    let running = null
    if (live && !live.replaced) {
      const elapsed = live.t0 === null ? 0 : Math.max((now() - live.t0) / 1000, 0)
      const estimates = { ...DEFAULT_SECONDS, ...live.estimates }
      const { seconds, overdue } = timeLeft({ steps: STEPS, currentId: live.step, elapsed, fraction: live.fraction, estimates })
      // The current step alone (no later steps), for the page's per-step "time left".
      const step = timeLeft({ steps: [live.step], currentId: live.step, elapsed, fraction: live.fraction, estimates })
      running = {
        slug: live.slug, runId: live.runId, step: live.step, label: STEPS.find((s) => s.id === live.step)?.label ?? null,
        elapsed: Math.round(elapsed), fraction: live.fraction, left: seconds, stepLeft: step.seconds, overdue, note: live.note,
      }
    }
    return { running, queued: waiting.map(({ slug, runId }) => ({ slug, runId })) }
  }

  const interrupt = (run, message) => {
    const open = run.steps.find((s) => s.status === 'running') || run.steps.find((s) => s.status === 'waiting')
    for (const s of run.steps) if (s.status === 'running') { s.status = 'failed'; s.finishedAt = new Date(now()).toISOString() }
    run.status = 'interrupted'
    run.finishedAt = new Date(now()).toISOString()
    run.error = { step: open?.id ?? null, message, logTail: [] }
    store.writeRun(run)
  }

  // Runs the disk says are still going were cut off by a stop or crash.
  const recoverInterrupted = () => {
    for (const run of store.allRuns()) {
      if (run.status === 'running') interrupt(run, STOPPED)
      else if (run.status === 'queued') interrupt(run, 'The Studio was stopped before this started.')
    }
  }

  // A stop or restart cut these off: each exercise whose newest run was interrupted runs again, once.
  // The old run records its restart; a run that was itself such a restart is never restarted, so
  // nothing can loop. Without usable pictures a run could only fail, so none is started.
  const restartInterrupted = () => {
    const restarted = []
    for (const { slug } of store.listExercises()) {
      const newest = store.listRuns(slug)[0]
      if (!newest || newest.status !== 'interrupted' || newest.restartedAs || newest.restartOf || newest.replacedBy) continue
      if (!store.hasUsablePictures(slug)) continue
      const run = enqueue(slug, { note: RESTARTED, restartOf: newest.id })
      store.writeRun({ ...newest, restartedAs: run.id, restartedAt: iso() })
      restarted.push({ slug, runId: run.id })
    }
    return restarted
  }

  const stop = () => {
    if (stopped) return
    stopped = true
    for (const t of timers) { clearTimeout(t); clearInterval(t) }
    timers.clear()
    for (const wake of wakers) wake()
    wakers.clear()
    killChild()
    if (live && !live.replaced) interrupt(live.run, STOPPED)
    for (const w of waiting.splice(0)) {
      const run = store.readRun(w.slug, w.runId)
      if (run) interrupt(run, 'The Studio was stopped before this started.')
    }
  }

  const forceKill = () => {
    try {
      child?.kill('SIGKILL')
    } catch { /* already gone */ }
  }

  return { enqueue, enqueueRecheck, state, recoverInterrupted, restartInterrupted, stop, forceKill }
}
