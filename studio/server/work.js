import { closeSync, existsSync, mkdirSync, openSync, readFileSync, readdirSync, renameSync, rmSync, statSync, writeFileSync } from 'node:fs'
import { randomBytes } from 'node:crypto'
import path from 'node:path'
import { DATA } from './paths.js'
import { slugOf, stamp } from './store.js'

// Work going on behind the scenes (Claude or a helper agent fixing the pipeline), one file per item
// in DATA/work/<id>.json. The CLI (scripts/work.mjs) writes them; the Studio reads them, shows them,
// and re-runs the listed exercises once an item is marked done.

export class WorkError extends Error {}

const ID = /^\d{8}-\d{6}-[0-9a-f]{4}$/
const ENDED = { done: 'marked done', failed: 'marked failed' }
const LOCK_WAIT_MS = 3000
const LOCK_STALE_MS = 10_000 // a read-modify-write takes milliseconds; older means its writer died
const WATCHER_WAIT_MS = 200 // the Studio never blocks long on a busy item: it tries again next look
const OTHERS_SHOWN = 50

const text = (value) => (typeof value === 'string' ? value.trim() : '')
const pause = (ms) => Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, ms)

export const createWork = (dataRoot = DATA) => {
  const dir = path.join(dataRoot, 'work')
  const fileOf = (id) => path.join(dir, `${id}.json`)

  // One writer per item at a time (CLI processes and the Studio): an O_EXCL lock file around each
  // read-modify-write, so no writer can undo another's change.
  const withLock = (id, fn, { waitMs = LOCK_WAIT_MS } = {}) => {
    mkdirSync(dir, { recursive: true })
    const lock = path.join(dir, `${id}.lock`)
    const end = Date.now() + waitMs
    for (;;) {
      try {
        closeSync(openSync(lock, 'wx'))
        break
      } catch (e) {
        if (e.code !== 'EEXIST') throw e
      }
      try {
        if (Date.now() - statSync(lock).mtimeMs > LOCK_STALE_MS) {
          rmSync(lock, { force: true })
          continue
        }
      } catch {
        continue // released meanwhile
      }
      if (Date.now() > end) throw new WorkError(`Work item ${id} is busy: another writer is changing it. Try again.`)
      pause(5 + Math.random() * 20)
    }
    try {
      return fn()
    } finally {
      rmSync(lock, { force: true })
    }
  }

  // Whole file or nothing: a reader never sees half an item. Unlocked: changes go through update().
  const write = (item) => {
    mkdirSync(dir, { recursive: true })
    const tmp = `${fileOf(item.id)}.${process.pid}.tmp`
    writeFileSync(tmp, JSON.stringify(item, null, 2))
    renameSync(tmp, fileOf(item.id))
    return item
  }

  const mustExist = (id, item) => {
    if (!item) throw new WorkError(`There is no work item "${id}". Use the id that "start" printed (or see "list").`)
    return item
  }

  // Read, change and write one item under its lock. `change` returns the new item, or null to leave it.
  const update = (id, change, lockOptions) => {
    if (!ID.test(String(id))) mustExist(id, null)
    return withLock(id, () => {
      const item = mustExist(id, self.read(id))
      const next = change(item)
      return next ? write(next) : item
    }, lockOptions)
  }

  const start = ({ title, exercises = [] } = {}) => {
    const clean = text(title)
    if (!clean) throw new WorkError('Give the work a title, e.g. --title "Fix skin folds at the shoulders".')
    for (;;) {
      const id = `${stamp()}-${randomBytes(2).toString('hex')}`
      const made = withLock(id, () => {
        if (existsSync(fileOf(id))) return null
        const at = new Date().toISOString()
        return write({
          id, title: clean, exercises: [...exercises], status: 'running', startedAt: at, updatedAt: at, finishedAt: null,
          log: [], summary: null, rerun: null,
        })
      })
      if (made) return made
    }
  }

  // A line can be added at any time (a note after the fact too); it never changes the status.
  const log = (id, line) => {
    const clean = text(line)
    if (!clean) throw new WorkError('Say what to log, e.g. log <id> "Measured the folds again".')
    return update(id, (item) => {
      const at = new Date().toISOString()
      return { ...item, log: [...(item.log || []), { at, text: clean }], updatedAt: at }
    })
  }

  // `opts.recheck` (done only): this fix only changed the check rules, so the Studio re-judges the
  // listed exercises' newest rendered run instead of re-rendering them (work watcher -> enqueueRecheck).
  const finish = (id, status, summary, opts = {}) => {
    if (status !== 'done' && status !== 'failed') throw new WorkError('A work item finishes as done or failed.')
    const clean = text(summary)
    return update(id, (item) => {
      if (item.status !== 'running') {
        throw new WorkError(`Work item ${item.id} is already ${ENDED[item.status] ?? item.status}. Start a new one for more work.`)
      }
      if (!clean) throw new WorkError(`Say what happened with --summary "…" when marking it ${status}.`)
      const at = new Date().toISOString()
      const recheck = status === 'done' && opts.recheck ? { recheck: true } : {}
      return { ...item, status, summary: clean, finishedAt: at, updatedAt: at, ...recheck }
    })
  }

  const names = () => {
    try {
      return readdirSync(dir).filter((n) => n.endsWith('.json'))
    } catch {
      return []
    }
  }

  const list = () => names().map((n) => self.read(n.slice(0, -5))).filter(Boolean)

  // The same items, re-reading a file only when its mtime or size changed, or when it was last seen
  // running (so a rewrite inside one mtime tick can't hide running -> done). The page and the
  // watcher both read through this.
  const seen = new Map() // id -> { key, item }
  const cached = () => {
    const out = []
    const present = new Set()
    for (const name of names()) {
      const id = name.slice(0, -5)
      present.add(id)
      let key
      try {
        const st = statSync(path.join(dir, name))
        key = `${st.mtimeMs}:${st.size}`
      } catch {
        continue
      }
      const last = seen.get(id)
      if (last && last.key === key && last.item.status !== 'running') {
        out.push(last.item)
        continue
      }
      const item = self.read(id)
      if (!item) continue // unreadable: try again next time
      seen.set(id, { key, item })
      out.push(item)
    }
    for (const id of seen.keys()) if (!present.has(id)) seen.delete(id)
    return out
  }

  const self = {
    dir,
    read: (id) => {
      if (!ID.test(String(id))) return null
      try {
        return JSON.parse(readFileSync(fileOf(id), 'utf8'))
      } catch {
        return null
      }
    },
    write, update, withLock, start, log, finish, list, cached,
  }
  return self
}

// Running first, then the newest.
export const sortWork = (items) =>
  [...items].sort((a, b) => {
    const ra = a.status === 'running' ? 0 : 1
    const rb = b.status === 'running' ? 0 : 1
    if (ra !== rb) return ra - rb
    if (a.startedAt !== b.startedAt) return a.startedAt < b.startedAt ? 1 : -1
    return a.id < b.id ? 1 : -1
  })

// What the page gets: every running item, then the newest 50 others.
export const recentWork = (items, others = OTHERS_SHOWN) => {
  const sorted = sortWork(items)
  const running = sorted.filter((w) => w.status === 'running')
  return [...running, ...sorted.filter((w) => w.status !== 'running').slice(0, others)]
}

// "--exercises a,b" (slugs or names) or "all" (every exercise with usable pictures) -> slugs.
export const resolveExercises = (store, spec) => {
  const all = store.listExercises()
  if (spec === undefined || spec === null || spec === true) return []
  if (String(spec).trim().toLowerCase() === 'all') return all.map((e) => e.slug).filter((s) => store.hasUsablePictures(s))
  const slugs = []
  for (const part of String(spec).split(',').map((p) => p.trim()).filter(Boolean)) {
    const slug = slugOf(part)
    if (!all.some((e) => e.slug === slug)) {
      throw new WorkError(`There is no exercise called "${part}". Use one of: ${all.map((e) => e.slug).join(', ')}, or "all".`)
    }
    if (!slugs.includes(slug)) slugs.push(slug)
  }
  return slugs
}

// Re-runs the exercises of every done item that hasn't been re-run yet, once: items that just went
// running -> done, and items found already done when the Studio starts (finished while it was
// closed or restarting). An exercise that already has a run made after the item finished isn't
// run again: that run is recorded instead (by hand, or another item's re-run).
export const createWorkWatcher = ({ work, store, queue, intervalMs = 2500 }) => {
  const handled = new Set() // ids re-run by this watcher, in case writing the record failed
  const ours = new Set() // run ids this watcher queued
  let timer = null

  const nameOf = (slug) => store.getExercise(slug)?.name ?? slug
  const automatic = (items) => new Set([...ours, ...items.flatMap((w) => (w.rerun ?? []).filter((r) => !r.byHand).map((r) => r.runId))])

  const rerunOne = (item, items) => {
    const auto = automatic(items)
    work.update(item.id, (fresh) => {
      if (fresh.status !== 'done' || fresh.rerunQueuedAt) return null
      handled.add(fresh.id) // from here on runs are queued: never again, even if the write below fails
      const rerun = []
      const noPictures = []
      const notStarted = []
      for (const slug of fresh.exercises || []) {
        if (!store.getExercise(slug) || !store.hasUsablePictures(slug)) {
          noPictures.push(nameOf(slug))
          continue
        }
        const since = fresh.finishedAt && store.listRuns(slug).find((r) => r.createdAt >= fresh.finishedAt)
        if (since) {
          rerun.push(auto.has(since.id) || since.restartOf ? { slug, runId: since.id } : { slug, runId: since.id, byHand: true })
          continue
        }
        try {
          // A recheck item re-judges the newest rendered run (fast, no Blender); otherwise a full run.
          const run = fresh.recheck ? queue.enqueueRecheck(slug) : queue.enqueue(slug)
          ours.add(run.id)
          rerun.push({ slug, runId: run.id })
        } catch (e) {
          console.error(`[studio] work ${fresh.id}: could not start a run for ${slug}:`, e)
          notStarted.push(nameOf(slug))
        }
      }
      const notes = []
      if (!fresh.exercises?.length) notes.push('No exercises were listed, so nothing re-ran.')
      else if (!rerun.length && noPictures.length === fresh.exercises.length) {
        notes.push('Nothing re-ran: none of its exercises have reference pictures yet.')
      } else if (noPictures.length) notes.push(`Not re-run, no reference pictures yet: ${noPictures.join(', ')}.`)
      if (notStarted.length) notes.push(`Could not start a run for ${notStarted.join(', ')}.`)
      return { ...fresh, rerun, rerunQueuedAt: new Date().toISOString(), rerunNote: notes.join(' ') || null }
    }, { waitMs: WATCHER_WAIT_MS })
  }

  const scan = () => {
    let items
    try {
      items = work.cached()
    } catch {
      return // no work folder yet, or the disk is unhappy: look again next time
    }
    for (const item of items) {
      if (item.status !== 'done' || item.rerunQueuedAt || handled.has(item.id)) continue
      try {
        rerunOne(item, items)
      } catch (e) {
        console.error(`[studio] work ${item.id}: could not record its re-run:`, e)
      }
    }
  }

  const start = () => {
    if (timer) return
    scan()
    timer = setInterval(scan, intervalMs)
    timer.unref?.()
  }

  const stop = () => {
    clearInterval(timer)
    timer = null
  }

  return { scan, start, stop }
}
