// Barbell back squat milestone, repeatable and non-destructive.
// Ensures the five reference images exist and are tagged (uploads only what is missing, patches
// only tags that differ, never deletes), starts one run unless the squat is already queued or
// running, then follows it until it ends. Exit 0 = the run finished (checks may still fail),
// 1 = the run failed or was interrupted (or the Studio could not be used).
// Usage: node studio/scripts/squat-milestone.mjs [--url http://localhost:5199] [--no-run]
import { linkSync, readFileSync, unlinkSync, writeFileSync } from 'node:fs'
import { createHash } from 'node:crypto'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'

const REPO = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..')
export const SLUG = 'barbell-back-squat'
export const DIR = path.join(REPO, '3d-character', 'reference-squat')
// Upload order is the display order.
export const PLAN = [
  { file: '14.png', want: { contextOnly: true } },
  { file: '15.png', want: { phase: 'start', view: 'side', contextOnly: false } },
  { file: '16.png', want: { phase: 'middle', view: 'side', contextOnly: false } },
  { file: '17.png', want: { phase: 'end', view: 'side', contextOnly: false } },
  { file: 'behind-thumb.png', want: { phase: 'start', view: 'behind', contextOnly: false } },
]

// The existing, not-deleted reference uploaded from this original file name, if any.
export const findRef = (refs, file) => refs.find((r) => !r.deleted && r.original === file) || null

// A deleted reference uploaded from this original file name (the user removed it on purpose).
export const findDeleted = (refs, file) => refs.find((r) => r.deleted && r.original === file) || null

// What to change on a picture: {patch, kept}. A tag that already matches is left alone. A phase or
// view the user set by hand is kept as is (listed in `kept`); only guesses and missing tags are patched.
export const tagPatch = (ref, want) => {
  const patch = {}
  const kept = []
  for (const [k, v] of Object.entries(want)) {
    if (ref[k] === v) continue
    if ((k === 'phase' || k === 'view') && ref[`${k}SetBy`] === 'user') kept.push({ key: k, has: ref[k], wanted: v })
    else patch[k] = v
  }
  return { patch, kept }
}

// An O_EXCL lock file per Studio URL + slug, so two copies cannot upload duplicates.
export const lockPath = (base, slug, dir = os.tmpdir()) =>
  path.join(dir, `squat-milestone-${createHash('sha1').update(`${base}|${slug}`).digest('hex').slice(0, 12)}.lock`)

// The lock appears already holding its pid: the pid goes into a temp file that is hard-linked to
// the lock path (fails if the lock exists). An unreadable or zero pid counts as held; only a
// readable pid of a dead process is taken over.
export const takeLock = (file, alive = (pid) => { try { process.kill(pid, 0); return true } catch { return false } }) => {
  const tmp = `${file}.${process.pid}.tmp`
  writeFileSync(tmp, String(process.pid))
  try {
    for (let tries = 0; tries < 2; tries++) {
      try {
        linkSync(tmp, file)
        return true
      } catch (e) {
        if (e.code !== 'EEXIST') throw e
        let pid = 0
        try { pid = Number(readFileSync(file, 'utf8')) } catch { /* unreadable: treat as held */ }
        if (!Number.isInteger(pid) || pid <= 0 || alive(pid)) return false
        try { unlinkSync(file) } catch { /* another copy took it */ }
      }
    }
    return false
  } finally {
    try { unlinkSync(tmp) } catch { /* already gone */ }
  }
}

// The squat's run that is already waiting or going, as {runId} (or null).
export const activeRun = (state, slug = SLUG) => {
  if (state.running?.slug === slug) return { runId: state.running.runId }
  const q = (state.queued || []).find((x) => x.slug === slug)
  return q ? { runId: q.runId } : null
}

export const fmt = (s) => `${Math.floor(s / 60)}m${String(Math.round(s % 60)).padStart(2, '0')}s`

export const rowLine = (r) =>
  `${String(r.status).toUpperCase().padEnd(9)} ${String(r.group).padEnd(8)} ${String(r.metric).padEnd(22)} target=${r.target ?? '-'} ours=${r.ours ?? '-'}${r.unit ? ' ' + r.unit : ''}${r.note ? '  | ' + r.note : ''}`

const args = process.argv.slice(2)
const argOf = (name) => { const i = args.indexOf(name); return i >= 0 ? args[i + 1] : null }
const BASE = (argOf('--url') || process.env.STUDIO_URL || 'http://localhost:5199').replace(/\/$/, '')
const sleep = (ms) => new Promise((r) => setTimeout(r, ms))

const call = async (method, url, { body, headers } = {}) => {
  const res = await fetch(BASE + url, { method, headers, body })
  const text = await res.text()
  let json = null
  try { json = text ? JSON.parse(text) : null } catch { /* not json */ }
  if (!res.ok) throw new Error(`${method} ${url} -> ${res.status} ${json?.error || text.slice(0, 200)}`)
  return json
}
const json = (method, url, obj) => call(method, url, { body: JSON.stringify(obj), headers: { 'content-type': 'application/json' } })

const refsNow = async () => (await call('GET', `/api/exercises/${SLUG}`)).references

// Deleted pictures are hidden by the API but kept in the exercise's references.json, which the
// Studio serves read-only under /files.
// Fails CLOSED: if we cannot tell what was deleted, nothing is uploaded. No references.json at all
// (404) just means nothing was ever uploaded, so nothing was deleted.
export class CannotCheckError extends Error {}
const deletedNow = async () => {
  try {
    return (await call('GET', `/files/references/${SLUG}/references.json`)).images.filter((i) => i.deleted)
  } catch (e) {
    if (/ -> 404 /.test(e.message)) return []
    throw new CannotCheckError("Couldn't check which pictures you deleted; not uploading anything")
  }
}

const ensureImages = async () => {
  const deleted = await deletedNow() // before any upload
  for (const { file } of PLAN) {
    const refs = await refsNow()
    const ref = findRef(refs, file)
    if (ref) console.log(`present   ${file} (${ref.id})`)
    else if (findDeleted(deleted, file)) console.log(`WARNING   ${file}: you deleted it; not uploading it again`)
    else {
      const added = await call('POST', `/api/exercises/${SLUG}/references`, {
        body: readFileSync(path.join(DIR, file)), headers: { 'x-filename': encodeURIComponent(file), 'content-type': 'application/octet-stream' },
      })
      console.log(`uploaded  ${file} -> ${added.id}`)
    }
  }
  // Tag only after every upload: guesses are recomputed as pictures arrive.
  for (const { file, want } of PLAN) {
    const ref = findRef(await refsNow(), file)
    if (!ref) continue
    const { patch, kept } = tagPatch(ref, want)
    for (const k of kept) console.log(`WARNING   ${file}: you set ${k.key} to ${k.has}; leaving it (plan says ${k.wanted})`)
    if (Object.keys(patch).length) {
      await json('PATCH', `/api/exercises/${SLUG}/references/${ref.id}`, patch)
      console.log(`tagged    ${file} ${JSON.stringify(patch)}`)
    }
  }
}

const follow = async (runId) => {
  let last = 0
  for (;;) {
    const state = await call('GET', '/api/state')
    const mine = state.running?.runId === runId
    const queued = (state.queued || []).some((q) => q.runId === runId)
    if (!mine && !queued) break
    if (Date.now() - last >= 10000) {
      last = Date.now()
      if (mine) {
        const r = state.running
        console.log(`[${new Date().toTimeString().slice(0, 8)}] step=${r.step} spent=${fmt(r.elapsed)} step-left=${fmt(r.stepLeft)} run-left=${fmt(r.left)}${r.overdue ? ' (longer than usual)' : ''}${r.note ? ' | ' + r.note : ''}`)
      } else {
        console.log(`[${new Date().toTimeString().slice(0, 8)}] queued, waiting for the Studio to start it`)
      }
    }
    await sleep(2000)
  }
  const run = await call('GET', `/api/runs/${SLUG}/${runId}`)
  console.log(`\nrun ${runId} status: ${run.status}`)
  if (run.error) console.log(`error: ${run.error.message}`)
  for (const s of run.steps || []) console.log(`  step ${s.id.padEnd(8)} ${s.status.padEnd(8)} ${s.seconds ?? '-'}s`)
  if (run.summary) console.log('summary:', JSON.stringify(run.summary))
  // This run's own checks, even if a newer run has started since (the Studio serves the run folder).
  let checks = null
  try { checks = await call('GET', `/files/runs/${SLUG}/${runId}/checks.json`) } catch { /* none written */ }
  if (checks?.rows) for (const r of checks.rows) console.log(rowLine(r))
  else if (run.status === 'done') console.log('(no checks.json found for this run)')
  return run.status === 'done' ? 0 : 1
}

const main = async () => {
  const lock = lockPath(BASE, SLUG)
  if (!takeLock(lock)) {
    console.error('another copy is already setting up this exercise')
    return 1
  }
  try {
    return await setup()
  } finally {
    try { unlinkSync(lock) } catch { /* already gone */ }
  }
}

const setup = async () => {
  await ensureImages()
  const active = activeRun(await call('GET', '/api/state'))
  let runId
  if (active) {
    runId = active.runId
    console.log(`squat already queued/running (${runId}); following it, not starting another`)
  } else if (args.includes('--no-run')) {
    console.log('--no-run: images ready, no run started')
    return 0
  } else {
    runId = (await call('POST', `/api/exercises/${SLUG}/runs`)).id
    console.log(`started run ${runId}`)
  }
  return follow(runId)
}

if (import.meta.url === pathToFileURL(process.argv[1] || '').href) {
  main().then((c) => process.exit(c), (e) => { console.error(e.message); process.exit(1) })
}
