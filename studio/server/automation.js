import { createHash, randomUUID } from 'node:crypto'
import { existsSync, mkdirSync, readFileSync, readdirSync, renameSync, statSync, writeFileSync } from 'node:fs'
import path from 'node:path'
import { BadInputError } from './store.js'
import { execFileSync } from 'node:child_process'
import { BLENDER, PYTHON, REPO } from './paths.js'
import { getReferenceSpec } from './reference-catalog.js'
import { fetchReference } from './reference-fetch.js'
import { automaticBuilder } from './kit.js'

const FINAL = new Set(['done', 'failed', 'interrupted'])
const json = (file) => { try { return JSON.parse(readFileSync(file, 'utf8')) } catch { return null } }
const write = (file, value) => {
  mkdirSync(path.dirname(file), { recursive: true })
  const temporary = `${file}.${process.pid}.tmp`
  writeFileSync(temporary, JSON.stringify(value, null, 2))
  renameSync(temporary, file)
}
const clone = (value) => structuredClone(value)
const pictureKey = (images) => JSON.stringify([...images].sort((a, b) => a.order - b.order).map((i) =>
  [i.id, i.file, i.phase, i.viewSetBy === 'user' ? i.view : null, Boolean(i.contextOnly)]))

// Reuse outputs only while the builder, model and input contents are unchanged.
const runtimeFingerprint = () => {
  let fitted = 'fitted runtime unavailable', reader = 'pose reader unavailable'
  try {
    const output = execFileSync(BLENDER, ['--background', '--python-exit-code', '1', '--python',
      path.join(REPO, 'studio/server/runtime_fingerprint.py')], { timeout: 30000, encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] })
    fitted = output.split('\n').find((line) => line.startsWith('FIT_RUNTIME_ID ')) || fitted
  } catch { /* A missing runtime must not match a completed run's known runtime. */ }
  try {
    reader = execFileSync(PYTHON, ['-c', 'import json; from importlib.metadata import version; print(json.dumps({p:version(p) for p in ("mediapipe","numpy","pillow")}))'],
      { timeout: 10000, encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] })
  } catch { /* Reference-only tasks can still report missing prerequisites. */ }
  return fitted + '\n' + reader
}
export const builderFingerprint = ({ root = REPO, execution = runtimeFingerprint } = {}) => {
  const hash = createHash('sha256')
  const add = (file) => {
    hash.update(path.relative(root, file)); hash.update(existsSync(file) ? readFileSync(file) : 'missing')
  }
  const walk = (dir) => {
    if (!existsSync(dir)) { hash.update('missing:' + path.relative(root, dir)); return }
    for (const entry of readdirSync(dir, { withFileTypes: true }).sort((a, b) => a.name.localeCompare(b.name))) {
      if (entry.name.startsWith('.') || ['__pycache__', 'vendor'].includes(entry.name)) continue
      const file = path.join(dir, entry.name)
      if (entry.isDirectory()) walk(file)
      else if (/\.(py|js|json)$/.test(entry.name) && !entry.name.endsWith('.test.js')) add(file)
    }
  }
  for (const dir of ['pipeline', 'studio/measure', 'studio/blender', 'reference/exercises']) walk(path.join(root, dir))
  for (const rel of ['studio/server/steps.js', 'studio/server/runtime_fingerprint.py', 'package-lock.json', 'studio/requirements.txt',
    '3d-character/3d cartoon boy.glb', 'studio/models/pose_landmarker_heavy.task']) add(path.join(root, rel))
  const clips = path.join(root, '3d-character/mixamo')
  if (existsSync(clips)) for (const name of readdirSync(clips).sort()) if (/\.fbx$/i.test(name)) add(path.join(clips, name))
  hash.update(execution())
  return hash.digest('hex')
}
const inputsFingerprint = (store, slug) => {
  const refs = store.listReferences(slug)
  const hash = createHash('sha256').update(pictureKey(refs))
  const exercise = store.getExercise(slug)
  hash.update(JSON.stringify([exercise.name, exercise.equipment, exercise.clip, exercise.pictureRefusal, automaticBuilder(exercise.name)]))
  for (const ref of refs) hash.update(readFileSync(path.join(store.root, 'references', slug, ref.file)))
  return hash.digest('hex')
}
const reusableOutputs = (store, run) => {
  const dir = store.runDir(run.slug, run.id)
  try {
    const gif = readFileSync(path.join(dir, 'animation.gif'))
    const checks = json(path.join(dir, 'checks.json'))
    const ours = json(path.join(dir, 'ours.json'))
    return gif.length > 1000 && /^GIF8[79]a$/.test(gif.subarray(0, 6).toString()) &&
      checks?.summary && JSON.stringify(checks.summary) === JSON.stringify(run.summary) &&
      statSync(path.join(dir, 'scene.blend')).size > 0 && Array.isArray(ours?.stills) && ours.stills.length > 0 &&
      ours.stills.every((still) => typeof still.file === 'string' && !still.file.includes('..') &&
        !path.isAbsolute(still.file) && statSync(path.join(dir, still.file)).size > 0)
  } catch { return false }
}

export const automationVerdict = (summary) => summary &&
  ['pass', 'fail', 'limited', 'uncertain', 'unmeasured'].every((key) => Number.isInteger(summary[key]) && summary[key] >= 0) &&
  summary.pass > 0 && ['fail', 'limited', 'uncertain', 'unmeasured'].every((key) => summary[key] === 0) ? 'passed' : 'needs-review'


export const createAutomation = ({ store, queue, referenceSpec = getReferenceSpec, download = fetchReference,
  fingerprint = builderFingerprint, pollMs = 250, maxRunMs = 45 * 60 * 1000, now = Date.now } = {}) => {
  const folder = path.join(store.root, 'automation')
  let live = null, stopped = false
  const iso = () => new Date(now()).toISOString()
  const fileFor = (id) => /^[a-f\d-]{36}$/.test(id || '') ? path.join(folder, `${id}.json`) : null
  const get = (id) => { const file = fileFor(id); return file ? json(file) : null }
  const save = (job) => write(fileFor(job.id), job)
  const history = () => existsSync(folder) ? readdirSync(folder).filter((f) => f.endsWith('.json'))
    .map((f) => json(path.join(folder, f))).filter(Boolean).sort((a, b) => b.createdAt.localeCompare(a.createdAt)).slice(0, 20) : []
  const summary = () => ({ active: live ? clone(live) : null, history: history() })
  const update = (job, item, patch) => { Object.assign(item, patch, { updatedAt: iso() }); save(job) }
  const busy = (slug) => { const state = queue.state(); return state.running?.slug === slug || state.queued.some((q) => q.slug === slug) }
  const pause = () => new Promise((resolve) => setTimeout(resolve, pollMs))
  const waitForRun = async (job, slug, id) => {
    const started = now()
    while (!stopped && job.status === 'running') {
      const run = store.readRun(slug, id)
      if (!run) throw Error('The queued run is missing. Check its history before retrying.')
      if (FINAL.has(run.status)) return run
      if (now() - started > maxRunMs) {
        job.cancelRequested = true
        job.timedOut = true
        throw Error('This run exceeded the automation time limit. It may still finish in Studio; later exercises were stopped.')
      }
      await pause()
    }
    return null
  }
  const blockedBuilder = (exercise) => exercise.pictureRefusal || (exercise.unheld
    ? `The builder does not yet create the required ${exercise.unheld} or its hand contacts.`
    : `No validated automatic motion builder exists for ${exercise.name} yet.`)

  const processItem = async (job, item) => {
    const exercise = store.getExercise(item.slug)
    if (busy(item.slug)) return update(job, item, { status: 'blocked', message: 'This exercise already has work running or queued. Its current work was left alone.' })
    const spec = referenceSpec(exercise.name)
    let refs = store.listReferences(item.slug)
    if (spec && refs.length && refs.every((r) => r.source?.catalogId) &&
        refs.some((r) => r.source.catalogId !== spec.id || r.source.commit !== spec.commit)) {
      return update(job, item, { status: 'blocked', message: 'The curated source changed. Review the previously imported references before replacing them.' })
    }
    const automaticOnly = spec && refs.length && refs.every((r) => r.source?.catalogId === spec.id && r.source?.commit === spec.commit)
    if (!refs.length || automaticOnly) {
      if (!spec) return update(job, item, { status: 'blocked', message: 'No exact curated reference source is mapped to this exercise. Add reference pictures in Studio.' })
      update(job, item, { status: 'fetching', message: 'Retrieving the curated reference photographs.', sourceUrl: spec.repository, sourceId: spec.id })
      const known = new Set(refs.map((r) => r.source?.url))
      // Download and verify the entire missing set before touching reference metadata.
      const fetched = []
      for (const image of spec.images.filter((image) => !known.has(image.url))) {
        let result
        for (let attempt = 1; attempt <= 2; attempt++) {
          try { result = await download(image); break }
          catch (error) {
            if (!error.retryable || attempt === 2 || stopped) throw error
            update(job, item, { message: 'The source is temporarily unavailable; retrying once.' })
            await new Promise((resolve) => setTimeout(resolve, 500))
          }
        }
        if (stopped) return
        fetched.push({ image, result })
      }
      refs = store.listReferences(item.slug)
      if (refs.length && !refs.every((r) => r.source?.catalogId === spec.id && r.source?.commit === spec.commit)) {
        item.message = 'Using reference pictures added while the download was running.'
      } else {
        const current = new Set(refs.map((r) => r.source?.url))
        for (const { image, result } of fetched) {
          if (current.has(image.url)) continue
          store.addReference(item.slug, result.bytes, `${spec.id}-${image.phase}.${result.ext}`, result.ext, {
            phase: image.phase, view: image.view, contextOnly: image.contextOnly,
            source: { ...result.provenance, catalogId: spec.id, commit: spec.commit, url: image.url,
              sha256: result.sha256, license: spec.license },
          })
          current.add(image.url)
        }
      }
    } else {
      update(job, item, { message: 'Using existing reference pictures; none were replaced.', sourceUrl: refs[0]?.source?.url })
    }
    if (stopped) return
    // A catalog's special setup/static-only pair cannot become invented motion frames.
    if (spec?.autoBuild === false && store.listReferences(item.slug).every((r) => r.source?.catalogId === spec.id)) {
      return update(job, item, { status: 'blocked', message: spec.reason || 'These source images do not provide the positions needed for an automatic motion cycle.' })
    }
    if (busy(item.slug)) return update(job, item, { status: 'blocked', message: 'New work started on this exercise. Automation left it alone.' })
    const inputs = inputsFingerprint(store, item.slug)
    const previous = !job.checkOnly && !job.forceRebuild && store.listRuns(item.slug).find((run) => !run.readOnly && run.status === 'done' &&
      run.automationFingerprint === job.builderFingerprint && run.automationInputs === inputs &&
      reusableOutputs(store, run))
    if (previous) {
      if (fingerprint() !== job.builderFingerprint) return update(job, item, { status: 'blocked', message: 'The builder changed during this batch. Start a fresh batch with the new code.' })
      return update(job, item, { status: automationVerdict(previous.summary), runId: previous.id,
        summary: previous.summary, reused: true, message: 'Reused the existing result: references and builder are unchanged.' })
    }

    update(job, item, { status: 'checking', message: 'Reading references and checking required inputs.' })
    const read = queue.enqueueRead(item.slug, { automationJobId: job.id })
    update(job, item, { referenceRunId: read.id })
    const checked = await waitForRun(job, item.slug, read.id)
    if (!checked || stopped) return
    const plan = json(path.join(store.runDir(item.slug, read.id), 'preflight.json'))
    const capability = automaticBuilder(exercise.name)
    if (!capability) return update(job, item, { status: 'blocked', message: blockedBuilder(exercise),
      preflight: plan, referenceMessage: checked.error?.message || null })
    if (checked.status !== 'done' || plan?.ready !== true) return update(job, item, { status: 'blocked', preflight: plan,
      message: plan?.blockers?.join(' ') || checked.error?.message || 'The reference check did not establish readiness.' })
    if (inputsFingerprint(store, item.slug) !== inputs) return update(job, item, { status: 'blocked', message: 'References changed during the check. Check the edited pictures before rebuilding.' })
    if (fingerprint() !== job.builderFingerprint) return update(job, item, { status: 'blocked', message: 'The builder changed during this batch. Start a fresh batch with the new code.' })
    if (job.checkOnly) return update(job, item, { status: 'ready', preflight: plan, message: 'References are ready. Rendering was not requested.' })
    const state = queue.state()
    if ((state.running?.slug === item.slug && state.running.runId !== read.id) || state.queued.some((pending) => pending.slug === item.slug)) {
      return update(job, item, { status: 'blocked', message: 'Manual work started during the reference check. Automation did not queue a duplicate build.' })
    }
    update(job, item, { status: 'building', preflight: plan, message: 'Building the supported animation and running its form checks.' })
    const run = queue.enqueue(item.slug, { automationJobId: job.id, automationFingerprint: job.builderFingerprint, automationInputs: inputs })
    update(job, item, { runId: run.id })
    const result = await waitForRun(job, item.slug, run.id)
    if (!result || stopped) return
    if (result.status !== 'done') return update(job, item, { status: 'failed', message: result.error?.message || 'The animation did not complete.' })
    update(job, item, { status: automationVerdict(result.summary), summary: result.summary,
      message: automationVerdict(result.summary) === 'passed' ? 'The completed animation passed its measured checks.'
        : 'The animation completed. Review its failed, uncertain or missing measurements in Studio.' })
  }

  const execute = async (job) => {
    for (const item of job.items) {
      if (stopped || job.status !== 'running') return
      if (job.cancelRequested) break
      try { await processItem(job, item) }
      catch (error) { if (!stopped) update(job, item, { status: 'failed', message: error.message }) }
    }
    if (stopped || job.status !== 'running') return
    for (const item of job.items) if (item.status === 'waiting') item.status = 'cancelled'
    job.status = job.timedOut ? 'interrupted' : job.cancelRequested ? 'cancelled' : 'done'
    job.finishedAt = iso(); save(job); live = null
  }
  const start = ({ slugs, checkOnly = false, forceRebuild = false } = {}) => {
    if (stopped) throw new BadInputError('Restart Studio before starting another batch.')
    if (live) throw Object.assign(Error('Another automation batch is already running.'), { status: 409 })
    if (typeof checkOnly !== 'boolean' || typeof forceRebuild !== 'boolean') throw new BadInputError('Batch options must be true or false.')
    const exercises = store.listExercises()
    if (slugs === undefined) slugs = exercises.map((e) => e.slug)
    if (!Array.isArray(slugs) || !slugs.length || slugs.length > 100 || slugs.some((s) => typeof s !== 'string' || !exercises.some((e) => e.slug === s))) {
      throw new BadInputError('Select existing exercises for the batch.')
    }
    const job = { id: randomUUID(), status: 'running', createdAt: iso(), finishedAt: null,
      checkOnly, forceRebuild, builderFingerprint: fingerprint(), cancelRequested: false,
      items: [...new Set(slugs)].map((slug) => ({ slug, name: store.getExercise(slug).name, status: 'waiting', message: 'Waiting for its turn.' })) }
    save(job); live = job
    setImmediate(() => execute(job).catch((error) => {
      job.status = 'interrupted'; job.error = error.message; job.finishedAt = iso()
      try { save(job) } catch { /* Recovery will mark the saved running job interrupted. */ }
      live = null
    }))
    return clone(job)
  }
  const cancel = (id) => {
    const job = live?.id === id ? live : get(id)
    if (!job) return null
    if (job.status === 'running') { job.cancelRequested = true; save(job) }
    return clone(job)
  }
  const recoverInterrupted = () => {
    for (const job of history()) if (job.status === 'running') {
      job.status = 'interrupted'; job.finishedAt = iso()
      for (const item of job.items) if (['waiting', 'fetching', 'checking', 'building'].includes(item.status)) item.status = 'interrupted'
      job.error = 'Studio stopped during this batch. Existing references and completed runs were kept; start a new batch to continue.'
      save(job)
    }
  }
  const stop = () => {
    stopped = true
    if (live) {
      live.status = 'interrupted'; live.finishedAt = iso(); live.error = 'Studio stopped during automation.'
      for (const item of live.items) if (['waiting', 'fetching', 'checking', 'building'].includes(item.status)) item.status = 'interrupted'
      try { save(live) } finally { live = null }
    }
  }
  return { start, get, summary, cancel, stop, recoverInterrupted }
}
