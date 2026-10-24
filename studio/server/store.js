import { copyFileSync, cpSync, existsSync, mkdirSync, readFileSync, readdirSync, renameSync, writeFileSync } from 'node:fs'
import { randomBytes } from 'node:crypto'
import path from 'node:path'
import { CLIPS_DIR, DATA, REPO } from './paths.js'
import { automationBlocker, pictureRefusal, unheld } from './kit.js'

// The "Save as" column of 3d-character/mixamo/README.md, keyed by the names in poses.json.
const CLIPS = {
  'Barbell Back Squat': 'squat.fbx',
  'Conventional Barbell Deadlift': 'deadlift.fbx',
  'Flat Barbell Bench Press': 'bench-press.fbx',
  'Standing Barbell Overhead Press (Shoulder Press)': 'overhead-press.fbx',
  'Barbell Bent-Over Row': 'row.fbx',
  'Pull-Up': 'pull-up.fbx',
  'Lat Pulldown (seated, to the front)': 'lat-pulldown.fbx',
  'Dumbbell Bicep Curl (standing, both arms)': 'bicep-curl.fbx',
  'Dumbbell Lateral Raise': 'lateral-raise.fbx',
  'Dumbbell Shoulder Press (seated overhead press)': 'db-shoulder-press.fbx',
  'Walking Lunge (dumbbells optional)': 'lunge.fbx',
  'Front Plank (forearm plank)': 'plank.fbx',
  'Tricep Pushdown (cable, bar or rope attachment)': 'tricep-pushdown.fbx',
  'Leg Press (seated machine)': 'leg-press.fbx',
}

const STEP_IDS = ['read', 'build', 'animate', 'render', 'stills', 'check']
const POSES = path.join(REPO, 'reference', 'exercises', 'poses.json')

export class NotFoundError extends Error {}
export class BadInputError extends Error {}

export const slugOf = (name) => (String(name).toLowerCase().match(/[a-z0-9]+/g) || []).join('-')

const equipmentOf = (name) => {
  const n = name.toLowerCase()
  for (const word of ['barbell', 'dumbbell', 'cable', 'machine']) if (n.includes(word)) return word
  if (n.includes('pulldown')) return 'cable'
  return 'bodyweight'
}

const readJson = (file, fallback) => {
  try {
    return JSON.parse(readFileSync(file, 'utf8'))
  } catch {
    return fallback
  }
}

const writeJson = (file, value) => {
  mkdirSync(path.dirname(file), { recursive: true })
  const tmp = `${file}.${process.pid}.tmp`
  writeFileSync(tmp, JSON.stringify(value, null, 2))
  renameSync(tmp, file)
}

export const stamp = (d = new Date()) => {
  const p = (n) => String(n).padStart(2, '0')
  return `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}-${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`
}

export const createStore = (dataRoot = DATA) => {
  const refDir = (slug) => path.join(dataRoot, 'references', slug)
  const refFile = (slug) => path.join(refDir(slug), 'references.json')
  const runDir = (slug, id) => path.join(dataRoot, 'runs', slug, id)

  const listExercises = () => {
    const base = readJson(POSES, { exercises: [] }).exercises.map((e) => ({
      name: e.name,
      equipment: equipmentOf(e.name),
      clip: CLIPS[e.name] || null,
    }))
    const added = readJson(path.join(dataRoot, 'added-exercises.json'), [])
    return [...base, ...added].map((e) => ({
      slug: slugOf(e.name),
      name: e.name,
      equipment: e.equipment || 'bodyweight',
      clip: e.clip && existsSync(path.join(CLIPS_DIR, e.clip)) ? e.clip : null,
      expectedClip: e.clip || null, // the file name the README table asks for, even when it is missing
      // without a clip: why the picture path can't animate it (null: it can), and what its hands would hold
      // that the path doesn't build (studio/measure/exercise-kit.json; task 9 review)
      pictureRefusal: pictureRefusal(e.name, e.equipment || 'bodyweight'),
      unheld: unheld(e.name, e.equipment || 'bodyweight'),
      automationBlocker: automationBlocker(e.name),
    }))
  }

  const getExercise = (slug) => listExercises().find((e) => e.slug === slug) || null

  const addExercise = ({ name, equipment, clip }) => {
    const clean = String(name || '').trim()
    if (!slugOf(clean)) throw new BadInputError('Give the exercise a name.')
    if (getExercise(slugOf(clean))) throw new BadInputError('That exercise is already in the list.')
    const file = path.join(dataRoot, 'added-exercises.json')
    writeJson(file, [...readJson(file, []), { name: clean, equipment: equipment || 'bodyweight', clip: clip || null }])
    return getExercise(slugOf(clean))
  }

  const load = (slug) => readJson(refFile(slug), { images: [] }).images
  const save = (slug, images) => writeJson(refFile(slug), { images })

  // Phases for images the user has not set: start, middle..., end over the non-context images in order.
  const guessPhases = (images) => {
    const live = images.filter((i) => !i.deleted && !i.contextOnly).sort((a, b) => a.order - b.order)
    live.forEach((img, k) => {
      if (img.phaseSetBy === 'user') return
      img.phase = k === 0 ? 'start' : k === live.length - 1 ? 'end' : 'middle'
      img.phaseSetBy = 'guess'
    })
    return images
  }

  const listReferences = (slug) => load(slug).filter((i) => !i.deleted).sort((a, b) => a.order - b.order)

  // Pictures a run can measure: at least one that is not context-only.
  const hasUsablePictures = (slug) => listReferences(slug).some((i) => !i.contextOnly)

  const addReference = (slug, bytes, originalName, ext = 'png', options = {}) => {
    const images = load(slug)
    let id
    do id = `img-${randomBytes(4).toString('hex')}`
    while (images.some((i) => i.id === id))
    const file = `${id}.${ext}`
    mkdirSync(refDir(slug), { recursive: true })
    writeFileSync(path.join(refDir(slug), file), bytes)
    const meta = {
      id, file, original: originalName, order: Math.max(0, ...images.map((i) => i.order)) + 1,
      phase: options.phase || 'start', phaseSetBy: options.phase ? 'user' : 'guess',
      view: options.view || null, viewSetBy: options.view ? 'user' : null,
      contextOnly: Boolean(options.contextOnly), deleted: false,
      ...(options.source ? { source: options.source } : {}),
    }
    images.push(meta)
    save(slug, guessPhases(images))
    return load(slug).find((i) => i.id === id)
  }

  const change = (slug, id, fn) => {
    const images = load(slug)
    const img = images.find((i) => i.id === id)
    if (!img) throw new NotFoundError('That picture is not in this exercise.')
    fn(img)
    save(slug, guessPhases(images))
    return load(slug).find((i) => i.id === id)
  }

  const updateReference = (slug, id, patch) =>
    change(slug, id, (img) => {
      if ('phase' in patch) { img.phase = patch.phase; img.phaseSetBy = 'user' }
      if ('view' in patch) { img.view = patch.view; img.viewSetBy = patch.view ? 'user' : null }
      if ('contextOnly' in patch) img.contextOnly = Boolean(patch.contextOnly)
    })

  const reorderReferences = (slug, ids) => {
    const images = load(slug)
    let n = 0
    for (const id of ids) {
      const img = images.find((i) => i.id === id)
      if (img) img.order = ++n
    }
    for (const img of images.filter((i) => !ids.includes(i.id)).sort((a, b) => a.order - b.order)) img.order = ++n
    save(slug, guessPhases(images))
  }

  const removeReference = (slug, id) => change(slug, id, (img) => { img.deleted = true })

  const applyDetectedViews = (slug, measurements) => {
    const images = load(slug)
    for (const img of images) {
      const view = measurements?.[img.id]?.view
      if (view && img.viewSetBy !== 'user') { img.view = view; img.viewSetBy = 'detector' }
    }
    save(slug, images)
  }

  const newStep = (id) => ({ id, status: 'waiting', startedAt: null, finishedAt: null, seconds: null, fraction: null, note: null })

  const writeRun = (run) => writeJson(path.join(runDir(run.slug, run.id), 'run.json'), run)
  const readRun = (slug, id) => readJson(path.join(runDir(slug, id), 'run.json'), null)

  // `extra` adds fields to the new run (a note for the page, what it restarts).
  const createRun = (slug, extra = {}) => {
    const exercise = getExercise(slug)
    if (!exercise) throw new NotFoundError('Unknown exercise.')
    let id
    do id = `${stamp()}-${randomBytes(2).toString('hex')}`
    while (existsSync(runDir(slug, id)))
    const dir = runDir(slug, id)
    const images = listReferences(slug).map((i) => ({ ...i, path: path.join(refDir(slug), i.file) }))
    writeJson(path.join(dir, 'references.json'), { exercise: exercise.name, slug, equipment: exercise.equipment, images })
    const run = {
      id, slug, name: exercise.name, clip: exercise.clip, status: 'queued',
      createdAt: new Date().toISOString(), startedAt: null, finishedAt: null,
      steps: (extra.readOnly ? ['read'] : STEP_IDS).map(newStep), error: null, summary: null, ...extra,
    }
    writeRun(run)
    return run
  }

  // Re-check a run without re-rendering (studio/scripts/recheck.mjs, the Studio's check-only re-run):
  // a change to the check rules re-judges the same measurements, so the character, animation and stills
  // are reused from `sourceRun` and only the check step runs again. These are the files the check reads
  // and the page shows; anything missing in the source is simply not copied.
  const RECHECK_COPY = ['targets.json', 'ours.json', 'measurements.json', 'references.json',
    'pipeline-targets.json', path.join('frames', 'retarget.json'), 'animation.gif']
  const RECHECK_COPY_DIRS = ['stills']

  const createRecheckRun = (slug, sourceRun, extra = {}) => {
    const exercise = getExercise(slug)
    if (!exercise) throw new NotFoundError('Unknown exercise.')
    if (!sourceRun?.id) throw new NotFoundError('No run to re-check.')
    const srcDir = runDir(slug, sourceRun.id)
    let id
    do id = `${stamp()}-${randomBytes(2).toString('hex')}`
    while (existsSync(runDir(slug, id)))
    const dir = runDir(slug, id)
    mkdirSync(dir, { recursive: true })
    for (const rel of RECHECK_COPY) {
      const from = path.join(srcDir, rel)
      if (!existsSync(from)) continue
      mkdirSync(path.dirname(path.join(dir, rel)), { recursive: true })
      copyFileSync(from, path.join(dir, rel))
    }
    for (const sub of RECHECK_COPY_DIRS) {
      const from = path.join(srcDir, sub)
      if (existsSync(from)) cpSync(from, path.join(dir, sub), { recursive: true })
    }
    // The steps whose outputs were rendered into the source and copied here are done; only check runs again.
    const steps = STEP_IDS.map((sid) => (sid === 'check' ? newStep(sid) : { ...newStep(sid), status: 'done' }))
    const run = {
      id, slug, name: exercise.name, clip: exercise.clip, status: 'queued',
      createdAt: new Date().toISOString(), startedAt: null, finishedAt: null,
      steps, error: null, summary: null, checkOnly: true, recheckOf: sourceRun.id, ...extra,
    }
    writeRun(run)
    return run
  }

  const listRuns = (slug) => {
    const dir = path.join(dataRoot, 'runs', slug)
    if (!existsSync(dir)) return []
    // Cached reference checks can finish within the same second. Random ID suffixes are not chronology.
    return readdirSync(dir).map((id) => readRun(slug, id)).filter(Boolean)
      .sort((a, b) => (b.createdAt || '').localeCompare(a.createdAt || '') || b.id.localeCompare(a.id))
  }

  const allRuns = () => {
    const dir = path.join(dataRoot, 'runs')
    if (!existsSync(dir)) return []
    return readdirSync(dir).flatMap(listRuns)
  }

  const stepHistory = (stepId) =>
    allRuns()
      .filter((r) => r.status === 'done')
      .sort((a, b) => (a.createdAt < b.createdAt ? -1 : 1))
      .map((r) => r.steps.find((s) => s.id === stepId))
      .filter((s) => s && s.status === 'done' && typeof s.seconds === 'number')
      .map((s) => s.seconds)

  return {
    root: dataRoot, runDir, listExercises, getExercise, addExercise, slugOf, listReferences, hasUsablePictures, addReference,
    updateReference, reorderReferences, removeReference, applyDetectedViews, createRun, createRecheckRun, readRun, writeRun,
    listRuns, allRuns, stepHistory,
  }
}
