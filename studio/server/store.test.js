import { afterEach, beforeEach, expect, test } from 'vitest'
import { mkdtemp, rm, readFile } from 'node:fs/promises'
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { createStore } from './store.js'

const PNG = Buffer.from('89504e470d0a1a0a0000000d49484452', 'hex')
let root, store
beforeEach(async () => {
  root = await mkdtemp(path.join(os.tmpdir(), 'studio-store-'))
  store = createStore(root)
})
afterEach(() => rm(root, { recursive: true, force: true }))

const upload = (n) => store.addReference('barbell-back-squat', PNG, `${n}.png`)

test('lists the 14 exercises with the squat clip', () => {
  const rows = store.listExercises()
  expect(rows).toHaveLength(14)
  expect(rows.find((r) => r.slug === 'barbell-back-squat').clip).toBe('squat.fbx')
  expect(rows.find((r) => r.name.startsWith('Conventional')).clip).toBeNull()
})

test('adds an exercise of our own', () => {
  store.addExercise({ name: 'Hip Thrust', equipment: 'barbell', clip: null })
  const row = store.listExercises().find((r) => r.slug === 'hip-thrust')
  expect(row).toMatchObject({ name: 'Hip Thrust', equipment: 'barbell', clip: null })
})

test('guesses phases from order', async () => {
  await upload(1)
  expect(store.listReferences('barbell-back-squat').map((i) => i.phase)).toEqual(['start'])
  await upload(2)
  expect(store.listReferences('barbell-back-squat').map((i) => i.phase)).toEqual(['start', 'end'])
  await upload(3)
  const imgs = store.listReferences('barbell-back-squat')
  expect(imgs.map((i) => i.phase)).toEqual(['start', 'middle', 'end'])
  expect(imgs[0].id).toMatch(/^img-[0-9a-f]{8}$/)
})

test('mark #2 context gives start, end for the others', async () => {
  for (const n of [1, 2, 3]) await upload(n)
  const imgs = store.listReferences('barbell-back-squat')
  store.updateReference('barbell-back-squat', imgs[1].id, { contextOnly: true })
  const after = store.listReferences('barbell-back-squat')
  expect([after[0].phase, after[2].phase]).toEqual(['start', 'end'])
})

test('keeps phases the user set', async () => {
  for (const n of [1, 2, 3]) await upload(n)
  const [a, b] = store.listReferences('barbell-back-squat')
  store.updateReference('barbell-back-squat', b.id, { phase: 'end' })
  await upload(4)
  const imgs = store.listReferences('barbell-back-squat')
  expect(imgs[1]).toMatchObject({ phase: 'end', phaseSetBy: 'user' })
  expect(imgs[0].phaseSetBy).toBe('guess')
  expect(a.id).toBe(imgs[0].id)
})

test('reorders and removes (soft delete, file kept)', async () => {
  const a = await upload(1), b = await upload(2)
  store.reorderReferences('barbell-back-squat', [b.id, a.id])
  expect(store.listReferences('barbell-back-squat').map((i) => i.id)).toEqual([b.id, a.id])
  store.removeReference('barbell-back-squat', a.id)
  expect(store.listReferences('barbell-back-squat').map((i) => i.id)).toEqual([b.id])
  expect(existsSync(path.join(root, 'references', 'barbell-back-squat', a.file))).toBe(true)
})

test('run snapshot leaves deleted images out and has absolute paths', async () => {
  const a = await upload(1), b = await upload(2)
  store.removeReference('barbell-back-squat', a.id)
  const run = store.createRun('barbell-back-squat')
  expect(run.id).toMatch(/^\d{8}-\d{6}-[0-9a-f]{4}$/)
  expect(run).toMatchObject({ slug: 'barbell-back-squat', status: 'queued', clip: 'squat.fbx' })
  expect(run.steps.map((s) => s.id)).toEqual(['read', 'build', 'animate', 'render', 'stills', 'check'])
  const snap = JSON.parse(await readFile(path.join(root, 'runs', run.slug, run.id, 'references.json'), 'utf8'))
  expect(snap.exercise).toBe('Barbell Back Squat')
  expect(snap.images.map((i) => i.id)).toEqual([b.id])
  expect(path.isAbsolute(snap.images[0].path)).toBe(true)
  expect(existsSync(snap.images[0].path)).toBe(true)
})

test('lists runs newest first and keeps step history', async () => {
  await upload(1)
  const r1 = store.createRun('barbell-back-squat')
  const r2 = { ...store.createRun('barbell-back-squat'), id: '29990101-000000-abcd' }
  store.writeRun(r2)
  r1.status = 'done'; r1.steps[0] = { ...r1.steps[0], status: 'done', seconds: 21 }
  store.writeRun(r1)
  expect(store.listRuns('barbell-back-squat')[0].id).toBe(r2.id)
  expect(store.allRuns()).toHaveLength(3)
  expect(store.stepHistory('read')).toEqual([21])
  expect(store.readRun('barbell-back-squat', r1.id).status).toBe('done')
})

test('applies detected views without overriding the user', async () => {
  const a = await upload(1), b = await upload(2)
  store.updateReference('barbell-back-squat', b.id, { view: 'behind' })
  store.applyDetectedViews('barbell-back-squat', {
    [a.id]: { view: 'side' },
    [b.id]: { view: 'frontal' },
  })
  const [x, y] = store.listReferences('barbell-back-squat')
  expect(x).toMatchObject({ view: 'side', viewSetBy: 'detector' })
  expect(y).toMatchObject({ view: 'behind', viewSetBy: 'user' })
})

test('expectedClip names the README file even when the clip is missing', () => {
  const ex = store.getExercise('pull-up')
  expect(ex.clip).toBeNull()
  expect(ex.expectedClip).toBe('pull-up.fbx')
})

test('each exercise says what the picture path refuses and what its hands would hold (task 9 review)', () => {
  const rows = Object.fromEntries(store.listExercises().map((r) => [r.name, r]))
  expect(rows['Walking Lunge (dumbbells optional)'].pictureRefusal)
    .toBe('No motion for this exercise: its feet step, which the picture path can\'t animate yet. Add lunge.fbx to 3d-character/mixamo (its README says how).')
  expect(rows['Flat Barbell Bench Press'].pictureRefusal).toBe('No motion for this exercise: it needs a bench the picture path doesn\'t build yet.')
  expect(rows['Conventional Barbell Deadlift']).toMatchObject({ pictureRefusal: null, unheld: 'bar' })
  expect(rows['Barbell Back Squat']).toMatchObject({ pictureRefusal: null, unheld: null })
  store.addExercise({ name: 'Back Squat from images', equipment: 'barbell', clip: null })
  expect(store.getExercise('back-squat-from-images')).toMatchObject({ pictureRefusal: null, unheld: 'bar' })
})

test('a run\'s picture snapshot carries the exercise\'s equipment word, for the checks', async () => {
  store.addExercise({ name: 'Back Squat from images', equipment: 'barbell', clip: null })
  const run = store.createRun('back-squat-from-images')
  const snap = JSON.parse(await readFile(path.join(root, 'runs', run.slug, run.id, 'references.json'), 'utf8'))
  expect(snap).toMatchObject({ exercise: 'Back Squat from images', equipment: 'barbell' })
})

// ---- re-checking a run without re-rendering (studio/scripts/recheck.mjs; the check-only re-run) ----

// A run that has been through the whole pipeline: the outputs a re-check (and the page) read.
const renderRun = (slug) => {
  const run = store.createRun(slug)
  const dir = path.join(root, 'runs', slug, run.id)
  writeFileSync(path.join(dir, 'targets.json'), '{"exercise":"Barbell Back Squat","phases":{}}')
  writeFileSync(path.join(dir, 'ours.json'), '{"stills":[{"file":"stills/a.png"}]}')
  writeFileSync(path.join(dir, 'measurements.json'), '{"m":1}')
  mkdirSync(path.join(dir, 'frames'), { recursive: true })
  writeFileSync(path.join(dir, 'frames', 'retarget.json'), '{"r":1}')
  writeFileSync(path.join(dir, 'animation.gif'), 'GIF89a')
  mkdirSync(path.join(dir, 'stills'), { recursive: true })
  writeFileSync(path.join(dir, 'stills', 'a.png'), 'PNGDATA')
  return { ...store.readRun(slug, run.id), summary: { pass: 1, fail: 0 } }
}

test('createRecheckRun makes a check-only run tied to its source, with only the check step left to run', () => {
  const source = renderRun('barbell-back-squat')
  const run = store.createRecheckRun('barbell-back-squat', source)
  expect(run.id).not.toBe(source.id)
  expect(run).toMatchObject({ checkOnly: true, recheckOf: source.id, status: 'queued' })
  const byId = Object.fromEntries(run.steps.map((s) => [s.id, s.status]))
  expect(byId).toMatchObject({ read: 'done', build: 'done', animate: 'done', render: 'done', stills: 'done', check: 'waiting' })
  expect(store.readRun('barbell-back-squat', run.id).checkOnly).toBe(true)
})

test('createRecheckRun copies the rendered outputs the check and the page need', () => {
  const source = renderRun('barbell-back-squat')
  const run = store.createRecheckRun('barbell-back-squat', source)
  const dir = path.join(root, 'runs', 'barbell-back-squat', run.id)
  for (const rel of ['targets.json', 'ours.json', 'measurements.json', 'frames/retarget.json', 'animation.gif', 'stills/a.png']) {
    expect(existsSync(path.join(dir, rel))).toBe(true)
  }
  expect(readFileSync(path.join(dir, 'frames', 'retarget.json'), 'utf8')).toBe('{"r":1}')
  expect(readFileSync(path.join(dir, 'stills', 'a.png'), 'utf8')).toBe('PNGDATA')
})
