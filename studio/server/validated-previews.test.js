import { afterEach, expect, test } from 'vitest'
import { cpSync, mkdtempSync, readFileSync, rmSync, symlinkSync, unlinkSync, writeFileSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { createStore } from './store.js'
import { createValidatedPreviews, validatedPreviews } from './validated-previews.js'
import { REPO } from './paths.js'

const roots = []
afterEach(() => { for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true }) })
const temp = () => { const root = mkdtempSync(path.join(os.tmpdir(), 'studio-previews-')); roots.push(root); return root }
const exerciseList = () => createStore(path.join(temp(), 'user-data')).listExercises()
const assetNames = ['animation.gif', 'start.png', 'end.png', 'checks.json']

test('all 14 exercises have readable committed examples with verified assets', () => {
  const rows = validatedPreviews.list(exerciseList())
  expect(rows).toHaveLength(14)
  expect(rows.filter((row) => row.available)).toHaveLength(14)
  for (const row of rows.filter((row) => row.available)) {
    expect(row.readOnly).toBe(true)
    expect(row.frames).toBeGreaterThan(0)
    expect(row.fps).toBeGreaterThan(0)
    expect(row.gifUrl).toBe(`/api/previews/${row.slug}/animation.gif`)
    for (const name of assetNames) {
      const asset = validatedPreviews.asset(row.slug, name)
      expect(asset.bytes.length).toBeGreaterThan(0)
      expect(asset.sha256).toMatch(/^[a-f0-9]{64}$/)
    }
  }
})

test('source uncertainty and row failures are retained rather than relabeled as approval', () => {
  const rows = validatedPreviews.list(exerciseList())
  const row = rows.find((r) => r.slug === 'barbell-bent-over-row')
  expect(row).toMatchObject({ summary: { pass: 18, fail: 2, uncertain: 0 }, reviewRequired: true, frames: 81, fps: 30 })
  expect(row.reason).toMatch(/disagree|failed checks/)
  const overhead = rows.find((r) => r.slug === 'standing-barbell-overhead-press-shoulder-press')
  expect(overhead).toMatchObject({ summary: { fail: 0, uncertain: 10 }, reviewRequired: true })
  expect(overhead.characterFit).toMatchObject({ applied: true, upper_scale: 1.8 })
  const plank = rows.find((r) => r.slug === 'front-plank-forearm-plank')
  expect(plank).toMatchObject({ summary: { fail: 0, uncertain: 0 }, reviewRequired: false })
  row.summary.fail = 0
  expect(validatedPreviews.list(exerciseList()).find((r) => r.slug === row.slug).summary.fail).toBe(2)
})

test('squat preset and one-way lunge keep their source modes while unknown exercises stay unavailable', () => {
  const rows = validatedPreviews.list([...exerciseList(), { slug: 'my-exercise', name: 'My exercise' }])
  const squat = rows.find((r) => r.slug === 'barbell-back-squat')
  expect(squat).toMatchObject({ available: true, sourceLabel: 'Checked movement preset', playback: 'loop', frames: 69, fps: 30 })
  expect(rows.find((r) => r.slug === 'walking-lunge-dumbbells-optional')).toMatchObject({
    available: true, sourceLabel: 'Observed one-way step', playback: 'once', frames: 42, fps: 16, reviewRequired: true,
  })
  expect(rows.find((r) => r.slug === 'my-exercise')).toMatchObject({ available: false, gifUrl: null })
})

test.each(['../package.json', '/etc/passwd', 'measurements.json', 'scene.blend', 'https://example.test/animation.gif'])('only whitelisted assets can be requested: %s', (name) => {
  expect(validatedPreviews.asset('barbell-bent-over-row', name)).toBeNull()
  expect(validatedPreviews.asset('../row', 'animation.gif')).toBeNull()
})

const fixture = () => {
  const root = temp(), base = path.join(root, 'reference/automation-validation')
  const source = path.join(REPO, 'reference/automation-validation')
  const manifest = JSON.parse(readFileSync(path.join(source, 'corrections/results.json'), 'utf8'))
  const record = manifest.runs[0]
  cpSync(path.join(source, 'corrections', record.slug), path.join(base, 'corrections', record.slug), { recursive: true })
  cpSync(path.join(source, 'row'), path.join(base, 'row'), { recursive: true })
  writeFileSync(path.join(base, 'corrections/results.json'), JSON.stringify({ runs: [record] }))
  return { root, slug: record.slug, folder: path.join(base, 'corrections', record.slug) }
}

test('missing or changed assets remain unavailable and recover only when authentic files return', () => {
  const { root, slug, folder } = fixture()
  const registry = createValidatedPreviews({ root })
  const exercises = [{ slug, name: 'Deadlift' }]
  expect(registry.list(exercises)[0].available).toBe(true)
  const file = path.join(folder, 'animation.gif'), original = readFileSync(file)
  writeFileSync(file, Buffer.from('GIF89a changed'))
  expect(registry.list(exercises)[0]).toMatchObject({ available: false, gifUrl: null })
  expect(registry.asset(slug, 'animation.gif')).toBeNull()
  writeFileSync(file, original)
  expect(registry.list(exercises)[0].available).toBe(true)
  unlinkSync(path.join(folder, 'checks.json'))
  expect(registry.list(exercises)[0].available).toBe(false)
})

test('symlinked assets and manifest path escapes cannot expose other files', () => {
  const { root, slug, folder } = fixture()
  const gif = path.join(folder, 'animation.gif'), outside = path.join(root, 'outside.gif')
  cpSync(gif, outside)
  unlinkSync(gif); symlinkSync(outside, gif)
  expect(createValidatedPreviews({ root }).asset(slug, 'animation.gif')).toBeNull()
  const manifestFile = path.join(root, 'reference/automation-validation/corrections/results.json')
  const manifest = JSON.parse(readFileSync(manifestFile, 'utf8'))
  manifest.runs[0].slug = '../outside'
  writeFileSync(manifestFile, JSON.stringify(manifest))
  expect(createValidatedPreviews({ root }).list([{ slug, name: 'Deadlift' }])[0].available).toBe(false)
})

test('reading examples never changes user references or creates a build history', () => {
  const store = createStore(temp())
  store.addReference('barbell-bent-over-row', Buffer.from('private image'), 'my-reference.jpg')
  const before = store.listReferences('barbell-bent-over-row')
  const runBefore = store.listRuns('barbell-bent-over-row')
  validatedPreviews.list(store.listExercises())
  validatedPreviews.asset('barbell-bent-over-row', 'animation.gif')
  expect(store.listReferences('barbell-bent-over-row')).toEqual(before)
  expect(store.listRuns('barbell-bent-over-row')).toEqual(runBefore)
})
