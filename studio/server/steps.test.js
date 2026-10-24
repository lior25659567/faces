import { afterEach, beforeEach, expect, test } from 'vitest'
import { mkdtemp, rm, writeFile, readFile } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { PROCESSES, STEPS, commandFor, readLine, requiredOutputs } from './steps.js'

let dir
beforeEach(async () => { dir = await mkdtemp(path.join(os.tmpdir(), 'studio-steps-')) })
afterEach(() => rm(dir, { recursive: true, force: true }))

const run = { slug: 'barbell-back-squat', name: 'Barbell Back Squat', clip: 'squat.fbx' }

test('lists the six steps and four processes', () => {
  expect(STEPS.map((s) => s.id)).toEqual(['read', 'build', 'animate', 'render', 'stills', 'check'])
  expect(STEPS[0].label).toBe('Reading the reference')
  expect(PROCESSES.map((p) => p.id)).toEqual(['read', 'animate', 'stills', 'check'])
})

test('reads the animate log into steps', async () => {
  const lines = (await readFile(new URL('./fixtures/animate-log.txt', import.meta.url), 'utf8')).split('\n')
  const state = {}
  const seen = []
  for (const line of lines) {
    const r = readLine('animate', line, state)
    if (r.total !== undefined) seen.push(['total', r.total])
    if (r.step) seen.push(['step', r.step])
  }
  expect(seen).toEqual([['total', 138], ['step', 'animate'], ['step', 'render']])
  expect(state.total).toBe(138)
})

test('passes --form-targets only when the file exists', async () => {
  const without = commandFor('animate', run, dir, run)
  expect(without.args).not.toContain('--form-targets')
  expect(without.args).toContain('--python-exit-code')
  expect(without.args).toContain('--side-view')
  await writeFile(path.join(dir, 'pipeline-targets.json'), '{}')
  const withIt = commandFor('animate', run, dir, run)
  const i = withIt.args.indexOf('--form-targets')
  expect(withIt.args[i + 1]).toBe(path.join(dir, 'pipeline-targets.json'))
  expect(withIt.args[withIt.args.indexOf('--mixamo') + 1]).toMatch(/mixamo\/squat\.fbx$/)
})

test('no clip and no key positions gives a plain error', () => {
  const ex = { ...run, clip: null, expectedClip: 'squat.fbx' }
  const r = commandFor('animate', ex, dir, ex)
  expect(r.error).toBe('No motion for this exercise yet. Add a Mixamo clip named squat.fbx to 3d-character/mixamo (its README says how).')
  expect(r.cmd).toBeUndefined()
  const none = commandFor('animate', { ...run, clip: null }, dir, { ...run, clip: null })
  expect(none.error).toMatch(/^No motion for this exercise yet\. Add a Mixamo clip for it/)
})

test('no clip moves between the pictures\' key positions (task 9)', async () => {
  const ex = { slug: 'back-squat-from-images', name: 'Back Squat from images', clip: null, expectedClip: null }
  await writeFile(path.join(dir, 'keyposes.json'), '{"phases": []}')
  const r = commandFor('animate', ex, dir, ex)
  expect(r.error).toBeUndefined()
  expect(r.args).not.toContain('--mixamo')
  expect(r.args[r.args.indexOf('--keyposes') + 1]).toBe(path.join(dir, 'keyposes.json'))
  expect(r.args[r.args.indexOf('--exercise') + 1]).toBe('Back Squat from images')
  for (const flag of ['--mpfb', '--wrap-skin', '--side-view', '--out', '--gif', '--save']) expect(r.args).toContain(flag)
  expect(r.args).not.toContain('--form-targets')
  await writeFile(path.join(dir, 'pipeline-targets.json'), '{}')
  expect(commandFor('animate', ex, dir, ex).args).toContain('--form-targets')
  // a clip, when there is one, still wins over the key positions
  expect(commandFor('animate', run, dir, run).args).not.toContain('--keyposes')
})

test('a key-position run reads its own frame count whatever the order of the lines (task 9 review)', () => {
  const rest = 'retarget: read squat.fbx: 69 frames at 30 fps (source frames 1..69)'
  const keys = 'retarget: read keyposes.json: 81 frames at 30 fps (the pictures\' key positions)'
  for (const lines of [[rest, keys], [keys, rest]]) {
    const state = {}
    for (const line of lines) readLine('animate', line, state)
    expect(state.total).toBe(162)
  }
  const clip = {}
  expect(readLine('animate', rest, clip)).toEqual({ total: 138 })      // a clip run: as before
})

test('no clip and the picture path refuses the exercise: its plain refusal, nothing run', async () => {
  const refusal = 'No motion for this exercise: it needs a bench the picture path doesn\'t build yet.'
  const ex = { slug: 'flat-barbell-bench-press', name: 'Flat Barbell Bench Press', clip: null, expectedClip: 'bench-press.fbx', pictureRefusal: refusal }
  await writeFile(path.join(dir, 'keyposes.json'), '{"phases": []}')
  const r = commandFor('animate', ex, dir, ex)
  expect(r.error).toBe(refusal)
  expect(r.cmd).toBeUndefined()
  expect(commandFor('animate', { ...ex, clip: 'bench-press.fbx' }, dir, { ...ex, clip: 'bench-press.fbx' }).args).toContain('--mixamo')
})

test('both Blender commands turn on line buffering before the script runs', () => {
  const a = commandFor('animate', run, dir, run).args
  const s = commandFor('stills', run, dir, run).args
  for (const args of [a, s]) {
    const expr = args.indexOf('--python-expr')
    expect(expr).toBeGreaterThan(-1)
    expect(args[expr + 1]).toContain('line_buffering=True')
    expect(expr).toBeLessThan(args.indexOf('--python'))
  }
})

test('stills require ours.json and a still for every non-context key image', async () => {
  const snapshot = { images: [{ id: 'img-1' }, { id: 'img-2', contextOnly: true }, { id: 'img-3' }] }
  expect(requiredOutputs('stills', dir, snapshot)).toEqual(['ours.json', 'stills/img-1.png', 'stills/img-3.png'])
  await writeFile(path.join(dir, 'ours.json'), JSON.stringify({ stills: [{ file: 'stills/img-1.png' }, { file: 'stills/extra.png' }] }))
  expect(requiredOutputs('stills', dir, snapshot)).toEqual(['ours.json', 'stills/img-1.png', 'stills/img-3.png', 'stills/extra.png'])
})

test('lists required outputs, including every still the script names', async () => {
  expect(requiredOutputs('read', dir)).toEqual(['measurements.json', 'targets.json'])
  expect(requiredOutputs('check', dir)).toEqual(['checks.json'])
  expect(requiredOutputs('animate', dir)).toEqual(['animation.gif', 'frames/retarget.json', 'scene.blend'])
  await writeFile(path.join(dir, 'ours.json'), JSON.stringify({ stills: [{ file: 'stills/a.png' }] }))
  expect(requiredOutputs('stills', dir, { images: [] })).toEqual(['ours.json', 'stills/a.png'])
})

test('the picture loop shows as a note while animating (task 19 review)', () => {
  const state = {}
  expect(readLine('animate', 'form: picture: reading our picture as the Studio will (its camera, five framings, the coach\'s detector)', state))
    .toEqual({ note: 'Reading our picture as the Studio will' })
  expect(readLine('animate', 'form: picture round 0 (frames start 1, middle 19, end 38): start torso_lean picture 7.9', state))
    .toEqual({ note: 'Matching the coach in our picture: reading 1' })
  expect(readLine('animate', 'form: picture round 2 (frames start 69, middle 57, end 37): …', state))
    .toEqual({ note: 'Matching the coach in our picture: reading 3' })
  expect(readLine('animate', 'form: picture: kept round 1 as it was solved and read: …', state)).toEqual({ note: null })
  expect(readLine('stills', 'form: picture round 0 (frames …)', state)).toEqual({})
  expect(readLine('animate', 'form: bar -0.16 of the torso below the shoulder joints', state)).toEqual({})
})

test('row images drive the movement even when a row clip exists', async () => {
  await writeFile(path.join(dir, 'keyposes.json'), '{}')
  const row = {name:'Barbell Bent-Over Row', clip:'row.fbx'}
  const command = commandFor('animate', row, dir, row)
  expect(command.args).toContain('--keyposes')
  expect(command.args).not.toContain('--mixamo')
})
