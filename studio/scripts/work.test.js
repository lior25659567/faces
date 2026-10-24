import { afterEach, beforeEach, expect, test } from 'vitest'
import { spawn, spawnSync } from 'node:child_process'
import { mkdtemp, rm } from 'node:fs/promises'
import { readFileSync, writeFileSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'
import { createStore } from '../server/store.js'

const CLI = path.join(path.dirname(fileURLToPath(import.meta.url)), 'work.mjs')
const PNG = Buffer.from('89504e470d0a1a0a0000000d49484452', 'hex')
let root

beforeEach(async () => {
  root = await mkdtemp(path.join(os.tmpdir(), 'studio-work-cli-'))
})
afterEach(() => rm(root, { recursive: true, force: true }))

// The CLI as an agent runs it, aimed at a temp data root (never the user's).
const cli = (...args) => {
  const r = spawnSync(process.execPath, [CLI, ...args], {
    cwd: os.tmpdir(), env: { ...process.env, STUDIO_DATA: root }, encoding: 'utf8', timeout: 20_000,
  })
  return { code: r.status, out: r.stdout.trim(), err: r.stderr.trim() }
}
const item = (id) => JSON.parse(readFileSync(path.join(root, 'work', `${id}.json`), 'utf8'))

test('start prints only the id; log, done and fail write the item', () => {
  const s = cli('start', '--title', 'Fix skin folds at the shoulders', '--exercises', 'barbell-back-squat,pull-up')
  expect(s).toMatchObject({ code: 0, err: '' })
  expect(s.out).toMatch(/^\d{8}-\d{6}-[0-9a-f]{4}$/)
  const id = s.out
  expect(item(id)).toMatchObject({ title: 'Fix skin folds at the shoulders', exercises: ['barbell-back-squat', 'pull-up'], status: 'running' })

  expect(cli('log', id, 'Measured the folds: 1,008 mm² at the left shoulder').code).toBe(0)
  expect(cli('log', id, 'Trying a corrective shape').code).toBe(0)
  expect(item(id).log.map((l) => l.text)).toEqual(['Measured the folds: 1,008 mm² at the left shoulder', 'Trying a corrective shape'])

  const d = cli('done', id, '--summary', 'Shoulder folds are down to 120 mm²')
  expect(d.code).toBe(0)
  expect(item(id)).toMatchObject({ status: 'done', summary: 'Shoulder folds are down to 120 mm²' })

  const f = cli('start', '--title=Another try')
  const failed = cli('fail', f.out, '--summary=It made the elbows worse')
  expect(failed.code).toBe(0)
  expect(item(f.out)).toMatchObject({ status: 'failed', exercises: [], summary: 'It made the elbows worse' })
})

test('--exercises all means every exercise with usable pictures', () => {
  createStore(root).addReference('pull-up', PNG, 'a.png')
  const s = cli('start', '--title', 'Fix', '--exercises', 'all')
  expect(s.code).toBe(0)
  expect(item(s.out).exercises).toEqual(['pull-up'])
})

test('list shows the items, running first', () => {
  const a = cli('start', '--title', 'First').out
  const b = cli('start', '--title', 'Second').out
  cli('done', a, '--summary', 'ok')
  const l = cli('list')
  expect(l.code).toBe(0)
  const lines = l.out.split('\n')
  expect(lines[0]).toContain(b)
  expect(lines[0]).toContain('running')
  expect(lines[1]).toContain(a)
  expect(lines[1]).toContain('done')
})

test.each([
  [['log', '20990101-000000-abcd', 'x'], /no work item "20990101-000000-abcd"/i],
  [['done', '20990101-000000-abcd', '--summary', 'x'], /no work item/i],
  [['start'], /title/i],
  [['start', '--title', 'X', '--exercises', 'squatt'], /no exercise called "squatt"/i],
  [['start', '--title', 'X', '--colour', 'red'], /--colour/],
  [['done'], /which work item/i],
  [['frobnicate'], /usage/i],
  [[], /usage/i],
])('plain errors: %j', (args, message) => {
  const r = cli(...args)
  expect(r.code).toBe(1)
  expect(r.err).toMatch(message)
  expect(r.err).not.toMatch(/at .*\.js:\d+/) // no stack traces
})

test('a done item cannot be finished again; a late log line is kept', () => {
  const id = cli('start', '--title', 'X').out
  cli('done', id, '--summary', 'ok')
  const r = cli('fail', id, '--summary', 'no')
  expect(r.code).toBe(1)
  expect(r.err).toMatch(/already marked done/)
  expect(cli('log', id, 'more').code).toBe(0)
  expect(item(id)).toMatchObject({ status: 'done', log: [{ text: 'more' }] })
})

// Each CLI process pauses 300 ms after reading the item, so two writers' read-modify-write windows
// overlap: without a lock the later write would undo the other's change.
const SLOW_READS = `
import fs from 'node:fs'
import { syncBuiltinESMExports } from 'node:module'
const read = fs.readFileSync
fs.readFileSync = function (file, ...rest) {
  const out = read.call(this, file, ...rest)
  if (/[\\/]work[\\/][^\\/]+\\.json$/.test(String(file))) Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, 300)
  return out
}
syncBuiltinESMExports()
`
const slowCli = (preload, ...args) => new Promise((resolve) => {
  const child = spawn(process.execPath, ['--import', pathToFileURL(preload).href, CLI, ...args], {
    cwd: os.tmpdir(), env: { ...process.env, STUDIO_DATA: root }, stdio: ['ignore', 'pipe', 'pipe'],
  })
  let err = ''
  child.stderr.on('data', (c) => (err += c))
  child.on('close', (code) => resolve({ code, err }))
})

test('two writers racing: log and done both land (done wins, both log lines survive)', async () => {
  const preload = path.join(root, 'slow-reads.mjs')
  writeFileSync(preload, SLOW_READS)
  for (let round = 0; round < 3; round++) {
    const id = cli('start', '--title', `Race ${round}`).out
    cli('log', id, 'first line')
    const [l, d] = await Promise.all([slowCli(preload, 'log', id, 'second line'), slowCli(preload, 'done', id, '--summary', 'finished')])
    expect(l, l.err).toMatchObject({ code: 0 })
    expect(d, d.err).toMatchObject({ code: 0 })
    expect(item(id)).toMatchObject({ status: 'done', summary: 'finished' })
    expect(item(id).log.map((x) => x.text)).toEqual(['first line', 'second line'])
  }
}, 30_000)

test('done --recheck marks the item as a check-only re-run', () => {
  const id = cli('start', '--title', 'Tighten the depth rule', '--exercises', 'barbell-back-squat').out
  const d = cli('done', id, '--summary', 'depth judged on the thigh angle', '--recheck')
  expect(d).toMatchObject({ code: 0, err: '' })
  expect(item(id)).toMatchObject({ status: 'done', summary: 'depth judged on the thigh angle', recheck: true })
})
