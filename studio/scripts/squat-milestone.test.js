import { expect, test } from 'vitest'
import { mkdtempSync, readFileSync, readdirSync, rmSync, writeFileSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { PLAN, activeRun, findDeleted, findRef, fmt, lockPath, tagPatch, takeLock } from './squat-milestone.mjs'

test('findRef ignores deleted and other names', () => {
  const refs = [{ id: 'a', original: '15.png', deleted: true }, { id: 'b', original: '15.png', deleted: false }]
  expect(findRef(refs, '15.png').id).toBe('b')
  expect(findRef(refs, '16.png')).toBeNull()
})
const U = { phaseSetBy: 'user', viewSetBy: 'user' }
test('tagPatch leaves matching tags alone, even guessed ones', () => {
  const ref = { phase: 'end', phaseSetBy: 'guess', view: 'side', viewSetBy: 'detector', contextOnly: false }
  expect(tagPatch(ref, PLAN[3].want)).toEqual({ patch: {}, kept: [] })
})
test('tagPatch patches guesses and missing tags', () => {
  const ref = { phase: 'middle', phaseSetBy: 'guess', view: null, viewSetBy: 'detector', contextOnly: true }
  expect(tagPatch(ref, PLAN[1].want).patch).toEqual({ phase: 'start', view: 'side', contextOnly: false })
})
test('tagPatch keeps what the user set and says so', () => {
  const ref = { phase: 'middle', view: 'side', contextOnly: false, ...U }
  const r = tagPatch(ref, PLAN[1].want)
  expect(r.patch).toEqual({})
  expect(r.kept).toEqual([{ key: 'phase', has: 'middle', wanted: 'start' }])
})
test('findDeleted finds only deleted ones', () => {
  const refs = [{ original: '17.png', deleted: true }, { original: '16.png', deleted: false }]
  expect(findDeleted(refs, '17.png')).toBeTruthy()
  expect(findDeleted(refs, '16.png')).toBeNull()
})
test('lock: second copy refused, dead copy taken over, same key same file', () => {
  const dir = mkdtempSync(path.join(os.tmpdir(), 'sq-lock-'))
  const f = lockPath('http://x', 'a', dir)
  expect(f).toBe(lockPath('http://x', 'a', dir))
  expect(f).not.toBe(lockPath('http://y', 'a', dir))
  expect(takeLock(f)).toBe(true)
  expect(takeLock(f)).toBe(false)
  rmSync(f)
  writeFileSync(f, '99999999')
  expect(takeLock(f, () => false)).toBe(true)
  rmSync(dir, { recursive: true })
})
test('lock: unreadable or zero pid counts as held, never taken over', () => {
  const dir = mkdtempSync(path.join(os.tmpdir(), 'sq-lock-'))
  const f = lockPath('http://z', 'a', dir)
  writeFileSync(f, '')
  expect(takeLock(f, () => false)).toBe(false)
  writeFileSync(f, '0')
  expect(takeLock(f, () => false)).toBe(false)
  expect(readdirSync(dir)).toEqual([path.basename(f)]) // no temp file left behind
  rmSync(dir, { recursive: true })
})
test('lock appears holding its own pid', () => {
  const dir = mkdtempSync(path.join(os.tmpdir(), 'sq-lock-'))
  const f = lockPath('http://w', 'a', dir)
  expect(takeLock(f)).toBe(true)
  expect(readFileSync(f, 'utf8')).toBe(String(process.pid))
  rmSync(dir, { recursive: true })
})
test('activeRun finds running or queued squat only', () => {
  expect(activeRun({ running: { slug: 'barbell-back-squat', runId: 'r1' }, queued: [] })).toEqual({ runId: 'r1' })
  expect(activeRun({ running: null, queued: [{ slug: 'barbell-back-squat', runId: 'r2' }] })).toEqual({ runId: 'r2' })
  expect(activeRun({ running: { slug: 'pull-up', runId: 'x' }, queued: [] })).toBeNull()
})
test('fmt', () => expect(fmt(95)).toBe('1m35s'))
