import { expect, test } from 'vitest'
import { DEFAULT_SECONDS, estimateStep, timeLeft } from './eta.js'

const IDS = ['read', 'build', 'animate', 'render', 'stills', 'check']

test('median of the last five, default without history', () => {
  expect(estimateStep('read', [10, 20, 30, 40, 50, 1000])).toBe(40)
  expect(estimateStep('read', [])).toBe(DEFAULT_SECONDS.read)
})

test('reports overdue instead of negative time', () => {
  const estimates = { ...DEFAULT_SECONDS, build: 60 }
  const r = timeLeft({ steps: IDS, currentId: 'build', elapsed: 90, fraction: null, estimates })
  const later = estimates.animate + estimates.render + estimates.stills + estimates.check
  expect(r).toEqual({ seconds: later, overdue: true })
})

test('uses the render fraction when there is one', () => {
  const estimates = { ...DEFAULT_SECONDS }
  const r = timeLeft({ steps: IDS, currentId: 'render', elapsed: 60, fraction: 0.5, estimates })
  expect(r.seconds).toBe(60 + estimates.stills + estimates.check)
  expect(r.overdue).toBe(false)
})

test('ignores a tiny fraction and falls back to the estimate', () => {
  const r = timeLeft({ steps: IDS, currentId: 'check', elapsed: 1, fraction: 0.01, estimates: DEFAULT_SECONDS })
  expect(r).toEqual({ seconds: 4, overdue: false })
})
