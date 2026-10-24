import { expect, test } from 'vitest'
import { hashFor, routeFromHash, slugFromHash } from './hash.js'

test('an exercise route is #/<slug>', () => {
  expect(slugFromHash('#/barbell-back-squat')).toBe('barbell-back-squat')
  expect(slugFromHash('#/romanian%20deadlift')).toBe('romanian deadlift')
})

test('in-page anchors and empty hashes are not routes', () => {
  expect(slugFromHash('#references')).toBeNull()
  expect(slugFromHash('#')).toBeNull()
  expect(slugFromHash('')).toBeNull()
  expect(slugFromHash('#/')).toBeNull()
  expect(slugFromHash(undefined)).toBeNull()
})

test('a broken escape is not a route', () => {
  expect(slugFromHash('#/%E0%A4%A')).toBeNull()
})

test('hashFor round-trips', () => {
  expect(hashFor('pull-up')).toBe('#/pull-up')
  expect(slugFromHash(hashFor('a b'))).toBe('a b')
})

test('a run route is #/<slug>/run/<id>, for "Re-ran automatically" links', () => {
  expect(hashFor('barbell-back-squat', '20261022-163941-bfde')).toBe('#/barbell-back-squat/run/20261022-163941-bfde')
  expect(routeFromHash('#/barbell-back-squat/run/20261022-163941-bfde')).toEqual({ slug: 'barbell-back-squat', runId: '20261022-163941-bfde' })
  expect(slugFromHash('#/barbell-back-squat/run/20261022-163941-bfde')).toBe('barbell-back-squat')
  expect(routeFromHash('#/pull-up')).toEqual({ slug: 'pull-up', runId: null })
  expect(routeFromHash('#/pull-up/other/x')).toBeNull()
  expect(routeFromHash('#references')).toBeNull()
})
