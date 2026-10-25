import { expect, test } from 'vitest'
import { previewStatus } from './preview-status.js'

const example = (summary) => ({ available: true, summary: { pass: 23, fail: 0, limited: 0, uncertain: 0, unmeasured: 0, ...summary } })

test('photo-depth uncertainty is review work despite zero failed checks', () => {
  expect(previewStatus(example({ uncertain: 10 }))).toEqual({ kind: 'review', label: 'Needs visual review' })
  expect(previewStatus(example({}))).toEqual({ kind: 'pass', label: 'Checks passed' })
})

test('the row keeps its real failures and missing measurements cannot look approved', () => {
  expect(previewStatus(example({ pass: 16, fail: 2 }))).toEqual({ kind: 'fail', label: '2 failed checks' })
  expect(previewStatus(example({ unmeasured: 1 })).kind).toBe('review')
  expect(previewStatus({ available: true, summary: { pass: 12 } }).kind).toBe('review')
  expect(previewStatus(example({ fail: -1 })).kind).toBe('review')
})

test('an unavailable exercise is shown without inventing a completed preview', () => {
  expect(previewStatus({ available: false, summary: { pass: 100 } })).toEqual({ kind: 'unavailable', label: 'No built preview' })
})
