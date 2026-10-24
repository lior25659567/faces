import { describe, expect, test } from 'vitest'
import {
  buildComparison, defaultRunId, groupRows, pairTally, phaseTally, samePictures, shouldMirror, stripReasons, tallyKind,
  tallyText,
} from './run-view.js'

describe('shouldMirror', () => {
  test('mirror ours only when both facings are known and differ', () => {
    expect(shouldMirror('left', 'right')).toBe(true)
    expect(shouldMirror('right', 'right')).toBe(false)
    expect(shouldMirror(null, 'right')).toBe(false)
    expect(shouldMirror('left', undefined)).toBe(false)
  })
})

describe('buildComparison', () => {
  const images = [
    { id: 'img-17', file: 'img-17.png', original: '17.png', order: 4, phase: 'end', view: 'side', contextOnly: false },
    { id: 'img-14', file: 'img-14.png', original: '14.png', order: 1, phase: 'start', view: 'side', contextOnly: true },
    { id: 'img-15', file: 'img-15.png', original: '15.png', order: 2, phase: 'start', view: 'side', contextOnly: false },
    { id: 'img-bt', file: 'img-bt.jpg', original: 'behind.jpg', order: 5, phase: 'start', view: 'behind', contextOnly: false },
  ]
  const ours = {
    stills: [
      { image: 'img-15', phase: 'start', view: 'side', frame: 68, file: 'stills/img-15.png', facing: 'left' },
      { image: 'img-17', phase: 'end', view: 'side', frame: 38, file: 'stills/img-17.png', facing: 'right' },
      { image: 'img-bt', phase: 'start', view: 'behind', frame: 68, file: 'stills/img-bt.png', facing: null },
    ],
  }
  const measurements = { 'img-15': { facing: 'right' }, 'img-17': { facing: 'right' }, 'img-bt': { facing: null } }
  const targets = {
    images: {
      'img-15': { used: true, why: null },
      'img-17': { used: true, why: null },
      'img-bt': { used: false, why: 'Another image already gives the widths' },
    },
  }
  const view = buildComparison({ slug: 'barbell-back-squat', runId: 'r1', images, ours, measurements, targets })

  test('pairs follow the reference order and skip context-only images', () => {
    expect(view.pairs.map((p) => p.id)).toEqual(['img-15', 'img-17', 'img-bt'])
  })

  test('each pair has the coach image and our still at the same moment', () => {
    expect(view.pairs[0]).toMatchObject({
      phase: 'start', view: 'side', frame: 68,
      refUrl: '/files/references/barbell-back-squat/img-15.png',
      oursUrl: '/files/runs/barbell-back-squat/r1/stills/img-15.png',
    })
  })

  test('ours is mirrored when it faces the other way from the coach', () => {
    expect(view.pairs[0].mirror).toBe(true)
    expect(view.pairs[1].mirror).toBe(false)
    expect(view.pairs[2].mirror).toBe(false)
  })

  test('unused images carry their reason', () => {
    expect(view.pairs[2]).toMatchObject({ used: false, why: 'Another image already gives the widths' })
    expect(view.pairs[0]).toMatchObject({ used: true, why: null })
  })

  test('context-only images go in their own row', () => {
    expect(view.context).toEqual([
      { id: 'img-14', original: '14.png', refUrl: '/files/references/barbell-back-squat/img-14.png' },
    ])
  })

  test('without a run output there are no stills, only the references', () => {
    const bare = buildComparison({ slug: 's', runId: 'r', images, ours: null, measurements: null, targets: null })
    expect(bare.pairs.map((p) => p.oursUrl)).toEqual([null, null, null])
    expect(bare.pairs.map((p) => p.used)).toEqual([null, null, null])
  })
})

describe('groupRows', () => {
  const rows = [
    { group: 'body', metric: 'cracks' },
    { group: 'end', metric: 'knee_included' },
    { group: 'start', metric: 'knee_included' },
    { group: 'all positions', metric: 'grip_ratio' },
    { group: 'start', metric: 'hip_included' },
    { group: 'mystery', metric: 'x' },
  ]
  test('Start, Middle, End, All positions, Body, then anything new', () => {
    const groups = groupRows(rows)
    expect(groups.map((g) => g.label)).toEqual(['Start', 'End', 'All positions', 'Body', 'mystery'])
    expect(groups[0].rows.map((r) => r.metric)).toEqual(['knee_included', 'hip_included'])
  })
})

test('phaseTally counts a position\'s judged checks', () => {
  const rows = [
    { group: 'end', status: 'pass' }, { group: 'end', status: 'fail' }, { group: 'end', status: 'limited' },
    { group: 'end', status: 'uncertain' }, { group: 'end', status: 'info' }, { group: 'start', status: 'pass' },
  ]
  expect(phaseTally(rows, 'end')).toEqual({ pass: 1, judged: 3, unmeasured: 0 })
  expect(phaseTally([], 'end')).toEqual({ pass: 0, judged: 0, unmeasured: 0 })
})

describe('a position\'s tally above not-measured rows (task 18)', () => {
  const rows = [
    { group: 'start', status: 'pass' }, { group: 'start', status: 'pass' },
    { group: 'start', status: 'info', kind: 'not-measured', metric: 'neck_vs_torso' },
    { group: 'start', status: 'info', kind: 'shown-only', metric: 'head_pitch' },
    // an older run's face-direction row said "not-measured" but was never counted: not counted here either
    { group: 'middle', status: 'pass' }, { group: 'middle', status: 'info', kind: 'not-measured', metric: 'head_pitch' },
    { group: 'end', status: 'pass' }, { group: 'end', status: 'fail' },
    { group: 'end', status: 'info', kind: 'not-measured', metric: 'knee_included' },
    { group: 'body', status: 'info', kind: 'not-measured', metric: 'cracks' },
  ]

  test('counts the rows that would be judged but were not measured', () => {
    expect(phaseTally(rows, 'start')).toEqual({ pass: 2, judged: 2, unmeasured: 1 })
    expect(phaseTally(rows, 'middle')).toEqual({ pass: 1, judged: 1, unmeasured: 0 })
    expect(phaseTally(rows, 'body')).toEqual({ pass: 0, judged: 0, unmeasured: 1 })
  })

  test('is amber, not green, when everything judged passes but something was not measured', () => {
    expect(tallyKind(phaseTally(rows, 'start'))).toBe('unmeasured')
    expect(tallyKind(phaseTally(rows, 'middle'))).toBe('pass')
    expect(tallyKind(phaseTally(rows, 'end'))).toBe('fail')            // a fail stays red
    expect(tallyKind(phaseTally(rows, 'body'))).toBe('unmeasured')
    expect(tallyKind({ pass: 0, judged: 0, unmeasured: 0 })).toBe('none')
  })

  test('says how many were not measured', () => {
    expect(tallyText(phaseTally(rows, 'start'))).toBe('2 of 2 pass · 1 not measured')
    expect(tallyText(phaseTally(rows, 'middle'))).toBe('1 of 1 pass')
    expect(tallyText(phaseTally(rows, 'body'))).toBe('1 not measured')
    expect(tallyText({ pass: 0, judged: 0, unmeasured: 0 })).toBeNull()
  })
})

describe('defaultRunId', () => {
  test('the newest finished run, so a run in progress does not hide the last result', () => {
    const runs = [{ id: 'r3', status: 'running' }, { id: 'r2', status: 'done' }, { id: 'r1', status: 'done' }]
    expect(defaultRunId(runs)).toBe('r2')
  })

  test('falls back to the newest run, then null', () => {
    expect(defaultRunId([{ id: 'r2', status: 'failed' }, { id: 'r1', status: 'interrupted' }])).toBe('r2')
    expect(defaultRunId([])).toBeNull()
  })
})

describe('samePictures', () => {
  const now = [
    { id: 'a', order: 1, phase: 'start', view: 'side', contextOnly: false },
    { id: 'b', order: 2, phase: 'end', view: 'side', contextOnly: false },
  ]
  const snapshot = [{ ...now[1], path: '/x/b.png' }, { ...now[0], path: '/x/a.png' }]

  test('the same pictures in the same order and tags match, whatever else the snapshot carries', () => {
    expect(samePictures(now, snapshot)).toBe(true)
  })

  test('a reorder, a new tag, a context change, an added or removed picture do not', () => {
    expect(samePictures([{ ...now[0], order: 3 }, now[1]], snapshot)).toBe(false)
    expect(samePictures([{ ...now[0], phase: 'middle' }, now[1]], snapshot)).toBe(false)
    expect(samePictures([{ ...now[0], view: 'front', viewSetBy: 'user' }, now[1]], snapshot)).toBe(false)
    expect(samePictures([{ ...now[0], contextOnly: true }, now[1]], snapshot)).toBe(false)
    expect(samePictures([now[0]], snapshot)).toBe(false)
  })

  test('a view the detector filled in after the run started is not a change', () => {
    const before = [{ ...now[0], view: null, viewSetBy: null }, { ...now[1], view: null, viewSetBy: null }]
    const after = [{ ...now[0], view: 'side', viewSetBy: 'detector' }, { ...now[1], view: 'side', viewSetBy: 'detector' }]
    expect(samePictures(after, before)).toBe(true)
  })

  test('without a snapshot we cannot say they changed', () => {
    expect(samePictures(now, null)).toBe(true)
  })
})

describe('cropped references', () => {
  const images = [
    { id: 'a', file: 'a.png', order: 1, phase: 'start', contextOnly: false },
    { id: 'b', file: 'b.png', order: 2, phase: 'end', contextOnly: false },
  ]
  test('each pair carries the crop the reading found; older runs without one show the full picture', () => {
    const measurements = { a: { crop: [0.3, 0.4, 0.55, 0.78], size: [1260, 2736] }, b: { facing: 'right' } }
    const { pairs } = buildComparison({ slug: 's', runId: 'r', images, ours: null, measurements, targets: null })
    expect(pairs[0].crop).toEqual({ box: [0.3, 0.4, 0.55, 0.78], size: [1260, 2736] })
    expect(pairs[1].crop).toBeNull()
    expect(buildComparison({ slug: 's', runId: 'r', images, ours: null, measurements: null, targets: null }).pairs[0].crop).toBeNull()
  })
})

// ---- the strip's "Not used" reasons come from one run's own files, and say which run (final fix 2) ----
describe('stripReasons', () => {
  const run = { id: 'r2', createdAt: '2026-10-22T14:03:00', status: 'failed' }
  const why = (images) => ({ images })

  test('a finished read: that run\'s targets.json reasons', () => {
    const r = stripReasons({ run, targets: why({ a: { used: false, why: 'Context only' } }), readProblems: null })
    expect(r).toEqual({ runId: 'r2', createdAt: run.createdAt, images: { a: { used: false, why: 'Context only' } } })
  })

  test('a read that stopped: that run\'s read-problems.json reasons', () => {
    const r = stripReasons({ run, targets: null, readProblems: why({ a: { used: false, why: 'No body found in this image' } }) })
    expect(r.images.a.why).toBe('No body found in this image')
  })

  test('a run that has not read yet shows none, never another run\'s', () => {
    expect(stripReasons({ run: { ...run, status: 'running' }, targets: null, readProblems: null })).toBeNull()
    expect(stripReasons({ run: null, targets: why({}), readProblems: null })).toBeNull()
  })
})

describe('the comparison\'s picture pair tally (task 19: as the checks table\'s)', () => {
  const rows = [
    { group: 'start', status: 'pass' }, { group: 'start', status: 'pass' },
    { group: 'start', status: 'info', kind: 'not-measured', metric: 'neck_vs_torso' },
    { group: 'middle', status: 'pass' }, { group: 'middle', status: 'uncertain' },
    { group: 'end', status: 'pass' }, { group: 'end', status: 'fail' },
    { group: 'end', status: 'info', kind: 'not-measured', metric: 'knee_included' },
  ]

  test('is amber with the not-measured count, never green, above a not-measured row', () => {
    expect(pairTally(rows, 'start')).toEqual({ kind: 'unmeasured', text: '2 of 2 pass · 1 not measured' })
  })

  test('is green only when every judged row passes and nothing is unmeasured; red on a miss', () => {
    expect(pairTally(rows, 'middle')).toEqual({ kind: 'pass', text: '1 of 1 pass' })
    expect(pairTally(rows, 'end')).toEqual({ kind: 'fail', text: '1 of 2 pass · 1 not measured' })
  })

  test('shows nothing for a position with nothing judged or unmeasured', () => {
    expect(pairTally(rows, 'body')).toBeNull()
    expect(pairTally([{ group: 'end', status: 'uncertain' }], 'end')).toBeNull()
  })

  test('shows a position whose only rows were not measured', () => {
    expect(pairTally([{ group: 'end', status: 'info', kind: 'not-measured', metric: 'torso_lean' }], 'end'))
      .toEqual({ kind: 'unmeasured', text: '1 not measured' })
  })
})
