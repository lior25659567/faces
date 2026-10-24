import { describe, expect, test } from 'vitest'
import {
  checksAside, checksLine, dotKind, exerciseLine, historyBar, historyLine, pollInterval, queueLine, scoreKind,
  unmeasuredLine,
} from './status.js'

const summary = { pass: 7, fail: 9, limited: 2, uncertain: 3, info: 4 }

describe('checksLine', () => {
  test('counts pass + fail + limited as the checks that were judged', () => {
    expect(checksLine(summary)).toBe('7 of 18 checks pass')
  })

  test('no summary, no line', () => {
    expect(checksLine(null)).toBeNull()
  })
})

describe('checksAside', () => {
  test('names the limited and uncertain counts that are not zero', () => {
    expect(checksAside(summary)).toBe('2 limited by proportions, 3 uncertain')
    expect(checksAside({ ...summary, limited: 0 })).toBe('3 uncertain')
    expect(checksAside({ ...summary, limited: 0, uncertain: 0 })).toBe('')
  })
})

describe('exerciseLine', () => {
  const idle = { running: null, queued: [] }
  const done = (id, s) => ({ id, status: 'done', summary: s })

  test('running shows the time left', () => {
    const state = { running: { slug: 'squat', runId: 'r1', left: 180, overdue: false }, queued: [] }
    expect(exerciseLine({ slug: 'squat', status: 'running' }, state, [])).toBe('running · about 3 min left')
  })

  test('running past its estimate', () => {
    const state = { running: { slug: 'squat', runId: 'r1', left: 0, overdue: true }, queued: [] }
    expect(exerciseLine({ slug: 'squat', status: 'running' }, state, [])).toBe('running · taking longer than usual')
  })

  test('queued behind another exercise', () => {
    const state = { running: { slug: 'deadlift', runId: 'r9', left: 60, overdue: false }, queued: [{ slug: 'squat', runId: 'r2' }] }
    expect(exerciseLine({ slug: 'squat', status: 'running' }, state, [])).toBe('waiting to start')
  })

  test('needs a reference', () => {
    expect(exerciseLine({ slug: 'squat', status: 'needs-reference' }, idle, [])).toBe('needs a reference')
  })

  test('passing and failing show the latest finished run', () => {
    const runs = [done('r2', summary), done('r1', { pass: 1, fail: 0, limited: 0 })]
    expect(exerciseLine({ slug: 'squat', status: 'failing' }, idle, runs)).toBe('7 of 18 pass')
    expect(exerciseLine({ slug: 'squat', status: 'passing' }, idle, [done('r3', { pass: 18, fail: 0, limited: 0 })]))
      .toBe('18 of 18 pass')
  })

  test('a failed latest run says so instead of an old pass count', () => {
    const runs = [{ id: 'r3', status: 'failed', summary: null }, done('r2', summary)]
    expect(exerciseLine({ slug: 'squat', status: 'failing' }, idle, runs)).toBe('last run failed')
  })

  test('an interrupted run is skipped like the server skips it', () => {
    const runs = [{ id: 'r3', status: 'interrupted', summary: null }, done('r2', summary)]
    expect(exerciseLine({ slug: 'squat', status: 'failing' }, idle, runs)).toBe('7 of 18 pass')
  })

  test('not run yet', () => {
    expect(exerciseLine({ slug: 'squat', status: 'not-run' }, idle, [])).toBe('not run yet')
  })
})

describe('not measured (final fix 1)', () => {
  const idle = { running: null, queued: [] }

  test('green only when nothing fails, everything judged was measured, and something passes', () => {
    expect(scoreKind({ pass: 5, fail: 0, limited: 0, unmeasured: 0 })).toBe('pass')
    expect(scoreKind({ pass: 5, fail: 0, limited: 0, unmeasured: 2 })).toBe('unmeasured')
    expect(scoreKind({ pass: 5, fail: 1, limited: 0, unmeasured: 2 })).toBe('fail')
    expect(scoreKind({ pass: 5, fail: 0, limited: 1, unmeasured: 0 })).toBe('limited')
    expect(scoreKind({ pass: 0, fail: 0, limited: 0, unmeasured: 0 })).toBe('none')
    expect(scoreKind({ pass: 5, fail: 0, limited: 0 })).toBe('pass') // runs from before the count existed
    expect(scoreKind(null)).toBe('none')
  })

  test('says how many', () => {
    expect(unmeasuredLine({ pass: 5, unmeasured: 3 })).toBe('3 not measured')
    expect(unmeasuredLine({ pass: 5, unmeasured: 0 })).toBeNull()
    expect(unmeasuredLine({ pass: 5 })).toBeNull()
    expect(unmeasuredLine(null)).toBeNull()
  })

  test('the list line and dot for an exercise held back by not-measured checks', () => {
    const runs = [{ id: 'r1', status: 'done', summary: { pass: 16, fail: 0, limited: 0, unmeasured: 2 } }]
    expect(exerciseLine({ slug: 'squat', status: 'unmeasured' }, idle, runs)).toBe('16 of 16 pass · 2 not measured')
    expect(exerciseLine({ slug: 'squat', status: 'unmeasured' }, idle, [])).toBe('not measured')
    expect(dotKind('unmeasured')).toBe('unmeasured')
  })
})

test('dotKind maps exercise statuses to status colours', () => {
  expect(dotKind('passing')).toBe('pass')
  expect(dotKind('failing')).toBe('fail')
  expect(dotKind('running')).toBe('running')
  expect(dotKind('needs-reference')).toBe('empty')
  expect(dotKind('not-run')).toBe('idle')
})

describe('historyBar (task 18)', () => {
  test('a run held back by not-measured checks shows amber in its bar, not all green', () => {
    expect(historyBar({ pass: 6, fail: 0, limited: 0, unmeasured: 2 })).toEqual([
      { kind: 'pass', share: 0.75 }, { kind: 'limited', share: 0 }, { kind: 'fail', share: 0 }, { kind: 'unmeasured', share: 0.25 },
    ])
    expect(historyBar({ pass: 2, fail: 1, limited: 1, unmeasured: 0 })).toEqual([
      { kind: 'pass', share: 0.5 }, { kind: 'limited', share: 0.25 }, { kind: 'fail', share: 0.25 }, { kind: 'unmeasured', share: 0 },
    ])
  })

  test('runs from before the count existed, and empty runs', () => {
    expect(historyBar({ pass: 1, fail: 1 }).map((s) => s.share)).toEqual([0.5, 0, 0.5, 0])
    expect(historyBar({ pass: 0, fail: 0, limited: 0, unmeasured: 0 })).toBeNull()
    expect(historyBar(null)).toBeNull()
  })
})

test('historyLine', () => {
  expect(historyLine({ status: 'done', summary })).toBe('7 of 18 pass')
  expect(historyLine({ status: 'failed', summary: null })).toBe('—')
})

test('reference-only checks do not replace the animation score in the exercise list', () => {
  const runs = [
    { status: 'done', readOnly: true, summary: null },
    { status: 'failed', readOnly: true, summary: null },
    { status: 'done', summary: { pass: 4, fail: 1 } },
  ]
  expect(exerciseLine({ slug: 'squat', status: 'failing' }, null, runs)).toBe('4 of 5 pass')
})

describe('pollInterval', () => {
  test('every second while something runs or waits, else every 5 s', () => {
    expect(pollInterval({ running: { slug: 'a' }, queued: [] })).toBe(1000)
    expect(pollInterval({ running: null, queued: [{ slug: 'a' }] })).toBe(1000)
    expect(pollInterval({ running: null, queued: [] })).toBe(5000)
    expect(pollInterval(null)).toBe(5000)
  })
})

test('queueLine', () => {
  expect(queueLine(0)).toBe('Starting now')
  expect(queueLine(1)).toBe('Waiting for 1 run to finish first')
  expect(queueLine(3)).toBe('Waiting for 3 runs to finish first')
})
