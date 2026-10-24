import { describe, expect, test } from 'vitest'
import {
  WORK_STATUS_LABELS, awaitingRerun, inProgressText, livePoll, logTail, logTime, rerunStatus, staleText, startedText,
  tookText, workForExercise,
} from './work.js'

const NOW = new Date('2026-10-22T16:00:00')
const ago = (s) => new Date(NOW.getTime() - s * 1000).toISOString()
const item = (over) => ({
  id: 'x', title: 'Fix', exercises: [], status: 'running', startedAt: ago(60), finishedAt: null, log: [], summary: null,
  rerun: null, ...over,
})

describe('top bar', () => {
  test('says how many fixes are in progress, or nothing', () => {
    expect(inProgressText([])).toBeNull()
    expect(inProgressText([item({ status: 'done' })])).toBeNull()
    expect(inProgressText([item({}), item({ status: 'failed' })])).toBe('1 fix in progress')
    expect(inProgressText([item({}), item({})])).toBe('2 fixes in progress')
  })

  test('a short form for phone widths', () => {
    expect(inProgressText([item({})], { short: true })).toBe('1 fix')
    expect(inProgressText([item({}), item({})], { short: true })).toBe('2 fixes')
    expect(inProgressText([], { short: true })).toBeNull()
  })
})

describe('polling', () => {
  test('1 s while a run or a fix is going, or a finished fix waits for its re-run; else 5 s', () => {
    expect(livePoll({ running: null, queued: [] }, [])).toBe(5000)
    expect(livePoll(null, null)).toBe(5000)
    expect(livePoll({ running: { runId: 'r' }, queued: [] }, [])).toBe(1000)
    expect(livePoll({ running: null, queued: [{}] }, [])).toBe(1000)
    expect(livePoll({ running: null, queued: [] }, [item({})])).toBe(1000)
    expect(livePoll({ running: null, queued: [] }, [item({ status: 'done' })])).toBe(1000)
    expect(livePoll({ running: null, queued: [] }, [item({ status: 'done', rerun: [], rerunQueuedAt: ago(1) })])).toBe(5000)
    expect(livePoll({ running: null, queued: [] }, [item({ status: 'failed' })])).toBe(5000)
  })

  test('a done item is waiting for its re-run until the server records it', () => {
    expect(awaitingRerun(item({ status: 'done' }))).toBe(true)
    expect(awaitingRerun(item({ status: 'done', rerun: [{ slug: 'a', runId: 'r' }], rerunQueuedAt: ago(1) }))).toBe(false)
    // An old note from before the catch-up existed: the Studio re-runs it on its next look.
    expect(awaitingRerun(item({ status: 'done', rerunNote: "Not re-run: the Studio wasn't running when this finished." }))).toBe(true)
    expect(awaitingRerun(item({ status: 'running' }))).toBe(false)
    expect(awaitingRerun(item({ status: 'failed' }))).toBe(false)
  })
})

describe('an exercise page', () => {
  test('shows the items that list it: all running ones, then the newest two finished', () => {
    const items = [
      item({ id: 'r', exercises: ['squat'] }),
      item({ id: 'd1', status: 'done', exercises: ['squat', 'row'] }),
      item({ id: 'other', exercises: ['row'] }),
      item({ id: 'd2', status: 'failed', exercises: ['squat'] }),
      item({ id: 'd3', status: 'done', exercises: ['squat'] }),
    ]
    const { shown, hidden } = workForExercise(items, 'squat')
    expect(shown.map((w) => w.id)).toEqual(['r', 'd1', 'd2'])
    expect(hidden).toBe(1)
    expect(workForExercise(items, 'pull-up')).toEqual({ shown: [], hidden: 0 })
    expect(workForExercise(null, 'squat')).toEqual({ shown: [], hidden: 0 })
  })
})

describe('words', () => {
  test('status labels never claim a pass', () => {
    expect(WORK_STATUS_LABELS).toEqual({ running: 'In progress', done: 'Finished', failed: 'Failed' })
  })

  test('time since start', () => {
    expect(startedText(ago(20), NOW)).toBe('Started less than a minute ago')
    expect(startedText(ago(4 * 60 + 50), NOW)).toBe('Started 4 min ago')
    expect(startedText(ago(2 * 3600 + 5 * 60), NOW)).toBe('Started 2 h 5 min ago')
    expect(startedText(ago(3 * 3600), NOW)).toBe('Started 3 h ago')
    expect(startedText('2026-10-20T09:05:00', NOW)).toBe('Started 20 Oct 09:05')
    expect(startedText(null, NOW)).toBe('')
  })

  test('how long a finished item took', () => {
    expect(tookText(item({ status: 'done', startedAt: '2026-10-22T10:00:00Z', finishedAt: '2026-10-22T10:18:30Z' }))).toBe('took 18 min 30 s')
    expect(tookText(item({}))).toBeNull()
  })

  test('the newest log lines, oldest of them first', () => {
    const log = [1, 2, 3, 4, 5].map((n) => ({ at: ago(100 - n), text: `line ${n}` }))
    expect(logTail(log).map((l) => l.text)).toEqual(['line 3', 'line 4', 'line 5'])
    expect(logTail(log, Infinity)).toHaveLength(5)
    expect(logTail(undefined)).toEqual([])
  })

  test('log times: the clock today, the date before', () => {
    expect(logTime('2026-10-22T09:05:00', NOW)).toBe('09:05')
    expect(logTime('2026-10-21T18:30:00', NOW)).toBe('Yesterday 18:30')
    expect(logTime('nonsense', NOW)).toBe('')
  })

  test('what the re-run is doing, from the queue and the runs on disk', () => {
    const state = { running: { runId: 'live', left: 200, overdue: false }, queued: [{ runId: 'wait' }] }
    expect(rerunStatus({ runId: 'live' }, state, [])).toBe('running, about 3 min left')
    expect(rerunStatus({ runId: 'wait' }, state, [])).toBe('waiting to start')
    const runs = [
      { id: 'ok', status: 'done', summary: { pass: 4, fail: 2, limited: 0 } },
      { id: 'bad', status: 'failed', summary: null },
      { id: 'cut', status: 'interrupted' },
    ]
    expect(rerunStatus({ runId: 'ok' }, state, runs)).toBe('finished: 4 of 6 checks pass')
    expect(rerunStatus({ runId: 'bad' }, state, runs)).toBe('stopped with an error')
    expect(rerunStatus({ runId: 'cut' }, state, runs)).toBe('interrupted')
    expect(rerunStatus({ runId: 'unknown' }, state, runs)).toBe('')
  })
})

// ---- a fix that went quiet (final fix 9) ----
describe('staleText', () => {
  const H = 3600
  test('two hours without an update: it may have stopped', () => {
    expect(staleText(item({ startedAt: ago(3 * H), updatedAt: ago(2 * H) }), NOW)).toBe('No update for 2 h — it may have stopped')
    expect(staleText(item({ startedAt: ago(9 * H), updatedAt: ago(5 * H + 59 * 60) }), NOW)).toBe('No update for 5 h — it may have stopped')
  })

  test('the newest log line counts as an update; without updatedAt the start does', () => {
    const log = [{ at: ago(30 * 60), text: 'Measured again' }]
    expect(staleText(item({ startedAt: ago(5 * H), updatedAt: ago(5 * H), log }), NOW)).toBeNull()
    expect(staleText(item({ startedAt: ago(2 * H + 1) }), NOW)).toBe('No update for 2 h — it may have stopped')
  })

  test('under two hours, or finished: nothing to say', () => {
    expect(staleText(item({ startedAt: ago(3 * H), updatedAt: ago(2 * H - 60) }), NOW)).toBeNull()
    expect(staleText(item({ status: 'done', startedAt: ago(9 * H), updatedAt: ago(9 * H) }), NOW)).toBeNull()
  })
})
