import { formatLeft } from './format.js'

const n = (v) => (typeof v === 'number' && Number.isFinite(v) ? v : 0)

// Judged checks are pass + fail + limited; uncertain and measured-only rows are not judged.
const counts = (summary) => ({ pass: n(summary.pass), judged: n(summary.pass) + n(summary.fail) + n(summary.limited) })

export const checksLine = (summary) => {
  if (!summary) return null
  const { pass, judged } = counts(summary)
  return `${pass} of ${judged} checks pass`
}

// Checks that would be judged but weren't measured on the character (checks.json summary `unmeasured`):
// any one keeps the exercise from passing.
export const unmeasuredLine = (summary) => (n(summary?.unmeasured) ? `${summary.unmeasured} not measured` : null)

// The header score's colour: green only when nothing fails, everything judged was measured, and
// something passes; amber when held back by not-measured checks or proportions.
export const scoreKind = (summary) => {
  if (!summary) return 'none'
  if (n(summary.fail)) return 'fail'
  if (n(summary.unmeasured)) return 'unmeasured'
  if (n(summary.limited)) return 'limited'
  return n(summary.pass) ? 'pass' : 'none'
}

export const checksAside = (summary) => {
  if (!summary) return ''
  const parts = []
  if (n(summary.limited)) parts.push(`${summary.limited} limited by proportions`)
  if (n(summary.uncertain)) parts.push(`${summary.uncertain} uncertain`)
  return parts.join(', ')
}

// The history bar's parts, as shares of every check that was judged or would have been: pass, limited,
// fail, and the not-measured ones in amber, so a run held back by them never shows all green.
export const historyBar = (summary) => {
  if (!summary) return null
  const parts = ['pass', 'limited', 'fail', 'unmeasured'].map((kind) => ({ kind, count: n(summary[kind]) }))
  const total = parts.reduce((t, p) => t + p.count, 0)
  if (!total) return null
  return parts.map(({ kind, count }) => ({ kind, share: count / total }))
}

export const historyLine = (run) => {
  if (!run?.summary) return '—'
  const { pass, judged } = counts(run.summary)
  return `${pass} of ${judged} pass`
}

// The run the server judges an exercise by: the newest that finished (done or failed).
const latestFinished = (runs = []) => runs.find((r) => r.status === 'done' || r.status === 'failed') || null

// The short line under each exercise in the list.
export const exerciseLine = (exercise, state, runs) => {
  const { slug, status } = exercise
  if (status === 'running') {
    const live = state?.running
    if (live?.slug === slug) return `running · ${formatLeft({ seconds: live.left, overdue: live.overdue })}`
    return 'waiting to start'
  }
  if (status === 'needs-reference') return 'needs a reference'
  if (status === 'unmeasured') {
    const latest = latestFinished(runs)
    return latest?.summary ? `${historyLine(latest)} · ${unmeasuredLine(latest.summary)}` : 'not measured'
  }
  if (status === 'passing' || status === 'failing') {
    const latest = latestFinished(runs)
    if (latest?.status === 'failed') return 'last run failed'
    if (latest?.summary) return historyLine(latest)
    return status === 'passing' ? 'passing' : 'failing'
  }
  return 'not run yet'
}

const DOTS = { passing: 'pass', failing: 'fail', unmeasured: 'unmeasured', running: 'running', 'needs-reference': 'empty' }
export const dotKind = (status) => DOTS[status] ?? 'idle'

export const pollInterval = (state) => (state?.running || state?.queued?.length ? 1000 : 5000)

export const queueLine = (ahead) => {
  if (!ahead) return 'Starting now'
  return `Waiting for ${ahead} ${ahead === 1 ? 'run' : 'runs'} to finish first`
}
