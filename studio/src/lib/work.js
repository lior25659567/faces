import { formatDuration, formatLeft, formatWhen } from './format.js'
import { historyLine, pollInterval } from './status.js'

// Work going on behind the scenes (GET /api/work). Neutral words: "Finished" says the work ended,
// not that anything passes; the re-run's own checks say that.
export const WORK_STATUS_LABELS = { running: 'In progress', done: 'Finished', failed: 'Failed' }

const isRunning = (w) => w?.status === 'running'

// Done, but the Studio hasn't recorded its re-run yet. It looks every few seconds (and on start).
export const awaitingRerun = (w) => w?.status === 'done' && !w.rerunQueuedAt

// The page polls every second while anything moves: a run, a fix, or a fix about to re-run.
export const livePoll = (state, items) =>
  pollInterval(state) === 1000 || (items ?? []).some((w) => isRunning(w) || awaitingRerun(w)) ? 1000 : 5000

// "1 fix in progress" for the top bar ("1 fix" where a phone has no room), or null.
export const inProgressText = (items, { short = false } = {}) => {
  const n = (items ?? []).filter(isRunning).length
  if (!n) return null
  return `${n} ${n === 1 ? 'fix' : 'fixes'}${short ? '' : ' in progress'}`
}

// On an exercise's page: every running item that lists it, then its newest finished ones.
export const workForExercise = (items, slug, finished = 2) => {
  const mine = (items ?? []).filter((w) => w.exercises?.includes(slug))
  const shown = [...mine.filter(isRunning), ...mine.filter((w) => !isRunning(w)).slice(0, finished)]
  return { shown, hidden: mine.length - shown.length }
}

const parse = (iso) => {
  const d = iso ? new Date(iso) : null
  return d && !Number.isNaN(d.getTime()) ? d : null
}

// "Started 4 min ago"; past half a day the date and time instead.
export const startedText = (iso, now = new Date()) => {
  const d = parse(iso)
  if (!d) return ''
  const minutes = Math.floor((now - d) / 60000)
  if (minutes < 1) return 'Started less than a minute ago'
  if (minutes < 60) return `Started ${minutes} min ago`
  if (minutes < 12 * 60) {
    const h = Math.floor(minutes / 60)
    const m = minutes % 60
    return `Started ${h} h${m ? ` ${m} min` : ''} ago`
  }
  return `Started ${formatWhen(iso, now).replace(/^(Today|Yesterday)/, (w) => w.toLowerCase())}`
}

// A fix still "in progress" with nothing new for 2 hours may have stopped without saying so: the page
// says that, while still counting it as in progress (only its writer can mark it done or failed).
const STALE_HOURS = 2
export const staleText = (w, now = new Date()) => {
  if (!isRunning(w)) return null
  const times = [w.updatedAt, w.startedAt, ...(w.log ?? []).map((l) => l.at)].map(parse).filter(Boolean)
  if (!times.length) return null
  const hours = Math.floor((now - Math.max(...times)) / 3_600_000)
  return hours >= STALE_HOURS ? `No update for ${hours} h — it may have stopped` : null
}

export const tookText = (w) => {
  const a = parse(w?.startedAt)
  const b = parse(w?.finishedAt)
  return a && b ? `took ${formatDuration((b - a) / 1000)}` : null
}

// The newest `n` lines, oldest of them first (the order they happened).
export const logTail = (log, n = 3) => (log ?? []).slice(-n)

const pad = (n) => String(n).padStart(2, '0')

// "09:05" today, "Yesterday 18:30" before.
export const logTime = (iso, now = new Date()) => {
  const d = parse(iso)
  if (!d) return ''
  const today = d.toDateString() === now.toDateString()
  return today ? `${pad(d.getHours())}:${pad(d.getMinutes())}` : formatWhen(iso, now)
}

// What a re-run is doing now: from the live queue first, then the run as saved on disk.
export const rerunStatus = ({ runId }, state, runs) => {
  if (state?.running?.runId === runId) return `running, ${formatLeft({ seconds: state.running.left, overdue: state.running.overdue })}`
  if (state?.queued?.some((q) => q.runId === runId)) return 'waiting to start'
  const run = (runs ?? []).find((r) => r.id === runId)
  if (!run) return ''
  if (run.status === 'done') return run.summary ? `finished: ${historyLine(run).replace(/ pass$/, ' checks pass')}` : 'finished'
  if (run.status === 'failed') return 'stopped with an error'
  if (run.status === 'interrupted') return 'interrupted'
  if (run.status === 'running') return 'running'
  return 'waiting to start'
}
