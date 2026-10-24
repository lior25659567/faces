import './run-history.css'
import StatusChip from './status-chip.jsx'
import { formatDuration, formatWhen } from '../lib/format.js'
import { RUN_STATUS_LABELS } from '../lib/labels.js'
import { historyBar, historyLine } from '../lib/status.js'

// A thin bar per run: pass / limited / fail / not measured (amber), so improvement shows at a glance.
const Bar = ({ summary }) => {
  const parts = historyBar(summary)
  if (!parts) return <span className="history-bar is-empty" aria-hidden="true" />
  return (
    <span className="history-bar" aria-hidden="true">
      {parts.map((p) => <span key={p.kind} className={`is-${p.kind}`} style={{ width: `${p.share * 100}%` }} />)}
    </span>
  )
}

const took = (run) => {
  if (!run.startedAt || !run.finishedAt) return null
  return (new Date(run.finishedAt) - new Date(run.startedAt)) / 1000
}

const RunHistory = ({ runs, shownId, onPick }) => (
  <div className="history">
    <div className="section-head">
      <h2 className="section-title" id="history-title">History</h2>
      <p className="section-note">Every run, newest first. Pick one to see its comparison and checks.</p>
    </div>
    <ol className="history-list">
      {runs.map((run) => {
        const seconds = took(run)
        return (
          <li key={run.id}>
            <button
              type="button"
              className={`history-item${run.id === shownId ? ' is-shown' : ''}`}
              aria-pressed={run.id === shownId}
              onClick={() => onPick(run.id)}
            >
              <span className="history-when num">{formatWhen(run.createdAt)}</span>
              <span className="history-score num">{historyLine(run)}</span>
              <Bar summary={run.summary} />
              <span className="history-took num">{seconds === null ? '' : `took ${formatDuration(seconds)}`}</span>
              <StatusChip kind={run.status}>{RUN_STATUS_LABELS[run.status] ?? run.status}</StatusChip>
            </button>
          </li>
        )
      })}
    </ol>
  </div>
)

export default RunHistory
