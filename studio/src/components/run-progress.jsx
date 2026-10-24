import './run-progress.css'
import { formatDuration, formatLeft } from '../lib/format.js'
import { STEP_LABELS } from '../lib/labels.js'
import { queueLine } from '../lib/status.js'

const StepIcon = ({ status }) => {
  if (status === 'done') return <span className="step-icon is-done" aria-label="Done">✓</span>
  if (status === 'failed') return <span className="step-icon is-failed" aria-label="Failed">✗</span>
  if (status === 'running') return <span className="step-icon is-running" aria-label="Running"><span className="step-spinner" /></span>
  return <span className="step-icon is-waiting" aria-label="Waiting" />
}

const stepLabel = (id) => STEP_LABELS[id] ?? id

// Seconds spent so far: finished steps plus the live one.
const spentSoFar = (run, live) =>
  run.steps.reduce((sum, s) => sum + (s.status !== 'running' && typeof s.seconds === 'number' ? s.seconds : 0), 0) +
  (live?.elapsed ?? 0)

const StepTime = ({ step, live }) => {
  if (step.status !== 'running') return <span className="step-time num">{formatDuration(step.seconds)}</span>
  if (!live || live.step !== step.id) return <span className="step-time num">running</span>
  if (live.note) return <span className="step-time is-note">{live.note}</span>
  const pct = typeof live.fraction === 'number' ? ` (${Math.round(live.fraction * 100)}%)` : ''
  return (
    <span className="step-time num">
      {formatDuration(live.elapsed)}{pct}
      <span className={`step-left${live.overdue ? ' is-overdue' : ''}`}>
        {formatLeft({ seconds: live.stepLeft, overdue: live.overdue })}
      </span>
    </span>
  )
}

const failedNote = (below, belowKnown) => {
  if (below === 'partial') return 'This run stopped early. Below is what it made before it stopped.'
  if (below) return `This run failed. The comparison below is from ${below}.`
  return belowKnown ? 'This run failed, so there is nothing to compare yet.' : 'This run failed.'
}

const Head = ({ run, live, ahead, below, belowKnown, titleId }) => {
  if (run.status === 'queued') {
    return (
      <>
        <h2 className="section-title" id={titleId}>Waiting to start</h2>
        <p className="progress-sub">{ahead === null ? 'In the queue' : queueLine(ahead)}</p>
        {run.note && <p className="progress-note">{run.note}</p>}
      </>
    )
  }
  if (run.status === 'running') {
    const spent = spentSoFar(run, live)
    const left = live ? formatLeft({ seconds: live.left, overdue: live.overdue }) : 'working out the time left'
    return (
      <>
        <h2 className="section-title" id={titleId}>Running</h2>
        <p className="progress-sub num">
          <strong className={live?.overdue ? 'is-overdue' : ''}>{left}</strong>
          <span>{formatDuration(spent)} so far</span>
        </p>
        {run.note && <p className="progress-note">{run.note}</p>}
      </>
    )
  }
  if (run.status === 'interrupted') {
    return (
      <>
        <h2 className="section-title is-failed" id={titleId}>Interrupted</h2>
        <p className="progress-sub">{run.error?.message ?? 'The Studio was stopped while this ran.'}</p>
        {run.restartOf && <p className="progress-note">It was already an automatic restart, so it doesn't start again on its own.</p>}
      </>
    )
  }
  return (
    <>
      <h2 className="section-title is-failed" id={titleId}>
        {STEP_LABELS[run.error?.step] ? `Stopped at “${stepLabel(run.error.step)}”` : 'The run stopped'}
      </h2>
      <p className="progress-sub">{failedNote(below, belowKnown)}</p>
    </>
  )
}

const RunProgress = ({ run, live, ahead, below, belowKnown, starting, onRunAgain, titleId = 'progress-title' }) => {
  const stopped = run.status === 'failed' || run.status === 'interrupted'
  const spent = spentSoFar(run, live)
  const share = live && spent + live.left > 0 ? Math.min(spent / (spent + live.left), 1) : null

  return (
    <div className={`progress is-${run.status}`}>
      <div className="section-head progress-head">
        <div>
          <Head run={run} live={live} ahead={ahead} below={below} belowKnown={belowKnown} titleId={titleId} />
        </div>
        {stopped && (
          <button type="button" className="btn btn-primary" onClick={onRunAgain} disabled={starting} aria-busy={starting}>
            {starting ? 'Starting…' : 'Run again'}
          </button>
        )}
      </div>

      {run.status === 'running' && (
        <div className="progress-bar" role="progressbar" aria-label="Whole run" aria-valuemin={0} aria-valuemax={100}
          aria-valuenow={share === null ? undefined : Math.round(share * 100)}>
          <span className={`progress-bar-fill${share === null ? ' is-unknown' : ''}`} style={share === null ? undefined : { width: `${share * 100}%` }} />
        </div>
      )}

      <ol className="progress-steps">
        {run.steps.map((step) => (
          <li key={step.id} className={`progress-step is-${step.status}`}>
            <StepIcon status={step.status} />
            <span className="step-label">{stepLabel(step.id)}</span>
            <StepTime step={step} live={live} />
          </li>
        ))}
      </ol>

      {run.status === 'failed' && run.error && (
        <div className="progress-error">
          <p className="progress-error-message">{run.error.message}</p>
          {run.error.logTail?.length > 0 && (
            <details className="progress-log" open>
              <summary>Last lines of the log</summary>
              <pre>{run.error.logTail.join('\n')}</pre>
            </details>
          )}
        </div>
      )}
    </div>
  )
}

export default RunProgress
