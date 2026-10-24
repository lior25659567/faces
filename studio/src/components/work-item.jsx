import { useState } from 'react'
import './work-item.css'
import { hashFor } from '../lib/hash.js'
import { WORK_STATUS_LABELS, awaitingRerun, logTail, logTime, rerunStatus, staleText, startedText, tookText } from '../lib/work.js'

const SHOWN_LINES = 3

const nameOf = (exercises, slug) => exercises?.find((e) => e.slug === slug)?.name ?? slug

// What happened after the work finished: the runs the Studio started on its own, or why none.
const Rerun = ({ item, exercises, state, runsBySlug }) => {
  if (item.status !== 'done') return null
  const rows = item.rerun ?? []
  return (
    <div className="work-rerun">
      {rows.length > 0 && (
        <ul className="work-rerun-list">
          {rows.map((r) => {
            const status = rerunStatus(r, state, runsBySlug?.[r.slug])
            return (
              <li key={`${r.slug}-${r.runId}`} className="work-rerun-row">
                <span className="work-rerun-lead">{r.byHand ? 'Already re-run by hand →' : 'Re-ran automatically →'}</span>{' '}
                <a className="work-rerun-link" href={hashFor(r.slug, r.runId)}>{nameOf(exercises, r.slug)}</a>
                {status && <span className="work-rerun-status">{status}</span>}
              </li>
            )
          })}
        </ul>
      )}
      {item.rerunNote && <p className="work-rerun-note">{item.rerunNote}</p>}
      {awaitingRerun(item) && <p className="work-rerun-note is-pending">Starting the re-runs…</p>}
    </div>
  )
}

// One piece of work behind the scenes: what it is, how far it got, and what re-ran after it.
const WorkItem = ({ item, exercises, state, runsBySlug, showExercises = false, headingLevel = 3 }) => {
  const [allLines, setAllLines] = useState(false)
  const now = new Date()
  const log = item.log ?? []
  const lines = allLines ? log : logTail(log, SHOWN_LINES)
  const running = item.status === 'running'
  const took = tookText(item)
  const stale = staleText(item, now)
  const Heading = `h${headingLevel}`
  const listed = item.exercises ?? []

  return (
    <article className={`work-item is-${item.status}`}>
      <div className="work-head">
        <Heading className="work-title">{item.title}</Heading>
        <span className={`work-status is-${item.status}`}>
          {running && <span className="work-status-dot" aria-hidden="true" />}
          {WORK_STATUS_LABELS[item.status] ?? item.status}
        </span>
      </div>

      <p className="work-meta num">
        <span>{startedText(item.startedAt, now)}</span>
        {took && <span>{took}</span>}
      </p>
      {stale && <p className="work-stale">{stale}</p>}

      {showExercises && listed.length > 0 && (
        <p className="work-for">
          {running ? 'Re-runs when finished: ' : 'For '}
          {listed.map((slug, i) => (
            <span key={slug}>
              {i > 0 && ', '}
              <a href={hashFor(slug)}>{nameOf(exercises, slug)}</a>
            </span>
          ))}
        </p>
      )}

      {log.length > 0 ? (
        <ol className={`work-log${running ? ' is-live' : ''}`} aria-label="Progress">
          {lines.map((line, i) => (
            <li key={`${line.at}-${i}`} className="work-log-line">
              <time className="work-log-time num" dateTime={line.at}>{logTime(line.at, now)}</time>
              <span className="work-log-text">{line.text}</span>
            </li>
          ))}
        </ol>
      ) : (
        running && <p className="work-empty">No progress written yet.</p>
      )}
      {log.length > SHOWN_LINES && (
        <button type="button" className="work-more" aria-expanded={allLines} onClick={() => setAllLines((v) => !v)}>
          {allLines ? `Show only the newest ${SHOWN_LINES}` : `Show all ${log.length} lines`}
        </button>
      )}

      {item.summary && !running && <p className="work-summary">{item.summary}</p>}

      <Rerun item={item} exercises={exercises} state={state} runsBySlug={runsBySlug} />
    </article>
  )
}

export default WorkItem
