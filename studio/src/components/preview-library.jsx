import { useState } from 'react'
import { previewStatus } from '../lib/preview-status.js'
import './preview-library.css'

const PreviewCard = ({ example, playing, onPlay, onOpenExercise }) => {
  const status = previewStatus(example)
  const summary = example.summary
  return (
    <article className="preview-card" data-exercise={example.slug}>
      <div className={`preview-card-media${example.available ? '' : ' is-unavailable'}`}>
        {example.available ? (
          <>
            <img src={playing ? example.gifUrl : example.posterUrl}
              alt={`${example.name}: ${playing ? 'animated movement' : 'starting pose'}`}
              loading="lazy" width="360" height="360" />
            <button type="button" className="preview-play btn" aria-pressed={playing}
              aria-label={`${playing ? 'Pause' : 'Play'} ${example.name} preview`} onClick={onPlay}>
              <span aria-hidden="true">{playing ? 'Ⅱ' : '▶'}</span> {playing ? 'Pause' : 'Play preview'}
            </button>
          </>
        ) : <p>{example.reason || 'Build this exercise after adding suitable references.'}</p>}
      </div>
      <div className="preview-card-content">
        <p className="preview-example-label">{example.available ? 'Validated example' : 'Awaiting a build'}</p>
        <h2>{example.name}</h2>
        <span className={`preview-status is-${status.kind}`}>{status.label}</span>
        {example.available && summary && (
          <p className="preview-check-count num">
            {summary.pass} passing checks{summary.uncertain ? ` · ${summary.uncertain} uncertain` : ''}
          </p>
        )}
        {example.available && example.reason && <p className="preview-review-note">{example.reason}</p>}
        <div className="preview-card-actions">
          <button type="button" className="btn btn-primary" onClick={() => onOpenExercise(example.slug)}>
            Open exercise
          </button>
          {example.available && <a className="btn btn-quiet" href={example.gifUrl} target="_blank" rel="noreferrer">Full size</a>}
          {example.available && <a className="btn btn-quiet" href={example.checksUrl} target="_blank" rel="noreferrer">View checks</a>}
        </div>
      </div>
    </article>
  )
}

const PreviewLibrary = ({ previews, error, onRetry, onOpenExercise }) => {
  const [playing, setPlaying] = useState(null)
  const available = previews?.filter((example) => example.available).length ?? 0
  return (
    <section className="preview-library" aria-labelledby="preview-library-title">
      <div className="preview-library-heading">
        <h1 id="preview-library-title">Built exercise previews</h1>
        {previews && <p className="preview-library-count num">{available} built previews · {previews.length} exercises</p>}
        <p>Browse completed example builds. Open an exercise to edit references and build your own.</p>
        <p className="preview-library-note">Review labels retain uncertain photo depth and failed checks.</p>
      </div>
      {error ? (
        <div className="preview-library-error" role="alert"><p>{error}</p><button type="button" className="btn" onClick={onRetry}>Retry previews</button></div>
      ) : !previews ? <p className="preview-library-loading">Loading the previews…</p> : (
        <div className="preview-library-grid">
          {previews.map((example) => <PreviewCard key={example.slug} example={example} playing={playing === example.slug}
            onPlay={() => setPlaying((current) => current === example.slug ? null : example.slug)} onOpenExercise={onOpenExercise} />)}
        </div>
      )}
    </section>
  )
}

export default PreviewLibrary
