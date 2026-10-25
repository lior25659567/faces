import { useCallback, useEffect, useRef, useState } from 'react'
import './exercise-page.css'
import ChecksTable from './checks-table.jsx'
import Comparison from './comparison.jsx'
import ReferenceStrip from './reference-strip.jsx'
import ReferencePreview from './reference-preview.jsx'
import RunHistory from './run-history.jsx'
import RunProgress from './run-progress.jsx'
import UploadZone from './upload-zone.jsx'
import WorkItem from './work-item.jsx'
import { api } from '../lib/api.js'
import { formatWhen } from '../lib/format.js'
import { motionSource } from '../lib/labels.js'
import { defaultRunId, samePictures, stripReasons } from '../lib/run-view.js'
import { checksAside, checksLine, scoreKind, unmeasuredLine } from '../lib/status.js'
import { workForExercise } from '../lib/work.js'
import { referenceCheck } from '../lib/reference-check.js'
import CharacterFit from './character-fit.jsx'
import { characterFitResult } from '../lib/character-fit.js'

const ACTIVE = new Set(['queued', 'running', 'failed', 'interrupted'])

// The header score is the newest finished run's. When that isn't the newest run, or the pictures
// changed after it, say which run it is.
const scoreSource = (latest, latestDone, picturesChanged) => {
  if (!latestDone) return null
  const from = `From the run of ${formatWhen(latestDone.createdAt)}`
  if (latest && latest.id !== latestDone.id) {
    if (latest.status === 'failed') return `${from}. The newest run failed.`
    if (latest.status === 'interrupted') return `${from}. The newest run was interrupted.`
    return `${from}. A newer run is in progress.`
  }
  return picturesChanged ? `${from}, before the pictures changed.` : null
}

const ExercisePage = ({ exercise, state, work, exercises, focusRun, onChanged, onDetail, onOpenWork }) => {
  const { slug } = exercise
  const [detail, setDetail] = useState(null)
  const [loadError, setLoadError] = useState(null)
  const [actionError, setActionError] = useState(null)
  const [pickedRunId, setPickedRunId] = useState(null)
  const [bundle, setBundle] = useState(null)
  const [doneSnapshot, setDoneSnapshot] = useState(null)
  const [referenceBundle, setReferenceBundle] = useState(null)
  const [busy, setBusy] = useState(false)
  const [pickedCharacterFit, setCharacterFit] = useState(null)
  const inFlight = useRef(false)
  const refsSection = useRef(null)
  const progressSection = useRef(null)
  const resultSection = useRef(null)
  const focusedRun = useRef(null)

  const load = useCallback(async () => {
    try {
      const d = await api.exercise(slug)
      setDetail(d)
      setLoadError(null)
      onDetail(slug, d)
    } catch (e) {
      setLoadError(e.message)
    }
  }, [slug, onDetail])

  // Refetch when this exercise's run starts, changes step, waits, or finishes.
  const live = state?.running?.slug === slug ? state.running : null
  const queued = state?.queued ?? []
  const queuedAt = queued.findIndex((q) => q.slug === slug)
  const runKey = `${live?.runId ?? ''}:${live?.step ?? ''}:${live?.note ?? ''}|${queued[queuedAt]?.runId ?? ''}|${exercise.status}`
  useEffect(() => {
    load()
  }, [load, runKey])

  // Reference checks have no animation or score; keep them out of result selection and build history.
  const allRuns = detail?.runs ?? []
  const runs = allRuns.filter((r) => !r.readOnly)
  const referenceRun = allRuns[0] ?? null
  const referenceRunId = referenceRun?.id
  const referenceStatus = referenceRun?.status
  const latest = runs[0] ?? null
  const characterFit = pickedCharacterFit ?? latest?.characterFit ?? 'auto'
  const latestDone = runs.find((r) => r.status === 'done') ?? null
  const shownId = pickedRunId && runs.some((r) => r.id === pickedRunId) ? pickedRunId : defaultRunId(runs)
  const shown = runs.find((r) => r.id === shownId) ?? null
  const shownStatus = shown?.status

  useEffect(() => {
    if (!referenceRunId) {
      setReferenceBundle(null)
      return undefined
    }
    let alive = true
    api.runBundle(slug, referenceRunId).then((b) => {
      if (alive) setReferenceBundle({ runId: referenceRunId, ...b })
    })
    return () => { alive = false }
  }, [slug, referenceRunId, referenceStatus, live?.step])

  useEffect(() => {
    if (!shownId) {
      setBundle(null)
      return undefined
    }
    let alive = true
    api.runBundle(slug, shownId).then((b) => {
      if (alive) setBundle({ runId: shownId, ...b })
    })
    return () => {
      alive = false
    }
  }, [slug, shownId, shownStatus])

  // The pictures the newest finished run used, to tell whether they changed since.
  const latestDoneId = latestDone?.id
  useEffect(() => {
    if (!latestDoneId) return undefined
    let alive = true
    api.runJson(slug, latestDoneId, 'references.json').then((snap) => {
      if (alive) setDoneSnapshot({ runId: latestDoneId, images: snap?.images ?? null })
    })
    return () => {
      alive = false
    }
  }, [slug, latestDoneId])

  const refresh = useCallback(async () => {
    await load()
    onChanged()
  }, [load, onChanged])

  // One request that starts a run at a time, until the page has the new run: a double-click must
  // not queue a second Blender job once the first has already been picked up.
  const guarded = async (work) => {
    if (inFlight.current) return
    inFlight.current = true
    setBusy(true)
    setActionError(null)
    try {
      await work()
    } catch (e) {
      setActionError(e.message)
    }
    try {
      await refresh()
    } finally {
      inFlight.current = false
      setBusy(false)
    }
  }

  // Save edits together before checking or committing to a complete animation build.
  const changeReferences = (action) => guarded(action)

  const runAgain = () => guarded(() => api.startRun(slug, { characterFit }))
  const checkReferences = () => guarded(() => api.checkReferences(slug))

  const scrollTo = (ref) => {
    const still = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches
    ref.current?.scrollIntoView({ block: 'start', behavior: still ? 'auto' : 'smooth' })
  }
  const scrollToReferences = () => scrollTo(refsSection)

  // A "Re-ran automatically" link: show that run, its progress while it goes and its result after.
  useEffect(() => {
    if (!focusRun || focusedRun.current === focusRun.n || !detail) return
    const run = detail.runs.find((r) => !r.readOnly && r.id === focusRun.runId)
    if (!run) return // not on the page yet: the next load brings it
    focusedRun.current = focusRun.n
    // Still going: its progress is the panel at the top. Otherwise show its result (or why there is none).
    const going = run.status === 'queued' || run.status === 'running'
    const newestStopped = detail.runs[0]?.id === run.id && ACTIVE.has(run.status)
    setPickedRunId(going ? null : run.id)
    // Wait a frame for the picked run's sections to render.
    requestAnimationFrame(() => scrollTo(going || newestStopped ? progressSection : resultSection))
  }, [focusRun, detail])

  const references = detail?.references ?? []
  const currentReferenceBundle = referenceBundle?.runId === referenceRun?.id && referenceBundle?.snapshot?.images &&
    samePictures(references, referenceBundle.snapshot.images) ? referenceBundle : null
  const check = referenceCheck(references, referenceRun, referenceBundle)
  const checking = referenceRun?.readOnly && ['queued', 'running'].includes(referenceRun.status)
  const working = Boolean(live || queuedAt >= 0)
  // A stopped build may have old pictures or unresolved reference problems. Retry the cheap
  // reference check first unless the current pictures have already passed it.
  const retryRun = check.canBuild ? runAgain : checkReferences
  const retryLabel = check.canBuild ? 'Build again' : 'Check references'
  const summary = latestDone?.summary ?? null
  const line = checksLine(summary)
  const aside = checksAside(summary)
  const unmeasured = unmeasuredLine(summary)
  const kind = scoreKind(summary)
  const ahead = queuedAt < 0 ? null : queuedAt + (state?.running ? 1 : 0)
  const showProgress = latest && ACTIVE.has(latest.status)
  // This exercise's run is going while a newer one waits behind it: show the one going too.
  const liveRun = live && live.runId !== latest?.id ? runs.find((r) => r.id === live.runId) ?? null : null
  const viewingOlder = shown && pickedRunId && shownId !== defaultRunId(runs)
  const bundleReady = bundle && bundle.runId === shownId
  const rows = bundleReady ? bundle.checks?.rows ?? [] : []
  // A run that never got as far as stills or checks has nothing to compare; its progress panel says why.
  const hasResult = shown && (shown.status === 'done' || (bundleReady && Boolean(bundle.ours || bundle.checks)))
  // What the failure panel can say about the comparison below it.
  const below = !hasResult ? null : shown.id === latest?.id ? 'partial' : `the run of ${formatWhen(shown.createdAt)}`
  const belowKnown = !shown || shown.status === 'done' || bundleReady
  const picturesChanged = Boolean(
    detail && doneSnapshot && doneSnapshot.runId === latestDone?.id && !samePictures(references, doneSnapshot.images),
  )
  const source = scoreSource(latest, latestDone, picturesChanged)
  const behind = workForExercise(work, slug)
  // "Not used" beside each picture: the reasons of the run the page is showing (a picked older run, else
  // the newest run, also when its read step stopped), never mixed with another run's.
  const reasons = viewingOlder
    ? (bundleReady ? stripReasons({ run: shown, targets: bundle.targets, readProblems: bundle.readProblems }) : null)
    : currentReferenceBundle
      ? stripReasons({ run: referenceRun, targets: currentReferenceBundle.targets, readProblems: currentReferenceBundle.readProblems })
      : null

  return (
    <article className="exercise-page">
      <header className="exercise-head">
        <div className="exercise-head-main">
          <h1 className="exercise-title">{exercise.name}</h1>
          <p className="exercise-meta">
            <span className="exercise-equipment">{exercise.equipment}</span>
            <span className={exercise.clip ? undefined : 'exercise-noclip'}>{motionSource(exercise)}</span>
            {references.length > 0 && (
              <button type="button" className="exercise-refs-link" onClick={scrollToReferences}>
                {references.length} reference {references.length === 1 ? 'picture' : 'pictures'}
              </button>
            )}
          </p>
        </div>
        {line && (
          <div className="exercise-score">
            <p className={`exercise-score-line num is-${kind}`}>{line}</p>
            {source && <p className="exercise-score-source">{source}</p>}
            {unmeasured && <p className="exercise-score-unmeasured num">{unmeasured}</p>}
            {aside && <p className="exercise-score-aside">{aside}</p>}
          </div>
        )}
      </header>

      {loadError && <p className="exercise-error" role="alert">{loadError}</p>}
      {actionError && <p className="exercise-error" role="alert">{actionError}</p>}

      {detail && (
        <section className="section exercise-section exercise-reference-check" aria-labelledby="reference-check-title">
          <div className="section-head">
            <div>
              <h2 className="section-title" id="reference-check-title">Prepare the movement</h2>
              <p className="section-note">Save your picture edits, check that Studio can use them, then build the animation.</p>
            </div>
            <div className="exercise-build-actions">
              <button type="button" className="btn btn-quiet" onClick={scrollToReferences}>Edit pictures</button>
              <button type="button" className="btn" onClick={checkReferences} disabled={busy || checking || !references.length}>
                {checking ? 'Checking references…' : 'Check references'}
              </button>
              <button type="button" className="btn btn-primary" onClick={runAgain} disabled={busy || working || !check.canBuild}>
                Build exercise
              </button>
            </div>
          </div>
          <CharacterFit value={characterFit} onChange={setCharacterFit} disabled={busy || working} />
          <div className={`exercise-check-feedback is-${check.state}`} role="status">
            {checking ? (
              <p>{referenceRun.status === 'queued' ? 'The reference check is waiting in the queue.' : 'Reading your pictures and checking what they can show…'}</p>
            ) : check.state === 'stale' ? (
              <p>Your pictures or tags changed. Check the references again before building.</p>
            ) : check.state === 'ready' ? (
              <p>The references are ready to build. The finished animation still needs its movement checks.</p>
            ) : check.state === 'blocked' ? (
              <p>Fix the reference problems below, then check again.</p>
            ) : check.state === 'failed' ? (
              <p>{check.message}</p>
            ) : (
              <p>{exercise.name === 'Barbell Bent-Over Row'
                ? 'Start with matching side or three-quarter views of the beginning and end. Label their positions and actual views, then check the references.'
                : 'Start with side views of the beginning and end of the movement. Label each position and view, then check the references.'}</p>
            )}
            {currentReferenceBundle?.preflight?.mode && <p className="exercise-check-mode">Movement source: {currentReferenceBundle.preflight.mode}.</p>}
            {check.blockers?.length > 0 && <ul>{check.blockers.map((message, i) => <li key={i}>{message}</li>)}</ul>}
            {check.warnings?.length > 0 && <ul className="exercise-check-warnings">{check.warnings.map((message, i) => <li key={i}>{message}</li>)}</ul>}
          </div>
        </section>
      )}

      {behind.shown.length > 0 && (
        <section className="section exercise-section" aria-labelledby="work-title">
          <div className="section-head">
            <h2 className="section-title" id="work-title">Behind the scenes</h2>
            <p className="section-note">Work on the pipeline that touches this exercise. When it finishes, this exercise runs again on its own.</p>
          </div>
          <ol className="exercise-work">
            {behind.shown.map((item) => (
              <li key={item.id} className="exercise-work-item">
                <WorkItem item={item} exercises={exercises} state={state} runsBySlug={{ [slug]: runs }} />
              </li>
            ))}
          </ol>
          {behind.hidden > 0 && (
            <button type="button" className="btn btn-quiet exercise-work-more" onClick={onOpenWork}>
              {behind.hidden} older in Behind the scenes
            </button>
          )}
        </section>
      )}

      {liveRun && (
        <section className="section exercise-section" aria-labelledby="progress-live-title">
          <RunProgress run={liveRun} live={live} ahead={null} titleId="progress-live-title"
            onRunAgain={retryRun} retryLabel={retryLabel} starting={busy || checking} />
        </section>
      )}

      {showProgress && (
        <section ref={progressSection} className="section exercise-section exercise-anchor" aria-labelledby="progress-title">
          <RunProgress
            run={latest}
            live={live?.runId === latest.id ? live : null}
            ahead={ahead}
            below={below}
            belowKnown={belowKnown}
            starting={busy || checking}
            onRunAgain={retryRun}
            retryLabel={retryLabel}
          />
        </section>
      )}

      <div ref={resultSection} className="exercise-anchor" />

      {viewingOlder && bundleReady && !hasResult && (
        <p className="exercise-older">
          The run from {formatWhen(shown.createdAt)} has nothing to compare
          {shown.error?.message ? `: ${shown.error.message}` : '.'}
          <button type="button" className="btn btn-quiet" onClick={() => setPickedRunId(null)}>Back to the latest result</button>
        </p>
      )}

      {hasResult && (
        <>
          {viewingOlder && (
            <p className="exercise-older">
              Showing the run from {formatWhen(shown.createdAt)}.
              <button type="button" className="btn btn-quiet" onClick={() => setPickedRunId(null)}>Back to the latest result</button>
            </p>
          )}
          <section className="section exercise-section" aria-labelledby="comparison-title">
            <p className="section-note">{characterFitResult(bundleReady ? bundle.characterFit : null, shown.characterFit)}</p>
            <Comparison
              key={shown.id}
              slug={slug}
              run={shown}
              bundle={bundleReady ? bundle : null}
              fallbackImages={references}
              rows={rows}
              newerRun={!pickedRunId && latest && latest.id !== shown.id}
            />
          </section>
          <section className="section exercise-section" aria-labelledby="checks-title">
            <ChecksTable run={shown} rows={rows} loading={!bundleReady} />
          </section>
        </>
      )}

      {!detail && !loadError && <p className="exercise-loading">Loading…</p>}

      {detail && (
        <section ref={refsSection} className="section exercise-section exercise-refs" aria-labelledby="references-title">
          <div className="section-head">
            <h2 className="section-title" id="references-title">Reference pictures</h2>
            {references.length > 0 && (
              <p className="section-note">In movement order, start to end. Changes are saved without rebuilding. Check the references when you finish editing.</p>
            )}
          </div>
          {references.length > 0 && (
            <ReferenceStrip
              slug={slug}
              references={references}
              reasons={reasons}
              busy={busy}
              onChange={changeReferences}
            />
          )}
          {currentReferenceBundle?.measurements && (
            <ReferencePreview slug={slug} images={currentReferenceBundle.snapshot.images}
              measurements={currentReferenceBundle.measurements} targets={currentReferenceBundle.targets} />
          )}
          <UploadZone slug={slug} compact={references.length > 0} onDone={refresh} />
        </section>
      )}

      {runs.length > 0 && (
        <section className="section exercise-section" aria-labelledby="history-title">
          <RunHistory runs={runs} shownId={shownId} onPick={setPickedRunId} />
        </section>
      )}
    </article>
  )
}

export default ExercisePage
