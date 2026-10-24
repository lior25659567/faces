import { useCallback, useEffect, useRef, useState } from 'react'
import './app.css'
import ExerciseList from './components/exercise-list.jsx'
import ExercisePage from './components/exercise-page.jsx'
import WorkPanel from './components/work-panel.jsx'
import { api } from './lib/api.js'
import { formatLeft } from './lib/format.js'
import { hashFor, routeFromHash } from './lib/hash.js'
import { usePoll } from './lib/use-poll.js'
import { inProgressText, livePoll } from './lib/work.js'

// A run link (#/<slug>/run/<id>) asks the page to show that run once.
let focusCount = 0
const focusFrom = (route) => (route?.runId ? { slug: route.slug, runId: route.runId, n: ++focusCount } : null)

// What changes when a run moves on: which run, which step, what is waiting.
const stateKey = (state) =>
  state ? `${state.running?.runId ?? ''}:${state.running?.step ?? ''}|${state.queued.map((q) => q.runId).join(',')}` : ''

const App = () => {
  const [state, setState] = useState(null)
  const [exercises, setExercises] = useState(null)
  const [runsBySlug, setRunsBySlug] = useState({})
  const [selected, setSelected] = useState(() => routeFromHash(window.location.hash)?.slug ?? null)
  const [focusRun, setFocusRun] = useState(() => focusFrom(routeFromHash(window.location.hash)))
  const [offline, setOffline] = useState(null)
  const [listOpen, setListOpen] = useState(false)
  const [work, setWork] = useState(null)
  const [workOpen, setWorkOpen] = useState(false)
  const workToggle = useRef(null)
  const fetchedStatus = useRef({})

  const loadState = useCallback(async () => {
    try {
      setState(await api.state())
      setOffline(null)
    } catch (e) {
      setOffline(e.message)
    }
  }, [])

  const loadExercises = useCallback(async () => {
    try {
      setExercises(await api.exercises())
    } catch (e) {
      setOffline(e.message)
    }
  }, [])

  const loadWork = useCallback(async () => {
    try {
      setWork(await api.work())
    } catch (e) {
      setOffline(e.message)
    }
  }, [])

  // Every second while a run, a fix, or a fix's re-run is moving; else every 5 s.
  const every = livePoll(state, work)
  usePoll(loadState, every)
  usePoll(loadWork, every)
  usePoll(loadExercises, 5000)

  // A run started, moved to the next step or finished: refresh the list's statuses at once.
  const key = stateKey(state)
  useEffect(() => {
    if (key) loadExercises()
  }, [key, loadExercises])

  // A fix finished and the Studio queued its re-runs: show them running at once.
  const rerunKey = (work ?? []).map((w) => w.rerunQueuedAt ?? '').join('|')
  useEffect(() => {
    if (rerunKey.replaceAll('|', '')) {
      loadState()
      loadExercises()
    }
  }, [rerunKey, loadState, loadExercises])

  // Each exercise's runs (for "7 of 18 pass" in the list), fetched again only when its status changes.
  useEffect(() => {
    if (!exercises) return
    const stale = exercises.filter((e) => fetchedStatus.current[e.slug] !== e.status)
    for (const e of stale) {
      fetchedStatus.current[e.slug] = e.status
      api.exercise(e.slug)
        .then((d) => setRunsBySlug((prev) => ({ ...prev, [e.slug]: d.runs })))
        .catch(() => { fetchedStatus.current[e.slug] = null })
    }
  }, [exercises])

  // A picture dropped beside the drop zone must not make the browser leave the page to show it.
  useEffect(() => {
    const guard = (e) => {
      if (Array.from(e.dataTransfer?.types ?? []).includes('Files')) e.preventDefault()
    }
    window.addEventListener('dragover', guard)
    window.addEventListener('drop', guard)
    return () => {
      window.removeEventListener('dragover', guard)
      window.removeEventListener('drop', guard)
    }
  }, [])

  useEffect(() => {
    // Only #/<slug> (or #/<slug>/run/<id>) changes the exercise; other hashes leave the selection alone.
    const onHash = () => {
      const route = routeFromHash(window.location.hash)
      if (!route) return
      setSelected(route.slug)
      setListOpen(false)
      setWorkOpen(false)
      // A run link focuses that run once; any other route clears it, so coming back doesn't jump again.
      setFocusRun(focusFrom(route))
      // Back to the plain exercise route, so the same link works again.
      if (route.runId) window.history.replaceState(null, '', hashFor(route.slug))
    }
    // Opened straight from a run link: the state above already holds it; tidy the address the same way.
    const first = routeFromHash(window.location.hash)
    if (first?.runId) window.history.replaceState(null, '', hashFor(first.slug))
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [])

  const select = (slug) => {
    window.location.hash = hashFor(slug)
    setSelected(slug)
    setListOpen(false)
  }

  const onDetail = useCallback((slug, detail) => setRunsBySlug((prev) => ({ ...prev, [slug]: detail.runs })), [])
  const refresh = useCallback(() => { loadState(); loadExercises() }, [loadState, loadExercises])

  const current =
    exercises?.find((e) => e.slug === selected) ||
    exercises?.find((e) => e.status === 'running') ||
    exercises?.[0] ||
    null

  const running = state?.running
  const runningName = running && exercises?.find((e) => e.slug === running.slug)?.name
  const fixes = inProgressText(work)
  const closeWork = useCallback(() => setWorkOpen(false), [])

  return (
    <div className="app">
      <header className="app-bar">
        <div className="app-brand">
          <span className="app-mark" aria-hidden="true" />
          <span className="app-name">Exercise Studio</span>
        </div>
        <button type="button" className="app-list-toggle" aria-expanded={listOpen} onClick={() => setListOpen((o) => !o)}>
          {listOpen ? 'Hide exercises' : 'All exercises'}
        </button>
        <p className={`app-queue${running ? ' is-running' : ''}`} aria-live="polite">
          {running ? (
            <>
              <span className="app-queue-dot" aria-hidden="true" />
              <span className="app-queue-text">
                {runningName || running.slug}:{' '}
                {running.note ? running.note.toLowerCase() : formatLeft({ seconds: running.left, overdue: running.overdue })}
                {state.queued.length ? `, ${state.queued.length} waiting` : ''}
              </span>
            </>
          ) : (
            <span className="app-queue-text">Nothing running</span>
          )}
        </p>
        {(fixes || work?.length > 0) && (
          <button
            type="button"
            ref={workToggle}
            className={`app-work${fixes ? ' is-running' : ''}`}
            aria-expanded={workOpen}
            aria-controls="work-panel"
            onClick={() => setWorkOpen((o) => !o)}
          >
            {fixes && <span className="app-work-ring" aria-hidden="true" />}
            <span className="app-work-long">{fixes ?? 'Behind the scenes'}</span>
            {fixes && <span className="app-work-short" aria-hidden="true">{inProgressText(work, { short: true })}</span>}
          </button>
        )}
      </header>

      <WorkPanel
        id="work-panel"
        open={workOpen}
        items={work}
        exercises={exercises}
        state={state}
        runsBySlug={runsBySlug}
        onClose={closeWork}
        toggleRef={workToggle}
      />

      {offline && <p className="app-offline" role="alert">{offline}</p>}

      <div className="app-body">
        <aside className={`app-side${listOpen ? ' is-open' : ''}`}>
          <ExerciseList
            exercises={exercises}
            state={state}
            runsBySlug={runsBySlug}
            selected={current?.slug}
            onSelect={select}
            onAdded={(e) => { loadExercises(); select(e.slug) }}
          />
        </aside>
        <main className="app-main">
          {current ? (
            <ExercisePage
              key={current.slug}
              exercise={current}
              state={state}
              work={work}
              exercises={exercises}
              focusRun={focusRun?.slug === current.slug ? focusRun : null}
              onChanged={refresh}
              onDetail={onDetail}
              onOpenWork={() => setWorkOpen(true)}
            />
          ) : (
            <p className="app-loading">{exercises ? 'No exercises yet. Add one on the left.' : 'Loading the exercises…'}</p>
          )}
        </main>
      </div>
    </div>
  )
}

export default App
