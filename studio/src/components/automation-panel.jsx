import { useCallback, useEffect, useRef, useState } from 'react'
import './automation-panel.css'
import { api } from '../lib/api.js'
import { automationCheckLine, automationItemState, automationJobLine, automationProgress, automationSourceUrl } from '../lib/automation.js'
import { formatWhen } from '../lib/format.js'
import { usePoll } from '../lib/use-poll.js'

const AutomationPanel = ({ open, exercises = [], onClose, toggleRef, onOpenExercise, onChanged, onActiveChange }) => {
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)
  const [busy, setBusy] = useState(false)
  const [selection, setSelection] = useState(null)
  const [viewedId, setViewedId] = useState(null)
  const [checkOnly, setCheckOnly] = useState(false)
  const panel = useRef(null)
  const heading = useRef(null)
  const requestVersion = useRef(0)
  const actionInFlight = useRef(false)
  const previousProgress = useRef(null)
  const mounted = useRef(true)

  useEffect(() => {
    mounted.current = true
    return () => { mounted.current = false; requestVersion.current += 1 }
  }, [])

  const load = useCallback(async () => {
    const version = ++requestVersion.current
    try {
      const next = await api.automation()
      if (!mounted.current || version !== requestVersion.current) return
      setData(next)
      setError(null)
      onActiveChange?.(next.active || null)
      const latest = next.active || next.history?.[0]
      const key = latest ? `${latest.id}:${latest.status}:${latest.items?.map((item) => `${item.slug}:${item.status}:${item.runId || ''}`).join('|')}` : ''
      if (key !== previousProgress.current) {
        previousProgress.current = key
        onChanged?.()
      }
    } catch (e) {
      if (mounted.current && version === requestVersion.current) setError(e.message)
    }
  }, [onChanged, onActiveChange])
  // Fetch even with the panel closed so a restored running batch stays visible in the header.
  usePoll(load, open ? 1500 : 5000)

  useEffect(() => {
    if (!open) return undefined
    heading.current?.focus()
    const key = (event) => {
      if (event.key === 'Escape') { onClose(); toggleRef?.current?.focus() }
    }
    const outside = (event) => {
      if (!panel.current?.contains(event.target) && !toggleRef?.current?.contains(event.target)) onClose()
    }
    window.addEventListener('keydown', key)
    window.addEventListener('pointerdown', outside)
    return () => {
      window.removeEventListener('keydown', key)
      window.removeEventListener('pointerdown', outside)
    }
  }, [open, onClose, toggleRef])

  const list = exercises || []
  const selected = selection === null ? list.map((exercise) => exercise.slug) : selection.filter((slug) => list.some((e) => e.slug === slug))
  const active = data?.active || null
  const jobs = [...new Map([active, ...(data?.history || []).filter((job) => job.id !== active?.id)].filter(Boolean).map((job) => [job.id, job])).values()]
  const shown = jobs.find((job) => job.id === viewedId) || active || jobs[0] || null

  const action = async (work) => {
    if (actionInFlight.current) return
    actionInFlight.current = true
    requestVersion.current += 1 // an older poll must not overwrite the new job
    setBusy(true)
    setError(null)
    try {
      const result = await work()
      if (!mounted.current) return
      setViewedId(result.id)
      setData((before) => ({ active: result.status === 'running' ? result : null,
        history: [result, ...(before?.history || []).filter((job) => job.id !== result.id)] }))
      onActiveChange?.(result.status === 'running' ? result : null)
      onChanged?.()
      await load()
    } catch (e) {
      if (mounted.current) setError(e.message)
    } finally {
      actionInFlight.current = false
      if (mounted.current) setBusy(false)
    }
  }
  const toggle = (slug) => setSelection(selected.includes(slug) ? selected.filter((s) => s !== slug) : [...selected, slug])

  if (!open) return null
  return (
    <section id="automation-panel" className="automation-panel" ref={panel} role="dialog" aria-labelledby="automation-title">
      <div className="automation-head">
        <div>
          <h2 id="automation-title" ref={heading} tabIndex={-1}>Automate exercises</h2>
          <p>Find missing references, check their poses, then build supported exercises. Existing references stay in place. Problems are kept for review.</p>
        </div>
        <button type="button" className="btn btn-quiet" aria-label="Close automation" onClick={onClose}>Close</button>
      </div>
      {error && <p className="automation-error" role="alert">{error}</p>}
      {!data && !error && <p className="automation-note">Loading batch progress…</p>}

      {!active && (
        <details className="automation-prepare" open={shown ? undefined : true}>
          <summary>Choose exercises{shown ? ' for another batch' : ''}<span>{selected.length} selected</span></summary>
          <div className="automation-selection-head">
            <p>Select which exercises to process</p>
            <div>
              <button type="button" className="btn btn-quiet" onClick={() => setSelection(null)} disabled={busy}>Select all</button>
              <button type="button" className="btn btn-quiet" onClick={() => setSelection([])} disabled={busy}>Clear</button>
            </div>
          </div>
          <fieldset className="automation-selection" disabled={busy}>
            <legend className="sr-only">Exercises to process</legend>
            {list.map((exercise) => (
              <label key={exercise.slug}>
                <input type="checkbox" checked={selected.includes(exercise.slug)} onChange={() => toggle(exercise.slug)} />
                <span>{exercise.name}</span>
              </label>
            ))}
          </fieldset>
          <label className="automation-check-only">
            <input type="checkbox" checked={checkOnly} disabled={busy} onChange={(event) => setCheckOnly(event.target.checked)} />
            <span>Find and check references only — skip animation builds</span>
          </label>
          <button type="button" className="btn btn-primary" disabled={busy || !data || !selected.length}
            onClick={() => action(() => api.startAutomation(selected, { checkOnly }))}>
            {busy ? 'Starting…' : `${checkOnly ? 'Start reference checks' : 'Start automation'} (${selected.length})`}
          </button>
          <p className="automation-note">Missing equipment, unsuitable pictures, and failed form checks need review. A completed batch does not mean every exercise passed.</p>
        </details>
      )}

      {shown && (
        <div className="automation-results">
          <div className="automation-results-head">
            <h3>{shown.id === active?.id ? 'Current batch' : 'Batch results'}</h3>
            {jobs.length > 1 && (
              <label className="automation-history">
                <span>Batch</span>
                <select className="field" value={shown.id} onChange={(e) => setViewedId(e.target.value)}>
                  {jobs.map((job) => <option key={job.id} value={job.id}>{formatWhen(job.createdAt)}{job.id === active?.id ? ' · running' : ''}</option>)}
                </select>
              </label>
            )}
          </div>
          {shown.checkOnly && <p className="automation-note">Reference checks only; animation builds were not requested.</p>}
          <p className="automation-job-line" role="status">{automationJobLine(shown)}</p>
          <progress className="automation-progress" value={automationProgress(shown).finished} max={Math.max(shown.items?.length || 0, 1)}
            aria-label="Exercises processed" />
          {active && shown.id === active.id && (
            <div className="automation-stop">
              <button type="button" className="btn" disabled={busy || active.cancelRequested}
                onClick={() => action(() => api.cancelAutomation(active.id))}>
                {active.cancelRequested ? 'Stopping after current exercise…' : 'Stop remaining exercises'}
              </button>
              <p>Closing this panel keeps the batch running. Stopping lets the current exercise finish.</p>
            </div>
          )}
          <ol className="automation-items">
            {(shown.items || []).map((item) => {
              const status = automationItemState(item)
              const checks = automationCheckLine(item.summary)
              const source = automationSourceUrl(item)
              return (
                <li key={item.slug} className={`automation-item is-${status.kind}`}>
                  <div className="automation-item-head">
                    <span className="automation-item-name">{item.name || item.slug}</span>
                    <span className="automation-status">{status.label}</span>
                  </div>
                  {item.message && <p>{item.message}</p>}
                  {checks && <p className="automation-checks num">{checks}</p>}
                  <div className="automation-item-links">
                    <button type="button" className="btn btn-quiet" onClick={() => { onOpenExercise(item.slug, item.runId); onClose() }}>
                      {item.runId ? 'Open result' : 'Open exercise'}
                    </button>
                    {source && <a href={source} target="_blank" rel="noreferrer">Reference source</a>}
                  </div>
                </li>
              )
            })}
          </ol>
        </div>
      )}
    </section>
  )
}

export default AutomationPanel
