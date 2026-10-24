import { useEffect, useRef } from 'react'
import './work-panel.css'
import WorkItem from './work-item.jsx'

const MAX_SHOWN = 20

// Everything going on behind the scenes, opened from the top bar. Running work first, then the newest.
const WorkPanel = ({ id, open, items, exercises, state, runsBySlug, onClose, toggleRef }) => {
  const panel = useRef(null)
  const heading = useRef(null)

  // Opening moves focus into the panel; Esc or a click outside closes it.
  useEffect(() => {
    if (!open) return undefined
    heading.current?.focus()
    const onKey = (e) => {
      if (e.key === 'Escape') {
        onClose()
        toggleRef?.current?.focus()
      }
    }
    const onDown = (e) => {
      if (panel.current?.contains(e.target) || toggleRef?.current?.contains(e.target)) return
      onClose()
    }
    window.addEventListener('keydown', onKey)
    window.addEventListener('pointerdown', onDown)
    return () => {
      window.removeEventListener('keydown', onKey)
      window.removeEventListener('pointerdown', onDown)
    }
  }, [open, onClose, toggleRef])

  if (!open) return null
  const list = items ?? []
  const shown = list.slice(0, MAX_SHOWN)

  return (
    <section id={id} ref={panel} className="work-panel" aria-labelledby={`${id}-title`}>
      <div className="work-panel-head">
        <div>
          <h2 className="work-panel-title" id={`${id}-title`} ref={heading} tabIndex={-1}>Behind the scenes</h2>
          <p className="work-panel-note">
            Work Claude is doing on the pipeline. When a fix finishes, the exercises it lists run again on their own.
          </p>
        </div>
        <button type="button" className="btn btn-quiet work-panel-close" onClick={onClose} aria-label="Close Behind the scenes">
          Close
        </button>
      </div>

      {list.length === 0 ? (
        <p className="work-panel-empty">Nothing yet. When Claude works on the pipeline, it shows here with its progress.</p>
      ) : (
        <ol className="work-panel-list">
          {shown.map((item) => (
            <li key={item.id} className="work-panel-row">
              <WorkItem item={item} exercises={exercises} state={state} runsBySlug={runsBySlug} showExercises />
            </li>
          ))}
        </ol>
      )}
      {list.length > MAX_SHOWN && (
        <p className="work-panel-more">Showing the newest {MAX_SHOWN} of {list.length}.</p>
      )}
    </section>
  )
}

export default WorkPanel
