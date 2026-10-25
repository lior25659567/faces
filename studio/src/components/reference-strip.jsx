import { useRef, useState } from 'react'
import './reference-strip.css'
import { api } from '../lib/api.js'
import { formatWhen } from '../lib/format.js'
import { PHASE_LABELS, VIEW_LABELS } from '../lib/labels.js'
import { referenceUrl } from '../lib/run-view.js'

const PICKABLE_VIEWS = ['side', 'front', 'behind', 'three-quarter']

const move = (ids, id, to) => {
  const rest = ids.filter((x) => x !== id)
  rest.splice(Math.max(0, Math.min(to, rest.length)), 0, id)
  return rest
}

const sameOrder = (a, b) => a.length === b.length && a.every((x, i) => x === b[i])

const Card = ({ slug, image, index, count, why, whyRun, dragging, busy, onPatch, onMove, onRemove, drag }) => {
  const views = image.view === 'frontal' ? ['frontal', ...PICKABLE_VIEWS] : PICKABLE_VIEWS
  const name = image.original || image.file
  return (
    <li className={`ref-card${dragging ? ' is-dragging' : ''}${image.contextOnly ? ' is-context' : ''}`} {...drag}>
      <div className="ref-thumb">
        <img src={referenceUrl(slug, image.file)} alt={`Reference ${index + 1}: ${name}`} draggable={false} />
        <span className="ref-order num">{index + 1}</span>
      </div>
      <p className="ref-name" title={name}>{name}</p>
      {image.source?.motionOnly && <span className="ref-motion-sample"
        title={`Sequence frame${Number.isFinite(image.source.timeSeconds) ? ` at ${image.source.timeSeconds.toFixed(3)} seconds` : ''}; excluded from comparison stills.`}>
        Motion sample
      </span>}

      {/* While a change is being saved, every control waits, so a double-click can't start two runs. */}
      <fieldset className="ref-controls" disabled={busy}>
        <label className="ref-field">
          <span>Position</span>
          <select
            className="field"
            value={image.phase}
            disabled={image.contextOnly}
            onChange={(e) => onPatch({ phase: e.target.value })}
          >
            {Object.entries(PHASE_LABELS).map(([k, v]) => <option key={k} value={k}>{v}</option>)}
          </select>
        </label>
        {!image.contextOnly && image.phaseSetBy !== 'user' && <span className="ref-hint">guessed from the order</span>}

        <label className="ref-field">
          <span>View</span>
          <select
            className={`field${image.view === 'frontal' ? ' needs-pick' : ''}`}
            value={image.view ?? ''}
            onChange={(e) => onPatch({ view: e.target.value || null })}
          >
            <option value="">Not known yet</option>
            {views.map((v) => <option key={v} value={v}>{VIEW_LABELS[v]}</option>)}
          </select>
        </label>
        {image.view === 'frontal' && <span className="ref-hint is-pick">The detector can't tell: pick front or behind</span>}
        {image.view && image.view !== 'frontal' && image.viewSetBy === 'detector' && <span className="ref-hint">detected</span>}

        <label className="ref-check">
          <input type="checkbox" checked={image.contextOnly} onChange={(e) => onPatch({ contextOnly: e.target.checked })} />
          <span>Context only</span>
        </label>

        {why && !image.contextOnly && <p className="ref-why">Not used in the run of {whyRun}: {why}</p>}

        <div className="ref-actions">
          <button type="button" className="btn btn-quiet ref-move" disabled={index === 0} onClick={() => onMove(index - 1)}
            aria-label={`Move ${name} earlier`} title="Move earlier">‹</button>
          <button type="button" className="btn btn-quiet ref-move" disabled={index === count - 1} onClick={() => onMove(index + 1)}
            aria-label={`Move ${name} later`} title="Move later">›</button>
          <button type="button" className="btn btn-quiet ref-remove" onClick={onRemove}>Remove</button>
        </div>
      </fieldset>
    </li>
  )
}

// The pictures in movement order. Every change saves; checking and building are explicit. `reasons` (run-view
// stripReasons) are one run's own "Not used" reasons, named by that run's time.
const ReferenceStrip = ({ slug, references, reasons, busy, onChange }) => {
  const [order, setOrder] = useState(null)
  const [dragId, setDragId] = useState(null)
  const dropped = useRef(false)
  const ids = references.map((r) => r.id)
  const shownIds = order ?? ids
  const items = shownIds.map((id) => references.find((r) => r.id === id)).filter(Boolean)

  const saveOrder = async (next) => {
    if (sameOrder(next, ids)) {
      setOrder(null)
      return
    }
    setOrder(next)
    await onChange(() => api.reorderReferences(slug, next))
    setOrder(null)
  }

  const dragProps = (image) => ({
    draggable: !busy,
    onDragStart: (e) => {
      if (busy) {
        e.preventDefault()
        return
      }
      dropped.current = false
      setDragId(image.id)
      e.dataTransfer.effectAllowed = 'move'
      e.dataTransfer.setData('text/plain', image.id)
    },
    onDragOver: (e) => {
      if (!dragId) return
      e.preventDefault()
      if (dragId === image.id) return
      const box = e.currentTarget.getBoundingClientRect()
      const after = e.clientX > box.left + box.width / 2
      const current = order ?? ids
      const without = current.filter((x) => x !== dragId)
      const at = without.indexOf(image.id) + (after ? 1 : 0)
      const next = move(current, dragId, at)
      if (!sameOrder(next, current)) setOrder(next)
    },
    onDrop: (e) => {
      if (!dragId) return
      e.preventDefault()
      dropped.current = true
    },
    onDragEnd: () => {
      const next = order
      setDragId(null)
      // Esc, or let go away from the pictures: put them back as they were.
      if (!dropped.current) setOrder(null)
      else if (next) saveOrder(next)
    },
  })

  return (
    <ol className="ref-strip">
      {items.map((image, index) => {
        const info = reasons?.images?.[image.id]
        return (
          <Card
            key={image.id}
            slug={slug}
            image={image}
            index={index}
            count={items.length}
            why={info && info.used === false ? info.why : null}
            whyRun={reasons ? formatWhen(reasons.createdAt) : null}
            dragging={dragId === image.id}
            busy={busy}
            drag={dragProps(image)}
            onPatch={(patch) => onChange(() => api.updateReference(slug, image.id, patch))}
            onMove={(to) => saveOrder(move(shownIds, image.id, to))}
            onRemove={() => {
              if (window.confirm(`Remove ${image.original || 'this picture'} from this exercise?`)) {
                onChange(() => api.removeReference(slug, image.id))
              }
            }}
          />
        )
      })}
    </ol>
  )
}

export default ReferenceStrip
