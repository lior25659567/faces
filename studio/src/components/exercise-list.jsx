import { useState } from 'react'
import './exercise-list.css'
import { api } from '../lib/api.js'
import { EQUIPMENT } from '../lib/labels.js'
import { dotKind, exerciseLine } from '../lib/status.js'

const AddExercise = ({ onAdded }) => {
  const [open, setOpen] = useState(false)
  const [name, setName] = useState('')
  const [equipment, setEquipment] = useState('barbell')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)

  const submit = async (e) => {
    e.preventDefault()
    setBusy(true)
    setError(null)
    try {
      const added = await api.addExercise({ name, equipment })
      setName('')
      setOpen(false)
      onAdded(added)
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy(false)
    }
  }

  if (!open) {
    return (
      <button type="button" className="btn exercise-add-open" onClick={() => setOpen(true)}>
        <span aria-hidden="true">+</span> Add exercise
      </button>
    )
  }

  return (
    <form className="exercise-add" onSubmit={submit}>
      <label className="exercise-add-label" htmlFor="new-exercise-name">Name</label>
      <input
        id="new-exercise-name"
        className="field"
        value={name}
        onChange={(e) => setName(e.target.value)}
        placeholder="e.g. Romanian Deadlift"
        autoFocus
        required
      />
      <label className="exercise-add-label" htmlFor="new-exercise-equipment">Equipment</label>
      <select id="new-exercise-equipment" className="field" value={equipment} onChange={(e) => setEquipment(e.target.value)}>
        {EQUIPMENT.map((k) => <option key={k} value={k}>{k}</option>)}
      </select>
      {error && <p className="exercise-add-error" role="alert">{error}</p>}
      <div className="exercise-add-actions">
        <button type="submit" className="btn btn-primary" disabled={busy || !name.trim()}>
          {busy ? 'Adding…' : 'Add exercise'}
        </button>
        <button type="button" className="btn btn-quiet" onClick={() => { setOpen(false); setError(null) }}>Cancel</button>
      </div>
    </form>
  )
}

const ExerciseList = ({ exercises, state, runsBySlug, selected, onSelect, onAdded }) => (
  <nav className="exercise-list" aria-label="Exercises">
    <div className="exercise-list-head">
      <h2 className="exercise-list-title">Exercises</h2>
      {exercises && <span className="exercise-list-count num">{exercises.length}</span>}
    </div>
    {!exercises && <p className="exercise-list-empty">Loading…</p>}
    <ul className="exercise-list-items">
      {exercises?.map((e) => {
        const kind = dotKind(e.status)
        return (
          <li key={e.slug}>
            <button
              type="button"
              className={`exercise-item${e.slug === selected ? ' is-selected' : ''}`}
              aria-current={e.slug === selected ? 'page' : undefined}
              onClick={() => onSelect(e.slug)}
            >
              <span className={`exercise-dot is-${kind}`} aria-hidden="true" />
              <span className="exercise-item-text">
                <span className="exercise-item-name">{e.name}</span>
                <span className={`exercise-item-line is-${kind}`}>{exerciseLine(e, state, runsBySlug[e.slug])}</span>
              </span>
            </button>
          </li>
        )
      })}
    </ul>
    <AddExercise onAdded={onAdded} />
  </nav>
)

export default ExerciseList
