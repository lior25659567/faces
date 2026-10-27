import { useId } from 'react'
import './character-fit.css'

const MovementSource = ({ presets, value, onChange, disabled }) => {
  const id = useId()
  if (!presets?.length) return null
  return (
    <div className="character-fit movement-source">
      <label htmlFor={id}>Movement source</label>
      <select id={id} className="field" value={value ?? ''} disabled={disabled}
        aria-describedby={`${id}-help`} onChange={(event) => onChange(event.target.value || null)}>
        {presets.map((preset) => <option key={preset.id} value={preset.id}>{preset.label}</option>)}
        <option value="">Reference pictures</option>
      </select>
      <p id={`${id}-help`}>{value
        ? 'Use the checked movement and fitted hand placement. Pictures are kept as visual references; none are required to start.'
        : 'Reconstruct the movement from your pictures. The required positions and camera views must pass the reference check.'}</p>
    </div>
  )
}

export default MovementSource
