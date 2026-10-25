import { useId } from 'react'
import { CHARACTER_FITS } from '../lib/character-fit.js'
import './character-fit.css'

const CharacterFit = ({ value, onChange, disabled = false }) => {
  const id = useId()
  const selected = CHARACTER_FITS.find((fit) => fit.value === value) || CHARACTER_FITS[0]
  return (
    <div className="character-fit">
      <label htmlFor={id}>Character proportions</label>
      <select id={id} className="field" value={selected.value} disabled={disabled}
        aria-describedby={`${id}-help`} onChange={(event) => onChange(event.target.value)}>
        {CHARACTER_FITS.map((fit) => <option key={fit.value} value={fit.value}>{fit.label}</option>)}
      </select>
      <p id={`${id}-help`}>{selected.description}</p>
    </div>
  )
}

export default CharacterFit
