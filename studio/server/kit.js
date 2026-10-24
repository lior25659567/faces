import { readFileSync } from 'node:fs'
import path from 'node:path'
import { REPO } from './paths.js'

// What each exercise holds and what holds it up, and what of that the animation path builds:
// studio/measure/exercise-kit.json, the one table (studio/measure/kit.py reads it for the pipeline and the checks;
// these are its sentences, word for word). An exercise added in the Studio goes by its equipment word.
const KIT = JSON.parse(readFileSync(path.join(REPO, 'studio', 'measure', 'exercise-kit.json'), 'utf8'))

const own = (table, key) => (key != null && Object.hasOwn(table, key) ? table[key] : null)

const kitOf = (name, equipment) => {
  const row = own(KIT.exercises, name) ?? own(KIT.by_equipment, equipment) ?? {}
  return { holds: row.holds ?? null, support: row.support ?? null, steps: Boolean(row.steps), built: Boolean(row.built), clip: row.clip ?? null }
}

// The plain sentence the picture path (no Mixamo clip) refuses the exercise with, or null when it can animate it.
export const pictureRefusal = (name, equipment) => {
  const row = kitOf(name, equipment)
  if (row.steps) return KIT.refusals.steps.replace('{clip}', row.clip || 'its clip')
  if (row.support) return KIT.refusals.support.replace('{support}', row.support)
  return null
}

// What the hands should hold that the animation path doesn't build ("bar", "dumbbells", ...), or null.
export const unheld = (name, equipment) => {
  const row = kitOf(name, equipment)
  return row.built ? null : row.holds
}
