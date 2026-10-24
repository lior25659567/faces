// Import the sourced example photos without replacing existing references or starting a build.
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { createStore } from '../server/store.js'
const store = createStore()
const slug = 'barbell-bent-over-row'
for (const phase of ['start', 'end']) {
  const name = `example-row-${phase}.jpg`
  const existing = store.listReferences(slug).find((image) => image.original === name)
  if (existing) { console.log(`${name}: already imported`); continue }
  const file = fileURLToPath(new URL(`../../reference/exercise-row/${phase}.jpg`, import.meta.url))
  const image = store.addReference(slug, readFileSync(file), name, 'jpg')
  store.updateReference(slug, image.id, { phase, view: 'three-quarter' })
  console.log(`${name}: saved as ${phase}, three-quarter view`)
}
