import { readFileSync } from 'node:fs'

const manifest = JSON.parse(readFileSync(new URL('../../reference/exercise-catalog.json', import.meta.url), 'utf8'))
const freeze = (value) => {
  if (value && typeof value === 'object') {
    for (const child of Object.values(value)) freeze(child)
    Object.freeze(value)
  }
  return value
}

// This is a reviewed, exact variant mapping. Unknown exercises need a curated source,
// not a fuzzy search that can silently turn a front squat into a back squat.
export const REFERENCE_SOURCE = freeze(manifest.source)
export const REFERENCE_CATALOG = freeze(Object.fromEntries(manifest.exercises.map((entry) => [entry.exercise, {
  ...entry, reason: entry.autoBuild === false ? entry.limitations.join(' ') : null,
  sourceId: entry.id, commit: manifest.source.commit, license: manifest.source.license,
  repository: manifest.source.repository, licenseUrl: manifest.source.licenseUrl,
  images: entry.images.map((image) => ({ ...image, sourceId: entry.id, commit: manifest.source.commit })),
}])))

export const getReferenceSpec = (exerciseName) =>
  typeof exerciseName === 'string' && Object.hasOwn(REFERENCE_CATALOG, exerciseName) ? REFERENCE_CATALOG[exerciseName] : null

const imagesByUrl = new Map(Object.values(REFERENCE_CATALOG).flatMap((spec) => spec.images.map((image) => [image.url, image])))

// Download callers may pass catalog image objects, but cannot change the trusted URL,
// expected content hash, view or source revision by editing their own copy of one.
export const referenceImageFor = (url) => typeof url === 'string' ? imagesByUrl.get(url) ?? null : null
