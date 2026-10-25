import { runFileUrl } from './run-view.js'

const OFFLINE = "The Studio isn't answering. Is `npm run dev` still running in the studio folder?"

const seg = (s) => encodeURIComponent(s)

// Errors come back as {error: message} in plain words; show them as given.
const request = async (url, options) => {
  let res
  try {
    res = await fetch(url, options)
  } catch {
    throw new Error(OFFLINE)
  }
  const text = await res.text()
  let body = null
  try {
    body = text ? JSON.parse(text) : null
  } catch {
    body = null
  }
  if (!res.ok) throw new Error(body?.error || `The Studio answered with an error (${res.status}).`)
  return body
}

const json = (method, body) => ({ method, headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) })

// A run file that may not exist yet (a failed or running run): null instead of an error.
const optionalJson = async (url) => {
  try {
    const res = await fetch(url)
    return res.ok ? await res.json() : null
  } catch {
    return null
  }
}

export const api = {
  state: () => request('/api/state'),
  work: () => request('/api/work'),
  automation: () => request('/api/automation'),
  automationJob: (id) => request(`/api/automation/${seg(id)}`),
  startAutomation: (slugs, options = {}) => request('/api/automation', json('POST', { slugs, ...options })),
  cancelAutomation: (id) => request(`/api/automation/${seg(id)}/cancel`, { method: 'POST' }),
  exercises: () => request('/api/exercises'),
  exercise: (slug) => request(`/api/exercises/${seg(slug)}`),
  addExercise: ({ name, equipment }) => request('/api/exercises', json('POST', { name, equipment })),
  uploadReference: (slug, file) =>
    request(`/api/exercises/${seg(slug)}/references`, {
      method: 'POST',
      headers: { 'x-filename': encodeURIComponent(file.name), 'content-type': file.type || 'application/octet-stream' },
      body: file,
    }),
  reorderReferences: (slug, ids) => request(`/api/exercises/${seg(slug)}/references/order`, json('PUT', { ids })),
  updateReference: (slug, id, patch) => request(`/api/exercises/${seg(slug)}/references/${seg(id)}`, json('PATCH', patch)),
  removeReference: (slug, id) => request(`/api/exercises/${seg(slug)}/references/${seg(id)}`, { method: 'DELETE' }),
  startRun: (slug, options = {}) => request(`/api/exercises/${seg(slug)}/runs`, json('POST', options)),
  checkReferences: (slug) => request(`/api/exercises/${seg(slug)}/read`, { method: 'POST' }),

  // One JSON file from a run's folder, or null when it isn't there.
  runJson: (slug, runId, file) => optionalJson(runFileUrl(slug, runId, file)),

  // Everything the comparison and checks need from one run's folder.
  runBundle: async (slug, runId) => {
    const [snapshot, ours, checks, targets, measurements, readProblems, preflight, retarget] = await Promise.all(
      ['references.json', 'ours.json', 'checks.json', 'targets.json', 'measurements.json', 'read-problems.json', 'preflight.json', 'frames/retarget.json']
        .map((file) => optionalJson(runFileUrl(slug, runId, file))),
    )
    return { snapshot, ours, checks, targets, measurements, readProblems, preflight, characterFit: retarget?.character_fit || null }
  },
}
