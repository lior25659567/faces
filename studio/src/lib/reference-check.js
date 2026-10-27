import { samePictures } from './run-view.js'

const bundleKey = (value) => value ? `${value.id}:${value.commit}` : null
export const sameReferenceSource = (snapshot, motionPreset = null, referenceBundle = null) =>
  (snapshot?.motionPreset ?? null) === (motionPreset ?? null) &&
  bundleKey(snapshot?.referenceBundle) === bundleKey(referenceBundle)

// A reference check is useful only for the exact pictures and tags currently being edited.
// It never constitutes a passing animation result.
export const referenceCheck = (references, run, bundle, motionPreset = null, referenceBundle = null) => {
  if (!run || !bundle || bundle.runId !== run.id || !bundle.snapshot?.images) return { state: 'unchecked', canBuild: false }
  if (!sameReferenceSource(bundle.snapshot, motionPreset, referenceBundle)) return { state: 'stale', canBuild: false }
  if (!samePictures(references, bundle.snapshot.images)) return { state: 'stale', canBuild: false }
  if (run.status === 'queued' || (run.status === 'running' && !bundle.preflight)) return { state: 'reading', canBuild: false }
  if (bundle.preflight) {
    const { ready, blockers = [], warnings = [] } = bundle.preflight
    const canBuild = ready === true && blockers.length === 0
    return { state: canBuild ? 'ready' : 'blocked', canBuild, blockers, warnings }
  }
  if (run.status === 'failed' || run.status === 'interrupted') {
    return { state: 'failed', canBuild: false, message: run.error?.message || 'The reference check stopped. Try checking again.' }
  }
  return { state: 'unchecked', canBuild: false }
}
