import { readFileSync } from 'node:fs'
import { isDeepStrictEqual } from 'node:util'

const freeze = (value) => {
  if (value && typeof value === 'object') {
    for (const child of Object.values(value)) freeze(child)
    Object.freeze(value)
  }
  return value
}

const directory = new URL('../../reference/exercise-lunge/', import.meta.url)
const sequence = JSON.parse(readFileSync(new URL('sequence.json', directory), 'utf8'))
const source = JSON.parse(readFileSync(new URL('sources.json', directory), 'utf8'))
const sourceId = `wger-${source.exerciseId}-video-${source.videoId}`
const anchors = new Map([[0, 'start'], [3, 'middle'], [8, 'end']])
const unavailableReason = 'Walking lunges require a chronological stepping/root-motion builder. These frontal frames do not supply calibrated depth, and the selection is not a closed cycle.'

if (!/^[a-f0-9]{64}$/.test(source.sha256) || !Array.isArray(sequence.frames) || sequence.frames.length !== 12) {
  throw Error('Invalid bundled lunge reference manifest.')
}

// These files and manifests are reviewed repository assets. Neither an exercise
// name nor a caller's image object can select a filesystem path.
const images = sequence.frames.map((frame, index) => {
  if (!/^lunge-\d{2}\.jpg$/.test(frame.file) || !/^[a-f0-9]{64}$/.test(frame.sha256) ||
      frame.sequenceIndex !== index || !Number.isInteger(frame.sourceFrameIndex) ||
      !Number.isFinite(frame.sourceTimestampSeconds) || frame.sourceTimestampSeconds < 0 ||
      index > 0 && (frame.sourceTimestampSeconds <= sequence.frames[index - 1].sourceTimestampSeconds ||
        frame.sourceFrameIndex <= sequence.frames[index - 1].sourceFrameIndex)) {
    throw Error('Invalid bundled lunge reference manifest.')
  }
  return {
    id: frame.id, sourceId, commit: source.sha256,
    url: `${source.url}#t=${frame.sourceTimestampSeconds}`,
    phase: anchors.get(frame.sequenceIndex) ?? 'middle', view: 'front', contextOnly: false,
    motionOnly: !anchors.has(frame.sequenceIndex), timeSeconds: frame.sourceTimestampSeconds,
    sequenceIndex: frame.sequenceIndex, sourceFrameIndex: frame.sourceFrameIndex,
    role: frame.role, leadLeg: frame.leadLeg, observedSupport: frame.observedSupport,
    footContacts: frame.footContacts, contactNote: frame.contactNote, sha256: frame.sha256,
  }
})

const provenance = freeze({
  provider: source.provider, repository: source.metadataUrl, commit: source.sha256,
  license: source.license, licenseUrl: source.licenseUrl, author: source.author,
  authorHistory: source.authorHistory, sourceVideoUrl: source.url, sourceVideoSha256: source.sha256,
  sourceMetadataUrl: source.metadataUrl, retrievedOn: source.retrievedOn,
  derivative: sequence.derivative, cropPixels: sequence.cropPixels,
})

const spec = freeze({
  exercise: sequence.exercise, id: sourceId, sourceId, commit: source.sha256,
  repository: source.metadataUrl, license: source.license, licenseUrl: source.licenseUrl,
  author: source.author, variant: sequence.variant, images, motion: 'sequence', autoBuild: false,
  limitations: sequence.limitations, reason: unavailableReason,
  sequenceInfo: {
    completeClosedCycle: sequence.completeClosedCycle, sourceInferredDepth: true,
    calibrated: false, camera: sequence.camera,
    initialStandingTimestampSeconds: sequence.initialStandingTimestampSeconds,
    finalStandingTimestampSeconds: sequence.finalStandingTimestampSeconds,
    uprightPassingTimestampSeconds: sequence.uprightPassingTimestampSeconds,
    contactAnnotationPolicy: sequence.contactAnnotationPolicy,
  },
})

const specs = new Map([[spec.exercise, spec]])
const imagesByUrl = new Map(images.map((image, index) => [image.url, freeze({
  image, fileUrl: new URL(sequence.frames[index].file, directory).href,
  provenance, sequenceInfo: spec.sequenceInfo,
})]))

export const getBundledReferenceSpec = (exercise) => typeof exercise === 'string' ? specs.get(exercise) ?? null : null

export const bundledReferenceImageFor = (url) => typeof url === 'string' ? imagesByUrl.get(url) ?? null : null

// A trusted URL is an identity, not permission to rewrite the frame's hash,
// timing, phase, source attribution, or path. Extra display fields are harmless.
export const matchesBundledReference = (candidate, trusted) => {
  if (!candidate || typeof candidate !== 'object') return false
  if (['path', 'file', 'fileUrl', 'localPath', 'filename'].some((key) => Object.hasOwn(candidate, key))) return false
  return Object.entries(trusted).every(([key, value]) => !Object.hasOwn(candidate, key) || isDeepStrictEqual(candidate[key], value))
}
