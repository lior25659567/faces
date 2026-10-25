import { createHash } from 'node:crypto'
import { readFileSync } from 'node:fs'
import { expect, test } from 'vitest'
import { bundledReferenceImageFor, getBundledReferenceSpec, matchesBundledReference } from './bundled-references.js'
import { getReferenceSpec } from './reference-catalog.js'
import { fetchReference } from './reference-fetch.js'

const exercise = 'Walking Lunge (dumbbells optional)'
const spec = getBundledReferenceSpec(exercise)
const manifest = JSON.parse(readFileSync(new URL('../../reference/exercise-lunge/sequence.json', import.meta.url), 'utf8'))
const source = JSON.parse(readFileSync(new URL('../../reference/exercise-lunge/sources.json', import.meta.url), 'utf8'))

test('reviewed stepping frames remain unavailable to the reversible-cycle builder', () => {
  expect(spec.motion).toBe('sequence')
  expect(spec.autoBuild).toBe(false)
  expect(spec.reason).toContain('stepping/root-motion builder')
  expect(spec.sequenceInfo).toMatchObject({
    calibrated: false, sourceInferredDepth: true, completeClosedCycle: false,
    initialStandingTimestampSeconds: 2.001667, finalStandingTimestampSeconds: null,
  })
  expect(spec.images[8].role).toBe('left_bottom')
  expect(spec.images.at(-1).role).toBe('next_right_heel_approach')
  // The existing catalog stays safe for callers that do not understand sequences.
  expect(getReferenceSpec(exercise).images.every((image) => image.contextOnly)).toBe(true)
  expect(getReferenceSpec(exercise).autoBuild).toBe(false)
})

test('sequence retains actual timestamps and observed occlusions instead of inventing floor contacts', () => {
  expect(spec.images).toHaveLength(12)
  expect(spec.images.map((image) => image.timeSeconds)).toEqual(manifest.frames.map((frame) => frame.sourceTimestampSeconds))
  expect(spec.images.map((image) => image.sourceFrameIndex)).toEqual([120, 150, 180, 210, 240, 270, 300, 330, 360, 390, 420, 450])
  expect(spec.images.filter((image) => !image.motionOnly).map((image) => [image.sequenceIndex, image.phase]))
    .toEqual([[0, 'start'], [3, 'middle'], [8, 'end']])
  expect(spec.images.every((image) => image.view === 'front' && image.contextOnly === false)).toBe(true)
  expect(spec.images[3]).toMatchObject({ observedSupport: 'right', footContacts: { left: 'uncertain', right: 'flat' } })
  expect(spec.images[8]).toMatchObject({ observedSupport: 'left', footContacts: { left: 'flat', right: 'uncertain' } })
  expect(spec.images[10].footContacts.right).toBe('airborne')
})

test('every bundled frame is present, has its reviewed checksum and carries derivative attribution', async () => {
  for (const [index, image] of spec.images.entries()) {
    const result = await fetchReference(image, { fetchImpl: () => { throw Error('local frames must not use the network') } })
    expect(result.ext).toBe('jpg')
    expect(result.sha256).toBe(manifest.frames[index].sha256)
    expect(createHash('sha256').update(result.bytes).digest('hex')).toBe(image.sha256)
    expect(result.provenance).toMatchObject({
      provider: 'wger', author: 'Goulart', license: 'CC BY-SA 4.0',
      licenseUrl: 'https://creativecommons.org/licenses/by-sa/4.0/',
      sourceVideoUrl: source.url, sourceVideoSha256: source.sha256,
      commit: source.sha256, sha256: image.sha256, timeSeconds: image.timeSeconds,
      sourceFrameIndex: image.sourceFrameIndex, observedSupport: image.observedSupport,
      derivative: manifest.derivative,
    })
    expect(result.provenance.sequenceInfo.calibrated).toBe(false)
  }
})

test('bundles use exact exercise and frame identities without arbitrary URL or path selection', async () => {
  for (const name of ['Walking Lunge', exercise + ' ', '__proto__', null, {}]) expect(getBundledReferenceSpec(name)).toBeNull()
  const image = spec.images[0]
  let reads = 0
  const readFileImpl = async () => { reads++; throw Error('must not read') }
  const fetchImpl = async () => { throw Error('must not fetch') }
  for (const url of [image.url.replace('#t=', '?t='), image.url + '0', source.url,
    'file:///etc/passwd', 'http://127.0.0.1/private', image.url.replace('wger.de', 'wger.de.evil.example')]) {
    expect(bundledReferenceImageFor(url)).toBeNull()
    await expect(fetchReference({ url }, { readFileImpl, fetchImpl })).rejects.toMatchObject({ code: 'UNTRUSTED_SOURCE' })
  }
  for (const key of ['path', 'file', 'fileUrl', 'localPath', 'filename']) {
    await expect(fetchReference({ ...image, [key]: '/etc/passwd' }, { readFileImpl, fetchImpl }))
      .rejects.toMatchObject({ code: 'UNTRUSTED_SOURCE' })
  }
  expect(reads).toBe(0)
})

test('callers cannot rewrite source timing, phase, checksum, or contact evidence', async () => {
  const image = spec.images[3]
  const overrides = {
    id: 'other', sourceId: 'other', commit: '0'.repeat(64), sha256: '0'.repeat(64),
    phase: 'start', view: 'side', contextOnly: true, motionOnly: true,
    timeSeconds: 0, sequenceIndex: 0, sourceFrameIndex: 0,
    role: 'standing_end', leadLeg: 'left', observedSupport: 'both',
    footContacts: { left: 'flat', right: 'flat' }, contactNote: 'calibrated',
  }
  for (const [key, value] of Object.entries(overrides)) {
    const candidate = { ...image, [key]: value }
    expect(matchesBundledReference(candidate, image)).toBe(false)
    await expect(fetchReference(candidate, { readFileImpl: () => { throw Error('must not read') } }))
      .rejects.toMatchObject({ code: 'UNTRUSTED_SOURCE' })
  }
  // A URL-only caller still receives the registry's full trusted attribution.
  expect((await fetchReference({ url: image.url })).provenance.footContacts.left).toBe('uncertain')
})

test('missing, corrupt, non-image, and oversized local assets fail without a network fallback', async () => {
  const image = spec.images[0]
  const bytes = readFileSync(new URL('../../reference/exercise-lunge/lunge-00.jpg', import.meta.url))
  const altered = Buffer.from(bytes); altered[altered.length - 10] ^= 1
  const fetchImpl = () => { throw Error('must not fetch') }
  await expect(fetchReference(image, { fetchImpl, readFileImpl: async () => { throw Error('ENOENT /private/path') } }))
    .rejects.toMatchObject({ code: 'BUNDLE_MISSING', retryable: false })
  await expect(fetchReference(image, { fetchImpl, readFileImpl: async () => altered }))
    .rejects.toMatchObject({ code: 'CHECKSUM_MISMATCH', retryable: false })
  await expect(fetchReference(image, { fetchImpl, readFileImpl: async () => Buffer.from('<h1>unavailable</h1>') }))
    .rejects.toMatchObject({ code: 'NOT_IMAGE' })
  await expect(fetchReference(image, { fetchImpl, maxBytes: bytes.length - 1 }))
    .rejects.toMatchObject({ code: 'TOO_LARGE' })
})

test('local frame loading observes cancellation and a bounded deadline', async () => {
  const image = spec.images[0]
  const controller = new AbortController()
  const result = fetchReference(image, { signal: controller.signal, readFileImpl: () => new Promise(() => {}) })
  controller.abort()
  await expect(result).rejects.toMatchObject({ code: 'ABORTED' })
  await expect(fetchReference(image, { timeoutMs: 10, readFileImpl: () => new Promise(() => {}) }))
    .rejects.toMatchObject({ code: 'TIMEOUT' })
  await expect(fetchReference(image, { signal: controller.signal, readFileImpl: () => { throw Error('must not read') } }))
    .rejects.toMatchObject({ code: 'ABORTED' })
})

test('nested source and contact records cannot be mutated by callers', () => {
  expect(() => { spec.images[3].footContacts.left = 'flat' }).toThrow()
  expect(() => { spec.sequenceInfo.completeClosedCycle = true }).toThrow()
  expect(() => { bundledReferenceImageFor(spec.images[0].url).fileUrl = 'file:///etc/passwd' }).toThrow()
})
