import { createHash } from 'node:crypto'
import { readFileSync, realpathSync, statSync } from 'node:fs'
import path from 'node:path'
import { isDeepStrictEqual } from 'node:util'
import { REPO } from './paths.js'

const ASSETS = {
  'animation.gif': 'image/gif', 'start.png': 'image/png', 'end.png': 'image/png',
  'checks.json': 'application/json',
}
const COUNTS = ['pass', 'fail', 'limited', 'uncertain', 'info', 'unmeasured']
const readJson = (file) => JSON.parse(readFileSync(file, 'utf8'))
const digest = (bytes) => createHash('sha256').update(bytes).digest('hex')
const validSummary = (summary) => summary && COUNTS.every((key) => Number.isInteger(summary[key]) && summary[key] >= 0)
const originalFit = { applied: false, label: 'Original proportions', source_model_unchanged: true }
const fitSummary = (fit) => fit ? Object.fromEntries(['applied', 'label', 'requested_mode', 'upper_scale', 'fore_scale',
  'source_model_unchanged', 'face_max_displacement_mm', 'reason'].filter((key) => Object.hasOwn(fit, key)).map((key) => [key, fit[key]])) : originalFit

const unavailableReason = (slug) => slug === 'barbell-back-squat'
  ? 'A suitable side view and a front or rear grip reference are still needed for a verified squat preview.'
  : slug === 'walking-lunge-dumbbells-optional'
    ? 'Walking-lunge rendering remains blocked: the source handoff does not establish a complete root path, and stepping needs validation.'
    : 'No bundled verified preview is available for this exercise yet.'

// Bundled examples are separate from the user's references and run history.
// Only reviewed repository manifests select paths; request parameters never do.
export const createValidatedPreviews = ({ root = REPO } = {}) => {
  const records = new Map(), cache = new Map()
  let manifestError = null
  let allowedRoot = null
  try {
    allowedRoot = realpathSync(path.join(root, 'reference/automation-validation'))
    const base = path.join(root, 'reference/automation-validation/corrections')
    const manifest = readJson(path.join(base, 'results.json'))
    if (!Array.isArray(manifest.runs)) throw Error('Missing preview records.')
    for (const run of manifest.runs) {
      if (!/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(run.slug || '') || records.has(run.slug) || !validSummary(run.summary) ||
          !Number.isInteger(run.frames) || run.frames < 1 || !Number.isFinite(run.fps) || run.fps <= 0) throw Error('Invalid preview record.')
      records.set(run.slug, { folder: path.join(base, run.slug), summary: run.summary, frames: run.frames,
        fps: run.fps, characterFit: fitSummary(run.character_fit), hashes: run.output_sha256,
        playback: run.reference_motion?.mode === 'sequence' ? 'once' : 'loop',
        sourceLabel: run.reference_motion?.source === 'preset' ? 'Checked movement preset'
          : run.reference_motion?.mode === 'sequence' ? 'Observed one-way step' : null })
    }
    const folder = path.join(root, 'reference/automation-validation/row')
    const row = readJson(path.join(folder, 'result.json'))
    if (!validSummary(row.summary) || !Number.isInteger(row.row_solver?.contact?.frames)) throw Error('Invalid row preview.')
    records.set('barbell-bent-over-row', { folder, summary: row.summary, frames: row.row_solver.contact.frames,
      fps: 30, characterFit: originalFit, hashes: row.artifacts, playback: 'loop', sourceLabel: null })
  } catch {
    records.clear()
    manifestError = 'Bundled preview metadata is unavailable. Restore the checked-in validation files.'
  }

  const verified = (slug) => {
    const record = records.get(slug)
    if (!record) return null
    try {
      const folder = realpathSync(record.folder)
      if (!folder.startsWith(allowedRoot + path.sep)) throw Error('Preview folder escaped the validation assets.')
      const files = Object.keys(ASSETS).map((name) => {
        const file = path.join(folder, name), real = realpathSync(file), stat = statSync(real)
        if (real !== file || !stat.isFile() || !stat.size) throw Error('Missing preview asset.')
        return { name, file, stat }
      })
      const key = files.map(({ name, stat }) => `${name}:${stat.size}:${stat.mtimeMs}:${stat.ctimeMs}`).join('|')
      const old = cache.get(slug)
      if (old?.key === key) return old
      const assets = Object.fromEntries(files.map(({ name, file }) => {
        const bytes = readFileSync(file), expected = record.hashes?.[name]
        if (!/^[a-f\d]{64}$/.test(expected || '') || digest(bytes) !== expected) throw Error('Changed preview asset.')
        if (name.endsWith('.gif') && !/^GIF8[79]a$/.test(bytes.subarray(0, 6).toString())) throw Error('Invalid GIF.')
        if (name.endsWith('.png') && !bytes.subarray(0, 8).equals(Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]))) throw Error('Invalid PNG.')
        return [name, { bytes, type: ASSETS[name], sha256: expected }]
      }))
      const checks = JSON.parse(assets['checks.json'].bytes.toString('utf8'))
      if (!isDeepStrictEqual(checks.summary, record.summary) || !Array.isArray(checks.rows)) throw Error('Mismatched preview checks.')
      const counted = Object.fromEntries(COUNTS.map((key) => [key, key === 'unmeasured'
        ? checks.rows.filter((row) => row.kind === 'not-measured').length
        : checks.rows.filter((row) => row.status === key).length]))
      if (!isDeepStrictEqual(counted, record.summary)) throw Error('Mismatched preview counts.')
      const value = { key, record, assets }
      cache.set(slug, value)
      return value
    } catch {
      cache.delete(slug)
      return null
    }
  }

  const list = (exercises) => exercises.map(({ slug, name }) => {
    const checked = verified(slug)
    const empty = { slug, name, available: false, readOnly: true, summary: null, frames: null, fps: null,
      characterFit: null, gifUrl: null, posterUrl: null, endUrl: null, checksUrl: null, reviewRequired: true,
      playback: null, sourceLabel: null }
    if (!checked) return { ...empty, reason: records.has(slug)
      ? 'The bundled preview files are missing or changed. Restore the checked-in validation files.'
      : manifestError || unavailableReason(slug) }
    const { record } = checked
    const reviewRequired = ['fail', 'limited', 'uncertain', 'unmeasured'].some((key) => record.summary[key] > 0)
    const url = (file) => `/api/previews/${slug}/${file}`
    return { ...empty, available: true, summary: { ...record.summary }, frames: record.frames, fps: record.fps,
      characterFit: { ...record.characterFit }, gifUrl: url('animation.gif'), posterUrl: url('start.png'),
      endUrl: url('end.png'), checksUrl: url('checks.json'), reviewRequired,
      playback: record.playback, sourceLabel: record.sourceLabel,
      reason: record.summary.fail > 0 ? slug === 'barbell-bent-over-row'
        ? `${record.summary.fail} source-photo comparisons still disagree with the steady row hinge. Review the retained failed checks.`
        : 'This example has failed checks that require review.'
        : reviewRequired ? 'Source-photo depth remains inferred. Review this example before approving its form.' : null }
  })
  const asset = (slug, name) => Object.hasOwn(ASSETS, name) ? verified(slug)?.assets[name] ?? null : null
  return { list, asset }
}

export const validatedPreviews = createValidatedPreviews()
