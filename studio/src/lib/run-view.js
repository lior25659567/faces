import { readCrop } from './crop.js'
import { GROUP_LABELS, statusChip } from './labels.js'

const GROUP_ORDER = ['start', 'middle', 'end', 'all positions', 'body']
const JUDGED = new Set(['pass', 'fail', 'limited'])

const enc = (...parts) => parts.map((p) => String(p).split('/').map(encodeURIComponent).join('/')).join('/')

export const referenceUrl = (slug, file) => `/files/${enc('references', slug, file)}`
export const runFileUrl = (slug, runId, file) => `/files/${enc('runs', slug, runId, file)}`

// Ours faces the other way from the coach: flip it so the two read the same way.
export const shouldMirror = (oursFacing, refFacing) => Boolean(oursFacing && refFacing && oursFacing !== refFacing)

// Comparison anchors in reference order; chronological motion samples stay in the reference strip.
export const buildComparison = ({ slug, runId, images = [], ours, measurements, targets }) => {
  const sorted = [...images].sort((a, b) => a.order - b.order)
  const stills = ours?.stills || []
  const pairs = sorted.filter((img) => !img.contextOnly && !img.motionOnly && !img.source?.motionOnly).map((img) => {
    const still = stills.find((s) => s.image === img.id) || null
    const info = targets?.images?.[img.id]
    const used = typeof info?.used === 'boolean' ? info.used : null
    return {
      id: img.id,
      original: img.original,
      phase: still?.phase ?? img.phase ?? null,
      view: still?.view ?? img.view ?? null,
      frame: still?.frame ?? null,
      refUrl: referenceUrl(slug, img.file),
      oursUrl: still?.file ? runFileUrl(slug, runId, still.file) : null,
      mirror: shouldMirror(still?.facing, measurements?.[img.id]?.facing),
      // The coach's box in the screenshot; null (older runs) shows the whole picture.
      crop: readCrop(measurements?.[img.id]),
      used,
      why: used === false ? info.why ?? null : null,
    }
  })
  const context = sorted
    .filter((img) => img.contextOnly)
    .map((img) => ({ id: img.id, original: img.original, refUrl: referenceUrl(slug, img.file) }))
  return { pairs, context }
}

export const groupRows = (rows = []) => {
  const known = GROUP_ORDER.filter((g) => rows.some((r) => r.group === g))
  const extra = [...new Set(rows.map((r) => r.group))].filter((g) => !GROUP_ORDER.includes(g))
  return [...known, ...extra].map((group) => ({
    group,
    label: GROUP_LABELS[group] ?? group,
    rows: rows.filter((r) => r.group === group),
  }))
}

// A group's judged checks, and its rows that would be judged but weren't measured on the character (the
// amber "Not measured" chips, counted exactly as the header counts them).
export const phaseTally = (rows = [], phase) => {
  const group = rows.filter((r) => r.group === phase)
  const mine = group.filter((r) => JUDGED.has(r.status))
  return {
    pass: mine.filter((r) => r.status === 'pass').length,
    judged: mine.length,
    unmeasured: group.filter((r) => statusChip(r).kind === 'not-measured').length,
  }
}

// The tally's colour: red when anything judged misses, amber (never green) when something wasn't measured.
export const tallyKind = ({ pass, judged, unmeasured }) => {
  if (pass < judged) return 'fail'
  if (unmeasured) return 'unmeasured'
  return judged ? 'pass' : 'none'
}

export const tallyText = ({ pass, judged, unmeasured }) => {
  const parts = []
  if (judged) parts.push(`${pass} of ${judged} pass`)
  if (unmeasured) parts.push(`${unmeasured} not measured`)
  return parts.length ? parts.join(' · ') : null
}

// The comparison's picture pair tally: the checks table's own kind and text for that position (task 19:
// it used to go green above a not-measured row). null when the position has nothing judged or unmeasured.
export const pairTally = (rows, phase) => {
  const tally = phaseTally(rows, phase)
  const text = tallyText(tally)
  return text ? { kind: tallyKind(tally), text } : null
}

// The reference strip's "Not used" reasons: one run's own (its targets.json, or read-problems.json when
// its read step stopped), with which run they are; null when that run hasn't read its pictures yet.
// Never another run's reasons, which may be about other pictures.
export const stripReasons = ({ run, targets, readProblems }) => {
  const images = targets?.images ?? readProblems?.images
  if (!run || !images) return null
  return { runId: run.id, createdAt: run.createdAt, images }
}

// Show the newest finished result by default, so a run in progress doesn't blank the page.
export const defaultRunId = (runs = []) => (runs.find((r) => r.status === 'done') || runs[0])?.id ?? null

// What a run used from the pictures: which ones, in what order, with which tags.
const picturesKey = (images) =>
  [...images]
    .sort((a, b) => a.order - b.order)
    // A view the detector filled in after the run started is not a change the user made.
    .map((i) => `${i.id}:${i.phase}:${i.viewSetBy === 'user' ? i.view : ''}:${i.contextOnly ? 1 : 0}`)
    .join('|')

// Have the pictures changed since the run's snapshot? Without a snapshot there is nothing to compare.
export const samePictures = (current = [], snapshot) => !snapshot || picturesKey(current) === picturesKey(snapshot)
