// Display the detector's pixel coordinates on the original image. This is a visual aid,
// not a verdict on the exercise's form or the accuracy of inferred 3D depth.
const JOINTS = ['shoulder', 'elbow', 'wrist', 'hip', 'knee', 'ankle']
const NAMES = ['left', 'right'].flatMap((side) => JOINTS.map((joint) => `${side}_${joint}`))
const BONES = [
  ...['left', 'right'].flatMap((side) => [
    ['shoulder', 'elbow'], ['elbow', 'wrist'], ['shoulder', 'hip'], ['hip', 'knee'], ['knee', 'ankle'],
  ].map(([a, b]) => [`${side}_${a}`, `${side}_${b}`])),
  ['left_shoulder', 'right_shoulder'], ['left_hip', 'right_hip'],
]

export const REFERENCE_VISIBILITY_WARNING = 0.5

export const referencePose = (measurement) => {
  const size = measurement?.size
  if (measurement?.error || !Array.isArray(size) || size.length !== 2 ||
      !size.every((v) => Number.isFinite(v) && v > 0)) return null
  const [width, height] = size
  const points = NAMES.flatMap((name) => {
    const p = measurement?.pixels?.[name]
    if (!Array.isArray(p) || !Number.isFinite(p[0]) || !Number.isFinite(p[1]) ||
        p[0] < 0 || p[0] > width || p[1] < 0 || p[1] > height) return []
    const raw = measurement?.visibility?.[name]
    const visibility = Number.isFinite(raw) && raw >= 0 && raw <= 1 ? raw : null
    return [{ name, x: p[0], y: p[1], visibility,
      confidence: visibility === null ? 'unknown' : visibility < REFERENCE_VISIBILITY_WARNING ? 'low' : 'seen' }]
  })
  if (!points.length) return null
  const byName = Object.fromEntries(points.map((p) => [p.name, p]))
  const bones = BONES.flatMap(([a, b]) => {
    if (!byName[a] || !byName[b]) return []
    const ends = [byName[a], byName[b]]
    return [{ id: `${a}-${b}`, a: ends[0], b: ends[1],
      confidence: ends.some((p) => p.confidence === 'low') ? 'low'
        : ends.some((p) => p.confidence === 'unknown') ? 'unknown' : 'seen' }]
  })
  return { width, height, points, bones,
    lowConfidence: points.filter((p) => p.confidence === 'low').length,
    unknownConfidence: points.filter((p) => p.confidence === 'unknown').length,
    missingJoints: NAMES.length - points.length }
}
