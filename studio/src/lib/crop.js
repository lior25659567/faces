// The reference screenshots are whole phone screens with the coach small in them. The reading
// (measure/read_pose.py) saves the body's box as `crop: [x0, y0, x1, y1]`, fractions of the picture,
// with the picture's `size: [w, h]`. Older runs have no crop: they show the whole picture.

const isNumber = (v) => typeof v === 'number' && Number.isFinite(v)
const clamp01 = (v) => Math.min(Math.max(v, 0), 1)
const pct = (v) => `${Math.round(v * 100 * 1000) / 1000}%`

// { box, size } from one image's measurement, or null to show the full picture.
export const readCrop = (measurement) => {
  const crop = measurement?.crop
  const size = measurement?.size
  if (!Array.isArray(crop) || crop.length !== 4 || !crop.every(isNumber)) return null
  if (!Array.isArray(size) || size.length !== 2 || !size.every((v) => isNumber(v) && v > 0)) return null
  const box = crop.map(clamp01)
  if (box[2] - box[0] <= 0.01 || box[3] - box[1] <= 0.01) return null
  return { box, size: [...size] }
}

// Width over height of the cropped part, in real pixels.
export const cropAspect = ({ box: [x0, y0, x1, y1], size: [w, h] }) => ((x1 - x0) * w) / ((y1 - y0) * h)

// object-view-box: inset(top right bottom left), percentages of the picture.
export const viewBoxInset = ([x0, y0, x1, y1]) => `inset(${pct(y0)} ${pct(1 - x1)} ${pct(1 - y1)} ${pct(x0)})`

// Where object-view-box isn't supported: the picture as a background, scaled so the box fills the
// element. A position of p% puts the picture's p% point on the element's p% point, so the box's
// start s (of a box spanning d) lands on the element's edge at p = s / (1 - d).
export const backgroundCrop = ([x0, y0, x1, y1]) => {
  const w = x1 - x0
  const h = y1 - y0
  const at = (start, span) => (span >= 1 ? 0 : start / (1 - span))
  return {
    backgroundSize: `${pct(1 / w)} ${pct(1 / h)}`,
    backgroundPosition: `${pct(at(x0, w))} ${pct(at(y0, h))}`,
  }
}

export const supportsViewBox = () => {
  try {
    return Boolean(globalThis.CSS?.supports?.('object-view-box', 'inset(0% 0% 0% 0%)'))
  } catch {
    return false
  }
}
