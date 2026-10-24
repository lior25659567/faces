const clamp = (value, min, max) => Math.min(max, Math.max(min, value))

// A full turn takes the pointer from the head to the viewport edge on that side,
// and never less than a quarter of the viewport, so a head near an edge isn't twitchy.
const turn = (pointer, head, size) => {
  const offset = pointer - head
  const toEdge = offset < 0 ? head : size - head
  return clamp(offset / Math.max(toEdge, size / 4), -1, 1)
}

// Where the head should look, each axis from -1 (left / up) to 1 (right / down).
export const pointerToGaze = (pointer, head, viewport) => ({
  u: turn(pointer.x, head.x, viewport.width),
  v: turn(pointer.y, head.y, viewport.height),
})

export const gazeToCell = ({ u, v }, cols, rows) => ({
  col: Math.round(((u + 1) / 2) * (cols - 1)),
  row: Math.round(((v + 1) / 2) * (rows - 1)),
})

// Frame-rate independent smoothing: half the remaining distance every `halfLife` ms.
export const ease = (current, target, elapsed, halfLife) =>
  current + (target - current) * (1 - 0.5 ** (elapsed / halfLife))
