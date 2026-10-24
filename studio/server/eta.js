export const DEFAULT_SECONDS = { read: 20, build: 150, animate: 90, render: 120, stills: 60, check: 5 }

// Median of the last five finished times for a step, else the default.
export const estimateStep = (id, history = []) => {
  const last = history.filter((n) => Number.isFinite(n)).slice(-5).sort((a, b) => a - b)
  if (!last.length) return DEFAULT_SECONDS[id] ?? 60
  const mid = Math.floor(last.length / 2)
  return last.length % 2 ? last[mid] : (last[mid - 1] + last[mid]) / 2
}

// Time left for the current step plus the estimates of the steps after it. Never negative:
// a step past its estimate is `overdue` (the page says "taking longer than usual").
export const timeLeft = ({ steps, currentId, elapsed, fraction, estimates }) => {
  const ids = steps.map((s) => (typeof s === 'string' ? s : s.id))
  const at = ids.indexOf(currentId)
  const estimate = estimates[currentId] ?? DEFAULT_SECONDS[currentId] ?? 60
  let current
  let overdue = false
  if (fraction > 0.05) {
    current = elapsed / Math.min(fraction, 1) - elapsed
  } else {
    current = Math.max(estimate - elapsed, 0)
    overdue = elapsed > estimate
  }
  const later = at < 0 ? [] : ids.slice(at + 1)
  const rest = later.reduce((sum, id) => sum + (estimates[id] ?? DEFAULT_SECONDS[id] ?? 60), 0)
  return { seconds: Math.max(Math.round(current + rest), 0), overdue }
}
