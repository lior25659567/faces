const DASH = '—'
const MINUS = '−' // a real minus sign, not a hyphen
const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

const isNumber = (v) => typeof v === 'number' && Number.isFinite(v)

// "45 s", "4 min 10 s", "1 h 2 min". A step that never ran has no time: "—", never "0 s".
export const formatDuration = (seconds) => {
  if (!isNumber(seconds)) return DASH
  const total = Math.max(Math.round(seconds), 0)
  if (total < 60) return `${total} s`
  if (total < 3600) {
    const m = Math.floor(total / 60)
    const s = total % 60
    return s ? `${m} min ${s} s` : `${m} min`
  }
  const minutes = Math.round(total / 60)
  const h = Math.floor(minutes / 60)
  const m = minutes % 60
  return m ? `${h} h ${m} min` : `${h} h`
}

// Minutes, rounded: an estimate shown to the second pretends to be more exact than it is.
const roughly = (seconds) => {
  const minutes = Math.max(Math.round(seconds / 60), 1)
  if (minutes < 60) return `${minutes} min`
  const h = Math.floor(minutes / 60)
  const m = minutes % 60
  return m ? `${h} h ${m} min` : `${h} h`
}

// "about 3 min left"; past the estimate it says so, never a negative or frozen "0 s left".
export const formatLeft = ({ seconds, overdue } = {}) => {
  if (overdue) return isNumber(seconds) && seconds >= 60 ? `at least ${roughly(seconds)} left` : 'taking longer than usual'
  if (!isNumber(seconds)) return 'working out the time left'
  if (seconds <= 0) return 'almost done'
  if (seconds < 60) return 'less than a minute left'
  return `about ${roughly(seconds)} left`
}

const signed = (v, digits) => {
  const text = Math.abs(v).toFixed(digits)
  return (v < 0 && Number(text) !== 0 ? MINUS : '') + text
}

// Whole numbers with thousands grouped: "1,613".
const grouped = (v) => {
  const text = Math.round(Math.abs(v)).toLocaleString('en-US')
  return (v < 0 && text !== '0' ? MINUS : '') + text
}

// "47°", "1.87×", "−0.05 torso", "0.65 thigh", "28.9 mm", "1,613 mm²", "13%", "3 frames".
export const formatValue = (value, unit) => {
  if (!isNumber(value)) return DASH
  switch (unit) {
    case 'deg': return `${signed(value, 0)}°`
    case 'x': return `${signed(value, 2)}×`
    case 'torso': return `${signed(value, 2)} torso`
    case 'thigh': return `${signed(value, 2)} thigh`
    case 'mm': return `${signed(value, 1)} mm`
    case 'mm2': return `${grouped(value)} mm²`
    case '%': return `${signed(value, 0)}%`
    case 'frames': return `${value} ${value === 1 ? 'frame' : 'frames'}`
    default: return unit ? `${value} ${unit}` : String(value)
  }
}

const pad = (n) => String(n).padStart(2, '0')
const sameDay = (a, b) => a.getFullYear() === b.getFullYear() && a.getMonth() === b.getMonth() && a.getDate() === b.getDate()

// "Today 09:05", "Yesterday 18:30", "3 Sep 07:00" (local time).
export const formatWhen = (iso, now = new Date()) => {
  if (!iso) return DASH
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return DASH
  const clock = `${pad(d.getHours())}:${pad(d.getMinutes())}`
  if (sameDay(d, now)) return `Today ${clock}`
  const yesterday = new Date(now.getFullYear(), now.getMonth(), now.getDate() - 1)
  if (sameDay(d, yesterday)) return `Yesterday ${clock}`
  return `${d.getDate()} ${MONTHS[d.getMonth()]} ${clock}`
}
