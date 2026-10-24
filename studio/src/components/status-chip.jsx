import './status-chip.css'

// Colour carries the status; the word always says it too.
const StatusChip = ({ kind, children }) => <span className={`status-chip is-${kind}`}>{children}</span>

export default StatusChip
