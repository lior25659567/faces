import './checks-table.css'
import StatusChip from './status-chip.jsx'
import { formatValue } from '../lib/format.js'
import { label, statusChip, uncertainHint } from '../lib/labels.js'
import { groupRows, phaseTally, tallyKind, tallyText } from '../lib/run-view.js'
import { checksLine, unmeasuredLine } from '../lib/status.js'

// Body checks compare against a limit, not the coach: "up to 1.0 mm", "none" for jumpy frames.
const coachText = (row) => {
  if (row.group !== 'body') return formatValue(row.target, row.unit)
  if (row.target === null || row.target === undefined) return '—'
  if (row.unit === 'frames' && row.target === 0) return 'none'
  return `up to ${formatValue(row.target, row.unit)}`
}

const Row = ({ row }) => {
  const chip = statusChip(row)
  return (
    <tr className={`checks-row is-${row.status}${chip.kind !== row.status ? ` is-${chip.kind}` : ''}`}>
      <th scope="row" className="checks-name">{label(row.metric)}</th>
      <td className="checks-num num" data-label={row.group === 'body' ? 'Allowed' : 'Coach'}>{coachText(row)}</td>
      <td className="checks-num num" data-label="Ours" title={row.frame !== null && row.frame !== undefined ? `Frame ${row.frame}` : undefined}>
        {formatValue(row.ours, row.unit)}
      </td>
      <td className="checks-status" data-label="Status">
        <StatusChip kind={chip.kind}>{chip.label}</StatusChip>
      </td>
      <td className="checks-note" data-label="Note">
        {row.note}
        {row.status === 'uncertain' && <span className="checks-hint">{uncertainHint(row)}</span>}
      </td>
    </tr>
  )
}

const ChecksTable = ({ run, rows, loading }) => {
  const groups = groupRows(rows)
  const line = checksLine(run.summary)
  const unmeasured = unmeasuredLine(run.summary)

  return (
    <div className="checks">
      <div className="section-head">
        <h2 className="section-title" id="checks-title">Checks</h2>
        {line && (
          <p className="section-note num">
            {line}
            {unmeasured && <span className="checks-unmeasured"> · {unmeasured}</span>}
          </p>
        )}
      </div>

      {rows.length === 0 ? (
        <p className="checks-empty">
          {loading
            ? 'Loading the checks…'
            : run.status === 'done'
              ? 'This run wrote no checks.'
              : run.status === 'failed' || run.status === 'interrupted'
                ? 'This run stopped before the checks.'
                : 'Checks appear when a run finishes.'}
        </p>
      ) : (
        <div className="checks-scroll">
          <table className="checks-table">
            <thead>
              <tr>
                <th scope="col">Check</th>
                <th scope="col" className="checks-num">Coach</th>
                <th scope="col" className="checks-num">Ours</th>
                <th scope="col">Status</th>
                <th scope="col">Note</th>
              </tr>
            </thead>
            {groups.map((g) => {
              const tally = phaseTally(rows, g.group)
              const text = tallyText(tally)
              return (
                <tbody key={g.group}>
                  <tr className="checks-group">
                    <th colSpan={5} scope="rowgroup">
                      <span>{g.label}</span>
                      {g.group === 'body' && <span className="checks-group-note">limits, not coach numbers</span>}
                      {text && <span className={`checks-group-tally num is-${tallyKind(tally)}`}>{text}</span>}
                    </th>
                  </tr>
                  {g.rows.map((r) => <Row key={`${r.group}-${r.metric}`} row={r} />)}
                </tbody>
              )
            })}
          </table>
        </div>
      )}
    </div>
  )
}

export default ChecksTable
