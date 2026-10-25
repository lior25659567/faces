const validCount = (value) => Number.isInteger(value) && value >= 0

// An example with inferred depth is still review work, even when nothing failed.
export const previewStatus = (example) => {
  if (!example?.available) return { kind: 'unavailable', label: 'No built preview' }
  const summary = example.summary
  if (!summary || !['pass', 'fail', 'limited', 'uncertain', 'unmeasured'].every((key) => validCount(summary[key]))) {
    return { kind: 'review', label: 'Review checks' }
  }
  if (summary.fail) return { kind: 'fail', label: `${summary.fail} failed ${summary.fail === 1 ? 'check' : 'checks'}` }
  if (summary.unmeasured) return { kind: 'review', label: `${summary.unmeasured} missing measurements` }
  if (summary.uncertain || summary.limited) return { kind: 'review', label: 'Needs visual review' }
  return summary.pass ? { kind: 'pass', label: 'Checks passed' } : { kind: 'review', label: 'Review checks' }
}
