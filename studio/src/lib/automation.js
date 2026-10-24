const STAGES = {
  waiting: ['Waiting', 'waiting'],
  ready: ['References ready (not built)', 'waiting'],
  fetching: ['Finding references', 'running'],
  checking: ['Checking references', 'running'],
  building: ['Building animation', 'running'],
  passed: ['Checks passed', 'pass'],
  'needs-review': ['Needs review', 'review'],
  blocked: ['Blocked', 'blocked'],
  failed: ['Failed', 'fail'],
  cancelled: ['Cancelled', 'waiting'],
}
const IN_PROGRESS = new Set(['waiting', 'fetching', 'checking', 'building'])
const count = (value) => Number.isInteger(value) && value >= 0 ? value : 0

// Finishing a process does not establish a passing exercise. Keep incomplete or
// uncertain checks out of the green automation results, even if a status is stale.
export const automationChecksPassed = (summary) => Boolean(summary && count(summary.pass) > 0 &&
  ['fail', 'limited', 'unmeasured', 'uncertain'].every((key) => Number.isInteger(summary[key]) && summary[key] === 0) &&
  count(summary.uncertain) === 0)

export const automationItemState = (item) => {
  const status = item?.status === 'passed' && !automationChecksPassed(item.summary) ? 'needs-review' : item?.status
  const [label, kind] = STAGES[status] || ['Needs review', 'review']
  return { status, label, kind }
}

export const automationProgress = (job) => {
  const items = job?.items || []
  const finished = items.filter((item) => !IN_PROGRESS.has(item.status)).length
  return { finished, total: items.length, text: `${finished} of ${items.length} processed` }
}

export const automationCheckLine = (summary) => {
  if (!summary || !Number.isInteger(summary.pass)) return null
  const parts = [`${count(summary.pass)} pass`]
  for (const [key, label] of [['fail', 'fail'], ['limited', 'limited by proportions'], ['uncertain', 'uncertain'], ['unmeasured', 'not measured']]) {
    if (count(summary[key])) parts.push(`${summary[key]} ${label}`)
  }
  return parts.join(' · ')
}

export const automationSourceUrl = (item) => {
  const candidate = item?.sourceUrl || item?.referenceSource?.url || item?.source?.repository
  try {
    const url = new URL(candidate)
    return url.protocol === 'https:' && !url.username && !url.password ? url.href : null
  } catch { return null }
}

export const automationJobLine = (job) => {
  if (!job) return 'No batch has run yet.'
  const progress = automationProgress(job).text
  if (job.status === 'running') return job.cancelRequested ? `${progress}. Stopping after the current exercise.` : `${progress}.`
  if (job.status === 'cancelled') return `${progress}. Remaining exercises were cancelled.`
  if (job.status === 'interrupted') return `${progress}. The Studio stopped before this batch finished.`
  const review = job.items?.filter((item) => ['review', 'blocked', 'fail'].includes(automationItemState(item).kind)).length || 0
  return `${progress}. ${review ? `${review} need attention.` : 'Review the results below.'}`
}
