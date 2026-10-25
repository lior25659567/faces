export const CHARACTER_FITS = [
  { value: 'auto', label: 'Automatic', description: 'Use longer arms when the movement needs more reach. Keep the face and original model.' },
  { value: 'original', label: 'Original', description: 'Keep the original proportions. Movements outside its reach will stop for review.' },
  { value: 'fitness', label: 'Fitness', description: 'Use longer arms for this build. Keep the face and original model.' },
]

export const characterFitLabel = (mode) => CHARACTER_FITS.find((fit) => fit.value === mode)?.label || 'Automatic'

export const characterFitResult = (report, requested) => {
  if (!report) return `Requested proportions: ${characterFitLabel(requested)}.`
  if (!report.applied) return 'Original proportions used.'
  const upper = report.upper_scale, fore = report.fore_scale
  const scale = Number.isFinite(upper) && upper > 1 && upper === fore ? ` (${upper.toFixed(1)}× arm length)` : ''
  return `Fitness proportions used${scale}. ${report.reason ? report.reason + ' ' : ''}The face and original model are unchanged.`
}
