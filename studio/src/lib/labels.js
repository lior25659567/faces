// Plain words for everything the run writes. The user is not a rigging specialist.

export const METRIC_LABELS = {
  knee_included: 'Knee angle',
  hip_included: 'Hip angle',
  torso_lean: 'Back lean from upright',
  elbow_included: 'Elbow angle',
  upperarm_vs_torso: 'Upper arm away from the body',
  grip_ratio: 'Grip width (× shoulder width)',
  elbow_width_ratio: 'Elbow width (× shoulder width)',
  hip_below_knee: 'Depth: hip below knee',
  seat_below_knee: 'Depth: seat below the top of the knee',
  thigh_outline: "Depth: the thigh's line, as it looks",     // runs checked before task 22
  // posture, measured on the side views like the coach's (studio/measure/posture.py)
  back_rounding: 'Back straight (rounding)',
  neck_vs_torso: 'Head in line with the back',
  head_pitch: 'Where the face points',
  cracks: 'Cracks in the skin',
  // checks.py writes foot_slide_*; the plan called them feet_slide_*. Both read the same.
  foot_slide_left: 'Left foot sliding',
  foot_slide_right: 'Right foot sliding',
  feet_slide_left: 'Left foot sliding',
  feet_slide_right: 'Right foot sliding',
  feet_in_floor: 'Feet sinking into the floor',
  hands_on_bar: 'Hands on the bar',
  equipment: 'Equipment in the hands',
  skin_stretch: 'Skin stretch at the deepest point',
  smooth_motion: 'Smooth motion (no jumps between frames)',
  folded_skin: 'Skin folding over',
}

export const label = (metric) => METRIC_LABELS[metric] ?? metric

export const STATUS_LABELS = {
  pass: 'Pass',
  fail: 'Fail',
  limited: 'Limited by proportions',
  uncertain: 'Uncertain',
  info: 'Not judged',
}

// Why an info row has no verdict: checks.json says (`kind`), the page never guesses from the note.
export const INFO_KIND_LABELS = {
  'not-measured': 'Not measured',
  'not-judged': 'Not judged',
  'shown-only': 'Shown only',
}

// Metrics only ever shown (checks.py SHOWN_ONLY_METRICS): never counted as not measured, so never the
// amber chip either. Older runs' checks.json gave them "not-measured" without a reading.
export const SHOWN_ONLY_METRICS = new Set(['head_pitch', 'skin_stretch'])

// The chip for one checks row: its status, or for an info row its kind. Rows from runs made before
// kinds existed have none: "Not judged" is true of every info row.
export const statusChip = (row) => {
  if (row.status !== 'info') return { kind: row.status, label: STATUS_LABELS[row.status] ?? row.status }
  let kind = INFO_KIND_LABELS[row.kind] ? row.kind : 'not-judged'
  if (kind === 'not-measured' && SHOWN_ONLY_METRICS.has(row.metric)) kind = 'shown-only'
  return { kind, label: INFO_KIND_LABELS[kind] }
}

export const PHASE_LABELS = { start: 'Start', middle: 'Middle', end: 'End' }

export const GROUP_LABELS = { ...PHASE_LABELS, 'all positions': 'All positions', body: 'Body' }

// The same words the server's steps.js uses, in run order.
export const STEP_LABELS = {
  read: 'Reading the reference',
  build: 'Building the character',
  animate: 'Animating',
  render: 'Rendering the GIF',
  stills: 'Taking the comparison stills',
  check: 'Checking against the reference',
}

// "frontal": the detector saw a front or a back but cannot tell which, so the user picks.
export const VIEW_LABELS = {
  side: 'Side',
  front: 'Front',
  behind: 'Behind',
  'three-quarter': 'Three-quarter',
  frontal: 'Front or behind?',
}

export const RUN_STATUS_LABELS = {
  queued: 'Waiting',
  running: 'Running',
  done: 'Done',
  failed: 'Failed',
  interrupted: 'Interrupted',
}

export const EQUIPMENT = ['barbell', 'dumbbells', 'bench', 'cable', 'machine', 'bodyweight', 'other']

// An uncertain row's hint: whose two readings disagree (checks.py `uncertain_why`, task 19): the coach's
// (flat and 3-D; rows written before the field are the coach's) or ours (our picture and our rig).
const UNCERTAIN_HINTS = {
  coach: 'The coach readings disagree, so judge this one by eye in the pictures above.',
  ours: 'Our picture and our rig disagree, so judge this one by eye in the pictures above.',
}

export const uncertainHint = (row) =>
  row?.status === 'uncertain' ? UNCERTAIN_HINTS[row.uncertain_why] ?? UNCERTAIN_HINTS.coach : null

// Where an exercise's motion comes from, for its page header: its Mixamo clip, or without one the reference
// pictures' key positions (studio/server/steps.js --keyposes; pipeline/blender/keypose_clip.py, task 9) - unless
// the picture path refuses it (its feet step, or a bench/seat it doesn't build: the server's `pictureRefusal`,
// studio/measure/exercise-kit.json), and saying when the hands hold nothing there (`unheld`).
export const motionSource = (exercise) => {
  if (exercise?.clip) return `Motion from ${exercise.clip}`
  if (exercise?.pictureRefusal) return exercise.pictureRefusal
  const holding = exercise?.unheld ? `, holding no ${exercise.unheld}` : ''
  const add = exercise?.expectedClip ? ` (add ${exercise.expectedClip} to 3d-character/mixamo for a captured one)` : ''
  return `No motion clip: moves between the pictures' positions${holding}${add}`
}
