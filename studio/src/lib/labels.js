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
  equipment_contact: 'Measured equipment contact',
  equipment_anchor_error: 'Hand anchor error',
  equipment_path_adaptation: 'Equipment path changed for reach',
  equipment_body_overlap: 'Equipment overlapping the body',
  body_support_contact: 'Measured body support',
  physical_profile: 'Required movement and contact checks',
  support_motion: 'Supported machine motion',
  sled_rail_deviation: 'Sled deviation from its rails',
  weight_head_body_overlap: 'Weight heads overlapping the body',
  overhead_shortfall: 'Handles below the completed overhead position',
  pullup_chin_shortfall: 'Chin below the top of the pull-up bar',
  pullup_extension_shortfall: 'Elbows short of a full dead hang',
  pullup_bar_drift: 'Pull-up bar moving',
  pullup_bar_rotation: 'Pull-up bar rotating',
  pullup_hang_body_adaptation: 'Body lowered to complete the dead hang',
  pullup_elbow_above_bar: 'Elbows above the pull-up bar',
  pullup_elbow_above_wrist: 'Elbows above the wrists',
  pullup_peak_elbow_above_shoulder: 'Elbows above the shoulders at the top',
  deadlift_elbow_extension_shortfall: 'Bent lifting arms',
  deadlift_start_hip_excess: 'Hips too upright at the start',
  deadlift_start_knee_excess: 'Knees too straight at the start',
  deadlift_hip_lockout_shortfall: 'Hips short of standing lockout',
  deadlift_knee_lockout_shortfall: 'Knees short of standing lockout',
  deadlift_lockout_torso_lean: 'Torso lean at deadlift lockout',
  deadlift_start_bar_height_shortfall: 'Bar too high at the start',
  deadlift_grip_outside_knees_shortfall: 'Grip too far inside the knees',
  deadlift_bar_travel_shortfall: 'Bar short of a full lift',
  deadlift_missing_loaded_plates: 'Missing full-size plates',
  deadlift_plate_start_floor_gap: 'Loaded plates above the floor at the start',
  deadlift_plate_floor_penetration: 'Loaded plates sinking into the floor',
  skin_stretch: 'Skin stretch at the deepest point',
  smooth_motion: 'Smooth motion (no jumps between frames)',
  folded_skin: 'Skin folding over',
}

export const label = (metric) => {
  if (METRIC_LABELS[metric]) return METRIC_LABELS[metric]
  const contact = /^(equipment_gap|equipment_penetration|foot_floor_gap)_(left|right)$/.exec(metric)
  if (contact) return `${contact[2] === 'left' ? 'Left' : 'Right'} ${
    { equipment_gap: 'hand gap at the handle', equipment_penetration: 'hand inside the handle', foot_floor_gap: 'foot above the floor' }[contact[1]]}`
  const support = /^support_(gap|penetration)_(.+)$/.exec(metric)
  if (support) return `${support[2].replaceAll('_', ' ')} ${support[1] === 'gap' ? 'gap at support' : 'inside support'}`
  return metric
}

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
  if (exercise?.name === 'Barbell Bent-Over Row') return "Reference motion with a stable torso and shared bar"
  if (exercise?.clip) return `Motion from ${exercise.clip}`
  if (exercise?.pictureRefusal) return exercise.pictureRefusal
  const holding = exercise?.unheld ? `, holding no ${exercise.unheld}` : ''
  const add = exercise?.expectedClip ? ` (add ${exercise.expectedClip} to 3d-character/mixamo for a captured one)` : ''
  return `No motion clip: moves between the pictures' positions${holding}${add}`
}
