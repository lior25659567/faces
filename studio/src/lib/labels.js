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
  pullup_hang_raise_adaptation: 'Extra body rise to clear the fixed bar',
  pullup_elbow_above_bar: 'Elbows above the pull-up bar',
  pullup_elbow_above_wrist: 'Elbows above the wrists',
  pullup_peak_elbow_above_shoulder: 'Elbows above the shoulders at the top',
  pullup_elbow_path_speed: 'Pull-up elbows snapping between frames',
  pullup_return_path_error: 'Pull-up lowering follows a different arm path',
  pulldown_elbow_above_wrist: 'Pulldown elbows above the wrists',
  pulldown_bottom_elbow_above_shoulder: 'Pulldown elbows not pulling down',
  pulldown_bottom_upperarm_excess: 'Pulldown upper arms not beside the torso',
  pulldown_top_extension_shortfall: 'Pulldown return short of arm extension',
  pulldown_travel_shortfall: 'Pulldown bar movement too short',
  press_lockout_shortfall: 'Shoulder press short of arm lockout',
  press_bottom_flexion_excess: 'Shoulder press missing its lower position',
  press_bottom_overfold: 'Shoulder press elbows folded too far',
  press_elbow_above_wrist: 'Shoulder press elbows above the grips',
  press_travel_shortfall: 'Shoulder press movement too short',
  press_peak_stack_excess: 'Shoulder press lockout away from the shoulders',
  press_lockout_head_shortfall: 'Shoulder press lockout below the head',
  press_torso_lean: 'Body leaning during the shoulder press',
  press_foot_slide: 'Shoulder press feet sliding',
  press_elbow_asymmetry: 'Shoulder press arms moving unevenly',
  press_hip_drift: 'Body moving during the shoulder press',
  bench_lockout_shortfall: 'Bench press short of arm lockout',
  bench_bottom_flexion_excess: 'Bench press missing the lower-chest position',
  bench_bottom_overfold: 'Bench press elbows folded too far',
  bench_travel_shortfall: 'Bench press range too short',
  bench_forearm_tilt_excess: 'Bench press forearms tilted away from vertical',
  bench_bar_torso_path_excess: 'Bench bar outside the chest path',
  bench_bottom_chest_shortfall: 'Bench bar not reaching the lower chest',
  bench_lockout_shoulder_offset: 'Bench lockout away from the shoulders',
  bench_stroke_reversal: 'Bench bar changing direction within a stroke',
  bench_stationary_body_drift: 'Supported body moving during the press',
  bench_elbow_asymmetry: 'Bench arms pressing unevenly',
  bench_foot_slide: 'Bench press feet sliding',
  isolation_body_lean_excess: 'Body leaning during the arm exercise',
  isolation_knee_extension_shortfall: 'Holding a crouch during the arm exercise',
  isolation_hip_extension_shortfall: 'Hips bent during the arm exercise',
  isolation_body_drift: 'Body moving during the arm exercise',
  isolation_elbow_asymmetry: 'Arms moving unevenly',
  isolation_upperarm_drift: 'Upper arms moving during the elbow exercise',
  isolation_upperarm_position_excess: 'Upper arms too far from the body',
  isolation_start_extension_shortfall: 'Curl starts with bent elbows',
  isolation_end_flexion_excess: 'Curl stops before full bend',
  isolation_end_overfold: 'Curl elbows folded too far',
  isolation_start_elevation_excess: 'Lateral raise starts too high',
  isolation_end_elevation_shortfall: 'Lateral raise stops below shoulder height',
  isolation_end_elevation_excess: 'Lateral raise goes above shoulder height',
  isolation_elbow_bend_excess: 'Lateral raise elbow bend outside its range',
  isolation_elbow_bend_variation: 'Elbows bending during the lateral raise',
  isolation_start_bend_excess: 'Pushdown starts with elbows too straight',
  isolation_start_overfold: 'Pushdown starts with elbows folded too far',
  isolation_end_extension_shortfall: 'Pushdown short of full elbow extension',
  sled_max_supported_body_drift: 'Supported body moving during the leg press',
  sled_max_hand_drift: 'Hands moving during the leg press',
  sled_top_extension_shortfall: 'Leg press short of knee extension',
  sled_bottom_flexion_shortfall: 'Leg press lowering too shallow',
  sled_bottom_overfold: 'Leg press knees folded too far',
  sled_max_knee_asymmetry: 'Legs pressing unevenly',
  squat_top_knee_extension_shortfall: 'Squat short of knee extension',
  squat_top_hip_extension_shortfall: 'Squat short of hip extension',
  squat_bottom_knee_flexion_excess: 'Squat knee bend too shallow',
  squat_bottom_hip_flexion_excess: 'Squat hip bend too shallow',
  squat_depth_shortfall: 'Squat stops above knee height',
  squat_pelvis_travel_shortfall: 'Squat movement too short',
  squat_knee_asymmetry: 'Squat legs moving unevenly',
  squat_bar_level_excess: 'Squat bar tilting',
  squat_spine_deviation: 'Squat back changing shape',
  squat_loop_position_error: 'Squat does not return to its starting position',
  squat_foot_slide: 'Squat feet sliding',
  lunge_max_contact_gap: 'Lunge heel or toe above its support',
  lunge_max_contact_penetration: 'Lunge heel or toe sinking into the floor',
  lunge_max_anchor_drift: 'Lunge planted foot sliding',
  lunge_max_swing_clearance_shortfall: 'Lunge swinging foot too low',
  lunge_max_leg_length_error: 'Lunge leg stretching',
  lunge_max_backward_knee: 'Lunge knee bending backwards',
  lunge_max_body_floor_penetration: 'Lunge body sinking into the floor',
  lunge_missing_forward_travel: 'Walking lunge missing its forward step',
  lunge_max_reverse_root: 'Walking lunge moving backwards',
  lunge_start_extension_shortfall: 'Lunge starts with bent knees',
  lunge_middle_knee_overfold: 'Lunge knee folded too far',
  lunge_middle_knee_depth_shortfall: 'Lunge lowered position too shallow',
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
  upper_limb_speed: 'Arms snapping between frames',
  upper_limb_turn_speed: 'Arm or wrist rotation snapping between frames',
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
  if (exercise?.motionPreset) return exercise.motionPresets?.find((preset) => preset.id === exercise.motionPreset)?.label || 'Checked movement preset'
  if (exercise?.motionPresets?.length) return 'Movement from reference pictures'
  if (exercise?.movementSequence) return exercise.movementSequence.label
  if (exercise?.name === 'Barbell Bent-Over Row') return "Reference motion with a stable torso and shared bar"
  if (exercise?.clip) return `Motion from ${exercise.clip}`
  if (exercise?.pictureRefusal) return exercise.pictureRefusal
  const holding = exercise?.unheld ? `, holding no ${exercise.unheld}` : ''
  const add = exercise?.expectedClip ? ` (add ${exercise.expectedClip} to 3d-character/mixamo for a captured one)` : ''
  return `No motion clip: moves between the pictures' positions${holding}${add}`
}
