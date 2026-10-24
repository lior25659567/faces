import { describe, expect, test } from 'vitest'
import {
  GROUP_LABELS, INFO_KIND_LABELS, METRIC_LABELS, PHASE_LABELS, RUN_STATUS_LABELS, STATUS_LABELS, STEP_LABELS, VIEW_LABELS,
  label, motionSource, statusChip, uncertainHint,
} from './labels.js'

describe('METRIC_LABELS', () => {
  test('plain words for every check', () => {
    expect(METRIC_LABELS).toMatchObject({
      knee_included: 'Knee angle',
      hip_included: 'Hip angle',
      torso_lean: 'Back lean from upright',
      elbow_included: 'Elbow angle',
      upperarm_vs_torso: 'Upper arm away from the body',
      grip_ratio: 'Grip width (× shoulder width)',
      elbow_width_ratio: 'Elbow width (× shoulder width)',
      hip_below_knee: 'Depth: hip below knee',
      cracks: 'Cracks in the skin',
      feet_slide_left: 'Left foot sliding',
      feet_slide_right: 'Right foot sliding',
      feet_in_floor: 'Feet sinking into the floor',
      hands_on_bar: 'Hands on the bar',
      skin_stretch: 'Skin stretch at the deepest point',
      smooth_motion: 'Smooth motion (no jumps between frames)',
      folded_skin: 'Skin folding over',
    })
  })

  test('posture checks read in plain words', () => {
    expect(METRIC_LABELS.back_rounding).toBe('Back straight (rounding)')
    expect(METRIC_LABELS.neck_vs_torso).toBe('Head in line with the back')
    expect(METRIC_LABELS.head_pitch).toBe('Where the face points')
  })

  test("the depth as a judge sees it has a label (task 22), and an older run's depth row keeps its own", () => {
    expect(METRIC_LABELS.seat_below_knee).toBe('Depth: seat below the top of the knee')
    expect(METRIC_LABELS.thigh_outline).toBe("Depth: the thigh's line, as it looks")
  })

  test('the hands holding nothing has a label (task 9 review)', () => {
    expect(METRIC_LABELS.equipment).toBe('Equipment in the hands')
  })

  test('the ids checks.py really emits for the feet have labels too', () => {
    expect(METRIC_LABELS.foot_slide_left).toBe('Left foot sliding')
    expect(METRIC_LABELS.foot_slide_right).toBe('Right foot sliding')
  })
})

describe('label', () => {
  test('known ids get their plain words', () => {
    expect(label('knee_included')).toBe('Knee angle')
  })

  test('unknown ids fall back to the id', () => {
    expect(label('wrist_twist')).toBe('wrist_twist')
  })
})

test('STATUS_LABELS', () => {
  // an info row has no verdict: never "Measured" (a not-measured row would read as measured)
  expect(STATUS_LABELS).toEqual({
    pass: 'Pass', fail: 'Fail', limited: 'Limited by proportions', uncertain: 'Uncertain', info: 'Not judged',
  })
})

describe('statusChip', () => {
  test('an info row shows its kind, as checks.json says it', () => {
    expect(INFO_KIND_LABELS).toEqual({ 'not-measured': 'Not measured', 'not-judged': 'Not judged', 'shown-only': 'Shown only' })
    expect(statusChip({ status: 'info', kind: 'not-measured' })).toEqual({ kind: 'not-measured', label: 'Not measured' })
    expect(statusChip({ status: 'info', kind: 'not-judged' })).toEqual({ kind: 'not-judged', label: 'Not judged' })
    expect(statusChip({ status: 'info', kind: 'shown-only' })).toEqual({ kind: 'shown-only', label: 'Shown only' })
  })

  test('an older run\'s info row (no kind) is "Not judged", never guessed from its note', () => {
    expect(statusChip({ status: 'info', note: 'Not measured on the character' })).toEqual({ kind: 'not-judged', label: 'Not judged' })
  })

  test('a face-direction or skin-stretch row never shows the amber "Not measured" (task 18)', () => {
    // checks.py never counts them as not measured; an older run's checks.json still said so on the row
    expect(statusChip({ status: 'info', kind: 'not-measured', metric: 'head_pitch' })).toEqual({ kind: 'shown-only', label: 'Shown only' })
    expect(statusChip({ status: 'info', kind: 'not-measured', metric: 'skin_stretch' })).toEqual({ kind: 'shown-only', label: 'Shown only' })
    expect(statusChip({ status: 'info', kind: 'not-measured', metric: 'neck_vs_torso' })).toEqual({ kind: 'not-measured', label: 'Not measured' })
  })

  test('judged rows keep their status', () => {
    expect(statusChip({ status: 'fail', kind: null })).toEqual({ kind: 'fail', label: 'Fail' })
    expect(statusChip({ status: 'limited' })).toEqual({ kind: 'limited', label: 'Limited by proportions' })
  })
})

test('PHASE_LABELS', () => {
  expect(PHASE_LABELS).toEqual({ start: 'Start', middle: 'Middle', end: 'End' })
})

test('GROUP_LABELS cover the groups checks.py writes', () => {
  expect(GROUP_LABELS).toEqual({
    start: 'Start', middle: 'Middle', end: 'End', 'all positions': 'All positions', body: 'Body',
  })
})

test('STEP_LABELS in the order the run does them', () => {
  expect(Object.entries(STEP_LABELS)).toEqual([
    ['read', 'Reading the reference'],
    ['build', 'Building the character'],
    ['animate', 'Animating'],
    ['render', 'Rendering the GIF'],
    ['stills', 'Taking the comparison stills'],
    ['check', 'Checking against the reference'],
  ])
})

test('VIEW_LABELS include the front-or-behind case the user must pick', () => {
  expect(VIEW_LABELS.side).toBe('Side')
  expect(VIEW_LABELS.frontal).toBe('Front or behind?')
  expect(VIEW_LABELS['three-quarter']).toBe('Three-quarter')
})

test('RUN_STATUS_LABELS', () => {
  expect(RUN_STATUS_LABELS).toEqual({
    queued: 'Waiting', running: 'Running', done: 'Done', failed: 'Failed', interrupted: 'Interrupted',
  })
})

describe('uncertainHint (task 19: whose two readings disagree)', () => {
  test('the coach\'s two readings, as before, also for rows written before the field', () => {
    expect(uncertainHint({ status: 'uncertain', uncertain_why: 'coach' }))
      .toBe('The coach readings disagree, so judge this one by eye in the pictures above.')
    expect(uncertainHint({ status: 'uncertain' }))
      .toBe('The coach readings disagree, so judge this one by eye in the pictures above.')
  })

  test('ours: our picture and our rig', () => {
    expect(uncertainHint({ status: 'uncertain', uncertain_why: 'ours' }))
      .toBe('Our picture and our rig disagree, so judge this one by eye in the pictures above.')
  })

  test('no hint on a row that is not uncertain', () => {
    expect(uncertainHint({ status: 'pass', uncertain_why: null })).toBeNull()
  })
})

describe('motionSource (task 9: no clip moves between the pictures)', () => {
  test('a clip names it', () => {
    expect(motionSource({ clip: 'squat.fbx', expectedClip: 'squat.fbx' })).toBe('Motion from squat.fbx')
  })

  test('no clip and the picture path refuses it: the refusal, never "moves" (task 9 review)', () => {
    const lunge = 'No motion for this exercise: its feet step, which the picture path can\'t animate yet. Add lunge.fbx to 3d-character/mixamo (its README says how).'
    expect(motionSource({ clip: null, expectedClip: 'lunge.fbx', pictureRefusal: lunge })).toBe(lunge)
    expect(motionSource({ clip: 'lunge.fbx', expectedClip: 'lunge.fbx', pictureRefusal: lunge })).toBe('Motion from lunge.fbx')
  })

  test('no clip, hands empty on this path: says what they don\'t hold', () => {
    expect(motionSource({ clip: null, expectedClip: 'deadlift.fbx', unheld: 'bar' }))
      .toBe('No motion clip: moves between the pictures\' positions, holding no bar (add deadlift.fbx to 3d-character/mixamo for a captured one)')
  })

  test('no clip: the pictures\' positions, and which clip would replace them', () => {
    expect(motionSource({ clip: null, expectedClip: 'deadlift.fbx' }))
      .toBe('No motion clip: moves between the pictures\' positions (add deadlift.fbx to 3d-character/mixamo for a captured one)')
    expect(motionSource({ clip: null, expectedClip: null })).toBe('No motion clip: moves between the pictures\' positions')
  })
})
