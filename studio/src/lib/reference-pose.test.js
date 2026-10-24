import { expect, test } from 'vitest'
import { referencePose } from './reference-pose.js'

const reading = () => ({ size: [640, 480], pixels: {
  left_shoulder: [100, 80], left_elbow: [120, 160], left_wrist: [140, 240],
}, visibility: { left_shoulder: 0.9, left_elbow: 0.3, left_wrist: 0.8 } })

test('overlay preserves original pixel coordinates and connects only measured joints', () => {
  const pose = referencePose(reading())
  expect([pose.width, pose.height]).toEqual([640, 480])
  expect(pose.points[0]).toMatchObject({ name: 'left_shoulder', x: 100, y: 80 })
  expect(pose.bones.map((b) => b.id)).toEqual(['left_shoulder-left_elbow', 'left_elbow-left_wrist'])
  expect(pose.missingJoints).toBe(9)
})

test('a low visibility joint marks both incident limbs uncertain', () => {
  const pose = referencePose(reading())
  expect(pose.lowConfidence).toBe(1)
  expect(pose.bones.every((b) => b.confidence === 'low')).toBe(true)
  expect(pose.points.find((p) => p.name === 'left_shoulder').confidence).toBe('seen')
})

test('old measurements without visibility remain explicitly unknown', () => {
  const raw = reading()
  delete raw.visibility
  const pose = referencePose(raw)
  expect(pose.lowConfidence).toBe(0)
  expect(pose.unknownConfidence).toBe(3)
  expect(pose.bones.every((b) => b.confidence === 'unknown')).toBe(true)
})

test('missing, failed or invalid size measurements cannot display a pose', () => {
  for (const raw of [null, {}, { ...reading(), error: 'No body' },
    ...[[0, 480], [640, Infinity], ['640', 480], [640], [-2, 480]].map((size) => ({ ...reading(), size }))]) {
    expect(referencePose(raw)).toBeNull()
  }
})

test('invalid or off-image joints are omitted rather than drawing plausible false endpoints', () => {
  const raw = reading()
  Object.assign(raw.pixels, { left_shoulder: [NaN, 80], left_elbow: [-10, 160], right_knee: [100, 481], right_hip: ['1', 2] })
  const pose = referencePose(raw)
  expect(pose.points.map((p) => p.name)).toEqual(['left_wrist'])
  expect(pose.bones).toEqual([])
  expect(referencePose({ ...reading(), pixels: {} })).toBeNull()
})

test('invalid visibility never appears as a confident detection', () => {
  const raw = reading()
  raw.visibility = { left_shoulder: NaN, left_elbow: 4, left_wrist: -1 }
  expect(referencePose(raw).unknownConfidence).toBe(3)
})
