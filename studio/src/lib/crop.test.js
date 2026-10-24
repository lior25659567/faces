import { expect, test } from 'vitest'
import { backgroundCrop, cropAspect, readCrop, viewBoxInset } from './crop.js'

const near = (text) => text.split(' ').map((p) => Number(p.replace('%', '')))

test('readCrop takes a box and the image size from a measurement; anything else shows the full image', () => {
  expect(readCrop({ crop: [0.3, 0.4, 0.55, 0.78], size: [1260, 2736] })).toEqual({ box: [0.3, 0.4, 0.55, 0.78], size: [1260, 2736] })
  // older runs: no crop at all
  expect(readCrop({ size: [1260, 2736] })).toBeNull()
  expect(readCrop(undefined)).toBeNull()
  expect(readCrop({ crop: null, size: [1, 1] })).toBeNull()
  // unusable boxes and sizes
  expect(readCrop({ crop: [0.5, 0.5, 0.5, 0.9], size: [10, 10] })).toBeNull()
  expect(readCrop({ crop: [0.1, 0.1, 0.9], size: [10, 10] })).toBeNull()
  expect(readCrop({ crop: [0.1, 0.1, 0.9, 'x'], size: [10, 10] })).toBeNull()
  expect(readCrop({ crop: [0.1, 0.1, 0.9, 0.9] })).toBeNull()
  // a box a little outside the picture is clamped to it
  expect(readCrop({ crop: [-0.1, 0.2, 1.2, 0.9], size: [100, 100] }).box).toEqual([0, 0.2, 1, 0.9])
})

test('the cropped picture keeps its real shape', () => {
  // 0.25 of 1260 wide by 0.38 of 2736 tall
  expect(cropAspect({ box: [0.3, 0.4, 0.55, 0.78], size: [1260, 2736] })).toBeCloseTo((0.25 * 1260) / (0.38 * 2736), 6)
})

test('object-view-box: inset(top right bottom left) as percentages of the picture', () => {
  expect(near(viewBoxInset([0.3, 0.4, 0.55, 0.78]).slice(6, -1))).toEqual([40, 45, 22, 30])
  expect(viewBoxInset([0, 0, 1, 1])).toBe('inset(0% 0% 0% 0%)')
})

test('background fallback: the box fills the element exactly', () => {
  const box = [0.3, 0.4, 0.55, 0.78]
  const { backgroundSize, backgroundPosition } = backgroundCrop(box)
  const [sw, sh] = near(backgroundSize)
  const [px, py] = near(backgroundPosition)
  expect(sw).toBeCloseTo(400, 3) // 1 / 0.25
  expect(sh).toBeCloseTo(100 / 0.38, 3)
  // For an element of width W, the picture is sw% * W wide and sits at px% * (W - picture width).
  // Its left edge must then be x0 of the picture to the left of the element: offset = -x0 * picture width.
  const W = 300, H = 456
  const pw = (sw / 100) * W, ph = (sh / 100) * H
  // (percentages are rounded to 1/1000 %: well under a hundredth of a pixel)
  expect((px / 100) * (W - pw)).toBeCloseTo(-0.3 * pw, 2)
  expect((py / 100) * (H - ph)).toBeCloseTo(-0.4 * ph, 2)
})

test('background fallback: a box as wide as the picture sits at 0%, never divides by zero', () => {
  const { backgroundSize, backgroundPosition } = backgroundCrop([0, 0.25, 1, 0.75])
  expect(near(backgroundSize)).toEqual([100, 200])
  expect(near(backgroundPosition)[0]).toBe(0)
  expect(near(backgroundPosition)[1]).toBeCloseTo(50, 6)
})
