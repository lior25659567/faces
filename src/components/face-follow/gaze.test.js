import { describe, expect, test } from 'vitest'
import { ease, gazeToCell, pointerToGaze } from './gaze.js'

const viewport = { width: 1000, height: 800 }
const head = { x: 300, y: 200 }

describe('pointerToGaze', () => {
  test('looks straight ahead when the pointer is on the head', () => {
    expect(pointerToGaze({ x: 300, y: 200 }, head, viewport)).toEqual({ u: 0, v: 0 })
  })

  test.each([
    ['left', { x: 0, y: 200 }, { u: -1, v: 0 }],
    ['right', { x: 1000, y: 200 }, { u: 1, v: 0 }],
    ['top', { x: 300, y: 0 }, { u: 0, v: -1 }],
    ['bottom', { x: 300, y: 800 }, { u: 0, v: 1 }],
  ])('turns fully when the pointer reaches the %s edge of the viewport', (_edge, pointer, want) => {
    expect(pointerToGaze(pointer, head, viewport)).toEqual(want)
  })

  test.each([
    ['halfway to the left edge', { x: 150, y: 200 }, { u: -0.5, v: 0 }],
    ['halfway to the right edge', { x: 650, y: 200 }, { u: 0.5, v: 0 }],
    ['halfway to the top edge', { x: 300, y: 100 }, { u: 0, v: -0.5 }],
    ['halfway to the bottom edge', { x: 300, y: 500 }, { u: 0, v: 0.5 }],
  ])('turns halfway when the pointer is %s', (_where, pointer, want) => {
    expect(pointerToGaze(pointer, head, viewport)).toEqual(want)
  })

  test('stops at a full turn when the pointer is outside the viewport', () => {
    expect(pointerToGaze({ x: -500, y: 2000 }, head, viewport)).toEqual({ u: -1, v: 1 })
  })

  test('a head close to an edge still takes a quarter of the viewport to turn fully', () => {
    const nearRightEdge = { x: 980, y: 400 }
    expect(pointerToGaze({ x: 1000, y: 400 }, nearRightEdge, viewport).u).toBeCloseTo(0.08)
  })

  test('a head on the very edge looks straight ahead at a pointer on top of it', () => {
    const onRightEdge = { x: 1000, y: 400 }
    expect(pointerToGaze({ x: 1000, y: 400 }, onRightEdge, viewport)).toEqual({ u: 0, v: 0 })
  })
})

describe('gazeToCell', () => {
  test('straight ahead is the middle cell', () => {
    expect(gazeToCell({ u: 0, v: 0 }, 11, 11)).toEqual({ col: 5, row: 5 })
  })

  test.each([
    ['up and left', { u: -1, v: -1 }, { col: 0, row: 0 }],
    ['up and right', { u: 1, v: -1 }, { col: 10, row: 0 }],
    ['down and left', { u: -1, v: 1 }, { col: 0, row: 10 }],
    ['down and right', { u: 1, v: 1 }, { col: 10, row: 10 }],
  ])('a full turn %s is that corner of the grid', (_direction, gaze, want) => {
    expect(gazeToCell(gaze, 11, 11)).toEqual(want)
  })

  test('snaps to the nearest cell', () => {
    expect(gazeToCell({ u: 0.09, v: 0 }, 11, 11).col).toBe(5)
    expect(gazeToCell({ u: 0.11, v: 0 }, 11, 11).col).toBe(6)
  })

  test('columns and rows are counted separately', () => {
    expect(gazeToCell({ u: 0.5, v: 1 }, 5, 3)).toEqual({ col: 3, row: 2 })
  })
})

describe('ease', () => {
  test('covers half the remaining distance in one half-life', () => {
    expect(ease(0, 1, 80, 80)).toBeCloseTo(0.5)
  })

  test('covers three quarters in two half-lives', () => {
    expect(ease(0, 1, 160, 80)).toBeCloseTo(0.75)
  })

  test('moves toward smaller targets too', () => {
    expect(ease(1, -1, 80, 80)).toBeCloseTo(0)
  })

  test('does not move when no time has passed', () => {
    expect(ease(0.3, 1, 0, 80)).toBe(0.3)
  })

  test('travels the same distance whether 32 ms arrives as one frame or two', () => {
    // 1 - 0.5 ** (32 / 80)
    const want = 0.242142
    expect(ease(0, 1, 32, 80)).toBeCloseTo(want, 5)
    expect(ease(ease(0, 1, 16, 80), 1, 16, 80)).toBeCloseTo(want, 5)
  })
})
