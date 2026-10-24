import { expect, test } from 'vitest'
import { pictureRefusal, unheld } from './kit.js'

// The same table and sentences as studio/measure/kit.py (studio/tests/test_kit.py pins them on the Python side).
const LUNGE = 'No motion for this exercise: its feet step, which the picture path can\'t animate yet. Add lunge.fbx to 3d-character/mixamo (its README says how).'
const needs = (what) => `No motion for this exercise: it needs a ${what} the picture path doesn't build yet.`

test('the picture path refuses the lunge and the exercises it has no support for', () => {
  expect(pictureRefusal('Walking Lunge (dumbbells optional)')).toBe(LUNGE)
  expect(pictureRefusal('Flat Barbell Bench Press')).toBe(needs('bench'))
  expect(pictureRefusal('Pull-Up')).toBe(needs('pull-up bar'))
  expect(pictureRefusal('Leg Press (seated machine)')).toBe(needs('leg-press sled'))
  expect(pictureRefusal('Barbell Back Squat')).toBeNull()
  expect(pictureRefusal('Conventional Barbell Deadlift')).toBeNull()
})

test('what the hands should hold that the path does not build', () => {
  expect(unheld('Conventional Barbell Deadlift')).toBe('bar')
  expect(unheld('Dumbbell Lateral Raise')).toBe('dumbbells')
  expect(unheld('Barbell Back Squat')).toBeNull()
  expect(unheld('Front Plank (forearm plank)')).toBeNull()
})

test('an exercise of our own goes by its equipment word', () => {
  expect(unheld('Back Squat from images', 'barbell')).toBe('bar')
  expect(pictureRefusal('Incline Press', 'bench')).toBe(needs('bench'))
  expect(pictureRefusal('Burpee', 'bodyweight')).toBeNull()
  expect(unheld('Burpee', undefined)).toBeNull()
})
