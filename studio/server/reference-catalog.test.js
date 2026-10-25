import { expect, test } from 'vitest'
import { readFileSync } from 'node:fs'
import { getReferenceSpec, referenceImageFor, REFERENCE_CATALOG, REFERENCE_SOURCE } from './reference-catalog.js'

const poses = JSON.parse(readFileSync(new URL('../../reference/exercises/poses.json', import.meta.url), 'utf8')).exercises

test('all fourteen known exercises have reviewed exact variants at one pinned revision', () => {
  expect(Object.keys(REFERENCE_CATALOG)).toHaveLength(14)
  for (const { name } of poses) {
    const spec = getReferenceSpec(name)
    expect(spec?.exercise).toBe(name)
    expect(spec.commit).toMatch(/^[a-f0-9]{40}$/)
    expect(spec.commit).toBe(REFERENCE_SOURCE.commit)
    expect(spec.instructions.length).toBeGreaterThan(0)
    for (const image of spec.images) {
      expect(image.url).toContain(`/free-exercise-db/${spec.commit}/exercises/${spec.id}/`)
      expect(image.sha256).toMatch(/^[a-f0-9]{64}$/)
      expect(image.viewVerified).toBe(true)
      expect(referenceImageFor(image.url)).toBe(image)
    }
  }
})

test('unknown or ambiguous variants do not silently fall back to similar exercises', () => {
  for (const name of ['Front Barbell Squat', 'Barbell Back Squat ', 'back squat', '__proto__', null, {}]) {
    expect(getReferenceSpec(name)).toBeNull()
  }
  expect(getReferenceSpec('Barbell Back Squat').id).toBe('Barbell_Full_Squat')
  expect(getReferenceSpec('Conventional Barbell Deadlift').id).toBe('Barbell_Deadlift')
  expect(getReferenceSpec('Lat Pulldown (seated, to the front)').id).toBe('Wide-Grip_Lat_Pulldown')
})

test('real view tags do not relabel oblique photos as side targets', () => {
  expect(getReferenceSpec('Barbell Back Squat').images.map((i) => i.view)).toEqual(['three-quarter', 'three-quarter'])
  expect(getReferenceSpec('Barbell Bent-Over Row').images.map((i) => i.view)).toEqual(['three-quarter', 'three-quarter'])
  expect(getReferenceSpec('Tricep Pushdown (cable, bar or rope attachment)').images.map((i) => i.view)).toEqual(['side', 'side'])
})

test('setup and alternate-lunge photos cannot become fabricated movement endpoints', () => {
  const plank = getReferenceSpec('Front Plank (forearm plank)')
  expect(plank.motion).toBe('hold')
  expect(plank.autoBuild).toBe(true)
  expect(plank.images.map((i) => i.contextOnly)).toEqual([true, false])
  expect(plank.images[1].phase).toBe('start')
  const lunge = getReferenceSpec('Walking Lunge (dumbbells optional)')
  expect(lunge.autoBuild).toBe(false)
  expect(lunge.images.every((i) => i.contextOnly)).toBe(true)
  expect(lunge.limitations.join(' ')).toContain('standing start')
})

test('catalog records cannot be mutated in a running process', () => {
  const row = getReferenceSpec('Barbell Bent-Over Row')
  expect(() => { row.images[0].view = 'side' }).toThrow()
  expect(() => { row.images.push({}) }).toThrow()
})
