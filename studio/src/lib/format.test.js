import { describe, expect, test } from 'vitest'
import { formatDuration, formatLeft, formatValue, formatWhen } from './format.js'

describe('formatDuration', () => {
  test('seconds, minutes and hours in plain words', () => {
    expect(formatDuration(45)).toBe('45 s')
    expect(formatDuration(250)).toBe('4 min 10 s')
    expect(formatDuration(3720)).toBe('1 h 2 min')
  })

  test('whole units drop the zero part', () => {
    expect(formatDuration(0)).toBe('0 s')
    expect(formatDuration(120)).toBe('2 min')
    expect(formatDuration(3600)).toBe('1 h')
  })

  test('rounds fractions of a second', () => {
    expect(formatDuration(44.6)).toBe('45 s')
    expect(formatDuration(59.6)).toBe('1 min')
  })

  test('a step that never ran shows a dash, not "0 s"', () => {
    expect(formatDuration(null)).toBe('—')
    expect(formatDuration(undefined)).toBe('—')
    expect(formatDuration(Number.NaN)).toBe('—')
  })
})

describe('formatLeft', () => {
  test('about N min left', () => {
    expect(formatLeft({ seconds: 180, overdue: false })).toBe('about 3 min left')
    expect(formatLeft({ seconds: 170, overdue: false })).toBe('about 3 min left')
    expect(formatLeft({ seconds: 3900, overdue: false })).toBe('about 1 h 5 min left')
  })

  test('under a minute', () => {
    expect(formatLeft({ seconds: 20, overdue: false })).toBe('less than a minute left')
    expect(formatLeft({ seconds: 0, overdue: false })).toBe('almost done')
  })

  test('overdue is never a negative or frozen 0 s', () => {
    expect(formatLeft({ seconds: 0, overdue: true })).toBe('taking longer than usual')
    expect(formatLeft({ seconds: -12, overdue: true })).toBe('taking longer than usual')
  })

  test('overdue step with later steps still to go', () => {
    expect(formatLeft({ seconds: 300, overdue: true })).toBe('at least 5 min left')
  })

  test('no estimate yet', () => {
    expect(formatLeft({ seconds: null, overdue: false })).toBe('working out the time left')
  })
})

describe('formatValue', () => {
  test('each unit the checks use', () => {
    expect(formatValue(47, 'deg')).toBe('47°')
    expect(formatValue(47.4, 'deg')).toBe('47°')
    expect(formatValue(1.8712, 'x')).toBe('1.87×')
    expect(formatValue(-0.05, 'torso')).toBe('−0.05 torso')
    expect(formatValue(0.1, 'torso')).toBe('0.10 torso')
    expect(formatValue(0.646, 'thigh')).toBe('0.65 thigh')         // the seat below the knee's top, in thigh lengths
    expect(formatValue(-0.05, 'thigh')).toBe('−0.05 thigh')
    expect(formatValue(28.91, 'mm')).toBe('28.9 mm')
    expect(formatValue(13.2, '%')).toBe('13%')
  })

  test('areas of folded skin in square millimetres, with thousands grouped', () => {
    expect(formatValue(0, 'mm2')).toBe('0 mm²')
    expect(formatValue(87.4, 'mm2')).toBe('87 mm²')
    expect(formatValue(1612.6, 'mm2')).toBe('1,613 mm²')
    expect(formatValue(null, 'mm2')).toBe('—')
    expect(formatValue(Number.NaN, 'mm2')).toBe('—')
  })

  test('frames of jumpy motion', () => {
    expect(formatValue(0, 'frames')).toBe('0 frames')
    expect(formatValue(1, 'frames')).toBe('1 frame')
    expect(formatValue(13, 'frames')).toBe('13 frames')
  })

  test('negative numbers use a real minus sign', () => {
    expect(formatValue(-5, 'deg')).toBe('−5°')
  })

  test('not measured shows a dash', () => {
    expect(formatValue(null, 'deg')).toBe('—')
    expect(formatValue(undefined, 'mm')).toBe('—')
  })

  test('unknown units keep the number and the unit', () => {
    expect(formatValue(2.5, 'kg')).toBe('2.5 kg')
    expect(formatValue(3)).toBe('3')
  })
})

describe('formatWhen', () => {
  const now = new Date(2026, 9, 22, 15, 0)
  test('today, yesterday, then the date', () => {
    expect(formatWhen(new Date(2026, 9, 22, 9, 5).toISOString(), now)).toBe('Today 09:05')
    expect(formatWhen(new Date(2026, 9, 21, 18, 30).toISOString(), now)).toBe('Yesterday 18:30')
    expect(formatWhen(new Date(2026, 8, 3, 7, 0).toISOString(), now)).toBe('3 Sep 07:00')
  })

  test('missing time shows a dash', () => {
    expect(formatWhen(null, now)).toBe('—')
  })
})
