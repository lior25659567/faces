import { expect, test } from 'vitest'
import { automationChecksPassed, automationItemState, automationProgress, automationCheckLine, automationSourceUrl, automationJobLine } from './automation.js'

const pass = { pass: 4, fail: 0, limited: 0, uncertain: 0, unmeasured: 0 }

test('a finished build cannot masquerade as a passing exercise', () => {
  expect(automationChecksPassed(pass)).toBe(true)
  for (const summary of [null, { pass: 1 }, { ...pass, pass: 0 }, { ...pass, fail: 1 }, { ...pass, limited: 1 }, { ...pass, unmeasured: 1 }, { ...pass, uncertain: 1 }]) {
    expect(automationItemState({ status: 'passed', summary }).kind).toBe('review')
  }
  expect(automationItemState({ status: 'passed', summary: pass }).kind).toBe('pass')
})

test('blocked and failed exercises are terminal progress without becoming passes', () => {
  const job = { items: [{ status: 'building' }, { status: 'blocked' }, { status: 'failed' }, { status: 'needs-review' }, { status: 'waiting' }] }
  expect(automationProgress(job)).toEqual({ finished: 3, total: 5, text: '3 of 5 processed' })
  expect(automationItemState(job.items[1]).label).toBe('Blocked')
  expect(automationItemState(job.items[2]).kind).toBe('fail')
})

test('summary retains failed, uncertain and missing checks', () => {
  expect(automationCheckLine({ ...pass, fail: 2, uncertain: 1, unmeasured: 3 })).toBe('4 pass · 2 fail · 1 uncertain · 3 not measured')
  expect(automationCheckLine(null)).toBeNull()
})

test('source links cannot execute scripts or expose URL credentials', () => {
  expect(automationSourceUrl({ sourceUrl: 'https://github.com/yuhonas/free-exercise-db' })).toBe('https://github.com/yuhonas/free-exercise-db')
  for (const sourceUrl of ['javascript:alert(1)', 'http://example.com', 'https://user:secret@example.com', 'not a url']) {
    expect(automationSourceUrl({ sourceUrl })).toBeNull()
  }
})

test('cancel request says current work finishes rather than claiming cancellation already complete', () => {
  expect(automationJobLine({ status: 'running', cancelRequested: true, items: [{ status: 'building' }, { status: 'waiting' }] }))
    .toBe('0 of 2 processed. Stopping after the current exercise.')
  expect(automationJobLine({ status: 'cancelled', items: [{ status: 'cancelled' }] })).toContain('Remaining exercises were cancelled')
})

test('done batch retains attention-needed count and unknown results cannot turn green', () => {
  const job = { status: 'done', items: [{ status: 'needs-review' }, { status: 'blocked' }, { status: 'passed', summary: null }] }
  expect(automationJobLine(job)).toBe('3 of 3 processed. 3 need attention.')
  expect(automationItemState({ status: 'unexpected' }).kind).toBe('review')
})


test('reference readiness is neutral and cannot claim a built animation', () => {
  expect(automationItemState({status: 'ready'})).toEqual({ status: 'ready', label: 'References ready (not built)', kind: 'waiting' })
  expect(automationProgress({items:[{status:'ready'}]}).finished).toBe(1)
})


test('uncertainty must be explicitly known zero before a batch can show green', () => {
  for(const uncertain of [undefined,NaN,-1,'0']) expect(automationChecksPassed({...pass,uncertain})).toBe(false)
})
