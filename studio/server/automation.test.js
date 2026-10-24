import { afterEach, beforeEach, expect, test } from 'vitest'
import { mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { createStore } from './store.js'
import { builderFingerprint, createAutomation } from './automation.js'

const ROW = 'barbell-bent-over-row', SQUAT = 'barbell-back-squat', UNSUPPORTED = 'standing-barbell-overhead-press-shoulder-press'
let root, store, automation, builds, reads, downloads, settings
beforeEach(() => {
  root = mkdtempSync(path.join(os.tmpdir(), 'studio-automation-test-'))
  store = createStore(root); builds = []; reads = []; downloads = []
  settings = { summary: { pass: 8, fail: 0, limited: 0, uncertain: 0, unmeasured: 0 }, ready: true, delay: 0, code: 'v1' }
})
afterEach(() => { automation?.stop(); rmSync(root, { recursive: true, force: true }) })
const until = async (fn) => {
  for (let n = 0; n < 500; n++) { if (fn()) return; await new Promise((r) => setTimeout(r, 2)) }
  throw Error('automation did not reach expected state')
}
const spec = (name) => ({ id: name, commit: 'pinned', license: 'Unlicense', repository: 'https://github.com/example/data',
  images: ['start', 'end'].map((phase) => ({ url: `https://example.test/${name}/${phase}.jpg`, phase, view: 'three-quarter' })) })
const put = (run, rel, value) => {
  const file = path.join(store.runDir(run.slug, run.id), rel)
  mkdirSync(path.dirname(file), { recursive: true }); writeFileSync(file, typeof value === 'string' ? value : JSON.stringify(value))
}
const fakeQueue = () => ({
  state: () => ({ running: null, queued: [] }),
  enqueueRead: (slug, extra) => {
    reads.push(slug); const run = store.createRun(slug, { ...extra, readOnly: true })
    setTimeout(() => {
      put(run, 'preflight.json', { ready: settings.ready, blockers: settings.ready ? [] : ['Need a front reference.'], warnings: [] })
      store.writeRun({ ...run, status: settings.ready ? 'done' : 'failed', error: settings.ready ? null : { message: 'Need a front reference.' } })
    }, settings.delay)
    return run
  },
  enqueue: (slug, extra) => {
    builds.push(slug); const run = store.createRun(slug, extra)
    put(run, 'animation.gif', 'GIF89a' + 'x'.repeat(1100)); put(run, 'checks.json', { summary: settings.summary, rows: settings.checkRows || [] })
    put(run, 'scene.blend', 'BLENDER'); put(run, 'ours.json', { stills: [{ file: 'stills/start.png' }] }); put(run, 'stills/start.png', 'PNG')
    store.writeRun({ ...run, status: 'done', summary: settings.summary })
    return run
  },
})
const make = (extra = {}) => {
  automation = createAutomation({ store, queue: fakeQueue(), referenceSpec: spec,
    download: async (image) => { downloads.push(image.url); return { bytes: Buffer.from('JPEG fixture'), ext: 'jpg', sha256: 'hash' } },
    fingerprint: () => settings.code, pollMs: 1, ...extra })
}
const finished = async (job) => { await until(() => automation.get(job.id).status !== 'running'); return automation.get(job.id) }

test('retrieves, checks and builds supported exercises, preserving source metadata', async () => {
  make(); const result = await finished(automation.start({ slugs: [ROW] }))
  expect(result.items[0].status).toBe('passed')
  expect(reads).toEqual([ROW]); expect(builds).toEqual([ROW]); expect(downloads).toHaveLength(2)
  const refs = store.listReferences(ROW)
  expect(refs.map((r) => r.phase)).toEqual(['start', 'end'])
  expect(refs.every((r) => r.view === 'three-quarter' && r.source.commit === 'pinned')).toBe(true)
  expect(store.readRun(ROW, result.items[0].runId).automationJobId).toBe(result.id)
})

test('existing user references are never replaced or relabeled', async () => {
  const image = store.addReference(ROW, Buffer.from('personal picture'), 'my-row.png')
  store.updateReference(ROW, image.id, { phase: 'end', view: 'side' })
  const before = store.listReferences(ROW)
  make(); await finished(automation.start({ slugs: [ROW] }))
  expect(downloads).toEqual([]); expect(store.listReferences(ROW)).toEqual(before)
  expect(readFileSync(path.join(root, 'references', ROW, image.file), 'utf8')).toBe('personal picture')
})

test('missing input and unsupported equipment prevent costly builds, and batch continues', async () => {
  settings.ready = false; make()
  const result = await finished(automation.start({ slugs: [SQUAT, UNSUPPORTED, ROW] }))
  expect(result.items.map((i) => i.status)).toEqual(['blocked', 'blocked', 'blocked'])
  expect(result.items[0].message).toMatch(/front reference/)
  expect(result.items[1].message).toMatch(/bar|hand contacts/)
  expect(builds).toEqual([])
})

test('a completed run with form failures is review work, never a pass', async () => {
  settings.summary.fail = 6; make()
  const result = await finished(automation.start({ slugs: [ROW] }))
  expect(result.items[0].status).toBe('needs-review'); expect(result.items[0].summary.fail).toBe(6)
})

test('unattended results name measured defects instead of sending the user into another blind rebuild', async () => {
  settings.summary.fail = 1
  settings.checkRows = [{ status: 'fail', note: 'Skin folds over at the left collarbone.' }]
  make(); const result = await finished(automation.start({ slugs: [ROW] }))
  expect(result.items[0].status).toBe('needs-review')
  expect(result.items[0].message).toContain('left collarbone')
})

test('reference-only mode does not render or claim animation success', async () => {
  make(); const result = await finished(automation.start({ slugs: [ROW], checkOnly: true }))
  expect(result.items[0].status).toBe('ready'); expect(builds).toEqual([])
})

test('unchanged full results are reused; changed code, references and force invalidate reuse', async () => {
  make(); await finished(automation.start({ slugs: [ROW] }))
  const repeated = await finished(automation.start({ slugs: [ROW] }))
  expect(repeated.items[0].reused).toBe(true); expect(builds).toHaveLength(1); expect(downloads).toHaveLength(2)
  settings.code = 'v2'; await finished(automation.start({ slugs: [ROW] })); expect(builds).toHaveLength(2)
  const image = store.listReferences(ROW)[0]; store.updateReference(ROW, image.id, { phase: 'middle' })
  await finished(automation.start({ slugs: [ROW] })); expect(builds).toHaveLength(3)
  await finished(automation.start({ slugs: [ROW], forceRebuild: true })); expect(builds).toHaveLength(4)
})

test.each(['running', 'queued'])('manual work %s during preflight prevents a duplicate automatic build', async (stage) => {
  settings.delay = 25
  const queue = fakeQueue()
  const state = { running: null, queued: [] }
  queue.state = () => state
  make({ queue })
  const job = automation.start({ slugs: [ROW] })
  await until(() => automation.get(job.id).items[0].status === 'checking')
  const manual = store.createRun(ROW)
  store.writeRun({ ...manual, status: stage })
  const pending = { slug: ROW, runId: manual.id }
  if (stage === 'running') state.running = pending
  else state.queued.push(pending)
  const before = store.readRun(ROW, manual.id)

  const result = await finished(job)

  expect(result.items[0].status).toBe('blocked')
  expect(result.items[0].message).toMatch(/duplicate build/)
  expect(builds).toEqual([])
  expect(reads).toEqual([ROW])
  expect(store.readRun(ROW, manual.id)).toEqual(before)
})

test('motion clip contents and availability invalidate the builder fingerprint', () => {
  const clips = path.join(root, '3d-character', 'mixamo')
  const clip = path.join(clips, 'squat.fbx')
  const fingerprint = () => builderFingerprint({ root, execution: () => 'fixed test runtime' })
  const absent = fingerprint()
  mkdirSync(clips, { recursive: true })
  expect(fingerprint()).toBe(absent)
  writeFileSync(clip, 'first motion clip')
  const first = fingerprint()
  expect(first).not.toBe(absent)
  expect(fingerprint()).toBe(first)
  writeFileSync(clip, 'revised motion clip')
  expect(fingerprint()).not.toBe(first)
  rmSync(clip)
  expect(fingerprint()).toBe(absent)
})

test('a builder change earlier in a batch prevents reuse of later cached results', async () => {
  const queue = fakeQueue()
  const enqueueRead = queue.enqueueRead
  queue.enqueueRead = (slug, extra) => {
    const run = enqueueRead(slug, extra)
    if (slug === SQUAT) settings.code = 'v2'
    return run
  }
  make({ queue })
  await finished(automation.start({ slugs: [ROW] }))

  const result = await finished(automation.start({ slugs: [SQUAT, ROW] }))

  expect(result.items[0].status).toBe('blocked')
  expect(result.items[1].status).toBe('blocked')
  expect(result.items[1].reused).not.toBe(true)
  expect(result.items[1].message).toMatch(/builder changed/)
  expect(builds).toEqual([ROW])
})

test('changing the selected exercise clip invalidates reuse even with unchanged builder and references', async () => {
  let selectedClip = null
  make({ store: { ...store, getExercise: (slug) => ({ ...store.getExercise(slug), clip: selectedClip }) } })
  const first = await finished(automation.start({ slugs: [SQUAT] }))
  const unchanged = await finished(automation.start({ slugs: [SQUAT] }))
  expect(unchanged.items[0].reused).toBe(true)

  selectedClip = 'alternate-squat.fbx'
  const changed = await finished(automation.start({ slugs: [SQUAT] }))

  expect(changed.items[0].status).toBe('passed')
  expect(changed.items[0].reused).not.toBe(true)
  expect(builds).toEqual([SQUAT, SQUAT])
  expect(store.readRun(SQUAT, changed.items[0].runId).automationInputs)
    .not.toBe(store.readRun(SQUAT, first.items[0].runId).automationInputs)
  expect(downloads).toHaveLength(2)
})

test('partially imported automatic references resume without duplicates', async () => {
  const source = spec(store.getExercise(ROW).name)
  store.addReference(ROW, Buffer.from('existing'), 'source-start.jpg', 'jpg', {
    phase: 'start', view: 'three-quarter', source: { catalogId: source.id, commit: source.commit, url: source.images[0].url },
  })
  make(); await finished(automation.start({ slugs: [ROW] }))
  expect(store.listReferences(ROW)).toHaveLength(2); expect(downloads).toHaveLength(1)
})

test('cancel completes the current exercise, then stops the rest', async () => {
  settings.delay = 25; make(); const job = automation.start({ slugs: [ROW, SQUAT] })
  await until(() => automation.get(job.id).items[0].status === 'checking')
  automation.cancel(job.id)
  const result = await finished(job)
  expect(result.status).toBe('cancelled'); expect(builds).toEqual([ROW]); expect(result.items[1].status).toBe('cancelled')
})

test('shutdown records interruption, does not launch future work or turn recovery into builds', async () => {
  settings.delay = 25; make(); const job = automation.start({ slugs: [ROW, SQUAT] })
  await until(() => automation.get(job.id).items[0].status === 'checking')
  automation.stop(); expect(automation.get(job.id).status).toBe('interrupted')
  await new Promise((r) => setTimeout(r, 35)); expect(builds).toEqual([])
  make(); automation.recoverInterrupted(); expect(builds).toEqual([]); expect(store.listReferences(ROW)).toHaveLength(2)
})

test('invalid selections and concurrent batches are rejected', async () => {
  settings.delay = 20; make()
  expect(() => automation.start({ slugs: ['invented'] })).toThrow(/existing exercises/)
  expect(() => automation.start({ checkOnly: 'yes' })).toThrow(/true or false/)
  const job = automation.start({ slugs: [ROW] })
  expect(() => automation.start({ slugs: [SQUAT] })).toThrow(/already running/)
  expect(automation.get('../escape')).toBeNull()
  await finished(job)
})


test('missing or corrupted cached outputs force a real rebuild', async () => {
  make(); const first=await finished(automation.start({slugs:[ROW]}))
  put(store.readRun(ROW, first.items[0].runId), 'animation.gif', 'broken')
  const next=await finished(automation.start({slugs:[ROW]}))
  expect(next.items[0].reused).not.toBe(true)
  expect(builds).toHaveLength(2)
})

test('static or setup-only catalog images are imported as context, never invented motion', async () => {
  make({referenceSpec: name => ({...spec(name),autoBuild:false,reason:'Only a static hold is shown.',
    images:spec(name).images.map(image=>({...image,contextOnly:true}))})})
  const result=await finished(automation.start({slugs:[ROW]}))
  expect(result.items[0].status).toBe('blocked');expect(builds).toEqual([]);expect(reads).toEqual([])
  expect(store.listReferences(ROW).every(i=>i.contextOnly)).toBe(true)
})


test.each([{pass:8,fail:0}, {pass:0.5,fail:0,limited:0,uncertain:0,unmeasured:0}])('an incomplete or invalid score cannot certify success: %j', async (summary) => {
  settings.summary = summary
  make();const result=await finished(automation.start({slugs:[ROW]}))
  expect(result.items[0].status).toBe('needs-review')
})

test('transient source errors retry once without repeated imports', async () => {
  let calls=0
  make({download:async()=>{calls++;if(calls===1)throw Object.assign(Error('temporary'),{retryable:true});return {bytes:Buffer.from('JPEG'),ext:'jpg',sha256:'hash'}}})
  const result=await finished(automation.start({slugs:[ROW]}))
  expect(result.items[0].status).toBe('passed');expect(calls).toBe(3);expect(store.listReferences(ROW)).toHaveLength(2)
})

test('integrity errors stop that exercise without changing refs and later items continue', async () => {
  let calls=0
  make({download:async()=>{calls++;if(calls===1)throw Error('Checksum mismatch');return {bytes:Buffer.from('JPEG'),ext:'jpg',sha256:'hash'}}})
  const result=await finished(automation.start({slugs:[ROW,SQUAT]}))
  expect(result.items[0].status).toBe('failed');expect(store.listReferences(ROW)).toEqual([])
  expect(result.items[1].status).toBe('passed');expect(builds).toEqual([SQUAT]);expect(calls).toBe(3)
})
