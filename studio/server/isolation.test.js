import { expect, test } from 'vitest'
import { spawnSync } from 'node:child_process'
import { rmSync } from 'node:fs'
import path from 'node:path'
import { REPO } from './paths.js'
import { liveDataRoot } from './live-data-root.js'

const STUDIO = path.join(REPO, 'studio')
const VITEST = path.join(REPO, 'node_modules', 'vitest', 'vitest.mjs')

test('a vitest run leaves a "running" run in the data root untouched', () => {
  const live = liveDataRoot()
  try {
    // The child is `npm test` as the user runs it (config found the usual way), aimed at the temp copy.
    const env = Object.fromEntries(Object.entries(process.env).filter(([k]) => !k.startsWith('VITEST')))
    const child = spawnSync(process.execPath, [VITEST, 'run', 'src/lib/hash.test.js'], {
      cwd: STUDIO, env: { ...env, STUDIO_DATA: live.root }, encoding: 'utf8', timeout: 60_000,
    })
    expect(child.status, child.stdout + child.stderr).toBe(0)
    expect(live.status()).toBe('running')
  } finally {
    rmSync(live.root, { recursive: true, force: true })
  }
}, 90_000)
