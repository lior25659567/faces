import { mkdirSync, mkdtempSync, readFileSync, writeFileSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'

// A data root holding one run that a live Studio is working on right now.
export const liveDataRoot = () => {
  const root = mkdtempSync(path.join(os.tmpdir(), 'studio-live-'))
  const dir = path.join(root, 'runs', 'barbell-back-squat', '20990101-000000-abcd')
  mkdirSync(dir, { recursive: true })
  const file = path.join(dir, 'run.json')
  writeFileSync(file, JSON.stringify({
    id: '20990101-000000-abcd', slug: 'barbell-back-squat', name: 'Barbell Back Squat', status: 'running',
    createdAt: '2099-01-01T00:00:00.000Z', startedAt: '2099-01-01T00:00:00.000Z', finishedAt: null,
    steps: [{ id: 'animate', status: 'running', startedAt: '2099-01-01T00:00:00.000Z', finishedAt: null, seconds: null, fraction: null, note: null }],
    error: null, summary: null,
  }))
  return { root, file, status: () => JSON.parse(readFileSync(file, 'utf8')).status }
}
