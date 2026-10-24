import { existsSync, readFileSync } from 'node:fs'
import path from 'node:path'
import { BLENDER, CLIPS_DIR, PYTHON, REPO, SKIN } from './paths.js'

const UNBUFFERED = 'import sys; sys.stdout.reconfigure(line_buffering=True); sys.stderr.reconfigure(line_buffering=True)'

export const STEPS = [
  { id: 'read', label: 'Reading the reference' },
  { id: 'build', label: 'Building the character' },
  { id: 'animate', label: 'Animating' },
  { id: 'render', label: 'Rendering the GIF' },
  { id: 'stills', label: 'Taking the comparison stills' },
  { id: 'check', label: 'Checking against the reference' },
]

export const PROCESSES = [
  { id: 'read', steps: ['read'] },
  { id: 'animate', steps: ['build', 'animate', 'render'] },
  { id: 'stills', steps: ['stills'] },
  { id: 'check', steps: ['check'] },
]

export const commandFor = (processId, run, runDir, exercise) => {
  const studio = (...p) => path.join(REPO, 'studio', ...p)
  if (processId === 'read') return { cmd: PYTHON, args: [studio('measure', 'run_read.py'), runDir], cwd: REPO }
  if (processId === 'check') return { cmd: PYTHON, args: [studio('measure', 'run_check.py'), runDir], cwd: REPO }
  if (processId === 'stills') {
    return {
      cmd: BLENDER,
      args: ['--background', path.join(runDir, 'scene.blend'), '--python-exit-code', '1',
        '--python-expr', UNBUFFERED, '--python', studio('blender', 'inspect_run.py'), '--', '--run', runDir],
      cwd: REPO,
    }
  }
  if (processId === 'animate') {
    const preflight = path.join(runDir, 'preflight.json')
    if (existsSync(preflight)) {
      try {
        const plan = JSON.parse(readFileSync(preflight, 'utf8'))
        if (plan.ready !== true) return { error: plan.blockers?.join(' ') || 'Check the reference images before building.' }
      } catch { return { error: 'The reference check is unreadable. Check the reference images again.' } }
    }
    const name = exercise?.name ?? run?.name
    // Row references must actually drive the motion even when a generic row clip exists.
    const clip = name === 'Barbell Bent-Over Row' ? null : exercise?.clip ?? run?.clip
    // No clip: the motion moves between the reference pictures' key positions, which the read step wrote
    // (studio/measure/run_read.py -> keyposes.json; pipeline/blender/keypose_clip.py, task 9).
    const keyposes = path.join(runDir, 'keyposes.json')
    // ... unless the picture path can't: its feet step, or it needs a bench/seat/bar the path doesn't build
    // (store.js pictureRefusal, from studio/measure/exercise-kit.json; animate.py refuses the same before building)
    const refusal = exercise?.pictureRefusal ?? run?.pictureRefusal
    if (!clip && refusal) return { error: refusal }
    if (!clip && !existsSync(keyposes)) {
      const expected = exercise?.expectedClip ?? run?.expectedClip
      return {
        error: expected
          ? `No motion for this exercise yet. Add a Mixamo clip named ${expected} to 3d-character/mixamo (its README says how).`
          : 'No motion for this exercise yet. Add a Mixamo clip for it to 3d-character/mixamo (its README says how).',
      }
    }
    const motion = clip ? ['--mixamo', path.join(CLIPS_DIR, clip)] : ['--keyposes', keyposes]
    const targets = path.join(runDir, 'pipeline-targets.json')
    return {
      cmd: BLENDER,
      args: ['--background', '--python-exit-code', '1', '--python-expr', UNBUFFERED,
        '--python', path.join(REPO, 'pipeline', 'blender', 'animate.py'), '--',
        '--mpfb', '--exercise', name, '--wrap-skin', SKIN, ...motion,
        ...(existsSync(targets) ? ['--form-targets', targets] : []),
        '--side-view', '--out', path.join(runDir, 'frames'), '--gif', path.join(runDir, 'animation.gif'),
        '--save', path.join(runDir, 'scene.blend')],
      cwd: REPO,
    }
  }
  return { error: `Unknown process ${processId}.` }
}

// Paths (relative to the run folder) that must exist and be non-empty once the process exits 0.
export const requiredOutputs = (processId, runDir, snapshot) => {
  if (processId === 'read') return ['measurements.json', 'targets.json']
  if (processId === 'check') return ['checks.json']
  if (processId === 'animate') return ['animation.gif', 'frames/retarget.json', 'scene.blend']
  if (processId === 'stills') {
    let files = []
    try {
      files = (JSON.parse(readFileSync(path.join(runDir, 'ours.json'), 'utf8')).stills || []).map((s) => s.file).filter(Boolean)
    } catch { /* a missing or broken ours.json is reported as missing by the caller */ }
    // A still for every key image the run measured, plus whatever ours.json lists.
    const keyImages = (snapshot?.images || []).filter((i) => !i.contextOnly && !i.deleted).map((i) => `stills/${i.id}.png`)
    return [...new Set(['ours.json', ...keyImages, ...files])]
  }
  return []
}

// Turns one line of animate's log into a step change and/or the expected frame total.
export const readLine = (processId, line, state = {}) => {
  if (processId !== 'animate') return {}
  const frames = /^retarget: read .*?: (\d+) frames/.exec(line)
  if (frames) {
    // A key-position run also reads the Mixamo rest (squat.fbx) first; its own line wins in any order (task 9 review).
    if (line.includes("(the pictures' key positions)")) state.keyposes = true
    else if (state.keyposes) return {}
    state.total = 2 * Number(frames[1]) // with the side view every frame is drawn twice
    return { total: state.total }
  }
  if (line.startsWith('bake_to_skin:')) return { step: 'animate' }
  if (line.startsWith('retarget report ->')) return { step: 'render' }
  // The solver's picture loop (pipeline/blender/mpfb_cage.py `match_picture`, task 19): a note on the step, so
  // the page shows what the minute is spent on
  if (line.startsWith('form: picture: reading our picture')) return { note: 'Reading our picture as the Studio will' }
  const round = /^form: picture round (\d+)/.exec(line)
  if (round) return { note: `Matching the coach in our picture: reading ${Number(round[1]) + 1}` }
  if (line.startsWith('form: picture: kept')) return { note: null }
  return {}
}
