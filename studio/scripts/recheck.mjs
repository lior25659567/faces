#!/usr/bin/env node
// Re-check a run that was already rendered, without re-rendering it.
//
// A change to the check rules (studio/measure/checks.py, build_targets.py, picture.py, posture.py …)
// does not change the character, the animation or the stills — only how the same measurements are
// judged. So there is no need to rebuild and re-animate (~8 min): run just the check step
// (studio/measure/run_check.py) against a run's existing measurements and rewrite its checks.json.
// On the squat that is ~0.2s instead of a full pipeline run.
//
//   node studio/scripts/recheck.mjs barbell-back-squat                      # newest rendered run
//   node studio/scripts/recheck.mjs "Barbell Back Squat"                    # a name works too
//   node studio/scripts/recheck.mjs barbell-back-squat --run 20261023-182924-f861
//
// It only rewrites checks.json in the run's own folder; it never renders, never starts Blender, and
// never changes a verdict's logic — it runs the exact same build_checks the Studio runs.
// STUDIO_DATA points it at another data root (a throwaway copy), exactly like work.mjs.
import { execFileSync } from 'node:child_process'
import { existsSync, readFileSync } from 'node:fs'
import path from 'node:path'
import { DATA, PYTHON, REPO } from '../server/paths.js'
import { createStore, slugOf } from '../server/store.js'

const CHECK = path.join(REPO, 'studio', 'measure', 'run_check.py')
// What the check step reads from a run's folder; all three exist only once the run has been rendered.
const NEEDED = ['targets.json', 'ours.json', path.join('frames', 'retarget.json')]

export class RecheckError extends Error {}

const USAGE = `Usage (from the repo root):
  node studio/scripts/recheck.mjs <exercise> [--run <id>]
    <exercise>  a slug or name, e.g. barbell-back-squat or "Barbell Back Squat"
    --run <id>  a specific run; without it, the newest run that has been rendered
  Re-runs only the check step against a run already rendered. Nothing is re-rendered.`

// "--run value" or "--run=value"; the first non-flag is the exercise.
const parse = (argv) => {
  let exercise
  let run
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i]
    if (arg === '--run' || arg.startsWith('--run=')) {
      const eq = arg.indexOf('=')
      run = eq < 0 ? argv[++i] : arg.slice(eq + 1)
      if (run === undefined) throw new RecheckError('"--run" needs a run id after it.')
    } else if (arg.startsWith('--')) {
      throw new RecheckError(`"${arg}" isn't something this command takes.`)
    } else if (exercise === undefined) {
      exercise = arg
    } else {
      throw new RecheckError(`Only one exercise at a time (got "${exercise}" and "${arg}").`)
    }
  }
  if (!exercise) throw new RecheckError('Say which exercise to re-check.')
  return { exercise, run }
}

// The run's folder has everything the check reads, i.e. it was rendered.
const checkable = (store, slug, id) => NEEDED.every((f) => existsSync(path.join(store.runDir(slug, id), f)))

const summaryOf = (dir) => {
  try {
    return JSON.parse(readFileSync(path.join(dir, 'checks.json'), 'utf8')).summary
  } catch {
    return null
  }
}

const fmtSummary = (s) =>
  s ? `${s.pass} pass, ${s.fail} fail, ${s.limited} limited, ${s.uncertain} uncertain` : '(no checks yet)'

// Resolve the exercise to a slug that exists, erroring with the list when it doesn't.
export const resolveSlug = (store, exercise) => {
  const slug = slugOf(exercise)
  if (!store.getExercise(slug)) {
    const all = store.listExercises().map((e) => e.slug).join(', ')
    throw new RecheckError(`There is no exercise called "${exercise}". Use one of: ${all}.`)
  }
  return slug
}

// The run to re-check: the one asked for by id, else the newest that has been rendered.
export const pickRun = (store, slug, wantId) => {
  const runs = store.listRuns(slug) // newest first
  if (!runs.length) throw new RecheckError(`No runs for "${slug}" yet. Render it once in the Studio first.`)
  if (wantId) {
    const run = runs.find((r) => r.id === wantId)
    if (!run) throw new RecheckError(`"${slug}" has no run "${wantId}". Its runs: ${runs.map((r) => r.id).join(', ')}.`)
    if (!checkable(store, slug, run.id)) {
      throw new RecheckError(`Run ${wantId} hasn't been rendered, so there is nothing to check. Render it first.`)
    }
    return run
  }
  const run = runs.find((r) => checkable(store, slug, r.id))
  if (!run) throw new RecheckError(`No rendered run for "${slug}" yet (none has its measurements). Render it once first.`)
  return run
}

export const recheck = (store, { exercise, run: wantId }, { python = PYTHON, log = console.log } = {}) => {
  const slug = resolveSlug(store, exercise)
  const run = pickRun(store, slug, wantId)
  const dir = store.runDir(slug, run.id)
  const before = summaryOf(dir)
  log(`Re-checking ${store.getExercise(slug).name} — run ${run.id}`)
  log(`  data root: ${store.root}`)
  try {
    execFileSync(python, [CHECK, dir], { stdio: ['ignore', 'ignore', 'inherit'] })
  } catch (e) {
    throw new RecheckError(`The check step failed for run ${run.id}: ${e?.message || e}`)
  }
  const after = summaryOf(dir)
  log(`  before: ${fmtSummary(before)}`)
  log(`  after:  ${fmtSummary(after)}`)
  return { slug, runId: run.id, before, after }
}

const main = (argv) => recheck(createStore(DATA), parse(argv))

if (import.meta.url === `file://${process.argv[1]}`) {
  try {
    main(process.argv.slice(2))
  } catch (e) {
    if (e instanceof RecheckError) {
      console.error(`recheck: ${e.message}\n\n${USAGE}`)
    } else {
      console.error(`recheck: something went wrong: ${e?.message || e}`)
    }
    process.exitCode = 1
  }
}
