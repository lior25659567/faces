#!/usr/bin/env node
// Show work going on behind the scenes in the Exercise Studio, live, with its progress.
// When an item is marked done, the Studio re-runs the exercises it lists on its own.
// Plain Node, no dependencies. It only writes files (DATA/work/<id>.json), so it works whether or not
// the Studio is running. STUDIO_DATA points it at another data root (tests, a throwaway Studio).
import { DATA } from '../server/paths.js'
import { createStore } from '../server/store.js'
import { WorkError, createWork, resolveExercises, sortWork } from '../server/work.js'

const USAGE = `Usage (from the repo root):
  node studio/scripts/work.mjs start --title "Fix skin folds at the shoulders" --exercises barbell-back-squat,pull-up
      prints the new item's id. --exercises all = every exercise with reference pictures.
  node studio/scripts/work.mjs log <id> "Measured the folds again: 1,008 mm² at the left shoulder"
  node studio/scripts/work.mjs done <id> --summary "Shoulder folds are down to 120 mm²"
      the Studio then re-runs the listed exercises on its own.
      add --recheck when the fix only changed the check rules: the Studio re-judges each
      exercise's newest rendered run (~0.2s, no Blender) instead of re-rendering it.
  node studio/scripts/work.mjs fail <id> --summary "Corrective shapes made the elbows worse"
      nothing re-runs.
  node studio/scripts/work.mjs list`

class UsageError extends WorkError {}

// "--name value" or "--name=value"; names in `booleans` take no value ("--recheck"). Everything
// else is positional.
const parse = (argv, allowed, booleans = []) => {
  const flags = {}
  const rest = []
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i]
    if (!arg.startsWith('--')) {
      rest.push(arg)
      continue
    }
    const eq = arg.indexOf('=')
    const name = eq < 0 ? arg.slice(2) : arg.slice(2, eq)
    if (!allowed.includes(name)) throw new UsageError(`"--${name}" isn't something this command takes.`)
    if (booleans.includes(name)) {
      flags[name] = true
      continue
    }
    const value = eq < 0 ? argv[++i] : arg.slice(eq + 1)
    if (value === undefined) throw new UsageError(`"--${name}" needs a value after it.`)
    flags[name] = value
  }
  return { flags, rest }
}

const needId = (rest, command) => {
  if (!rest[0]) throw new UsageError(`Say which work item to ${command}: the id that "start" printed.`)
  return rest[0]
}

const main = (argv) => {
  const [command, ...args] = argv
  const work = createWork(DATA)
  switch (command) {
    case 'start': {
      const { flags, rest } = parse(args, ['title', 'exercises'])
      if (rest.length) throw new UsageError(`Put the title in --title "…" (got "${rest.join(' ')}").`)
      const exercises = resolveExercises(createStore(DATA), flags.exercises)
      const item = work.start({ title: flags.title, exercises })
      if (flags.exercises && !exercises.length) {
        console.error('Note: no exercise has reference pictures yet, so nothing will re-run when this is done.')
      }
      console.log(item.id)
      return
    }
    case 'log': {
      const { rest } = parse(args, [])
      work.log(needId(rest, 'log to'), rest.slice(1).join(' '))
      return
    }
    case 'done':
    case 'fail': {
      const { flags, rest } = parse(args, ['summary', 'recheck'], ['recheck'])
      work.finish(needId(rest, `mark ${command === 'done' ? 'done' : 'failed'}`), command === 'done' ? 'done' : 'failed',
        flags.summary, { recheck: flags.recheck })
      return
    }
    case 'list': {
      for (const w of sortWork(work.list())) console.log(`${w.id}  ${w.status.padEnd(7)}  ${w.title}`)
      return
    }
    default:
      throw new UsageError(command ? `"${command}" isn't a command.` : 'Say what to do.')
  }
}

try {
  main(process.argv.slice(2))
} catch (e) {
  if (!(e instanceof WorkError)) {
    console.error(`work: something went wrong: ${e?.message || e}`)
  } else {
    console.error(`work: ${e.message}`)
    if (e instanceof UsageError) console.error(`\n${USAGE}`)
  }
  process.exitCode = 1
}
