import { execFile, spawn } from 'node:child_process'
import { readFileSync } from 'node:fs'
import { mkdir, rename, rm, writeFile } from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { parseEnv, promisify } from 'node:util'

const run = promisify(execFile)

const root = fileURLToPath(new URL('..', import.meta.url))
const python = path.join(root, 'pipeline/.venv/bin/python')
const generator = path.join(root, 'pipeline/generate-frames.py')
const restyler = path.join(root, 'pipeline/restyle.py')

const MAX_BYTES = 25 * 1024 * 1024
const PNG_START = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a])

// Exit statuses the two scripts use for failures a person can do something about.
const NO_FACE = 3
const ALLOWANCE_USED = 4
const REFUSED = 5
const TOO_DARK = 6

const PROBLEMS = {
  [NO_FACE]: 'No face found in that image. Use a front-facing photo in even light.',
  [ALLOWANCE_USED]: "Today's free restyles are used up. They come back at midnight UTC. Until then, switch 3D style off.",
  [REFUSED]: 'Cloudflare refused the restyle. Check the two CLOUDFLARE values in .env. Details are in the terminal running npm run dev.',
  [TOO_DARK]: 'That photo is too dark to see the face. Turn on a light or face a window, then try again.',
}
const NO_FACE_AFTER_RESTYLE = 'The restyled picture has no face the animator can find. Try again: each restyle comes out different.'
const STOPPED = 'It stopped unexpectedly. The reason is in the terminal running npm run dev.'
const NOT_INSTALLED = 'The frame generator is not installed. Run npm run setup, then try again.'
const NOT_SET_UP = '3D style is not set up. Add the two CLOUDFLARE values to .env. The README shows how.'
const NOT_A_PICTURE = 'That file is not a picture this Mac can open. Choose a photo or an image file.'
const NO_SUCH_OUTFIT = 'That outfit is not one this page knows.'

// The gym outfits of a whole-body avatar. What each one looks like is in restyle.py.
const OUTFITS = ['tee-pants', 'tank-shorts']

// A coarse pass so the character shows up in under a minute, then the smooth one.
// `steps` is frames across; the generator works out how many rows go with them.
const PASSES = [
  { name: 'preview', steps: 5 },
  { name: 'full', steps: 7 },
]

// The folder a character is saved under: lowercase words joined by hyphens, never a path.
export const faceName = (raw) =>
  String(raw ?? '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 40) || 'me'

// The outfit a request asks for, or null for a portrait in the person's own clothes.
export const gymOutfit = (raw) => (OUTFITS.includes(raw) ? raw : null)

// generate-frames.py prints `frame 7/25` after each frame when its output is piped.
export const readProgress = (line) => {
  const match = /^frame (\d+)\/(\d+)$/.exec(line)
  return match ? { done: Number(match[1]), total: Number(match[2]) } : null
}

// The three formats the pipeline reads as they are, told apart by their first bytes.
// Anything else (an iPhone HEIC, a TIFF, a GIF) is converted first, see savePicture.
export const imageKind = (bytes) => {
  if (bytes.length >= 3 && bytes[0] === 0xff && bytes[1] === 0xd8 && bytes[2] === 0xff) return 'jpg'
  if (bytes.subarray(0, 8).equals(PNG_START)) return 'png'
  if (bytes.toString('latin1', 0, 4) === 'RIFF' && bytes.toString('latin1', 8, 12) === 'WEBP') return 'webp'
  return null
}

// Saves an upload as input/<name>.<ext> and returns that path, or null if it is not a picture.
// Formats the pipeline cannot read are converted to JPEG by sips, which ships with macOS.
const savePicture = async (bytes, name) => {
  const folder = path.join(root, 'input')
  await mkdir(folder, { recursive: true })

  const kind = imageKind(bytes)
  if (kind) {
    const photo = path.join(folder, `${name}.${kind}`)
    await writeFile(photo, bytes)
    return photo
  }

  const original = path.join(folder, `${name}.upload`)
  const converted = `${original}.jpg`
  await writeFile(original, bytes)
  try {
    await run('sips', ['-s', 'format', 'jpeg', original, '--out', converted], { timeout: 30_000 })
    const photo = path.join(folder, `${name}.jpg`)
    await rename(converted, photo) // sips exits 0 even when it read nothing, so a missing file is the failure
    return photo
  } catch {
    return null
  } finally {
    await Promise.all([rm(original, { force: true }), rm(converted, { force: true })])
  }
}

// What the restyle step needs, from the text of .env. Null until both Cloudflare values are filled in.
export const styleKeys = (envText) => {
  const env = parseEnv(envText)
  return env.CLOUDFLARE_ACCOUNT_ID && env.CLOUDFLARE_API_TOKEN ? env : null
}

// Read fresh each time, so adding the keys takes effect without a restart.
const readStyleKeys = () => {
  try {
    return styleKeys(readFileSync(path.join(root, '.env'), 'utf8'))
  } catch {
    return null
  }
}

const reply = (response, status, body) => {
  response.writeHead(status, { 'Content-Type': 'application/json' })
  response.end(JSON.stringify(body))
}

// Dev-server bridge between the page and the pipeline. One job at a time; a new image replaces it.
//   POST /api/faces?name=me[&style=3d]   a picture's bytes as the body, in any format macOS can open:
//                  [&outfit=tee-pants]   saved to input/, restyled first if asked, then rendered into src/faces/me/.
//                                        An outfit makes a whole-body avatar in gym clothes, which is always restyled.
//   GET  /api/faces/job                  { state: idle | running | done | error, id, name, pass, done, total,
//                                          ready, error, canRestyle }
//                                        `pass` is restyle, preview or full. `ready` turns true once the page
//                                        has frames to show. `canRestyle` says whether .env has the keys.
export const framesApi = () => {
  let job = { state: 'idle' }
  let child = null
  let jobs = 0

  const status = () => ({ ...job, canRestyle: Boolean(readStyleKeys()) })

  const fail = (error, detail) => {
    if (detail?.trim()) console.error(`\n[frames] ${detail.trim()}\n`)
    child = null
    job = { ...job, state: 'error', error }
  }

  // Runs one pipeline script as the job's current step. Nothing here fires once a newer image has taken over.
  const step = (script, args, { env, onLine, done }) => {
    const running = spawn(python, [script, ...args], { cwd: root, env: { ...process.env, ...env } })
    child = running
    let unfinished = ''
    let complaints = ''

    running.stdout.setEncoding('utf8').on('data', (chunk) => {
      const lines = (unfinished + chunk).split(/[\r\n]+/)
      unfinished = lines.pop()
      if (child === running) lines.forEach((line) => onLine?.(line))
    })
    running.stderr.setEncoding('utf8').on('data', (chunk) => {
      complaints = (complaints + chunk).slice(-4000)
    })
    running.on('error', () => {
      if (child === running) fail(NOT_INSTALLED)
    })
    running.on('close', (code) => {
      if (child !== running) return
      if (code === 0) return done()
      fail(code === NO_FACE && job.restyled ? NO_FACE_AFTER_RESTYLE : (PROBLEMS[code] ?? STOPPED), complaints)
    })
  }

  const render = (source, pass) => {
    const { name, steps } = PASSES[pass]
    job = { ...job, pass: name, done: 0, total: 0 } // the generator reports the total with its first frame
    step(generator, [source, '--name', job.name, '--steps', String(steps)], {
      onLine: (line) => {
        const progress = readProgress(line)
        if (progress) job = { ...job, ...progress }
      },
      done: () => {
        job = { ...job, ready: true }
        if (pass + 1 < PASSES.length) render(source, pass + 1)
        else job = { ...job, state: 'done' }
      },
    })
  }

  const restyle = (photo, keys) => {
    const styled = photo.replace(/\.[^.]+$/, '-3d.png')
    job = { ...job, pass: 'restyle', done: 0, total: 0 }
    step(restyler, [photo, styled, '--name', job.name, ...(job.outfit ? ['--outfit', job.outfit] : [])], {
      env: keys,
      done: () => render(styled, 0),
    })
  }

  const handle = async (request, response, next) => {
    const url = new URL(request.url, 'http://localhost')
    if (request.method === 'GET' && url.pathname === '/job') return reply(response, 200, status())
    if (request.method !== 'POST' || url.pathname !== '/') return next()

    if (request.headers['sec-fetch-site'] === 'cross-site') return reply(response, 403, { error: 'Not from this page.' })

    const chunks = []
    let size = 0
    for await (const chunk of request) {
      size += chunk.length
      if (size <= MAX_BYTES) chunks.push(chunk)
    }
    if (size > MAX_BYTES) return reply(response, 413, { error: 'That image is over 25 MB. Use a smaller one.' })
    if (size === 0) return reply(response, 400, { error: 'The image arrived empty. Try again.' })

    const asked = url.searchParams.get('outfit')
    const outfit = gymOutfit(asked)
    if (asked && !outfit) return reply(response, 400, { error: NO_SUCH_OUTFIT })

    const restyled = url.searchParams.get('style') === '3d' || Boolean(outfit)
    const keys = restyled ? readStyleKeys() : null
    if (restyled && !keys) return reply(response, 409, { error: NOT_SET_UP })

    const name = faceName(url.searchParams.get('name'))
    const photo = await savePicture(Buffer.concat(chunks), name)
    if (!photo) return reply(response, 415, { error: NOT_A_PICTURE })

    child?.kill()
    job = { id: ++jobs, name, restyled, outfit, state: 'running', ready: false, error: null }
    if (restyled) restyle(photo, keys)
    else render(photo, 0)
    return reply(response, 202, status())
  }

  return {
    name: 'frames-api',
    apply: 'serve',
    configureServer(server) {
      server.middlewares.use('/api/faces', (request, response, next) => handle(request, response, next).catch(next))
      server.httpServer?.once('close', () => {
        const unfinished = child
        child = null // so the step's handlers stay quiet: this is a shutdown, not a failure
        unfinished?.kill()
      })
    },
  }
}
