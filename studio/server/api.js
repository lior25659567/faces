import { createReadStream, existsSync, readFileSync, statSync } from 'node:fs'
import path from 'node:path'
import { MAX_BYTES, TOO_BIG_MESSAGE, UploadError, prepareUpload, refuseEarly } from './images.js'
import { BadInputError, NotFoundError, characterFitOf, motionPresetOf, referenceBundleOf } from './store.js'
import { recentWork } from './work.js'
import { validatedPreviews } from './validated-previews.js'

const TYPES = {
  '.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.webp': 'image/webp',
  '.gif': 'image/gif', '.json': 'application/json', '.log': 'text/plain; charset=utf-8',
  '.txt': 'text/plain; charset=utf-8', '.blend': 'application/octet-stream',
}
const LOCAL_HOST = /^(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$/
const STUDIO_ORIGIN = 'http://localhost:5199'
const FOREIGN = `The Studio only accepts changes from its own page (${STUDIO_ORIGIN}).`
const PHASES = ['start', 'middle', 'end']
const VIEWS = ['side', 'front', 'behind', 'three-quarter', 'frontal']

class HttpError extends Error {
  constructor(status, message) {
    super(message)
    this.status = status
  }
}

// Reads the whole body. With `upload`, it refuses early (name, Content-Length, first bytes), stops
// keeping bytes once refused, and drains the rest so the plain message reaches the client.
const readBody = (req, upload) =>
  new Promise((resolve, reject) => {
    const chunks = []
    let size = 0
    let refusal = null
    // Too big by Content-Length: stop keeping bytes at once, but let the first bytes say "video" first.
    const oversize = upload && Number(req.headers['content-length']) > MAX_BYTES
    if (upload) {
      try {
        refuseEarly(Buffer.alloc(0), upload.name)
      } catch (e) {
        refusal = e
      }
    }
    req.on('data', (c) => {
      if (refusal) return
      if (upload && size === 0) {
        try { refuseEarly(c, upload.name) } catch (e) { refusal = e; chunks.length = 0; return }
        if (oversize) { refusal = new UploadError(TOO_BIG_MESSAGE, 413); return }
      }
      size += c.length
      if (upload && size > MAX_BYTES) { refusal = new UploadError(TOO_BIG_MESSAGE, 413); chunks.length = 0; return }
      chunks.push(c)
    })
    req.on('end', () => (refusal ? reject(refusal) : resolve(Buffer.concat(chunks))))
    req.on('error', reject)
  })

const readJsonBody = async (req) => {
  const raw = (await readBody(req)).toString('utf8')
  let body
  try {
    body = raw ? JSON.parse(raw) : {}
  } catch {
    throw new HttpError(400, 'That request was not valid JSON.')
  }
  if (!body || typeof body !== 'object' || Array.isArray(body)) throw new HttpError(400, 'That request needs a JSON object.')
  return body
}

const sendJson = (res, status, body) => {
  res.statusCode = status
  res.setHeader('content-type', 'application/json')
  res.end(JSON.stringify(body))
}

const readIf = (file) => (existsSync(file) ? JSON.parse(readFileSync(file, 'utf8')) : null)

export const studioApi = ({ store, queue, work, watcher, automation, onListening }) => {
  const exerciseOr404 = (slug) => {
    const exercise = store.getExercise(slug)
    if (!exercise) throw new HttpError(404, 'That exercise does not exist.')
    return exercise
  }

  // running | needs-reference | passing | failing | unmeasured | not-run — each from the run's own files.
  // Passing only when nothing fails AND everything that would be judged was measured on the character.
  const statusOf = (slug) => {
    const { running, queued } = queue.state()
    if (running?.slug === slug || (Array.isArray(queued) && queued.some((q) => q.slug === slug))) return 'running'
    if (!store.listReferences(slug).some((i) => !i.contextOnly)) return 'needs-reference'
    const latest = store.listRuns(slug).find((r) => !r.readOnly && (r.status === 'done' || r.status === 'failed'))
    if (!latest) return 'not-run'
    if (latest.status === 'failed') return 'failing'
    const { pass = 0, fail = 0, unmeasured = 0, limited = 0, uncertain = 0 } = latest.summary || {}
    if (fail > 0) return 'failing'
    if (unmeasured > 0) return 'unmeasured'
    if (limited > 0 || uncertain > 0 || (latest.automationJobId && ['pass','fail','limited','uncertain','unmeasured'].some((key) => !Number.isFinite(latest.summary?.[key])))) return 'review'
    return pass > 0 ? 'passing' : 'not-run'
  }

  // A change (anything but a read) from a browser must come from the Studio's own page: any other web
  // page the user opens could otherwise send one to localhost. Scripts send no Origin and go through.
  const fromOwnPage = (req) => {
    const origin = req.headers.origin
    return origin === undefined || origin === STUDIO_ORIGIN || origin === `http://${req.headers.host}`
  }

  const serveFile = (req, res, urlPath) => {
    let rel
    try {
      rel = decodeURIComponent(urlPath.slice('/files/'.length))
    } catch {
      throw new HttpError(404, 'Not found.')
    }
    const root = path.resolve(store.root)
    const file = path.resolve(root, rel)
    if (rel.includes('\0') || !file.startsWith(root + path.sep) || !existsSync(file) || !statSync(file).isFile()) {
      throw new HttpError(404, 'Not found.')
    }
    res.statusCode = 200
    res.setHeader('content-type', TYPES[path.extname(file).toLowerCase()] || 'application/octet-stream')
    res.setHeader('cache-control', 'no-cache')
    createReadStream(file).pipe(res)
  }

  const route = async (req, res, pathname) => {
    const method = req.method
    let seg
    try { seg = pathname.split('/').filter(Boolean).map(decodeURIComponent) } // ['api', ...]
    catch { throw new HttpError(404, 'Not found.') }
    if (seg[0] === 'files') return serveFile(req, res, pathname)

    if (seg[1] === 'previews') {
      if (seg.length === 2 && method === 'GET') return sendJson(res, 200, validatedPreviews.list(store.listExercises()))
      if (seg.length === 4 && (method === 'GET' || method === 'HEAD')) {
        const asset = validatedPreviews.asset(seg[2], seg[3])
        if (!asset) throw new HttpError(404, 'That bundled preview asset does not exist.')
        res.statusCode = 200
        res.setHeader('content-type', asset.type)
        res.setHeader('content-length', asset.bytes.length)
        res.setHeader('cache-control', 'no-cache')
        res.setHeader('etag', `"${asset.sha256}"`)
        return res.end(method === 'HEAD' ? undefined : asset.bytes)
      }
      throw new HttpError(404, 'That bundled preview does not exist.')
    }

    if (seg[1] === 'automation' && automation) {
      if (seg.length === 2 && method === 'GET') return sendJson(res, 200, automation.summary())
      if (seg.length === 2 && method === 'POST') return sendJson(res, 201, automation.start(await readJsonBody(req)))
      const job = automation.get(seg[2])
      if (!job) throw new HttpError(404, 'That automation batch does not exist.')
      if (seg.length === 3 && method === 'GET') return sendJson(res, 200, job)
      if (seg.length === 4 && seg[3] === 'cancel' && method === 'POST') return sendJson(res, 200, automation.cancel(seg[2]))
    }

    if (seg[1] === 'state' && method === 'GET') return sendJson(res, 200, queue.state())

    // Work going on behind the scenes: every running item, then the newest 50 others (mtime cache).
    if (seg[1] === 'work' && seg.length === 2 && method === 'GET') return sendJson(res, 200, work ? recentWork(work.cached()) : [])

    if (seg[1] === 'runs' && method === 'GET' && seg.length === 4) {
      const run = store.readRun(seg[2], seg[3])
      if (!run) throw new HttpError(404, 'That run does not exist.')
      return sendJson(res, 200, run)
    }

    if (seg[1] === 'exercises') {
      if (seg.length === 2) {
        if (method === 'GET') return sendJson(res, 200, store.listExercises().map((e) => ({ ...e, status: statusOf(e.slug) })))
        if (method === 'POST') {
          return sendJson(res, 201, store.addExercise(await readJsonBody(req)))
        }
      }
      const slug = seg[2]
      if (seg.length === 3 && method === 'GET') {
        const exercise = exerciseOr404(slug)
        const runs = store.listRuns(slug)
        let latest = null
        if (runs[0]) {
          const dir = store.runDir(slug, runs[0].id)
          latest = {
            run: runs[0],
            checks: readIf(path.join(dir, 'checks.json')),
            ours: readIf(path.join(dir, 'ours.json')),
            targets: readIf(path.join(dir, 'targets.json')),
            preflight: readIf(path.join(dir, 'preflight.json')),
            measurements: readIf(path.join(dir, 'measurements.json')),
            snapshot: readIf(path.join(dir, 'references.json')),
            // why each picture went unused when the read step stopped (no targets.json then)
            readProblems: readIf(path.join(dir, 'read-problems.json')),
          }
        }
        return sendJson(res, 200, { exercise, references: store.listReferences(slug), runs, latest })
      }
      if (seg[3] === 'references') {
        exerciseOr404(slug)
        if (seg.length === 4 && method === 'POST') {
          const original = decodeURIComponent(String(req.headers['x-filename'] || 'picture'))
          const { ext, bytes } = await prepareUpload(await readBody(req, { name: original }), original)
          return sendJson(res, 201, store.addReference(slug, bytes, original, ext))
        }
        if (seg.length === 5 && seg[4] === 'order' && method === 'PUT') {
          const { ids } = await readJsonBody(req)
          if (!Array.isArray(ids)) throw new HttpError(400, 'Send the picture ids in order.')
          store.reorderReferences(slug, ids)
          return sendJson(res, 200, store.listReferences(slug))
        }
        if (seg.length === 5 && method === 'PATCH') {
          const { phase, view, contextOnly } = await readJsonBody(req)
          const patch = {}
          if (phase !== undefined) {
            if (!PHASES.includes(phase)) throw new HttpError(400, 'Phase must be start, middle or end.')
            patch.phase = phase
          }
          if (view !== undefined) {
            if (view !== null && !VIEWS.includes(view)) throw new HttpError(400, 'View must be side, front, behind, three-quarter or frontal.')
            patch.view = view
          }
          if (contextOnly !== undefined) {
            if (typeof contextOnly !== 'boolean') throw new HttpError(400, 'Context-only must be true or false.')
            patch.contextOnly = contextOnly
          }
          return sendJson(res, 200, store.updateReference(slug, seg[4], patch))
        }
        if (seg.length === 5 && method === 'DELETE') {
          store.removeReference(slug, seg[4])
          return sendJson(res, 200, { ok: true })
        }
      }
      if (seg[3] === 'read' && seg.length === 4 && method === 'POST') {
        const exercise = exerciseOr404(slug)
        const body = await readJsonBody(req)
        return sendJson(res, 201, queue.enqueueRead(slug, { motionPreset: motionPresetOf(exercise.name, body.motionPreset),
          referenceBundle: referenceBundleOf(exercise.name, body.referenceBundle) }))
      }
      if (seg[3] === 'runs' && seg.length === 4 && method === 'POST') {
        const exercise = exerciseOr404(slug)
        const body = await readJsonBody(req)
        return sendJson(res, 201, queue.enqueue(slug, { characterFit: characterFitOf(body.characterFit),
          motionPreset: motionPresetOf(exercise.name, body.motionPreset),
          referenceBundle: referenceBundleOf(exercise.name, body.referenceBundle) }))
      }
    }
    throw new HttpError(404, 'Not found.')
  }

  return {
    name: 'exercise-studio-api',
    configureServer(server) {
      // Runs left "running" on disk are recovered (and runs a stop cut off are restarted), and work
      // items watched (and re-run), only once this server really listens. A second `vite` that loses
      // the port, or a vitest run (no HTTP server), never touches them.
      server.httpServer?.once('listening', () => {
        automation?.recoverInterrupted()
        queue.recoverInterrupted?.()
        queue.restartInterrupted?.()
        watcher?.start()
        onListening?.()
      })
      // The Blender child must not outlive the Studio.
      server.httpServer?.once('close', () => {
        watcher?.stop()
        automation?.stop()
        queue.stop?.()
      })
      server.middlewares.use(async (req, res, next) => {
        // Raw path: URL parsing would fold %2e%2e segments away before the /files guard sees them.
        const pathname = req.url.split('?')[0].split('#')[0]
        if (!pathname.startsWith('/api/') && !pathname.startsWith('/files/')) return next()
        try {
          // Reference images belong to other companies: answer only to localhost names.
          if (!LOCAL_HOST.test(req.headers.host || '')) throw new HttpError(403, 'The Studio only answers on localhost.')
          if (req.method !== 'GET' && req.method !== 'HEAD' && !fromOwnPage(req)) throw new HttpError(403, FOREIGN)
          await route(req, res, pathname)
        } catch (e) {
          const status = e instanceof NotFoundError ? 404 : e instanceof BadInputError ? 400 : e.status || 500
          if (res.headersSent) return res.end()
          sendJson(res, status, { error: e.message })
        }
      })
    },
  }
}
