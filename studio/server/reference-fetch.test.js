import { expect, test } from 'vitest'
import { readFileSync } from 'node:fs'
import { configureReferenceProxy, fetchReference } from './reference-fetch.js'
import { getReferenceSpec } from './reference-catalog.js'

const image = getReferenceSpec('Barbell Bent-Over Row').images[0]
const bytes = readFileSync(new URL('../../reference/exercise-row/start.jpg', import.meta.url))
const reply = (body = bytes, options = {}) => new Response(body, { status: 200, headers: { 'content-type': 'image/jpeg' }, ...options })

test('downloads pinned bytes and returns matching provenance without writing files', async () => {
  let request
  const result = await fetchReference(image, { fetchImpl: async (...args) => { request = args; return reply() } })
  expect(request[0]).toBe(image.url)
  expect(request[1].redirect).toBe('error')
  expect(result.bytes).toEqual(bytes)
  expect(result.ext).toBe('jpg')
  expect(result.sha256).toBe(image.sha256)
  expect(result.provenance).toMatchObject({ sourceId: 'Bent_Over_Barbell_Row', commit: image.commit, view: 'three-quarter', contextOnly: false })
})

test('rejects arbitrary URLs before any network request', async () => {
  let calls = 0
  for (const url of ['http://127.0.0.1/private', image.url + '?raw=1', image.url.replace(image.commit, 'main'),
    image.url.replace('raw.githubusercontent.com', 'raw.githubusercontent.com.evil.example'),
    image.url.replace('/exercises/', '/exercises/../')]) {
    await expect(fetchReference({ url }, { fetchImpl: async () => { calls++; return reply() } })).rejects.toMatchObject({ code: 'UNTRUSTED_SOURCE' })
  }
  expect(calls).toBe(0)
})

test('the caller cannot override the catalog checksum', async () => {
  const altered = Buffer.from(bytes); altered[altered.length - 10] ^= 1
  await expect(fetchReference({ ...image, sha256: '0'.repeat(64) }, { fetchImpl: async () => reply(altered) }))
    .rejects.toMatchObject({ code: 'CHECKSUM_MISMATCH', retryable: false })
})

test('refuses HTML/error payloads instead of importing them as pictures', async () => {
  await expect(fetchReference(image, { fetchImpl: async () => reply('<h1>Denied</h1>', { headers: { 'content-type': 'text/html' } }) }))
    .rejects.toMatchObject({ code: 'NOT_IMAGE' })
  await expect(fetchReference(image, { fetchImpl: async () => reply('not an image') }))
    .rejects.toMatchObject({ code: 'NOT_IMAGE' })
  await expect(fetchReference(image, { fetchImpl: async () => reply(null) }))
    .rejects.toMatchObject({ code: 'NOT_IMAGE' })
})

test('refuses declared and streaming oversize bodies', async () => {
  await expect(fetchReference(image, { maxBytes: 100, fetchImpl: async () => reply(bytes, { headers: { 'content-length': '101' } }) }))
    .rejects.toMatchObject({ code: 'TOO_LARGE' })
  const stream = new ReadableStream({ start(controller) { controller.enqueue(new Uint8Array(80)); controller.enqueue(new Uint8Array(80)); controller.close() } })
  await expect(fetchReference(image, { maxBytes: 100, fetchImpl: async () => reply(stream) }))
    .rejects.toMatchObject({ code: 'TOO_LARGE' })
})

test('keeps authorization denial distinct from temporary source failures', async () => {
  await expect(fetchReference(image, { fetchImpl: async () => reply('denied', { status: 403 }) }))
    .rejects.toMatchObject({ code: 'HTTP', status: 403, retryable: false })
  await expect(fetchReference(image, { fetchImpl: async () => reply('busy', { status: 503 }) }))
    .rejects.toMatchObject({ code: 'HTTP', status: 503, retryable: true })
})

test('does not follow a redirect even from an injected response', async () => {
  const response = reply()
  Object.defineProperty(response, 'url', { value: 'https://unreviewed.example/image.jpg' })
  await expect(fetchReference(image, { fetchImpl: async () => response })).rejects.toMatchObject({ code: 'UNTRUSTED_SOURCE' })
})

test('bounds the whole download even when a transport ignores abort', async () => {
  await expect(fetchReference(image, { timeoutMs: 15, fetchImpl: () => new Promise(() => {}) }))
    .rejects.toMatchObject({ code: 'TIMEOUT', retryable: true })
})

test('batch cancellation terminates a hanging transport promptly', async () => {
  const controller = new AbortController()
  const result = fetchReference(image, { signal: controller.signal, timeoutMs: 1000, fetchImpl: () => new Promise(() => {}) })
  controller.abort()
  await expect(result).rejects.toMatchObject({ code: 'ABORTED', retryable: false })
})

test('normalizes transport and interrupted-stream errors for the batch report', async () => {
  await expect(fetchReference(image, { fetchImpl: async () => { throw new Error('offline') } })).rejects.toMatchObject({ code: 'NETWORK', retryable: true })
  const stream = new ReadableStream({ start(c) { c.error(Error('interrupted')) } })
  await expect(fetchReference(image, { fetchImpl: async () => reply(stream) })).rejects.toMatchObject({ code: 'NETWORK', retryable: true })
})


test('uses the provided environment proxy route and preserves NO_PROXY', () => {
  const environment = { HTTPS_PROXY: 'http://proxy.example:8080', NO_PROXY: 'localhost,127.0.0.1' }
  let received
  expect(configureReferenceProxy({ environment, httpModule: { setGlobalProxyFromEnv: (value) => { received = value } } })).toBe(true)
  expect(received).toBe(environment)
  expect(configureReferenceProxy({ environment: {}, httpModule: { setGlobalProxyFromEnv: () => { throw Error('must not run') } } })).toBe(false)
})

test('proxy configuration failures do not echo potentially credentialed settings', () => {
  expect(() => configureReferenceProxy({ environment: { HTTPS_PROXY: 'configured' }, httpModule: {
    setGlobalProxyFromEnv: () => { throw Error('credential-bearing details') },
  } })).toThrow('The environment network proxy could not be configured.')
})


test('timeout cancels a hanging response stream, not only its request', async () => {
  let cancelled = false
  const stream = new ReadableStream({ cancel() { cancelled = true } })
  await expect(fetchReference(image, { timeoutMs: 15, fetchImpl: async () => reply(stream) }))
    .rejects.toMatchObject({ code: 'TIMEOUT' })
  expect(cancelled).toBe(true)
})
