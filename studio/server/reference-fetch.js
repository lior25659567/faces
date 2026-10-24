import { createHash } from 'node:crypto'
import * as http from 'node:http'
import { imageKind } from '../../pipeline/frames-api.js'
import { REFERENCE_SOURCE, referenceImageFor } from './reference-catalog.js'

export const MAX_REFERENCE_BYTES = 25 * 1024 * 1024
export class ReferenceFetchError extends Error {
  constructor(message, code, { status = null, retryable = false } = {}) {
    super(message)
    this.name = 'ReferenceFetchError'
    this.code = code
    this.status = status
    this.retryable = retryable
  }
}

const failure = (message, code, details) => new ReferenceFetchError(message, code, details)
const PROXY_READY = Symbol.for('exercise-studio.reference-proxy-ready')

// Node 24 offers the same proxy routing used by --use-env-proxy at runtime. Honor
// the platform's existing route and NO_PROXY exclusions without inspecting/logging
// credentials or introducing an undeclared HTTP-client package. Plain local setups
// need no proxy configuration.
export const configureReferenceProxy = ({ environment = process.env, httpModule = http } = {}) => {
  const native = environment === process.env && httpModule === http
  if (native && globalThis[PROXY_READY]) return true
  if (!['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy'].some((name) => Boolean(environment[name])) ||
      typeof httpModule.setGlobalProxyFromEnv !== 'function') return false
  try {
    httpModule.setGlobalProxyFromEnv(environment)
  } catch {
    throw failure('The environment network proxy could not be configured.', 'PROXY_CONFIG')
  }
  if (native) globalThis[PROXY_READY] = true
  return true
}

export const fetchReference = async (image, {
  fetchImpl = globalThis.fetch, timeoutMs = 20_000, maxBytes = MAX_REFERENCE_BYTES, signal,
} = {}) => {
  const trusted = referenceImageFor(image?.url)
  if (!trusted) throw failure('This reference URL is not in the reviewed pinned catalog.', 'UNTRUSTED_SOURCE')
  if (!Number.isInteger(maxBytes) || maxBytes < 1 || maxBytes > MAX_REFERENCE_BYTES ||
      !Number.isFinite(timeoutMs) || timeoutMs < 1 || timeoutMs > 120_000) {
    throw failure('Invalid reference download limits.', 'INVALID_LIMITS')
  }
  if (signal?.aborted) throw failure('Reference download cancelled.', 'ABORTED')
  if (fetchImpl === globalThis.fetch) configureReferenceProxy()
  const controller = new AbortController()
  let rejectCancelled
  const cancelled = new Promise((_, reject) => { rejectCancelled = reject })
  const cancel = () => {
    controller.abort()
    rejectCancelled(failure('Reference download cancelled.', 'ABORTED'))
  }
  signal?.addEventListener('abort', cancel, { once: true })
  let timer
  const deadline = new Promise((_, reject) => {
    timer = setTimeout(() => {
      controller.abort()
      reject(failure('The reference download timed out.', 'TIMEOUT', { retryable: true }))
    }, timeoutMs)
  })
  const download = async () => {
    let response
    try {
      response = await fetchImpl(trusted.url, { signal: controller.signal, redirect: 'error', headers: { accept: 'image/jpeg,image/png,image/webp' } })
    } catch (error) {
      if (signal?.aborted) throw failure('Reference download cancelled.', 'ABORTED')
      throw failure('Could not download the reference using the current network route.', 'NETWORK', { retryable: true })
    }
    if (!response.ok) {
      throw failure(`Reference source returned HTTP ${response.status}.`, 'HTTP', {
        status: response.status, retryable: response.status === 429 || response.status >= 500,
      })
    }
    if (response.url && response.url !== trusted.url) throw failure('The reference source redirected to an unreviewed URL.', 'UNTRUSTED_SOURCE')
    const declared = Number(response.headers.get('content-length'))
    if (Number.isFinite(declared) && declared > maxBytes) {
      controller.abort()
      throw failure('The reference image is larger than the download limit.', 'TOO_LARGE')
    }
    const mime = (response.headers.get('content-type') || '').split(';')[0].trim().toLowerCase()
    if (mime && !['image/jpeg', 'image/png', 'image/webp', 'application/octet-stream'].includes(mime)) {
      controller.abort()
      throw failure('The reference source did not return an image.', 'NOT_IMAGE')
    }
    const reader = response.body?.getReader()
    if (!reader) throw failure('The reference source returned an empty body.', 'NOT_IMAGE')
    const chunks = []
    let size = 0
    const abortReader = () => { reader.cancel().catch(() => {}) }
    controller.signal.addEventListener('abort', abortReader, { once: true })
    try {
      while (true) {
        const { value, done } = await reader.read()
        if (done) break
        size += value.byteLength
        if (size > maxBytes) {
          controller.abort()
          reader.cancel().catch(() => {})
          throw failure('The reference image is larger than the download limit.', 'TOO_LARGE')
        }
        chunks.push(Buffer.from(value))
      }
    } catch (error) {
      if (error instanceof ReferenceFetchError) throw error
      if (signal?.aborted) throw failure('Reference download cancelled.', 'ABORTED')
      throw failure('The reference image stream stopped before it completed.', 'NETWORK', { retryable: true })
    } finally {
      controller.signal.removeEventListener('abort', abortReader)
      reader.releaseLock()
    }
    const bytes = Buffer.concat(chunks, size)
    const ext = imageKind(bytes)
    if (!ext) throw failure('The reference body is not a supported image.', 'NOT_IMAGE')
    const sha256 = createHash('sha256').update(bytes).digest('hex')
    if (sha256 !== trusted.sha256) throw failure('The reference image does not match the pinned source checksum.', 'CHECKSUM_MISMATCH')
    return {
      bytes, ext, sha256, sourceUrl: trusted.url,
      provenance: {
        ...REFERENCE_SOURCE, sourceId: trusted.sourceId, sourceUrl: trusted.url, sha256,
        phase: trusted.phase, view: trusted.view, contextOnly: trusted.contextOnly,
      },
    }
  }
  try {
    return await Promise.race([download(), deadline, cancelled])
  } finally {
    clearTimeout(timer)
    signal?.removeEventListener('abort', cancel)
  }
}
