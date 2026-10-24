import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { execFile } from 'node:child_process'
import { promisify } from 'node:util'
import os from 'node:os'
import path from 'node:path'
import { imageKind } from '../../pipeline/frames-api.js'

const run = promisify(execFile)
export const MAX_BYTES = 25 * 1024 * 1024
const VIDEO_NAME = /\.(mov|mp4|m4v)$/i

export class UploadError extends Error {
  constructor(message, status = 415) {
    super(message)
    this.name = 'UploadError'
    this.status = status
  }
}

export const VIDEO_MESSAGE = "Videos aren't supported yet. Upload still images of the key positions."
export const TOO_BIG_MESSAGE = 'That picture is over 25 MB.'
export const NOT_A_PICTURE_MESSAGE = "That file isn't a picture this Mac can open."


const isVideo = (bytes, name) =>
  (bytes.length >= 8 && bytes.toString('latin1', 4, 8) === 'ftyp') || VIDEO_NAME.test(name || '')

// An iPhone HEIC has the same 'ftyp' marker as a video, so a brand check tells them apart.
const isHeif = (bytes) =>
  bytes.length >= 12 && /^(heic|heix|hevc|hevx|mif1|msf1|heim|heis)$/.test(bytes.toString('latin1', 8, 12))

// Cheap refusals that need only the file name and the first bytes, so the API can decide
// before it buffers a big upload.
export const refuseEarly = (bytes, name) => {
  if (isVideo(bytes, name) && !(isHeif(bytes) && !VIDEO_NAME.test(name || ''))) throw new UploadError(VIDEO_MESSAGE)
  if (bytes.length >= 4 && bytes.toString('latin1', 0, 4) === '%PDF') throw new UploadError(NOT_A_PICTURE_MESSAGE)
}

export const prepareUpload = async (bytes, name) => {
  refuseEarly(bytes, name)
  if (bytes.length > MAX_BYTES) throw new UploadError(TOO_BIG_MESSAGE, 413)

  const kind = imageKind(bytes)
  if (kind) return { ext: kind, bytes }

  const cannot = new UploadError(NOT_A_PICTURE_MESSAGE)
  const dir = await mkdtemp(path.join(os.tmpdir(), 'studio-upload-'))
  try {
    const source = path.join(dir, 'in.upload')
    const target = path.join(dir, 'out.jpg')
    await writeFile(source, bytes)
    await run('sips', ['-s', 'format', 'jpeg', source, '--out', target], { timeout: 30_000 })
    // sips exits 0 even when it read nothing, so the output file is the proof
    const converted = await readFile(target)
    if (imageKind(converted) !== 'jpg') throw cannot
    return { ext: 'jpg', bytes: converted }
  } catch {
    throw cannot
  } finally {
    await rm(dir, { recursive: true, force: true })
  }
}
