import { expect, test } from 'vitest'
import { readFile, writeFile, mkdtemp } from 'node:fs/promises'
import { execFileSync } from 'node:child_process'
import os from 'node:os'
import path from 'node:path'
import { prepareUpload, UploadError } from './images.js'

const SAMPLE = new URL('../../3d-character/reference-squat/15.png', import.meta.url)

test('passes a png through', async () => {
  const bytes = await readFile(SAMPLE)
  const out = await prepareUpload(bytes, '15.png')
  expect(out.ext).toBe('png')
})

test('refuses videos with a plain message', async () => {
  const bytes = Buffer.concat([Buffer.from([0, 0, 0, 24]), Buffer.from('ftypmp42'), Buffer.alloc(32)])
  await expect(prepareUpload(bytes, 'clip.bin')).rejects.toThrow(UploadError)
  await expect(prepareUpload(bytes, 'clip.bin')).rejects.toThrow(
    "Videos aren't supported yet. Upload still images of the key positions.",
  )
  await expect(prepareUpload(Buffer.alloc(10), 'x.MOV')).rejects.toThrow(/Videos aren't supported/)
})

test('refuses over 25 MB and non-pictures', async () => {
  await expect(prepareUpload(Buffer.alloc(26 * 1024 * 1024), 'big.png')).rejects.toThrow('That picture is over 25 MB.')
  await expect(prepareUpload(Buffer.from('hello world'), 'a.pdf')).rejects.toThrow(
    "That file isn't a picture this Mac can open.",
  )
})

test('converts heic with sips', async () => {
  const dir = await mkdtemp(path.join(os.tmpdir(), 'studio-heic-'))
  const heic = path.join(dir, 'a.heic')
  execFileSync('sips', ['-s', 'format', 'heic', fileURLPath(SAMPLE), '--out', heic])
  const out = await prepareUpload(await readFile(heic), 'IMG_0001.HEIC')
  expect(out.ext).toBe('jpg')
  expect(out.bytes[0]).toBe(0xff)
})

function fileURLPath(u) { return decodeURIComponent(u.pathname) }

test('refuses a real PDF before sips can render it', async () => {
  const pdf = await readFile(new URL('./fixtures/minimal.pdf', import.meta.url))
  expect(pdf.toString('latin1', 0, 5)).toBe('%PDF-')
  await expect(prepareUpload(pdf, 'plan.PDF')).rejects.toThrow("That file isn't a picture this Mac can open.")
  await expect(prepareUpload(pdf, 'renamed.png')).rejects.toThrow("That file isn't a picture this Mac can open.")
})
