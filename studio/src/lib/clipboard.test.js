import { describe, expect, test } from 'vitest'
import { imagesFromClipboard } from './clipboard.js'

const png = (name, type = 'image/png') => new File([new Uint8Array([137, 80, 78, 71])], name, { type })
const item = (file) => ({ kind: 'file', type: file.type, getAsFile: () => file })
const textItem = { kind: 'string', type: 'text/plain', getAsFile: () => null }
const at = new Date(2026, 9, 22, 13, 5, 9)

describe('imagesFromClipboard', () => {
  test('a copied image (screenshot, browser image) with no real name gets one', () => {
    const shot = png('image.png')
    const files = imagesFromClipboard({ files: [], items: [item(shot)] }, at)
    expect(files.map((f) => f.name)).toEqual(['pasted-20261022-130509-1.png'])
    expect(files[0].type).toBe('image/png')
    expect(files[0].size).toBe(4)
  })

  test('the name matches the picture type', () => {
    const files = imagesFromClipboard({ files: [png('', 'image/jpeg'), png('', 'image/webp')] }, at)
    expect(files.map((f) => f.name)).toEqual(['pasted-20261022-130509-1.jpg', 'pasted-20261022-130509-2.webp'])
  })

  test('files copied in Finder keep their names', () => {
    const files = imagesFromClipboard({ files: [png('15.png'), png('16.jpg', 'image/jpeg')], items: [] }, at)
    expect(files.map((f) => f.name)).toEqual(['15.png', '16.jpg'])
  })

  test('the same picture listed as a file and as an item is taken once', () => {
    const shot = png('image.png')
    expect(imagesFromClipboard({ files: [shot], items: [item(shot)] }, at)).toHaveLength(1)
  })

  test('named non-pictures from Finder go through, so the server can refuse them in plain words', () => {
    const clip = new File(['x'], 'clip.mov', { type: 'video/quicktime' })
    expect(imagesFromClipboard({ files: [clip] }, at).map((f) => f.name)).toEqual(['clip.mov'])
  })

  test('a text-only paste has no pictures', () => {
    expect(imagesFromClipboard({ files: [], items: [textItem] }, at)).toEqual([])
    expect(imagesFromClipboard({ files: [], items: [] }, at)).toEqual([])
    expect(imagesFromClipboard(null, at)).toEqual([])
  })
})
