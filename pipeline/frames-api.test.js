import { describe, expect, test } from 'vitest'
import { faceName, gymOutfit, imageKind, readProgress, styleKeys } from './frames-api.js'

describe('faceName', () => {
  test.each([
    ['me', 'me'],
    ['My Photo (2)', 'my-photo-2'],
    ['IMG_1234', 'img-1234'],
    ['  --weird__name--  ', 'weird-name'],
    ['../../etc/passwd', 'etc-passwd'],
  ])('turns %j into the folder name %j', (raw, want) => {
    expect(faceName(raw)).toBe(want)
  })

  test.each([[''], [null], ['///'], ['...']])('falls back to "me" when %j has nothing usable in it', (raw) => {
    expect(faceName(raw)).toBe('me')
  })

  test('keeps folder names to 40 characters', () => {
    expect(faceName('a'.repeat(80))).toBe('a'.repeat(40))
  })
})

describe('readProgress', () => {
  test("reads the generator's progress line", () => {
    expect(readProgress('frame 7/25')).toEqual({ done: 7, total: 25 })
  })

  test.each([
    ['Load motion_extractor from pretrained_weights done.'],
    ['frames:  78%|███████▊  | 7/9 [00:18<00:04,  2.14s/frame]'],
    ['frame x/9'],
    ['the frame 7/25 is done'],
    [''],
  ])('ignores %j', (line) => {
    expect(readProgress(line)).toBeNull()
  })
})

describe('styleKeys', () => {
  test('returns what .env holds when both Cloudflare values are there', () => {
    const env = 'CLOUDFLARE_ACCOUNT_ID=abc123\nCLOUDFLARE_API_TOKEN="s3cret"\nRESTYLE_MODEL=@cf/black-forest-labs/flux-2-klein-4b\n'
    expect(styleKeys(env)).toEqual({
      CLOUDFLARE_ACCOUNT_ID: 'abc123',
      CLOUDFLARE_API_TOKEN: 's3cret',
      RESTYLE_MODEL: '@cf/black-forest-labs/flux-2-klein-4b',
    })
  })

  test.each([
    ['the token is missing', 'CLOUDFLARE_ACCOUNT_ID=abc123\n'],
    ['the account id is missing', 'CLOUDFLARE_API_TOKEN=s3cret\n'],
    ['a value is left blank, as in .env.example', 'CLOUDFLARE_ACCOUNT_ID=\nCLOUDFLARE_API_TOKEN=s3cret\n'],
    ['the file is empty', ''],
  ])('returns null when %s', (_case, env) => {
    expect(styleKeys(env)).toBeNull()
  })
})

describe('imageKind', () => {
  const bytes = (...values) => Buffer.from(values)
  const ascii = (text) => [...Buffer.from(text, 'latin1')]

  test.each([
    ['a JPEG', bytes(0xff, 0xd8, 0xff, 0xe0, 0x00, 0x10, ...ascii('JFIF')), 'jpg'],
    ['a PNG', bytes(0x89, ...ascii('PNG'), 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00), 'png'],
    ['a WebP', bytes(...ascii('RIFF'), 0x24, 0x00, 0x00, 0x00, ...ascii('WEBPVP8 ')), 'webp'],
  ])('recognizes %s by its first bytes', (_label, data, want) => {
    expect(imageKind(data)).toBe(want)
  })

  test.each([
    ['an iPhone HEIC photo, which has to be converted first', bytes(0x00, 0x00, 0x00, 0x18, ...ascii('ftypheic'), 0x00, 0x00)],
    ['a WAV sound, which shares the RIFF wrapper with WebP', bytes(...ascii('RIFF'), 0x24, 0x00, 0x00, 0x00, ...ascii('WAVEfmt '))],
    ['plain text', Buffer.from('not a picture at all')],
    ['nothing', Buffer.alloc(0)],
  ])('returns null for %s', (_label, data) => {
    expect(imageKind(data)).toBeNull()
  })
})

describe('gymOutfit', () => {
  test.each([['tee-pants'], ['tank-shorts']])('knows the outfit %j', (name) => {
    expect(gymOutfit(name)).toBe(name)
  })

  test.each([
    ['no outfit, which means a portrait', null],
    ['an empty one', ''],
    ['an outfit nobody has written a prompt for', 'swimwear'],
    ['a known name in other letter case', 'Tee-Pants'],
    ['a known name with more after it', 'tee-pants --out /tmp/x'],
  ])('returns null for %s', (_case, name) => {
    expect(gymOutfit(name)).toBeNull()
  })
})
