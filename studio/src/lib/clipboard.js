const EXT = {
  'image/png': 'png', 'image/jpeg': 'jpg', 'image/jpg': 'jpg', 'image/webp': 'webp', 'image/gif': 'gif',
  'image/heic': 'heic', 'image/heif': 'heif', 'image/tiff': 'tiff', 'image/bmp': 'bmp',
}

// A picture copied from a browser or a screenshot comes with a stock name ("image.png") or none.
const STOCK_NAME = /^(image\.[a-z0-9]+)?$/i

const pad = (v) => String(v).padStart(2, '0')
const stamp = (d) =>
  `${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}-${pad(d.getHours())}${pad(d.getMinutes())}${pad(d.getSeconds())}`

// The files in a paste, ready to upload like dropped files. Finder files keep their names (and
// non-pictures among them go through, so the server refuses them in the same plain words as a drop);
// unnamed copied pictures are named pasted-<YYYYMMDD-HHMMSS>-<n>.<type>. Text alone gives [].
export const imagesFromClipboard = (clipboardData, now = new Date()) => {
  if (!clipboardData) return []
  const listed = Array.from(clipboardData.files ?? [])
  // Browsers list a copied picture under both `files` and `items`; take one of the two, not both.
  const raw = listed.length
    ? listed
    : Array.from(clipboardData.items ?? [])
      .filter((i) => i.kind === 'file')
      .map((i) => i.getAsFile())
      .filter(Boolean)

  let n = 0
  return [...new Set(raw)].flatMap((file) => {
    if (!STOCK_NAME.test(file.name ?? '')) return [file]
    if (!file.type?.startsWith('image/')) return []
    n += 1
    return [new File([file], `pasted-${stamp(now)}-${n}.${EXT[file.type] ?? 'png'}`, { type: file.type })]
  })
}
