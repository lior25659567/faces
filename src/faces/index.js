// Every folder that pipeline/generate-frames.py writes next to this file is picked up here.
const manifests = import.meta.glob('./*/manifest.json', { eager: true, import: 'default' })
const images = import.meta.glob('./*/{base,atlas}.webp', { eager: true, query: '?url', import: 'default' })

// A re-render keeps the file names, so the manifest's version goes into the URL to get past the cache.
const versioned = (url, version) => `${url}${url.includes('?') ? '&' : '?'}v=${version}`

const faces = Object.entries(manifests).map(([path, manifest]) => {
  const folder = path.slice(0, path.lastIndexOf('/'))
  return {
    name: folder.slice(2),
    base: versioned(images[`${folder}/base.webp`], manifest.version),
    atlas: versioned(images[`${folder}/atlas.webp`], manifest.version),
    ...manifest,
  }
})

export default faces
