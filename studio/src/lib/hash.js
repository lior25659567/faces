// Exercise routes are #/<slug>; #/<slug>/run/<id> also points at one run (the "Re-ran automatically"
// links). Anything else (an in-page anchor like #references) is not a route.
export const routeFromHash = (hash) => {
  const m = /^#\/([^/]+)(?:\/run\/([^/]+))?$/.exec(hash ?? '')
  if (!m) return null
  try {
    const slug = decodeURIComponent(m[1])
    return slug ? { slug, runId: m[2] ? decodeURIComponent(m[2]) : null } : null
  } catch {
    return null
  }
}

export const slugFromHash = (hash) => routeFromHash(hash)?.slug ?? null

export const hashFor = (slug, runId = null) =>
  `#/${encodeURIComponent(slug)}${runId ? `/run/${encodeURIComponent(runId)}` : ''}`
