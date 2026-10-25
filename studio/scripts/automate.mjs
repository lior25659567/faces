#!/usr/bin/env node
// A headless client for Studio's persisted batch runner; Studio owns the processes.
const args = process.argv.slice(2)
const slugs = []
let checkOnly = false, forceRebuild = false, characterFit = 'auto'
for (let i = 0; i < args.length; i++) {
  if (args[i] === '--exercise' && args[i + 1]) slugs.push(args[++i])
  else if (args[i] === '--references-only') checkOnly = true
  else if (args[i] === '--force') forceRebuild = true
  else if (args[i] === '--character-fit' && ['auto', 'original', 'fitness'].includes(args[i + 1])) characterFit = args[++i]
  else if (args[i] === '--all') { /* the default */ }
  else if (args[i] === '--help') {
    console.log('Start Studio, then: npm run automate -- --all [--references-only] [--force] [--character-fit auto|original|fitness]\nSelect a subset with repeated --exercise <slug>. Automatic proportions adapt arm reach when needed while preserving the original model. Exit 0: passed/ready; 2: review or blocked; 1: error/interrupted.')
    process.exit(0)
  } else { console.error(`Unknown or incomplete argument: ${args[i]}`); process.exit(1) }
}
const base = process.env.STUDIO_URL || 'http://127.0.0.1:5199'
const request = async (url, body) => {
  let response
  try { response = await fetch(new URL(url, base), body ? { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) } : {}) }
  catch { throw Error('Studio is not answering. Start npm run dev in studio/ first.') }
  const result = await response.json()
  if (!response.ok) throw Error(result.error || `Studio returned ${response.status}`)
  return result
}
try {
  let job = await request('/api/automation', { ...(slugs.length ? { slugs } : {}), checkOnly, forceRebuild, characterFit })
  process.once('SIGINT', async () => {
    try { await request(`/api/automation/${job.id}/cancel`, {}); console.log('Stop requested. Studio will finish the current exercise and stop the rest.') }
    finally { process.exit(130) }
  })
  let previous = ''
  for (;;) {
    const text = job.items.map((item) => `${item.name}: ${item.status} — ${item.message}`).join('\n')
    if (text !== previous) { console.log(text); previous = text }
    if (job.status !== 'running') break
    await new Promise((resolve) => setTimeout(resolve, 1000))
    job = await request(`/api/automation/${job.id}`)
  }
  console.log(`Batch ${job.id}: ${job.status}`)
  process.exitCode = job.status !== 'done' || job.items.some((i) => i.status === 'failed') ? 1
    : job.items.some((i) => ['blocked', 'needs-review'].includes(i.status)) ? 2 : 0
} catch (error) { console.error(error.message); process.exitCode = 1 }
