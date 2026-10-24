import { useEffect, useRef } from 'react'

// Calls `fn` now and then every `ms` after each call settles (never two at once).
// Changing `ms` restarts the timer with an immediate call.
export const usePoll = (fn, ms) => {
  const latest = useRef(fn)
  useEffect(() => {
    latest.current = fn
  }, [fn])

  useEffect(() => {
    if (!ms) return undefined
    let alive = true
    let timer = null
    const tick = async () => {
      try {
        await latest.current()
      } catch {
        // the caller shows its own errors; keep polling
      }
      if (alive) timer = setTimeout(tick, ms)
    }
    tick()
    return () => {
      alive = false
      clearTimeout(timer)
    }
  }, [ms])
}
