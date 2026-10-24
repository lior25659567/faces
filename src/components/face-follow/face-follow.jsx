import { useEffect, useRef } from 'react'
import { ease, gazeToCell, pointerToGaze } from './gaze.js'
import './face-follow.css'

const HALF_LIFE = 70 // ms for the head to close half the distance to the pointer
const SETTLED = 0.001
const DISSOLVE = 120 // ms for a new head angle to fade in over the one before it
const ON_EDGE = 0.002 // how close to the still's edge a side of the head counts as sitting on it

// A character still whose head turns toward the pointer.
// `face` is one folder written by pipeline/generate-frames.py: { base, atlas, width, height, head, cols, rows, gutter }.
const FaceFollow = ({ face, alt = '', className = '' }) => {
  const headRef = useRef(null)
  const angleRef = useRef(null)
  const beforeRef = useRef(null)
  const { base, atlas, width, height, head, cols, rows, gutter } = face

  useEffect(() => {
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return undefined

    const angle = angleRef.current
    const before = beforeRef.current
    const listeners = new AbortController()
    const gaze = { u: 0, v: 0 }
    let target = { u: 0, v: 0 }
    let shown = gazeToCell(gaze, cols, rows)
    let dissolve = null
    let pointer = null
    let frame = 0
    let last = 0

    // The atlas holds a few dozen angles, so each new one fades in over the last instead of jumping to it.
    const show = ({ col, row }) => {
      if (col === shown.col && row === shown.row) return
      before.style.setProperty('--col', shown.col)
      before.style.setProperty('--row', shown.row)
      angle.style.setProperty('--col', col)
      angle.style.setProperty('--row', row)
      dissolve?.cancel()
      dissolve = angle.animate([{ opacity: 0 }, { opacity: 1 }], { duration: DISSOLVE, easing: 'ease-out' })
      shown = { col, row }
    }

    const tick = (now) => {
      const elapsed = last ? Math.min(now - last, 100) : 16
      gaze.u = ease(gaze.u, target.u, elapsed, HALF_LIFE)
      gaze.v = ease(gaze.v, target.v, elapsed, HALF_LIFE)
      show(gazeToCell(gaze, cols, rows))

      const moving = Math.abs(target.u - gaze.u) > SETTLED || Math.abs(target.v - gaze.v) > SETTLED
      last = moving ? now : 0
      frame = moving ? requestAnimationFrame(tick) : 0
    }

    const aim = () => {
      if (pointer) {
        const box = headRef.current.getBoundingClientRect()
        const center = { x: box.left + box.width / 2, y: box.top + box.height / 2 }
        target = pointerToGaze(pointer, center, { width: window.innerWidth, height: window.innerHeight })
      } else {
        target = { u: 0, v: 0 }
      }
      frame ||= requestAnimationFrame(tick)
    }

    const follow = (event) => {
      pointer = { x: event.clientX, y: event.clientY }
      aim()
    }

    const rest = () => {
      pointer = null
      aim()
    }

    const options = { passive: true, signal: listeners.signal }
    window.addEventListener('pointermove', follow, options)
    window.addEventListener('pointerdown', follow, options)
    window.addEventListener('scroll', aim, options)
    window.addEventListener('resize', aim, options)
    document.documentElement.addEventListener('pointerleave', rest, options)

    return () => {
      listeners.abort()
      cancelAnimationFrame(frame)
      dissolve?.cancel()
      for (const layer of [angle, before]) {
        layer.style.removeProperty('--col')
        layer.style.removeProperty('--row')
      }
    }
  }, [cols, rows])

  const cells = { backgroundImage: `url(${atlas})` }
  // A side of the head that sits on the still's own edge is not feathered: the base would show
  // through there, and in a close-up that is the top of the hair, left behind when the head moves.
  const solid = (onEdge) => (onEdge ? '#000' : undefined)

  return (
    <div className={`face-follow ${className}`} style={{ aspectRatio: `${width} / ${height}` }}>
      <img className="face-follow__base" src={base} alt={alt} width={width} height={height} />
      <div
        ref={headRef}
        className="face-follow__head"
        aria-hidden="true"
        style={{
          left: `${head.x * 100}%`,
          top: `${head.y * 100}%`,
          width: `${head.width * 100}%`,
          height: `${head.height * 100}%`,
          '--cols': cols,
          '--rows': rows,
          '--gutter-x': gutter?.x,
          '--gutter-y': gutter?.y,
          '--edge-left': solid(head.x < ON_EDGE),
          '--edge-top': solid(head.y < ON_EDGE),
          '--edge-right': solid(head.x + head.width > 1 - ON_EDGE),
          '--edge-bottom': solid(head.y + head.height > 1 - ON_EDGE),
        }}
      >
        <div ref={beforeRef} className="face-follow__angle" style={cells} />
        <div ref={angleRef} className="face-follow__angle" style={cells} />
      </div>
    </div>
  )
}

export default FaceFollow
