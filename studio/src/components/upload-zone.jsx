import { useEffect, useRef, useState } from 'react'
import './upload-zone.css'
import { api } from '../lib/api.js'
import { imagesFromClipboard } from '../lib/clipboard.js'

const hasFiles = (e) => Array.from(e.dataTransfer?.types ?? []).includes('Files')

const PASTE_KEY = /Mac|iPhone|iPad/.test(globalThis.navigator?.platform ?? '') ? '⌘V' : 'Ctrl+V'
const NO_PICTURE = "There's no picture in what you pasted."
const NOTICE_MS = 4000
const TEXT_INPUTS = new Set(['text', 'search', 'email', 'url', 'tel', 'password', 'number', ''])

// Pasting into a text field (the Add exercise name, say) must stay a normal text paste.
const typingIn = (el) =>
  Boolean(el?.closest?.('textarea, [contenteditable]:not([contenteditable="false"])')) ||
  (el?.tagName === 'INPUT' && TEXT_INPUTS.has(el.getAttribute('type') ?? ''))

// Save a batch of pictures; the exercise page starts a reference check or build explicitly.
const UploadZone = ({ slug, compact, onDone }) => {
  const input = useRef(null)
  const zone = useRef(null)
  const busyRef = useRef(false)
  const noticeTimer = useRef(null)
  const [over, setOver] = useState(false)
  const [progress, setProgress] = useState(null) // {done, total}
  const [problems, setProblems] = useState([])
  const [started, setStarted] = useState(null)
  const [notice, setNotice] = useState(null)

  const showNotice = (text) => {
    clearTimeout(noticeTimer.current)
    setNotice(text)
    noticeTimer.current = setTimeout(() => setNotice(null), NOTICE_MS)
  }

  const upload = async (fileList) => {
    // Browsers hand over dropped files in no set order; names like 14.png, 15.png… carry the movement order.
    const files = Array.from(fileList ?? []).sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }))
    if (!files.length || busyRef.current) return
    busyRef.current = true
    clearTimeout(noticeTimer.current)
    setNotice(null)
    setProblems([])
    setStarted(null)
    const failed = []
    let added = 0
    for (const [i, file] of files.entries()) {
      setProgress({ done: i, total: files.length })
      try {
        await api.uploadReference(slug, file)
        added += 1
      } catch (e) {
        failed.push({ name: file.name, message: e.message })
      }
    }
    setProgress(null)
    if (added) {
      setStarted(`${added} ${added === 1 ? 'picture' : 'pictures'} saved. Set their positions and views, then check the references.`)
    }
    setProblems(failed)
    // The whole batch is saved before another paste can begin.
    busyRef.current = false
    onDone()
  }

  // ⌘V / Ctrl+V anywhere on the page adds the pasted pictures to this exercise, like a drop.
  const latestUpload = useRef(upload)
  useEffect(() => {
    latestUpload.current = upload
  })

  useEffect(() => {
    const onPaste = (e) => {
      const target = e.target instanceof Element ? e.target : document.activeElement
      if (typingIn(target)) return
      const files = imagesFromClipboard(e.clipboardData)
      // The paste can come while the zone is scrolled away: bring it into view so its answer is seen.
      const still = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches
      zone.current?.scrollIntoView({ block: 'nearest', behavior: still ? 'auto' : 'smooth' })
      if (!files.length) {
        showNotice(NO_PICTURE)
        return
      }
      e.preventDefault()
      if (busyRef.current) {
        showNotice('Wait for this upload to finish, then paste again.')
        return
      }
      latestUpload.current(files)
    }
    document.addEventListener('paste', onPaste)
    return () => {
      document.removeEventListener('paste', onPaste)
      clearTimeout(noticeTimer.current)
    }
  }, [])

  const onDrop = (e) => {
    if (!hasFiles(e)) return
    e.preventDefault()
    setOver(false)
    upload(e.dataTransfer.files)
  }

  const busy = Boolean(progress)

  return (
    <div className="upload">
      <div
        ref={zone}
        className={`upload-zone${compact ? ' is-compact' : ''}${over ? ' is-over' : ''}${busy ? ' is-busy' : ''}`}
        role="button"
        tabIndex={0}
        aria-disabled={busy}
        aria-describedby="upload-help"
        onClick={() => !busy && input.current?.click()}
        onKeyDown={(e) => {
          if ((e.key === 'Enter' || e.key === ' ') && !busy) {
            e.preventDefault()
            input.current?.click()
          }
        }}
        onDragEnter={(e) => { if (hasFiles(e)) { e.preventDefault(); setOver(true) } }}
        onDragOver={(e) => { if (hasFiles(e)) { e.preventDefault(); e.dataTransfer.dropEffect = 'copy' } }}
        onDragLeave={(e) => { if (!e.currentTarget.contains(e.relatedTarget)) setOver(false) }}
        onDrop={onDrop}
      >
        <span className="upload-icon" aria-hidden="true" />
        <span className="upload-text">
          {busy ? (
            <strong className="num">Uploading {progress.done + 1} of {progress.total}…</strong>
          ) : (
            <strong>Drop pictures here, click to choose, or paste ({PASTE_KEY})</strong>
          )}
          {notice ? (
            <span className="upload-notice" role="status">{notice}</span>
          ) : (
            <span id="upload-help" className="upload-help">
              {compact
                ? 'They go to the end of the list. Check the references when your edits are finished.'
                : 'Add side views of the start and end positions; include the middle when available. You can edit them before building.'}
            </span>
          )}
        </span>
        <input
          ref={input}
          type="file"
          accept="image/*,.heic,.heif"
          multiple
          hidden
          onChange={(e) => {
            upload(e.target.files)
            e.target.value = ''
          }}
        />
      </div>

      {started && <p className="upload-ok" role="status">{started}</p>}
      {problems.length > 0 && (
        <ul className="upload-problems" role="alert">
          {problems.map((p, i) => (
            <li key={`${p.name}-${i}`}>
              <strong>{p.name}</strong>: {p.message}
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

export default UploadZone
