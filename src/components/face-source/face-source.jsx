import { useEffect, useId, useRef, useState } from 'react'
import './face-source.css'

const POLL_MS = 600
const TIMER_SECONDS = 10 // time to walk back into the frame for a whole-body photo

// The gym outfits of a whole-body avatar. The dev server knows the same names (pipeline/frames-api.js).
const OUTFITS = [
  { name: 'tee-pants', label: "Men's: T-shirt and pants", wears: 'a T-shirt and pants' },
  { name: 'tank-shorts', label: "Women's: tank top and shorts", wears: 'a tank top and shorts' },
]

const phase = (job) => {
  if (job.pass === 'restyle') return 'Drawing the 3D avatar'
  if (job.pass === 'full') return 'Rendering full quality'
  return job.done ? 'Rendering the preview' : 'Finding the face'
}

// Makes a character from the camera or an image file, through the dev server's pipeline
// (pipeline/frames-api.js). Calls `onReady(name)` once that character has frames to show.
const FaceSource = ({ onReady, className = '' }) => {
  const dialogRef = useRef(null)
  const videoRef = useRef(null)
  const seenRunning = useRef(null)
  const shown = useRef(null)
  const titleId = useId()
  const [job, setJob] = useState({ state: 'idle' })
  const [canRestyle, setCanRestyle] = useState(false)
  const [wantsStyle, setWantsStyle] = useState(true)
  const [wantsOutfit, setWantsOutfit] = useState('')
  const [problem, setProblem] = useState(null)
  const [cameraProblem, setCameraProblem] = useState(null)
  const [live, setLive] = useState(false)
  const [countdown, setCountdown] = useState(null)
  // A gym outfit is drawn by the restyle, so a whole-body avatar is always restyled.
  const outfit = canRestyle ? OUTFITS.find((candidate) => candidate.name === wantsOutfit) : undefined
  const restyles = canRestyle && (wantsStyle || Boolean(outfit))

  // Learn whether 3D style is set up, and pick up a render that is already under way.
  // One that finished or failed earlier is old news and stays off the page.
  useEffect(() => {
    fetch('/api/faces/job')
      .then((response) => response.json())
      .then((current) => {
        setCanRestyle(current.canRestyle)
        if (current.state === 'running') setJob(current)
      })
      .catch(() => {})
  }, [])

  useEffect(() => {
    if (job.state !== 'running') return undefined
    const timer = setInterval(() => {
      fetch('/api/faces/job')
        .then((response) => response.json())
        .then(setJob)
        .catch(() => {})
    }, POLL_MS)
    return () => clearInterval(timer)
  }, [job.state])

  // Show each render once, as soon as it has frames, if this page watched it run.
  useEffect(() => {
    if (job.state === 'running') seenRunning.current = job.id
    if (job.ready && seenRunning.current === job.id && shown.current !== job.id) {
      shown.current = job.id
      onReady(job.name)
    }
  }, [job, onReady])

  const send = async (image, name) => {
    setProblem(null)
    try {
      const query = new URLSearchParams({
        name,
        ...(restyles && { style: '3d' }),
        ...(outfit && { outfit: outfit.name }),
      })
      const response = await fetch(`/api/faces?${query}`, {
        method: 'POST',
        headers: { 'Content-Type': image.type || 'application/octet-stream' },
        body: image,
      })
      const result = await response.json()
      if (response.ok) setJob(result)
      else setProblem(result.error)
    } catch {
      setProblem('The dev server did not answer. Check that npm run dev is still running.')
    }
  }

  const chooseFile = (event) => {
    const [file] = event.target.files
    if (file) send(file, file.name.replace(/\.[^.]+$/, ''))
    event.target.value = ''
  }

  const openCamera = async () => {
    setCameraProblem(null)
    dialogRef.current.showModal()
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user', width: { ideal: 1920 }, height: { ideal: 1080 } },
      })
      // The dialog may have been closed while the browser was asking for permission.
      if (dialogRef.current.open) videoRef.current.srcObject = stream
      else stream.getTracks().forEach((track) => track.stop())
    } catch (error) {
      setCameraProblem(
        error.name === 'NotAllowedError'
          ? 'The browser blocked the camera. Allow it for this page, or upload an image instead.'
          : 'No camera is available. Upload an image instead.',
      )
    }
  }

  const closeCamera = () => {
    videoRef.current.srcObject?.getTracks().forEach((track) => track.stop())
    videoRef.current.srcObject = null
    setLive(false)
    setCountdown(null)
  }

  // Saves what the viewfinder shows: the middle of the camera frame, not mirrored,
  // cropped to 3:4 for a portrait or 9:16 for a whole body.
  const takePhoto = () => {
    const video = videoRef.current
    const [across, down] = outfit ? [9, 16] : [3, 4]
    const width = Math.min(video.videoWidth, Math.round((video.videoHeight * across) / down))
    const height = Math.min(video.videoHeight, Math.round((width * down) / across))
    const canvas = Object.assign(document.createElement('canvas'), { width, height })
    const left = (video.videoWidth - width) / 2
    const top = (video.videoHeight - height) / 2
    canvas.getContext('2d').drawImage(video, left, top, width, height, 0, 0, width, height)
    // A scan is kept apart from the portrait, so neither replaces the other.
    canvas.toBlob((photo) => send(photo, outfit ? 'body' : 'camera'), 'image/jpeg', 0.92)
    dialogRef.current.close()
  }

  // The timer of a whole-body scan: one tick a second, then the photo.
  useEffect(() => {
    if (countdown === null) return undefined
    const tick = setTimeout(() => (countdown > 1 ? setCountdown(countdown - 1) : takePhoto()), 1000)
    return () => clearTimeout(tick)
  }, [countdown])

  const message = problem ?? (job.state === 'error' ? job.error : null)
  const destination = restyles
    ? 'The photo is sent to Cloudflare to be drawn as a 3D avatar, then animated on this computer.'
    : 'The photo stays on this computer and is animated as it is.'

  return (
    <div className={`face-source ${className}`}>
      <div className="face-source__choices">
        <label className="face-source__style">
          <input
            type="checkbox"
            role="switch"
            checked={restyles}
            disabled={!canRestyle || Boolean(outfit)}
            onChange={(event) => setWantsStyle(event.target.checked)}
          />
          3D style
        </label>
        <select
          className="face-source__button face-source__avatar"
          aria-label="Avatar"
          value={outfit?.name ?? ''}
          disabled={!canRestyle}
          onChange={(event) => setWantsOutfit(event.target.value)}
        >
          <option value="">Portrait, own clothes</option>
          <optgroup label="Whole body, gym clothes">
            {OUTFITS.map(({ name, label }) => (
              <option key={name} value={name}>
                {label}
              </option>
            ))}
          </optgroup>
        </select>
        <button type="button" className="face-source__button" onClick={openCamera}>
          {outfit ? 'Scan body' : 'Use camera'}
        </button>
        <label className="face-source__button">
          Upload image
          <input className="face-source__file" type="file" accept="image/*,.heic,.heif" onChange={chooseFile} />
        </label>
      </div>

      {!canRestyle && (
        <p className="face-source__hint">
          3D style and whole-body avatars need a free Cloudflare key. The README has the steps.
        </p>
      )}
      {outfit && (
        <p className="face-source__hint">
          Use a photo that shows you from head to feet. You get an avatar wearing {outfit.wears}.
        </p>
      )}

      {message && (
        <p className="face-source__status" role="alert">
          {message}
        </p>
      )}
      {!message && job.state === 'running' && (
        <div className="face-source__status">
          <p role="status">{phase(job)}</p>
          {job.total ? (
            <>
              <progress value={job.done} max={job.total} aria-label={phase(job)} />
              <span className="face-source__count" aria-hidden="true">
                {job.done} of {job.total}
              </span>
            </>
          ) : (
            <progress aria-label={phase(job)} />
          )}
        </div>
      )}

      <dialog ref={dialogRef} className="face-source__camera" aria-labelledby={titleId} onClose={closeCamera}>
        <h2 id={titleId}>{outfit ? 'Scan your whole body' : 'Take a photo'}</h2>
        <div className={`face-source__viewfinder ${outfit ? 'face-source__viewfinder--body' : ''}`}>
          <video ref={videoRef} autoPlay playsInline muted onLoadedMetadata={() => setLive(true)} />
          {countdown !== null && (
            <p className="face-source__countdown" role="timer" aria-live="assertive">
              {countdown}
            </p>
          )}
        </div>
        {cameraProblem ? (
          <p role="alert">{cameraProblem}</p>
        ) : outfit ? (
          <p>
            Stand the screen upright, start the timer, and step back until you fit from head to feet. Face the
            camera with your arms a little away from your sides. {destination}
          </p>
        ) : (
          <p>Face the camera with your head level. {destination}</p>
        )}
        <div className="face-source__actions">
          <button type="button" className="face-source__button" onClick={() => dialogRef.current.close()}>
            Cancel
          </button>
          {outfit ? (
            <button
              type="button"
              className="face-source__button face-source__button--primary"
              onClick={() => setCountdown(TIMER_SECONDS)}
              disabled={!live || countdown !== null}
            >
              {countdown === null ? `Start ${TIMER_SECONDS} second timer` : 'Hold still'}
            </button>
          ) : (
            <button
              type="button"
              className="face-source__button face-source__button--primary"
              onClick={takePhoto}
              disabled={!live}
            >
              Take photo
            </button>
          )}
        </div>
      </dialog>
    </div>
  )
}

export default FaceSource
