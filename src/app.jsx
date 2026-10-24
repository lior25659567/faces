import { useState } from 'react'
import FaceFollow from './components/face-follow/face-follow.jsx'
import FaceSource from './components/face-source/face-source.jsx'
import faces from './faces/index.js'
import './app.css'

// Making a character needs the frame generator behind the dev server, so the controls are dev-only.
const canMakeFaces = import.meta.env.DEV

// The manifest lists the still's backdrop as evenly spaced bands, top to bottom; each stop sits mid-band.
const backdropGradient = (stops) =>
  `linear-gradient(${stops.map((color, index) => `${color} ${((index + 0.5) / stops.length) * 100}%`).join(', ')})`

const NoFaces = ({ onReady }) => (
  <main className="stage stage--empty">
    <div className="empty">
      <h1>No character yet</h1>
      {canMakeFaces && (
        <>
          <p>Take a photo or upload an image to make the first one.</p>
          <FaceSource onReady={onReady} />
        </>
      )}
      <p>From the terminal, save a front-facing still in the input folder, then run:</p>
      <pre>npm run frames -- input/me.png</pre>
    </div>
  </main>
)

const App = () => {
  const [name, setName] = useState(faces[0]?.name)
  const face = faces.find((candidate) => candidate.name === name) ?? faces[0]

  if (!face) return <NoFaces onReady={setName} />

  return (
    <main
      className="stage"
      style={{
        '--backdrop': face.backdrop[0],
        '--backdrop-gradient': backdropGradient(face.backdrop),
        '--ratio': face.width / face.height,
      }}
    >
      <FaceFollow face={face} alt={`The ${face.name} character`} className="stage__figure" />

      <div className="stage__bar">
        {faces.length > 1 && (
          <nav className="stage__picker" aria-label="Characters">
            {faces.map((candidate) => (
              <button
                key={candidate.name}
                type="button"
                aria-pressed={candidate === face}
                onClick={() => setName(candidate.name)}
              >
                {candidate.name}
              </button>
            ))}
          </nav>
        )}
        {canMakeFaces && <FaceSource onReady={setName} className="stage__source" />}
      </div>

      <p className="stage__still-notice">Head tracking is off because this device asks for reduced motion.</p>
    </main>
  )
}

export default App
