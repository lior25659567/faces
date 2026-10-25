import { useState } from 'react'
import { PHASE_LABELS } from '../lib/labels.js'
import { referencePose } from '../lib/reference-pose.js'
import { referenceUrl } from '../lib/run-view.js'
import './reference-preview.css'

const ReferencePreview = ({ slug, images = [], measurements = {}, targets }) => {
  const [overlay, setOverlay] = useState(true)
  const shown = images.filter((image) => !image.contextOnly && !image.deleted).sort((a, b) => a.order - b.order)
  if (!shown.length) return null
  return (
    <div className="reference-preview">
      <div className="reference-preview-head">
        <p>Detected poses — check that the joints line up with the person before building.</p>
        <button className="btn btn-quiet" type="button" aria-pressed={overlay} onClick={() => setOverlay((on) => !on)}>
          {overlay ? 'Hide joints' : 'Show joints'}
        </button>
      </div>
      <p className="reference-preview-guide">
        Amber dots and dashed lines mean the detector is unsure; gray means confidence was not recorded.
        If the joints are misplaced, choose a clearer picture. Check each picture’s Position and View above.
        This overlay shows the detected 2D pose, not the accuracy of its 3D motion.
      </p>
      <ul className="reference-preview-cards">
        {shown.map((image) => {
          const reading = measurements?.[image.id]
          const pose = referencePose(reading)
          const used = targets?.images?.[image.id]
          const name = image.original || image.file
          return (
            <li className="reference-preview-card" key={image.id}>
              <p className="reference-preview-phase" title={image.source?.motionOnly ? 'Sequence frame; excluded from comparison stills.' : undefined}>
                {image.source?.motionOnly ? 'Motion sample' : PHASE_LABELS[image.phase] || 'Position not set'}
                {image.source?.motionOnly && Number.isFinite(image.source.timeSeconds) ? ` · ${image.source.timeSeconds.toFixed(3)} s` : ''}
              </p>
              <div className="reference-preview-picture">
                {pose ? (
                  <svg viewBox={`0 0 ${pose.width} ${pose.height}`} role="img" aria-label={`${name}: detected pose`}>
                    <image href={referenceUrl(slug, image.file)} width={pose.width} height={pose.height} />
                    {overlay && (
                      <g>
                        {pose.bones.map(({ id, a, b, confidence }) => (
                          <line key={id} className={`reference-pose-line is-${confidence}`}
                            x1={a.x} y1={a.y} x2={b.x} y2={b.y} vectorEffect="non-scaling-stroke" />
                        ))}
                        {pose.points.map(({ name: joint, x, y, confidence }) => (
                          <circle key={joint} className={`reference-pose-point is-${confidence}`}
                            cx={x} cy={y} r={Math.max(pose.width, pose.height) * 0.009}>
                            <title>{joint.replace('_', ' ')}{confidence === 'low' ? ': uncertain' : confidence === 'unknown' ? ': confidence unknown' : ''}</title>
                          </circle>
                        ))}
                      </g>
                    )}
                  </svg>
                ) : <img src={referenceUrl(slug, image.file)} alt={name} />}
              </div>
              <p className="reference-preview-name" title={name}>{name}</p>
              {!pose && <p className="reference-preview-warning">{reading?.error || 'No readable joint positions for this picture.'}</p>}
              {pose?.lowConfidence > 0 && <p className="reference-preview-warning">{pose.lowConfidence} joints are uncertain.</p>}
              {pose?.missingJoints > 0 && <p className="reference-preview-warning">{pose.missingJoints} joints are missing or outside the picture.</p>}
              {pose?.unknownConfidence > 0 && <p className="reference-preview-guide">Confidence was not recorded for {pose.unknownConfidence} joints.</p>}
              {used?.used === false && <p className="reference-preview-warning">Not used: {used.why}</p>}
            </li>
          )
        })}
      </ul>
    </div>
  )
}

export default ReferencePreview
