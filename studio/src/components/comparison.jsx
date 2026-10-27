import { useState } from 'react'
import './comparison.css'
import { backgroundCrop, cropAspect, supportsViewBox, viewBoxInset } from '../lib/crop.js'
import { formatWhen } from '../lib/format.js'
import { PHASE_LABELS, VIEW_LABELS } from '../lib/labels.js'
import { buildComparison, pairTally, runFileUrl } from '../lib/run-view.js'

const viewText = (view) => {
  if (!view) return 'view not set'
  if (view === 'frontal') return 'front or behind, not set'
  return `${(VIEW_LABELS[view] ?? view).toLowerCase()} view`
}

const VIEW_BOX = supportsViewBox()

// The coach cut to the body's box. object-view-box where the browser has it, else a background
// scaled and placed so the box fills the element. Either way the element has the box's shape.
const CroppedPicture = ({ src, alt, crop }) => {
  const aspectRatio = cropAspect(crop)
  if (VIEW_BOX) {
    return <img src={src} alt={alt} loading="lazy" className="pair-cropped" style={{ aspectRatio, objectViewBox: viewBoxInset(crop.box) }} />
  }
  return (
    <div role="img" aria-label={alt} className="pair-cropped is-background"
      style={{ aspectRatio, backgroundImage: `url("${src}")`, ...backgroundCrop(crop.box) }} />
  )
}

// Each side grows in proportion to its picture's width/height, so both pictures end up the same height.
const Side = ({ src, alt, label, mirror, ours, fallbackAspect, crop }) => {
  const [loadedAspect, setLoadedAspect] = useState(fallbackAspect)
  const aspect = crop ? cropAspect(crop) : loadedAspect
  return (
    <div className={`pair-side${ours ? ' is-ours' : ''}`} style={{ flexGrow: aspect }}>
      {src && crop ? (
        <CroppedPicture src={src} alt={alt} crop={crop} />
      ) : src ? (
        <img
          src={src}
          alt={alt}
          className={mirror ? 'is-mirrored' : undefined}
          loading="lazy"
          onLoad={(e) => {
            const { naturalWidth: w, naturalHeight: h } = e.currentTarget
            if (w && h) setLoadedAspect(w / h)
          }}
        />
      ) : (
        <div className="pair-missing" style={{ aspectRatio: aspect }}>No still from this run</div>
      )}
      <span className="pair-label">{label}</span>
    </div>
  )
}

const Pair = ({ pair, rows }) => {
  const [full, setFull] = useState(false)
  const phase = PHASE_LABELS[pair.phase] ?? pair.phase ?? 'No position'
  const tally = pair.used === false ? null : pairTally(rows, pair.phase)
  return (
    <figure className={`pair${pair.used === false ? ' is-unused' : ''}`}>
      <figcaption className="pair-head">
        <span className="pair-phase">{phase}</span>
        <span className="pair-view">{viewText(pair.view)}</span>
        {tally && <span className={`pair-tally num is-${tally.kind}`}>{tally.text}</span>}
      </figcaption>
      <div className="pair-images">
        <Side
          src={pair.refUrl}
          alt={`Coach, ${phase.toLowerCase()} position`}
          label="Coach"
          fallbackAspect={0.6}
          crop={full ? null : pair.crop}
        />
        <Side
          src={pair.oursUrl}
          alt={`Our character at the ${phase.toLowerCase()} position`}
          label={pair.mirror ? 'Ours, flipped to match' : 'Ours'}
          mirror={pair.mirror}
          ours
          fallbackAspect={pair.oursUrl ? 1 : 0.6}
        />
      </div>
      {(pair.frame !== null || pair.used === false || pair.crop) && (
        <div className="pair-foot">
          <div className="pair-foot-row">
            {pair.frame !== null && <span className="num">Frame {pair.frame}</span>}
            {pair.crop && (
              <button type="button" className="pair-crop-toggle" aria-pressed={full} onClick={() => setFull((f) => !f)}>
                {full ? 'Show the coach close up' : 'Show full screenshot'}
              </button>
            )}
          </div>
          {pair.used === false && <p className="pair-why">Not used for the targets: {pair.why ?? 'no reason given'}</p>}
        </div>
      )}
    </figure>
  )
}

const Comparison = ({ slug, run, bundle, fallbackImages, rows, newerRun }) => {
  const [gifOk, setGifOk] = useState(true)
  const [replay, setReplay] = useState(0)
  const once = bundle?.targets?.motion?.mode === 'sequence' || bundle?.snapshot?.motion?.playback === 'once'
  // The run's own snapshot of the pictures, so later edits don't change an old comparison.
  const preset = bundle?.snapshot?.motionPreset
  const recordedImages = bundle?.snapshot?.images ?? fallbackImages ?? []
  const selectedBundle = bundle?.snapshot?.referenceBundle
  const images = recordedImages.map((image) => ({ ...image, contextOnly: image.contextOnly || Boolean(preset) || Boolean(selectedBundle &&
    (image.source?.catalogId !== selectedBundle.id || image.source?.commit !== selectedBundle.commit)) }))
  const { pairs, context } = buildComparison({
    slug, runId: run.id, images, ours: bundle?.ours, measurements: bundle?.measurements, targets: bundle?.targets,
  })
  const gif = runFileUrl(slug, run.id, 'animation.gif')

  return (
    <div className="comparison">
      <div className="section-head">
        <h2 className="section-title" id="comparison-title">Comparison</h2>
        <p className="section-note">
          {newerRun
            ? `From the run of ${formatWhen(run.createdAt)}, the newest that finished.`
            : preset ? `Checked movement preset. Pictures are visual references. Run of ${formatWhen(run.createdAt)}.`
              : `Coach beside our character at the same moment. Run of ${formatWhen(run.createdAt)}.`}
        </p>
      </div>

      {!bundle && <p className="comparison-empty">Loading the comparison…</p>}
      {bundle && !bundle.ours && <p className="comparison-empty">This run saved no comparison stills.</p>}

      {bundle && (
        <div className="comparison-sheet">
          {gifOk && (
            <figure className="pair pair-gif">
              <figcaption className="pair-head">
                <span className="pair-phase">Animation</span>
                <span className="pair-view">{once ? 'observed step · plays once' : 'the whole movement'}</span>
              </figcaption>
              <div className="pair-images">
                <div className="pair-side is-gif">
                  <img src={gif + (once ? `?play=${replay}` : '')} alt={`Our character doing the ${run.name ?? 'exercise'}`} onError={() => setGifOk(false)} />
                </div>
              </div>
              {once && <button type="button" className="btn btn-quiet" onClick={() => setReplay((value) => value + 1)}>Replay step</button>}
            </figure>
          )}
          {pairs.map((p) => <Pair key={p.id} pair={p} rows={rows} />)}
        </div>
      )}

      {bundle && context.length > 0 && (
        <div className="comparison-context">
          <h3 className="comparison-context-title">Context</h3>
          <p className="section-note">Shown for reference only; not measured.</p>
          <div className="comparison-context-row">
            {context.map((c) => (
              <figure key={c.id} className="context-item">
                <img src={c.refUrl} alt={`Context picture ${c.original ?? ''}`} loading="lazy" />
                <figcaption>{c.original}</figcaption>
              </figure>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

export default Comparison
