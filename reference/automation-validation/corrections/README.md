# Exercise movement examples and measured checks

This review has 14 completed example records. Completed records
have no failed mechanical/body checks and no missing measurements. They are not all approved:
the row retains two source-photo comparison failures, triceps pushdown retains four, and most
photographed movements retain uncertain depth. A passing count alone did not detect the earlier
bent-arm bench lockout, pull-up branch switch or inverted pulldown.

| Exercise | Pass | Fail | Uncertain | Frames / fps | Media and review scope |
| --- | ---: | ---: | ---: | ---: | --- |
| Barbell Back Squat | 36 | 0 | 0 | 69 / 30 | [GIF](barbell-back-squat/animation.gif) · checked clip/preset; photos are context |
| Conventional Barbell Deadlift | 28 | 0 | 10 | 32 / 12 | [GIF](conventional-barbell-deadlift/animation.gif) · unchanged GIF, current movement recheck |
| Flat Barbell Bench Press | 38 | 0 | 10 | 32 / 12 | [GIF](flat-barbell-bench-press/animation.gif) · corrected full press and bar path |
| Standing Barbell Overhead Press | 27 | 0 | 10 | 32 / 12 | [GIF](standing-barbell-overhead-press-shoulder-press/animation.gif) · upright stance, loaded bar, full lockout |
| Barbell Bent-Over Row | 18 | 2 | 0 | 81 / 30 | [GIF](barbell-bent-over-row/animation.gif) · unchanged GIF, current row and arm replays |
| Pull-Up | 25 | 0 | 11 | 43 / 16 | [GIF](pull-up/animation.gif) · continuous elbows and controlled lowering |
| Lat Pulldown | 20 | 0 | 10 | 32 / 12 | [GIF](lat-pulldown-seated-to-the-front/animation.gif) · elbows pull down toward the torso |
| Dumbbell Bicep Curl | 25 | 0 | 10 | 32 / 12 | [GIF](dumbbell-bicep-curl-standing-both-arms/animation.gif) · upright body and fixed upper arms |
| Dumbbell Lateral Raise | 25 | 0 | 10 | 32 / 12 | [GIF](dumbbell-lateral-raise/animation.gif) · shoulder-height raise with a fixed elbow bend |
| Seated Dumbbell Shoulder Press | 35 | 0 | 10 | 32 / 12 | [GIF](dumbbell-shoulder-press-seated-overhead-press/animation.gif) · supported upright body, flat soles, full lockout |
| Walking Lunge | 25 | 0 | 15 | 42 / 16 | [GIF](walking-lunge-dumbbells-optional/animation.gif) · one observed forward step; plays once |
| Front Plank | 25 | 0 | 0 | 12 / 12 | [GIF](front-plank-forearm-plank/animation.gif) · unchanged static hold, current movement recheck |
| Tricep Pushdown | 30 | 4 | 2 | 32 / 12 | [GIF](tricep-pushdown-cable-bar-or-rope-attachment/animation.gif) · mechanical form passes; four photo comparisons still fail |
| Leg Press | 30 | 0 | 10 | 32 / 12 | [GIF](leg-press-seated-machine/animation.gif) · fixed supported body and leg-only rail motion |

Counts and modes come from each exercise's `result.json` and `checks.json`; the aggregate manifest
is [results.json](results.json). Zero uncertainty on the squat means its photographs were not motion
targets, not that their depth became calibrated. Lunge retains inferred source depth and ends during the observed swing-through.

Each completed folder contains its actual animation, recorded start/end frames, final Checks,
inspector measurements, geometry report, source provenance and hashed replay evidence. Photo builds
also retain their input measurements/targets. Full `.blend` scenes remain at the run paths recorded in
the results; they are not committed. These examples do not overwrite user references or run history.

## Sources and retained disagreements

Most corrected photo profiles use the observed exercise positions plus declared fitted mechanics.
Canonical lockout, upright body posture, stable grip and support constraints correct unreliable
monocular depth; they are reported as corrections, not photographic 3D measurements. Source
comparisons use the recorded generated frames and retain their own failures and uncertainty.

The checked squat preset uses the pinned `squat.fbx` body motion and an authored fitted back-bar
grip. Photos are optional context. **Reference pictures** remains a separate strict source mode:
it still needs usable side start/bottom poses and a front/rear grip view. The source search did not
establish that complete photographic set; see the [source review](../motion-quality/squat-source-review.json).
The corrected 69-frame scene and its PNGs were left unchanged during the later metadata/coverage
and planned-camera recheck. The GIF was re-exported only after the current checks passed.

The walking source is a separate observed step, selected from real chronological video frames with
named heel/toe contacts. Its one-way playback must not become a reversed or repeated cycle, and
uncertain support observations do not become flat-foot contact claims. All 42 actual fitted frames,
the final airborne foot and the GIF's single playback were independently replayed.

The row's held hinge intentionally disagrees with the demonstration's changing end hip angle and
torso lean. Its [two failed comparisons](barbell-bent-over-row/checks.json) remain failures.
The pushdown's [four retained failures](tricep-pushdown-cable-bar-or-rope-attachment/checks.json)
compare start torso lean/upper-arm angle and end hip/elbow angle against the photos. Mechanical
passes do not erase those differences or approve the exact source match.

The [historical reference-only batch](reference-batch.json) retrieved 38 images and reported
12 ready/2 blocked under the older photo-only capabilities. Those historical counts describe source
retrieval, not this review's current preset/sequence modes or animation approval.

## What was re-rendered and what was retained

Bench, pull-up, pulldown, curl, lateral raise, pushdown, squat, both shoulder presses and leg press
have corrected renders in this review. Row, deadlift and plank retain their existing scene/GIF
geometry. The row has current contact/hinge and arm replays; deadlift and plank have current
all-frame arm-motion rechecks with their prior source comparisons and earlier physical evidence
retained. Their old render timings are labeled as original work in `result.json`.

The [movement-quality report](../motion-quality/README.md) gives concrete measurements and replay
links. The original GLB, face and hand shapes remain intact; any fitness proportions are fixed rest
lengths, not per-frame stretching. Visible skin checks use the planned preview and comparison
cameras. Hidden folds remain informational; no camera or threshold is changed to turn a defect green.

## Faster iteration and verification scope

Required exercise range, full-frame equipment/support measurements and arm continuity run before
rendering. Missing coverage blocks the build. The shared arm gate measures elbow/wrist positions
and arm/forearm/hand rotations, including loop seams only for cyclic movements. Endpoint contact
alone is insufficient. Final source comparisons remain separate from these mechanical checks.

Picture readings and fitted characters are cached. An estimated middle no longer launches five
extra camera searches without a source picture. Automation hashes current solver/gate code,
runtime, source selection, inputs and proportions before reusing a completed result; old runs remain
in history when changed code triggers a new build.

The leg-press body/leg fitter took **8.35 seconds** inside the final run. Its full animate process,
including the other animation work and GIF rendering, took **155.85 seconds**. These are different
scopes. CPU was shared; these are observed run durations.

Portable tests and actual Blender scene replays have separate scopes. Final aggregate test counts
and command exclusions must be read from [verification.json](verification.json), not inferred from
the Checks counts above.

Final verification: **741 portable Python tests passed (27 explicit skips), 365 portable Studio tests passed, 2 targeted API/queue tests passed, and the Studio build passed**. Desktop/mobile playback, source selection, all 14 verified assets and real isolated source automation also passed; exact commands and exclusions are in the linked verification record.
