# Exercise Studio

From the repository root, install JavaScript dependencies with `npm ci`. Set up the pose reader with
`cd studio && npm run setup`, then start Studio with `npm run dev` from `studio/`.
The animation pipeline needs Blender 5.2 with MPFB and the mascot assets. On macOS it uses
`/Applications/Blender.app`; on Linux it uses `blender` or the `STUDIO_BLENDER` executable override.
The cloud environment has a tested Blender wrapper under `/workspace/.tools/bin/blender-faces`.

## Prepare references before building

1. Upload pictures, then set their Position and View. Edits are saved without launching a render.
2. Click **Check references**. This reads the pictures without starting Blender. Inspect the detected-joint
   overlays, missing-input messages, and uncertainty warnings. Recheck after changing a picture or tag.
3. Click **Build exercise** once the current references pass the check. This still runs the movement and
   contact checks; a successful reference check is not approval of the finished animation.

For a squat, provide clear side images at the start and bottom, preferably also in the middle, and a
front or rear image showing both shoulders, elbows and hands for grip/arm width. Incomplete squat
correction inputs now stop before Blender instead of silently rendering the unchanged source clip.
A joint overlay shows what the reader detected in the photograph; it does not certify inferred 3D depth.

Bent-over rows use image key poses even when a motion clip exists. The starting torso and lower-body
pose are held throughout the rep while the arms move. The visible reference arm drives both sides, preventing hidden-arm depth noise from weakening the pull.
A separate row solver puts both closed hands
on one level bar, keeps the grip width fixed, checks each hand's actual skin contact, and reports any
reach adjustment needed for the mascot's proportions. It also reports torso/thigh collision candidates.

Use Start and End references photographed from the same view: both side or both three-quarter.
Three-quarter references use explicitly marked inferred 3D depth, so their measurements are approximate.
A stationary hinge can intentionally differ from a demonstration that raises its torso as it pulls.
Studio uses `--fast` for 360 px, four-sample GIF previews; comparison stills retain their usual settings.
Preview framing includes the held bar so its ends remain visible.
The implemented reference-motion profiles also use edited picture poses when a generic clip exists.
Comparison stills use the recorded source phase frames; the inspector cannot choose a different,
closer-looking animation frame to hide an error. The latest row replay has **16 passing checks and
2 retained source conflicts**: a held hinge differs from the demonstration's changing torso/hip angle.
See the [row report](../reference/automation-validation/row/README.md) for the actual images and replay.

To add the sourced example photos without replacing your own references or starting a build:

```bash
node studio/scripts/import-row-references.mjs
```

The command is repeatable. Original photos, licenses and provenance are in `reference/exercise-row/`.
They are three-quarter views, never relabeled as side views. Select **Barbell Bent-Over Row** in Studio,
check the references, then build. On Linux set `STUDIO_BLENDER` to a Blender 5.2 executable or wrapper;
macOS retains the standard application path by default. The cloud snapshot's tested wrapper is
`/workspace/.tools/bin/blender-faces`.

A completed run can be independently replayed and checked without fitting/rendering it again:

```bash
blender --background RUN/scene.blend --python-exit-code 1 \
  --python studio/tests/blender/test_row_contact.py -- --run RUN
```

## Automate a batch

Start the Studio backend with `npm run dev` from `studio/`. Open **Automate exercises**, choose the
exercises, and start the batch. **Find and check references only** retrieves missing references and
checks their usability without rendering. Full automation also builds supported exercises and runs their
form checks. Progress and past batches remain available in the panel; a finished batch can contain
items that are blocked or need review.

The same runner is available from another terminal in `studio/`:

```bash
npm run automate -- --all
npm run automate -- --all --references-only
npm run automate -- --exercise barbell-bent-over-row
npm run automate -- --exercise barbell-bent-over-row --force
npm run automate -- --all --character-fit auto
```

These are alternative invocations. Repeat `--exercise <slug>` to select several exercises.
`STUDIO_URL` changes the backend address; the default is the local Studio on port 5199. The backend
must already be running. `--force` rebuilds an unchanged completed result, while still enforcing
reference and builder requirements. Exit code 0 means passed/ready, 2 means review or blocked items,
and 1 means an error or interrupted batch.

Automatic reference retrieval covers all 14 mapped exercises through the
[reviewed catalog](../reference/exercise-catalog.md): exact exercise variants at a pinned source
revision, with verified image hashes, provenance, and actual camera views. The lunge uses its reviewed
local video-frame bundle. Retrieving a source does not establish that its views or contacts can support
an accurate animation.
Manual references are kept. Missing parts of an earlier automatic import can be completed without
duplicating its pictures. Unchanged valid completed results are reused unless `--force` is given.

The implemented profiles include finite hand handles, bench/seat support, a fixed pull-up bar,
cable equipment, a leg-press rail and footplate, and a static plank. Seated and lying poses check
their actual supports; standing poses keep their feet planted. The plank checks forearms, toes,
body alignment and whole-body floor clearance. The leg press checks its seat, backrest, headrest and
both soles throughout the rail movement. Its two side previews expose contacts that the footplate
would hide from the old foot-end camera. Supports, props and the whole horizontal body are included
in preview framing.

Each attempt still has to pass reference preflight and the physical checks. The catalog's squat pair
supplies only three-quarter views, so it cannot satisfy the required side positions and front/rear
grip-width reference on its own. Missing views, unsuitable source poses and failed form checks stay
visible as blockers or review items. The plank's kneeling setup is context only; the usable pose makes
a static hold. The lunge remains blocked at an unobserved support handoff instead of becoming an
invented forward/reverse cycle.

**Stop remaining exercises**, or Ctrl-C in the CLI, requests a stop after the current exercise finishes.
Closing the panel does not stop the batch. A backend restart marks unfinished batches interrupted;
they do not resume automatically. Start a new batch to continue, preserving imported references and
completed results.

The profiles preserve bilateral symmetry and keep the torso/legs still on isolation movements.
Fast reference previews normally use 12 fps; pull-ups use 16 fps and the row keeps 30 fps. Comparison stills keep
560 px resolution with four EEVEE samples. Photo depth inferred from an oblique/frontal view stays
uncertain, even when contact passes. Impossible contact stops before rendering and saves its
measurements in Checks. Overhead completion and weight/head collision are separate from grip.

**Character proportions** is saved with each manual run and automation batch. **Automatic** uses
a fixed longer-arm fitness variant when the original short-arm character cannot satisfy the
supported reach profile. **Original** keeps the fitted character's proportions; impossible reach
still stops before rendering. **Fitness** selects the variant explicitly. The face, hand shapes,
original GLB and fitted cache are preserved. This is a rest-pose body choice, not changing arm
lengths during a rep. The result states which proportions were actually used; retries, queue
merging and completed-result reuse keep the choices separate.

The same option is available with `npm run automate -- --all --character-fit auto` (or `original`
or `fitness`) and Blender's `--character-fit`. Reference-driven Studio previews include a local,
rest-bound collar correction that keeps authored face/hand normals, UVs, weights and materials.
Skeletal-only USD export rejects that correction explicitly; the validated output is the Blender
scene/GIF.

Before rendering a GIF, Studio scans every posed frame for skin folds and movement jumps using
the same detector, camera visibility and limits as final Checks. It includes the planned GIF cameras
and each active source comparison camera at its recorded frame. Visible fold defects, missing camera
coverage and unreadable geometry stop the build before rendering; hidden folds remain informational.
The [saved-scene parity tests](../reference/automation-validation/skin-preflight/README.md) took about
1.8–3.8 seconds for this geometry scan alone, excluding character building and rendering.

For two-photo profiles, the estimated middle no longer launches five cartoon camera searches:
15 detector-rendered pictures become 10. Estimated middle angles stay informational and identify
their rig source. Actual photographed middle positions retain their comparisons. This avoids
spending render/reader time on a middle pose for which there is no source photograph.

The 12 reviewed [walking-lunge video frames](../reference/exercise-lunge/README.md) preserve source
timestamps and observed contacts. They are not a closed cycle and contain an unobserved support
handoff, so automatic stepping remains blocked rather than inventing forward travel. The squat
still needs suitable side and front/rear references. Motion-only frames are labeled **Motion sample**
with their source timestamps; they are separate from the three comparison anchors.

To enable another exercise, implement its motion/equipment constraints and validate real rendered
poses and contact checks first. Only then opt its named entry into `automatic: true` in
`studio/measure/exercise-kit.json`. Adding that flag alone does not establish accuracy.

## Reuse expensive work

Successful image readings are cached per exercise under `runs/<exercise>/.pose-cache/`. Image contents,
reader code, model and dependency versions form the cache key. Phase/view changes rebuild targets from
cached detections; changing an image or the detector invalidates its reading. Failed readings are retried.
The cache may be deleted safely. It contains the same local pose data as the run measurements.

The fitted-character cache defaults to `pipeline/vendor/fitted-cache/`. It reuses the completed,
unposed fit, skin binding and hand-grip setup; animation and rendering still run for the selected
exercise. Character files, fitting code/assets, options and runtime versions are fingerprinted, and
invalid or incomplete entries trigger a fresh fit. Set `STUDIO_FIT_CACHE=0` when starting the Studio
backend to disable cache reads and writes for a fresh-fit check. `--force` affects completed exercise
results and does not disable this character cache.

## Validation

From the repository root:

```bash
npm test --prefix studio -- server/preflight.test.js server/preflight-api.test.js server/steps.test.js src/lib
npm test --prefix studio -- server/automation.test.js server/automation-api.test.js server/reference-catalog.test.js server/reference-fetch.test.js
npm run build --prefix studio
studio/.venv/bin/python -m pytest studio/tests/test_preflight.py studio/tests/test_build_targets.py studio/tests/test_pose_cache.py studio/tests/test_picture_round.py studio/tests/test_motion_profiles.py
```

The Python checks need `pytest`, `numpy`, and `pillow`; the real photo-reader CLI additionally needs the
Studio model and MediaPipe. Some older tests require private reference pictures excluded from Git or
macOS image tools. Blender render quality and contact checks must run with the full pipeline installed.

The [latest complete validation report](../reference/automation-validation/corrections/README.md) has zero failed Checks for bench press, overhead press,
lat pulldown, seated dumbbell press, pull-up, forearm plank, leg press, curl, lateral raise, triceps pushdown
and deadlift. This includes skin visibility, motion, supports, equipment contacts and completion;
the earlier deadlift collar defect and short-arm overhead limits have concrete corrections.
Original-proportion attempts still stop when they cannot meet the unchanged physical requirements.

Source depth remains uncertain for oblique/frontal photographs. A run with zero failures can still
need review because its 3D pose is inferred rather than calibrated. The row retains the two source
conflicts described above. The final pull-up passes all 43 frames, including full range, elbow
direction, source widths, grip, visible skin and motion. Its 16 fps sampling keeps the 1.5-second
raise and one-second return; no motion threshold was changed to accept the earlier 12 fps candidate.

The [earlier profile report](../reference/automation-validation/profiles/README.md) retains the baseline
GIFs and failure evidence; the newer [row](../reference/automation-validation/row/README.md) and
[pre-render scan](../reference/automation-validation/skin-preflight/README.md) reports document their
replays. Read `studio/measure/exercise-kit.json` for the current capability flags. Restart Studio after
updating that file to refresh its loaded table; changing a flag alone never establishes accuracy.
