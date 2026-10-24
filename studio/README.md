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
Other non-squat exercises with existing clips are compared against their references without changing
the clip's motion.

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
```

These are alternative invocations. Repeat `--exercise <slug>` to select several exercises.
`STUDIO_URL` changes the backend address; the default is the local Studio on port 5199. The backend
must already be running. `--force` rebuilds an unchanged completed result, while still enforcing
reference and builder requirements. Exit code 0 means passed/ready, 2 means review or blocked items,
and 1 means an error or interrupted batch.

Automatic references use the [reviewed catalog](../reference/exercise-catalog.md): exact exercise
variants at a pinned source revision, with verified image hashes, provenance, and actual camera views.
Manual references are kept. Missing parts of an earlier automatic import can be completed without
duplicating its pictures. Unchanged valid completed results are reused unless `--force` is given.

Only the row and back-squat builders are currently enabled for automatic rendering, and each still
has to pass preflight. The catalog's squat pair supplies only three-quarter views, so it cannot satisfy
the required side positions and front/rear grip-width reference on its own. Unsupported equipment,
missing views, unsuitable source poses, and failed form checks stay visible as blockers or review
items. The plank's static hold and the lunge's alternate lowered stances are not invented into motion
cycles.

**Stop remaining exercises**, or Ctrl-C in the CLI, requests a stop after the current exercise finishes.
Closing the panel does not stop the batch. A backend restart marks unfinished batches interrupted;
they do not resume automatically. Start a new batch to continue, preserving imported references and
completed results.

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
npm run build --prefix studio
python -m pytest studio/tests/test_preflight.py studio/tests/test_build_targets.py studio/tests/test_pose_cache.py studio/tests/test_picture_round.py
```

The Python checks need `pytest`, `numpy`, and `pillow`; the real photo-reader CLI additionally needs the
Studio model and MediaPipe. Some older tests require private reference pictures excluded from Git or
macOS image tools. Blender render quality and contact checks must run with the full pipeline installed.
