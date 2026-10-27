# Exercise Studio

From the repository root, install JavaScript dependencies with `npm ci`. Set up the pose reader with
`cd studio && npm run setup`, then start Studio with `npm run dev` from `studio/`.
The animation pipeline needs Blender 5.2 with MPFB and the mascot assets. On macOS it uses
`/Applications/Blender.app`; on Linux it uses `blender` or the `STUDIO_BLENDER` override.
The cloud environment's tested wrapper is `/workspace/.tools/bin/blender-faces`.

## Browse built previews

Studio opens **Built previews**, which lists all 14 mapped exercises. Each available example has a
playable animation, actual Checks and a link to its builder. Pending examples remain marked as pending.
Click **Play preview** to animate a card; one plays at a time. **Full size** opens the GIF and
**View checks** opens its measurements. **Open exercise** switches to your references and run history;
return with **Built previews**. Browsing bundled media needs neither Blender nor reference downloads.

These are draft examples. Passing mechanical checks does not establish calibrated photo depth or
approval of exercise technique. The row retains two source-comparison failures and triceps pushdown
retains four. The [current build table](../reference/automation-validation/corrections/README.md)
distinguishes completed renders, unchanged media with current rechecks, and pending work.
Bundled examples do not replace user references or create entries in user run history.

## Choose a movement source

Where **Movement source** is available, the source choice is saved with the reference check and run.
Changing the source, pictures or tags makes the previous check stale.

- **Checked squat movement preset** uses the pinned 69-frame 3D clip and a declared fitted back-bar
  grip. Pictures remain optional visual context, not motion targets. A changed or missing clip stops
  the preset. Every build still checks depth, bilateral range, spine motion, bar/hand contact, floor
  clearance, skin and movement continuity.
- **Reference pictures** reconstructs the named exercise from photographed positions. Label each
  picture's actual Position and View, run **Check references**, inspect the joint overlays and
  warnings, then build. A readable pose is not proof of accurate depth or finished animation.
- The reviewed walking-lunge input is a chronological **one-way observed step**, not a repeated
  forward/backward lift. Its 26 real video frames cover 2.001667–4.503333 seconds and produce a
  42-frame, 16 fps sequence. Its fitted movement, actual contacts and saved-scene replay pass;
  inferred source depth remains uncertain. **Replay step** plays the observed step again.

Squat photo reconstruction still requires clear side views at the start and bottom, preferably also
the middle, plus a front or rear image showing both shoulders, elbows and hands for grip width.
The curated three-quarter pair does not satisfy those requirements. The checked preset is an
explicit alternative source, not a relabeling of those pictures. See the
[source review](../reference/automation-validation/motion-quality/squat-source-review.json).

The walking source preserves presentation timestamps and named heel/toe support observations.
Uncertain touchdown or toe-off stays uncertain; it cannot silently become a flat planted foot.
The sequence does not invent a return, a standing finish or motion across an unobserved handoff.
The original wider [12-frame bundle](../reference/exercise-lunge/README.md) remains source history;
the selected [one-step sequence](../reference/exercise-lunge/one-step/sequence.json) is a separate input.

## Build and check

1. Select the movement source and add or edit references when that mode needs them.
2. Check the inputs without rendering. Inspect missing-input messages, source views and uncertainty.
3. Build once the current input check is ready. Required range, contact, support and movement checks
   run before the expensive GIF render. Inspect the full animation and final source comparisons.

Comparison stills use recorded movement phase frames. The inspector cannot choose a different,
better-looking frame to conceal an error. In two-photo builds, an estimated middle remains
informational; it does not launch five extra picture-camera searches for an unphotographed pose.

The fitted profiles correct unreliable inferred depth with declared exercise mechanics: bench
lockout and lower-chest path; continuous pull-up elbows and a stable hanging body; a downward
pulldown branch; upright standing isolation movements; full overhead lockout; and a fixed supported
leg-press body. Their fitted grips, roll and body corrections are reported explicitly. They are not
claims that a photograph supplied calibrated 3D anatomy.

The row holds its starting hinge and lower body while its arms pull. Both closed hands stay on one
level bar, and actual hand skin contact is measured. A source demonstration that raises its torso
disagrees with this steady-hinge variant, so its two failed comparisons remain visible.
To add the attributed row photos without replacing user references or starting a render:

```bash
node studio/scripts/import-row-references.mjs
```

The command is repeatable. Original photos, licenses and provenance are in `reference/exercise-row/`.
They remain three-quarter views. See the [row history](../reference/automation-validation/row/README.md)
and the [current replay evidence](../reference/automation-validation/corrections/barbell-bent-over-row/evidence/row-replay.json).

## Automate a batch

Open **Automate exercises**, choose the exercises and start the batch. **Find and check references
only** checks the selected movement source and retrieves missing references when that mode needs
them. Full automation also builds supported movements and runs final Checks. A finished batch can
contain blocked items, uncertain results or source-comparison failures; completion is not approval.

The same runner is available from another terminal in `studio/`:

```bash
npm run automate -- --all
npm run automate -- --all --references-only
npm run automate -- --exercise barbell-bent-over-row
npm run automate -- --exercise barbell-bent-over-row --force
npm run automate -- --all --character-fit auto
```

These are alternative commands. Repeat `--exercise <slug>` for several exercises. The backend must
already be running; `STUDIO_URL` changes its default address, `http://localhost:5199`.
Exit code 0 means passed/ready, 2 means review or blocked items, and 1 means an error or interruption.
`--force` rebuilds instead of reusing a completed result, while still enforcing the input and form checks.

Automation selects the checked squat preset. Other source retrieval follows the
[reviewed catalog](../reference/exercise-catalog.md), pinned hashes, attribution and actual camera views.
The lunge uses its explicitly selected local chronological bundle. Retrieving a source does not prove
that its depth, hidden joints or contacts can support an accurate reconstruction. Manual references
and edits are preserved; incompatible manual inputs are not silently replaced.

**Stop remaining exercises**, or Ctrl-C in the CLI, stops after the current exercise finishes.
Closing the panel does not stop the batch. A backend restart marks unfinished batches interrupted;
they do not resume automatically. A new batch preserves references and completed history.

## Proportions, movement gates and reuse

**Character proportions** is saved with each manual run and automation batch. **Automatic** chooses
the fixed fitness variant when the original short arms cannot satisfy a supported reach profile.
**Original** keeps the original fitted proportions and blocks impossible reach; **Fitness** selects
the variant explicitly. The result states what was used. Face and hand shapes and the original GLB
are preserved. Arm lengths do not stretch during a repetition.

The profiles measure actual equipment contact, body support and exercise-specific range across
every generated frame. Seated press checks its finite backrest and full heel/toe patches; leg press
checks seat, backrest, headrest, soles and rail travel. Plank remains a static supported hold.
Both elbows/wrists and arm/forearm/hand rotations are checked before rendering and in final Checks.
The arm-snap caps are two character heights per second and 720 degrees per second; they are generous
continuity checks, not a technique classifier. Loop seams are checked for cycles. A one-way sequence
is checked as a sequence, not against an invented loop seam.

Visible skin defects are checked from the planned preview and any active source-comparison cameras using the
same detector and limits before rendering and in the inspector. Missing required measurements,
impossible reach, failed physical form or visible folds stop the preview. Hidden folds remain
informational. Mechanical passes do not erase source-comparison failures or inferred depth.

Fast profile GIFs normally use 12 fps; pull-ups and walking sequences use 16 fps, and the row and
checked squat clip use 30 fps. Comparison stills use 560 px and four EEVEE samples. Pull-ups,
pulldowns and shoulder presses use a declared one-second effort, 0.2-second hold and 1.5-second
controlled return. That is demonstration tempo, not timing recovered from two photographs.

Successful picture readings are cached per exercise under `runs/<exercise>/.pose-cache/`, keyed by
image bytes, reader/model code and dependency versions. Phase/view edits rebuild targets from the
cached readings; changed images or detector code invalidate them. Failed readings are retried.
The fitted-character cache at `pipeline/vendor/fitted-cache/` reuses the unposed fit, skin binding
and hand setup. `STUDIO_FIT_CACHE=0` disables that cache; `--force` only disables completed-result reuse.

Completed automation reuse fingerprints current solver/measurement code, runtime, inputs, source
selection and proportions. A changed solver or gate queues a fresh build and preserves the earlier
run; unchanged current results can be reused. Changed preset or sequence selection cannot reuse a
result from a different source mode.

The local rest-bound collar correction preserves authored face/hand normals, UVs, weights and
materials. Skeletal-only USD export explicitly rejects that correction; the checked artifacts here
are Blender scenes and rendered GIFs.

## Validation

The [current example table](../reference/automation-validation/corrections/README.md) links the
rendered media, retained failures and per-exercise evidence. The
[movement-quality report](../reference/automation-validation/motion-quality/README.md) records actual
before/after measurements and distinguishes fresh renders from unchanged media that were rechecked.

From the repository root:

```bash
npm test --prefix studio -- server/preflight.test.js server/preflight-api.test.js server/steps.test.js src/lib
npm test --prefix studio -- server/automation.test.js server/automation-api.test.js server/reference-catalog.test.js server/reference-fetch.test.js
npm run build --prefix studio
studio/.venv/bin/python -m pytest studio/tests -q --ignore=studio/tests/test_read_pose.py
```

The Python checks need `pytest`, `numpy` and `pillow`; real image reading additionally needs the
Studio model and MediaPipe. Older tests requiring private reference photos or macOS image tools
have a separate scope. Actual scene replays require Blender, MPFB and the full assets. For example:

```bash
blender --background RUN/scene.blend --python-exit-code 1 \
  --python studio/tests/blender/test_overhead_press_scene.py -- --run RUN
```

Restart Studio after changing its capability/preset tables. A capability flag alone never establishes
that an exercise has a correct motion or completed render.
