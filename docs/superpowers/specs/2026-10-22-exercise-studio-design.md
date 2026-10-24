# Exercise Studio — design

Date: 2026-10-22. Status: approved by the user ("that good build it"); builder review folded in.

## Intent and why

Faces is a fitness app. Every exercise in it needs an animated demonstration on the house cartoon
character that a user can copy their form from. The user's quality bar is a real coach doing the
exercise correctly (their reference: four screenshots of a barbell back squat in a commercial
fitness app, saved in `3d-character/reference-squat/`). An animation that is "almost right" — hands
off the bar, elbows flared, wrong depth — is a failure.

Until now each exercise was fixed by hand, one round at a time, and no squat has yet matched the
reference. That does not scale to every exercise the app needs. The Studio turns it into a
repeatable flow the user drives themselves: **upload reference images of an exercise → watch the
waiting time → when it's done, see the animation as a GIF next to the reference, with a pass/fail
check.**

## Goals

1. **A local web app, "Studio"**, layout **A · Workspace** (chosen by the user): exercise list on the
   left with a status per exercise (passing / failing / running / needs a reference); one exercise
   open on the right with everything about it on one page.
2. **Upload reference images** onto an exercise (drag and drop). Images are uploaded in the order of
   the movement (start → end); the page shows them in that order and lets the user reorder them.
3. **Processing starts automatically on upload.** No other buttons are needed in the normal flow.
4. **Waiting time is visible**: each step shown with time spent and an estimate of time left,
   estimated from how long the same step took on previous runs.
5. **Images produce the animation** (user's decision: images first, not video). The key positions
   are read from the reference images; the movement between them is generated with real-lift
   logic. A Mixamo clip may be used for the natural feel of the movement where one exists.
6. **The result is a GIF** of the character doing the exercise (fast preview quality).
7. **The comparison**: when a run finishes, the page shows each reference image next to our
   character at the matching moment, the GIF, and a table of checks — the reference's value, ours,
   and pass/fail.
8. **History ("progress")**: every run of an exercise is kept with its pass count, so the user can
   see an exercise improve from run to run.
9. **The flow is captured as a reusable project skill** ("add / animate an exercise") so it runs the
   same way every time. (The user asked for this alongside the design.)

## Non-goals (for now)

- Video as an input. Planned for later: the same reading step run on every frame would make the
  video itself the motion. Not in this build.
- Putting the animation into the iOS app (SceneKit). The user has deprioritised the app; GIFs and
  the `.usdz` files are enough to judge.
- Cloth physics (shirt drape). Known limitation, separate feature.
- Importing the 876-exercise open library (free-exercise-db). Later; the Studio starts with the
  existing 14 exercises plus "add exercise" by name and equipment.
- Hosting online or multiple users. Local, one user.
- High-quality (Cycles) renders.

## Anti-goals (would make it a failure even if it "works")

- **Showing a pass that isn't one.** This project has repeatedly had checks pass while the thing
  they protected was broken. Every status in the Studio must come from a measurement made on the
  run's actual output; a step that fails or produces nothing must show as failed, never as done.
- **Needing a person to hand-fix an exercise** for it to pass. If the flow can't make an exercise
  pass, it reports why; it doesn't hide it.
- **Breaking or slowing the live face demo.** The repo's existing Vite dev server and camera demo
  are in use by the user; the Studio must not share its port, entry point, or watch its folders.
- **Shipping the reference images.** The coach screenshots belong to another company. They are an
  internal check only and must never be copied into anything the app ships.

## Constraints

- **Runs on the user's Mac** (Apple Silicon, 8 GB RAM). Blender and the pipeline live there.
  Memory is tight — earlier runs pushed the machine into heavy swapping — so **jobs run one at a
  time** from a queue.
- **Free and safe for a commercial app**: MPFB/MakeHuman output is CC0; MediaPipe is Apache-2.0;
  Mixamo animations are royalty-free in projects; Blender output is unrestricted.
- **The user's stack** (global preferences): React + Vite, vanilla CSS in per-component files,
  components in PascalCase, files in kebab-case, const arrow function components.
- **No git** in this repo — no commits; back up files before editing.
- **Fast previews only**: EEVEE, ~560 px, with lights (Workbench is not available under that name
  in this Blender).
- Blender 5.2.2 LTS at `/Applications/Blender.app/Contents/MacOS/Blender`, run `--background`.
  MPFB is installed as a Blender extension (`bl_ext.user_default.mpfb`).

## Decided (by the user)

- Layout A · Workspace.
- Upload → automatic processing → waiting time → comparison. No intermediate steps for the user.
- Images are the input for now; the animation is generated from them.
- Output is a GIF.
- Visible character: the house cartoon boy (`3d-character/3d cartoon boy.glb`) wrapped over an
  MPFB skeleton (the "cartoon mascot on top" decision), driven by the existing pipeline.
- Acceptance: the animation must match the reference images — hand position on the bar, grip,
  elbows, depth, posture — not merely "look like the exercise".

## The flow (what happens after an upload)

1. **Read the reference.** For each image, find the body's joints (MediaPipe Pose Landmarker,
   heavy model) and compute the measurements: joint angles (knee, hip, elbow, upper arm vs torso),
   torso lean, hip vs knee height, grip and elbow width relative to shoulder width (only from a
   straight-front or straight-behind view — a side view hides the far shoulder), wrist and elbow
   height vs the shoulder. Save them as the exercise's targets (the squat's are already in
   `3d-character/reference-squat/targets.json`, with definitions and tolerances; that file is the
   format to follow). If an image has no readable body, the step fails and says which image.
2. **Build the key positions.** Pose the character to each reference image: aim each bone along
   the direction the reference's matching bone points (our bone lengths kept), plant the feet on
   the floor, and put the hands on the equipment (grip points on the bar / handles).
3. **Generate the movement between the positions** with real-lift logic: the lowering phase
   slower than the lifting phase, eased in and out, the bar travelling straight over the middle of
   the foot, the body's balance kept over the feet, feet locked to the floor where they don't
   move. Use a Mixamo clip as the base feel when the exercise has one; otherwise interpolate
   between the key positions.
4. **Render the GIF** (fast preview, the same camera views as the existing staging), plus a still
   of our character at each reference image's moment.
5. **Check it.** Measure our character from its own rig joints (MediaPipe cannot read the cartoon
   reliably — it found no body in one frame and returned impossible angles in another) using the
   same definitions as the targets, compare against the targets with their tolerances, and record
   pass/fail per check. Also run the existing integrity checks: no cracks, feet not sliding or
   sinking, hands on the equipment.
6. **Show it.** The page switches to the comparison when the run finishes.

## Existing work to build on (not rebuild)

- `pipeline/blender/mpfb_cage.py` — MPFB skeleton fitted to the cartoon skin, bar grip solve.
- `pipeline/blender/skin_wrap.py` — wraps the cartoon skin on the skeleton and bakes it to a
  normal skinned mesh (crack guard included).
- `pipeline/blender/retarget.py` — Mixamo motion onto the rig, per-frame grounding, foot lock.
- `pipeline/blender/animate.py` — the end-to-end runner (`--mpfb`, `--mixamo`, `--fast`).
- `pipeline/blender/equipment.py`, `exercise_stage.py` — props and cameras per exercise.
- `scratchpad/pose/measure_ref.py` (move into `pipeline/`) — the reference measurement used for
  the squat targets.
- `reference/exercises/poses.json` and `form-guidelines.json` — the 14 exercises and their sourced
  form data.
- Reports in `.sdd/` (task 12 in particular) describe what each piece does and its known limits.

## Left to the builder

- Server technology. A small local server behind the Vite app (the repo already uses a Vite plugin,
  `pipeline/frames-api.js`, as an in-process API — a separate Node server or a Vite plugin on the
  Studio's own dev server are both fine) that saves uploads, runs the job queue as subprocesses,
  and reports progress. Python or Node is the builder's call; keep the user's front-end stack.
- Where the Studio lives (e.g. `studio/` with its own Vite config and port) and where uploads and
  run results are stored (e.g. `3d-character/references/<exercise>/` and
  `3d-character/runs/<exercise>/<run-id>/`).
- How progress reaches the page (polling or server-sent events) and how the time-left estimate is
  computed (e.g. median of the last few runs of the same step).
- How the movement between key positions is generated in detail, as long as it follows the
  real-lift logic above and passes the checks.
- The exact list of checks per exercise beyond those defined in the squat targets file.

## Calls made after the builder review (Claude's calls — the user can overturn any)

1. **Checks a character's proportions can't meet.** The cartoon boy's upper arm is about half a
   human's length relative to his shoulder width, so not every coach arm number can hold at once
   (with elbows down at the coach's angle and forearms vertical, grip tops out near 1.7× shoulder
   width vs the coach's 1.87×). When a target is geometrically impossible for the character, the
   check aims for the closest achievable value and reports the gap as **"limited by proportions"**
   — a separate status from fail, shown with the numbers. Priority goes to what the eye reads:
   elbows down, forearms vertical, hands wrapped on the bar, depth, back angle. (Told to the user;
   their alternative was lengthening the character's arms.)
2. **Measurements that look wrong.** Every angle is measured two ways: in 3D (MediaPipe world
   landmarks) and in the image plane. Where they disagree by more than 10°, the check is
   **"uncertain"**: shown with the reference and our still side by side, not counted as pass or
   fail. Measured on the squat: standing knee reads 175° flat vs 160° in 3D (he is visibly
   standing tall — flat is right); elbow in side view reads 6–9° flat vs 82–100° in 3D (the
   forearm hides behind the upper arm — 3D is right). The user's reference images are the
   authority over `form-guidelines.json` where they conflict (squat depth: parallel, as in the
   reference).
3. **What each image is.** Every uploaded image carries a **phase** (start / middle / end, guessed
   from upload order, editable), a **view** (side / front / behind / three-quarter, detected from
   the body, editable), and can be marked **context only** (not a key position). Widths (grip,
   elbows) are measured only from front or behind views. For the squat milestone: 15/16/17 are
   the key positions, the straight-behind view (cropped from the thumbnail inside 15.png,
   `scratchpad/pose/behind-thumb.png`) is uploaded as a behind view of the start, and 14.png is
   context only (it shows the walkout from the rack, not a squat phase).
4. **A bad image fails only itself.** The run fails only if fewer than two key positions remain.
5. **Working alongside the squat fix (task 12).** Task 12 owns `mpfb_cage.py` and `animate.py`
   until it reports. The Studio is built first in the parts that don't touch them — app, server,
   job queue, upload, measurement, comparison page — and runs the pipeline through `animate.py`'s
   command line (task 12 added `--form-targets`). When an exercise has a Mixamo clip, the motion
   comes from the clip corrected toward the targets (task 12's solver); when it has none, from
   aiming the bones at the reference images.
6. **Skin stretch at full depth** is out of scope for this build: measured and shown as a number,
   not pass/fail, and a known limitation (fixing it needs corrective shapes, a separate piece of
   work). The coach squats to parallel, shallower than the Mixamo clip, so it shows less.

## First milestone

The barbell back squat, end to end through the Studio, using the user's four reference images:
upload → waiting time → GIF + comparison, with every check passing except those marked "limited
by proportions" or "uncertain" (each shown with its numbers), and the comparison looking the same
as the reference. This is the user's acceptance test for the whole
approach.
