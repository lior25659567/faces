---
name: animate-exercise
description: Use when animating a gym exercise or checking it against reference photos in the Faces Exercise Studio, when the Studio shows failing, limited or uncertain checks, when a deadlift/squat/any exercise "looks wrong" or "hunched", or before touching the pose/animation pipeline.
---

# Animate an exercise (Faces Exercise Studio)

**The Studio is the only way in. Checks report honestly; you never bend them.** Violating the letter of these rules is violating the spirit.

## Start
- `cd studio && npm run dev` -> http://localhost:5199 (this Mac only). First time only: `npm run setup` (Python env + model).
- Is a Studio already running on 5199? It is the user's live one. Look, never test against it.
- Repeatable squat setup: `node studio/scripts/squat-milestone.mjs` (non-destructive; talks to a Studio at `--url`, default 5199 = the live one, so pass the throwaway instance's URL when testing).

## Add references
Drag, click, or paste (Cmd+V) onto the exercise. Processing starts by itself. Tag each image:
- phase: start / middle / end; view: side / front / behind / three-quarter; or context only (ignored by measuring).
- **Side views give angles. A straight front or behind view gives widths.** The coach's back is readable only when arms don't hide it.
- Fix a wrong tag in the Studio, then re-run. Do not delete the user's pictures.

## Statuses -> action
| Status | Meaning | Do |
|---|---|---|
| pass | measured and within tolerance | nothing |
| fail | measured and off | report row, ours vs target, numbers, frame. **Never** edit tolerances, targets, poses or rows to turn it green. A real fix is pipeline work (see Work log). |
| limited | rare: a human-proportioned arm meets the coach's number, ours can't | say it is the character's proportions; offer the user longer arms (their call) |
| uncertain | two readings disagree: the coach's (flat vs 3D), or ours - our picture (read with the coach's detector) vs our rig, on the lean and hip angle, when they lie either side of the coach or one of them passes; the note gives both numbers and the hint says whose | open the stills and judge by eye; report what you saw and both numbers. Not a fail. |
| info: Not measured | would be judged, but the character wasn't measured there (amber; keeps the exercise from passing: "N not measured") | report it as a gap, never as a pass; finding why it wasn't measured is pipeline work |
| info: Not judged | the reason is on the coach's side (back hidden, no front/behind picture, estimated middle) or folds hidden from every view | report the reason; a better reference picture may let it be judged |
| info: Shown only | never judged by design (face direction, skin stretch, "Used to find this moment") | report only |

Body checks: cracks, feet sliding/sinking, hands on bar, smooth motion (jumps), skin folding (creases in deep bends are normal), skin stretch (info). Posture: back straight (only when form notes require it), head in line (from the ears), face direction (shown only).

**No clip** (page: "No motion clip: moves between the pictures' positions"): the run still animates, between the side pictures' start / middle / end (`keyposes.json` from the read step -> `animate.py --keyposes`; `pipeline/blender/keypose_clip.py`). Start -> end 1.5 s, hold 0.2 s, back 1.0 s. Side angles follow the pictures (the flat readings); left-right is made symmetric and the hidden arm mirrors the seen one, because MediaPipe's depth in a side picture is a guess. What each exercise holds and stands on is one table, `studio/measure/exercise-kit.json`:
- refused in plain words, before anything is built ("No motion for this exercise: it needs a bench / seat / pull-up bar / leg-press sled the picture path doesn't build yet", or the lunge's "its feet step"): bench press, seated dumbbell press, lat pulldown, leg press, pull-up, walking lunge. Tell the user; the fix is the clip or new props, never a workaround.
- moves but the hands hold nothing (page: "holding no bar"): deadlift, overhead press, row, curls, lateral raise, pushdown - and any exercise you add with barbell/dumbbells/cable. The body row "Equipment in the hands" is **not measured**, so it can never show green. Report it as that gap.
- only "Barbell Back Squat" gets the bar, grip and solver (form targets, picture loop): from the squat's pictures 18 pass / 5 fail / 3 limited / 4 uncertain (its clip: 20 / 6 / 0 / 4). Elsewhere expect the judged hip and lean to read off: our picture reads the trunk more upright than the rig (Task 19) and nothing aims at it. Collarbones only follow the chest.
A captured clip (`3d-character/mixamo/README.md`, e.g. `deadlift.fbx`) is still the better motion. Do not substitute another exercise's clip.
**No motion yet** ("No motion for this exercise yet. Add a Mixamo clip ..."): no clip AND the run has no `keyposes.json` (the read step writes it from a start and an end side picture with a body found; without those the read step already fails, naming each picture). Tell the user which pictures or which clip to add.

## "Every check passes but it looks wrong" (e.g. hunched squat)
The user's eye is data. Never say "the checks pass". Measure what they see (from the stills/our spine and ears), then add or adjust a check honestly so it fails on the current animation, then fix. Tell the user what you measured.
- Measure from `ours.json` and the stills (spine line, ears), and show the numbers.
- "Adjust honestly" = the new check measures what the user sees and **fails on the current animation first**. Loosening or removing a row is bending it.
- User in a hurry ("just make it pass", "presenting soon"): still no. Offer: show the evidence, or present as-is labelled known-hunched. The user may choose that; you may not change the check to hide it.

## Work log (standing requirement: the user watches live)
Any pipeline fix is opened and logged, before the first edit:
```
node studio/scripts/work.mjs start --title "..." --exercises <slug>|all   # prints id
node studio/scripts/work.mjs log <id> "what I measured/changed"
node studio/scripts/work.mjs done <id> --summary "..."   # triggers the automatic re-run
node studio/scripts/work.mjs fail <id> --summary "..."   # nothing re-runs
```
Verify a fix with full `animate.py` runs in your own scratch folder (one Blender at a time, EEVEE), never against the live Studio's data or `3d-character/studio`. Then call `done`, which makes the user-visible re-run; never skip it. Logging is never skipped for speed, even for a one-line fix.

## Constraints that bite
- One Blender at a time (8 GB Mac): `pgrep -fl Blender` first; wait if one runs.
- EEVEE/Workbench previews only, never Cycles.
- No git here: copy a file to `name.bak-<tag>` before editing it.
- Reference images are internal QA, never shipped or copied into the app.
- Tests: throwaway instance, `STUDIO_DATA=<tmp> npx vite --port <other>`; never write into `3d-character/studio`.
- Off limits: the live face demo (root vite on 5173, `src/`, `input/`).

## Where things are
Runs: `3d-character/studio/runs/<slug>/<run-id>/` (run.json, checks.json, ours.json, stills/, animation.gif, logs/). Target rules: `studio/measure/build_targets.py`. Checks: `studio/measure/checks.py`.

## Red flags -> stop
"Widen the tolerance a bit", "drop that row", "uncertain is basically a fail", "I'll run it against the live Studio / skip `done`", "Cycles for a nicer render", "all green so it's fine", "I'll edit the Studio's data directly".
