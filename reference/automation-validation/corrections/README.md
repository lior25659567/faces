# Exercise builder correction validation

Eleven complete source-driven animations have **zero failed and zero missing Checks**. Most source photographs still have uncertain depth, so those animations remain review results.

| Exercise | Pass | Fail | Uncertain | Frames / fps | Animation |
| --- | ---: | ---: | ---: | ---: | --- |
| Conventional Barbell Deadlift | 26 | 0 | 10 | 32 / 12 | [GIF](conventional-barbell-deadlift/animation.gif) |
| Flat Barbell Bench Press | 16 | 0 | 10 | 32 / 12 | [GIF](flat-barbell-bench-press/animation.gif) |
| Standing Barbell Overhead Press | 15 | 0 | 10 | 32 / 12 | [GIF](standing-barbell-overhead-press-shoulder-press/animation.gif) |
| Pull-Up | 20 | 0 | 11 | 43 / 16 | [GIF](pull-up/animation.gif) |
| Lat Pulldown | 13 | 0 | 10 | 32 / 12 | [GIF](lat-pulldown-seated-to-the-front/animation.gif) |
| Dumbbell Bicep Curl | 13 | 0 | 10 | 32 / 12 | [GIF](dumbbell-bicep-curl-standing-both-arms/animation.gif) |
| Dumbbell Lateral Raise | 13 | 0 | 10 | 32 / 12 | [GIF](dumbbell-lateral-raise/animation.gif) |
| Seated Dumbbell Shoulder Press | 12 | 0 | 10 | 32 / 12 | [GIF](dumbbell-shoulder-press-seated-overhead-press/animation.gif) |
| Front Plank | 23 | 0 | 0 | 12 / 12 | [GIF](front-plank-forearm-plank/animation.gif) |
| Tricep Pushdown | 22 | 0 | 2 | 32 / 12 | [GIF](tricep-pushdown-cable-bar-or-rope-attachment/animation.gif) |
| Leg Press | 22 | 0 | 10 | 32 / 12 | [GIF](leg-press-seated-machine/animation.gif) |

Each folder contains actual start/end previews, source provenance and hashes, key poses, targets, final Checks, measured physical/skin reports and output hashes. The full Blender scenes remain in the workspace paths recorded in `result.json`; they are not committed. These are isolated validation runs, not replacements for user references.

## What was verified

- The row has separate [81-frame evidence](../row/README.md): both hands and feet stay fixed, both elbow endpoints match, and the torso holds its hinge. Its two source-photo conflicts remain failures.
- The deadlift uses a declared fixed fitness variant, a loaded floor pull, straight arms throughout, a grip outside the knees, planted soles and simultaneous neutral lockout. Actual plate polygons and floor vertices are measured.
- Pull-up elbows stay below the bar/wrists and below the shoulders at the top. Chin clearance, full hang, source widths, fixed bar, grip, visible skin and motion pass. A disclosed 2.406 mm grip-span fit stays inside the original reference tolerance. The 43-frame, 16 fps preview preserves the 1.5-second raise and one-second return; movement thresholds were not changed to accept the earlier 12 fps candidate.
- Presses and pulldowns reach overhead without weight/head overlap. Seated/lying supports and feet are checked on the evaluated mesh. Leg press adds fixed back/head pads, full heel/toe patches and straight rails; opposite side cameras show the complete apparatus. Plank preserves the measured source angles and fits the whole horizontal body.
- The original GLB/cache, face shape and hand shapes are preserved. Fitness arms have fixed rest lengths, with no per-frame stretch. Local rest-bound surface corrections preserve authored weights, UVs and face/hand normals.

## Faster failure detection

The [new pre-render gate](../skin-preflight/README.md) runs the exact existing all-frame skin and motion checks against both the planned preview cameras and the actual source-comparison cameras. Five real-scene parity controls took about 1.8–3.8 seconds each, excluding fitting/rendering. It rejects missing measurements and catches the old hidden-view elbow defect and the previous end-of-cycle motion outlier before GIF rendering.

The complete CLI exercised this gate before the final Plank and Pull-Up renders. An independent CLI failure fixture also saved Checks/scene/report with zero renderer or GIF calls. Older positive runs in this table were completed before the gate was wired; their final Checks and saved scenes provide the same post-render evidence. The `pre_render_gate_exercised_in_cli` field distinguishes these cases.

Estimated middle positions skip five unnecessary picture-camera renders (15 → 10 searches) while retaining all-frame geometry, folds, actual source stills and judged results. Fitted-character and detector caches are reused, and queued/cached builds retain their selected character proportions.

## Retrieval and remaining source limits

The real backend reference-only batch retrieved and verified **38 images (2,870,618 bytes)**: **12 exercises ready, 2 blocked**. See [reference-batch.json](reference-batch.json). It did not render or claim animation approval.

- Squat still lacks the required real side positions and front/rear grip view in its curated source.
- Walking lunge has 12 attributed chronological video frames, but an unobserved support handoff prevents a complete trustworthy root path. Stepping rendering remains blocked; no reversed/in-place cycle was fabricated.
- The row source swings its torso while the corrected motion holds a hinge. The two disagreement rows are kept.

Oblique/frontal depth remains inferred throughout. Positive physical checks do not turn those photographs into calibrated 3D measurements. Skeletal-only USD export explicitly rejects the post-skin surface correction; the validated outputs are Blender scenes and GIFs.

Actual CLI step times are recorded, but runs shared cloud CPU resources and are not isolated performance benchmarks.

Independent replay scripts live under `studio/tests/blender/`, including `test_deadlift_motion_scene.py`, `test_pullup_range_scene.py`, `test_tricep_contact.py`, `test_support_geometry.py` and `test_pre_render_gate_scene.py`. The row and pre-render reports include their replay commands.

## Automated verification

597 portable Python tests passed (18 explicit skips), 315 portable Studio tests passed, 59 main-app tests passed, and both production builds passed. Two new API/queue proportion tests also passed independently. Legacy tests requiring private squat photos or macOS image tools were excluded; Blender scene replays ran separately. Exact commands and scope are in [verification.json](verification.json).
