# Movement-quality correction evidence

The previous examples passed many contact checks while their exercise movement was still wrong.
This review adds exercise-specific range and body constraints, then measures the actual keyed
character across every frame. The [example table](../corrections/README.md) records each completed
render's real failures and uncertainty. All fourteen mapped examples now have completed media and checks.

## Measured corrections

| Movement | Confirmed earlier problem | Current measured result and replay |
| --- | --- | --- |
| Bench press | Original lockout was about 126°; the bar reversed direction within its stroke. | Both elbows exceed 175°, bar travel 211.7 mm, forearms within 22.9° of vertical; loaded plates, hand skin and soles checked. [Replay](../corrections/flat-barbell-bench-press/evidence/bench-motion-replay.json) |
| Pull-up | A roughly 200 mm elbow branch switch escaped the old endpoint checks. | Stable body, fixed overhand grip, both bottom elbows at least 175°, chin proxy clears by 0.999 mm; pull/lowering elbow-path error 0.0349 upper-arm lengths. [Replay](../corrections/pull-up/evidence/pullup-range-replay.json) |
| Lat pulldown | At the recorded bottom the old elbows were over 111 mm above their shoulders and over 88 mm above their wrists. | No inverted elbow samples; overhead extension and upper-chest endpoint pass, bar travel 398.1 mm. [Form replay](../corrections/lat-pulldown-seated-to-the-front/evidence/pulldown-form-replay.json) |
| Standing shoulder press | Old top elbows were 158.0°/164.6°; the body also needed an upright stance. | Top elbows 178.35°/177.52°, torso approximately vertical, loaded bar; actual grip stays within 10.001° of its starting roll. [Replay](../corrections/standing-barbell-overhead-press-shoulder-press/evidence/overhead-range-replay.json) |
| Seated shoulder press | Old top elbows were about 145°, torso lean about 32°, heel patches up to 43.1 mm above the floor. | Top elbows 177.97°/177.82°, torso lean 1.03°, heel/toe patches within 2.10 mm, actual backrest contact; eased wrist motion. [Replay](../corrections/dumbbell-shoulder-press-seated-overhead-press/evidence/overhead-range-replay.json) |
| Curl and lateral raise | Inferred depth created a crouched/leaning standing body and insufficient exercise-specific range checks. | Upright stable body; curl upper arms remain still; raises reach shoulder height with a fixed slight elbow bend. [Curl replay](../corrections/dumbbell-bicep-curl-standing-both-arms/evidence/isolation-form-replay.json), [raise replay](../corrections/dumbbell-lateral-raise/evidence/isolation-form-replay.json) |
| Triceps pushdown | The supported reference fit needed explicit upright-body, upper-arm and full elbow-range checks. | Mechanical range/contact replay passes. Four photo comparisons remain failed and two uncertain. [Replay](../corrections/tricep-pushdown-cable-bar-or-rope-attachment/evidence/isolation-form-replay.json) |
| Leg press | Source-inferred body rocking and a near-extension knee jump could survive support checks. | Supported torso/head and hands remain fixed; knees move about 65.7° to 173.3°/176.0° along the rail, with full sole support. [Replay](../corrections/leg-press-seated-machine/evidence/sled-motion-replay.json) |
| Squat | A complete usable side-photo start/bottom and grip-view set was not established. | Explicit pinned 69-frame clip/preset with fitted back-bar grip, full range/depth/contact checks; pelvis travel 289.1 mm. Photos are context. [Replay](../corrections/barbell-back-squat/evidence/squat-preset-replay.json) |
| Walking lunge | The original wider photo selection crossed an unobserved support handoff and could not define a trustworthy closed cycle. | Pending: a separately selected 26-source-frame observed step, planned as 42 rendered frames at 16 fps with one-way playback. No completed geometry claim yet. |

The chin value is a geometric lower-face proxy on this fitted mascot. The torso comparisons use the
same actual hip-mid to shoulder-mid axis before and after. Fitted grip roll and humerus twist are
declared character constraints; photographs did not supply those 3D measurements.

The original negative controls remain available:

- [Actual original bench failures](bench-original-failures.json) and
  [raw original bench joints](../../../studio/tests/fixtures/bench-original-range.json).
- [All-frame arm baseline across eleven original scenes](arm-continuity-baseline.json): the old
  pull-up reached 3.93 character heights/s and 1,933°/s; the other ten were below 0.855 heights/s
  and 297°/s. The generous shared caps remain 2 heights/s and 720°/s.
- [Original seated full heel/toe measurements](overhead-original-seated-feet.json), plus raw
  [standing](../../../studio/tests/fixtures/overhead-original-standing.json) and
  [seated](../../../studio/tests/fixtures/overhead-original-seated.json) press geometry.
- [Squat source search review](squat-source-review.json), including unsuitable/occluded candidates
  and the missing genuine side start/bottom pair. Those candidates were not relabeled as usable sides.

## Source modes and unchanged artifacts

Photo-driven corrections retain source provenance, actual source views, input measurements and the
recorded comparison frames. Their inferred depths stay uncertain. Mechanical form constraints can
correct an implausible pose while still disagreeing with the photograph; that disagreement remains
visible. The row retains two such failures and pushdown four. Contact and authored-movement passes
leave failed source comparisons intact.

The squat uses a different explicit source: the pinned existing 3D clip plus a fitted back-bar grip.
Its photos are optional context. Selecting **Reference pictures** still requires the strict photo
views; preset success does not certify that incomplete photographic set. All 69 corrected scene
frames were remeasured under the current coverage metadata and planned cameras before GIF
re-export. That recheck did not change the corrected scene geometry or PNGs.

The walking-lunge input preserves source chronology, timestamps and named heel/toe support
observations. It is one observed step and does not invent a reversed/in-place loop or standing
finish. Uncertain contact is distinct from flat support. Its 42-frame fitted sequence has 25 passing
checks, no failures or missing measurements, and 15 uncertain comparisons. The independent
[contact replay](../corrections/walking-lunge-dumbbells-optional/evidence/lunge-contacts-replay.json)
checks actual heel/toe support, fixed lengths, the airborne ending and absence of a GIF loop extension.

Row, deadlift and plank GIFs/scenes were retained unchanged in this review. The
[row replay](../corrections/barbell-bent-over-row/evidence/row-replay.json) remeasures its steady hinge,
contact and pull; its [arm replay](../corrections/barbell-bent-over-row/evidence/arm-replay.json)
checks continuity. The [deadlift](../corrections/conventional-barbell-deadlift/evidence/movement-quality-recheck.json)
and [plank](../corrections/front-plank-forearm-plank/evidence/movement-quality-recheck.json) have current
arm-motion rechecks on unchanged scenes, with their earlier full physical/source evidence retained.
Their result records distinguish the original render timings from the current rechecks.

## Strict gates and iteration time

Required raw samples must cover every generated frame and the same recorded movement phases.
Checks recompute range from joint coordinates rather than trusting a saved `ready` flag or maximum.
Physical gates include exercise range, finite prop contact, supports, floor clearance and stable
body regions. The common arm gate also checks actual wrist/forearm/upper-arm rotations, which a
hips/knees-only jump detector missed. Cycles include their last-to-first transition; sequences are
checked over their actual one-way duration.

The pre-render skin/motion gate uses the same cameras, visibility policy and limits as final Checks.
Visible defects and missing measurements stop rendering; hidden folds stay informational. Actual
scene replays independently check keyed geometry, props and supports. Passing these engineering
checks does not establish calibrated 3D source anatomy or replace visual form review.

The leg-press fitter now uses one fixed supported body and direct leg-only rail IK instead of a
per-frame torso/head search. Its final measured fitting stage was **8.35 s**; the full animate
process was **155.85 s**, before the separately recorded 55.42 s comparison-still step. Fitting and
full-animation timings measure different work. CPU was shared; these are observed run durations.

Reference-reader and fitted-character caches are reused, and an estimated middle avoids five
unnecessary photo-camera searches. Completed automation reuse hashes actual current builder/gate
code, runtime, source choice, proportions and inputs, so corrected code queues a fresh build while
earlier run history remains intact.

## Reproduce the scene checks

The full saved scenes remain in the run directories listed in each result; GIFs, comparisons,
measurements, hashes and replay outputs are committed. For example:

```bash
blender --background RUN/scene.blend --python-exit-code 1 \
  --python studio/tests/blender/test_overhead_press_scene.py -- --run RUN
```

Other actual-scene replays are
[bench](../../../studio/tests/blender/test_bench_motion_scene.py),
[pull-up](../../../studio/tests/blender/test_pullup_range_scene.py),
[pulldown](../../../studio/tests/blender/test_pulldown_form_scene.py),
[isolation](../../../studio/tests/blender/test_isolation_form_scene.py),
[sled](../../../studio/tests/blender/test_sled_motion_scene.py),
[squat preset](../../../studio/tests/blender/test_squat_preset_scene.py) and
[lunge contact](../../../studio/tests/blender/test_lunge_contact_scene.py).
Each script documents its own arguments and scope. Portable test counts and exclusions belong to
the final [verification record](../corrections/verification.json), separately from scene replay results.

Final verification: **741 portable Python tests passed (27 explicit skips), 365 portable Studio tests passed, 2 targeted API/queue tests passed, and the Studio build passed**. Desktop/mobile playback, source selection, all 14 verified assets and real isolated source automation also passed; exact commands and exclusions are in the linked verification record.
