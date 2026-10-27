# Bent-over row validation

The corrected row passes the strict saved-scene contact, movement and render replay.
It still needs reference review: Checks retains 2 failures. A passing
contact replay is separate from matching the source photographs.

[Actual two-camera animation](animation.gif) · [Start](start.png) · [End](end.png)

The original mascot proportions were rebuilt from the verified fitted cache and
posed from the two pinned Free Exercise DB photographs. This run used Blender
5.2.2 LTS/MPFB 2.0.18, 81 frames at 30 fps, two 360 px preview cameras and 560 px
comparison stills. The completed run ID is `row-stance-validated`. Construction used the
production key-pose, retarget and row-contact modules; the queue itself was not
part of this replay. Source image URLs, hashes, code hashes and actual step times
are in [result.json](result.json). These timings include concurrent cloud test load
and are not an isolated speed benchmark.

The row holds the starting torso hinge. Its constant stance balances the observed
thigh directions and knee bends instead of copying the source's lower-body motion.
The feet retain their starting turns. A straight starting knee without a defined
bend plane retains its starting stance. The shared bar stays level with fixed
grip width. Reference angle preferences and fine hand-roll refinement preserve
both elbow bends while maintaining exact, unstrained hand contact; limb lengths
and the five-degree angle tolerance are unchanged.

Start/middle/end are judged at the actual generated frames 1/24/46. The middle is
estimated because neither source photograph records it. The inspector cannot
choose a different, better-fitting frame to hide an error.

The final inspector skips the five cartoon camera searches for the estimated
middle, reducing these searches from 15 to 10. A real CLI follow-up verified
identical all-frame rig and fold measurements, source-still frames/cameras,
judged Checks, verdicts and measurement coverage. Only two informational middle
readings now show rig angles instead of cartoon-detector estimates; their notes
identify that source. Actual photographs and measured middle positions keep
their existing comparisons.

The full replay checks both elbow endpoints against the same five-degree limit,
both evaluated hands against the finite shaft on all 81 frames, bar/body clearance,
fixed feet, ordinary transitions and the loop seam. Maximum hand gaps are below
0.052 mm and maximum bar penetration below 0.81 mm. The source bar path is moved
forward by up to 50.92 mm to clear the mascot's torso; this adaptation is retained
in the report. The chosen grip width (1.486× shoulder width) comes from adapted
arm paths; there is no front/behind photograph to validate that width.

Checks: **18 pass, 2 fail,
0 uncertain, 9 information,
0 not measured**. Raw results are in [checks.json](checks.json).

| Retained failure | Reference | Character | Difference |
| --- | ---: | ---: | ---: |
| end / hip_included | 126.7° | 96.8° | 30.0° |
| end / torso_lean | 34.9° | 55.5° | 20.5° |

The source photographs are three-quarter views with inferred depth, not calibrated
3D measurements. Their torso lean changes from 55.94° to 34.95° in inferred depth;
the projected photo reading also changes by about 12°. Holding a steady row hinge
conflicts with reproducing that torso swing. These differences remain visible and
are not relabeled as passes. A genuine side view and a front/behind grip view are
still needed for reliable form approval.

The earlier full candidate (`row-accuracy-final-kpwmr17v`) retained three failures;
the balanced stance reduces its starting hip discrepancy from 6.2° to about 4.0°,
within the unchanged five-degree tolerance. A finer roll search also
fixes the far arm's 7.6° starting elbow error, which a visible-side-only comparison
missed. The refinement is local to promising coarse solutions for speed, with a
full fine fallback when the coarse search finds no reachable roll.

Replay a completed run with:

```sh
blender --background RUN/scene.blend --python-exit-code 1 \
  --python studio/tests/blender/test_row_contact.py -- --run RUN
```

The upstream dataset declares the Unlicense;
its README credits wrkout/exercises.json without naming photographers. That
provenance and the pinned source revision are retained in `result.json`.
