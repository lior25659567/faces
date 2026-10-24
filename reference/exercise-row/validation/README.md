# Rendered row validation

`row.gif` is the actual 81-frame row rendered from two cameras at 320×320 per view,
EEVEE four samples, with Blender 5.2.2 and MPFB 2.0.18. The source photographs are
`../start.jpg` and `../end.jpg`; their provenance is in `../sources.json`. The pose
reader used the checksum-pinned MediaPipe model from `studio/setup.sh`.

`row-validation.json` comes from reopening the completed scene and running
`studio/tests/blender/test_row_contact.py`. It measures both hand anchors and actual
skin contacts, elbow endpoints, upward stroke, planted feet, hinge stability, loop
continuity, and full-bar visibility in both cameras. The test also checks the
rendered image files and GIF. `solver-report.json` records contact, surface-clearance
adjustments and runtime settings. Studio’s separate form report uses its usual tolerances.

Observed results include both hands touching in all 81 frames, grip-anchor error
below 0.005 mm, no measured foot slide, no detected torso/thigh intersection, and
37.7 mm of bar rise. Actual elbows are about 165° at Start and 100° at the pull;
the oblique reference estimates are 172° and 108°. The torso intentionally keeps
the starting hinge, while the demonstration rises between photographs. Source
angle estimates and the mascot's proportions still differ; Studio's form checks
retain those differences rather than awarding an automatic passing score.

The full replay command, after building a run in Studio, is:

```bash
blender --background RUN/scene.blend --python-exit-code 1 \
  --python studio/tests/blender/test_row_contact.py -- --run RUN
```

For faster development before rendering, the same test supports explicit
`--geometry-only --scene POSE.blend --report REPORT.json`. That mode does not check
or certify rendered output and prints a distinct result.

The photo-comparison report (`photo-checks.json`) records 10 passes and 6 angle
differences. Two end-position differences reflect the deliberately held torso;
the other four are approximately 6–8° offsets in inferred reference/mascot angles.
There are no unmeasured required checks in this run. These differences remain
visible in Studio.
