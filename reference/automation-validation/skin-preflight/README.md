# Movement and skin checks before rendering

The preflight checks every posed frame from the planned GIF cameras and each
active reference comparison camera at its recorded pose time. It reuses the
inspector's fold detector, jump/step measurements and existing Checks classifiers without changing limits.
It creates no renders and preserves the scene's frame, camera and geometry.

Five actual saved scenes reproduce their complete existing fold and movement rows exactly:

| Control | Skin result | Movement result | Preflight ready | Seconds |
| --- | --- | --- | --- | ---: |
| Overhead press | Hidden folds, information | Pass | Yes | 3.53 |
| Forearm plank | Hidden folds, information | Pass | Yes | 1.75 |
| Previous pull-up | Visible elbow defects, fail | Pass | No | 3.53 |
| Previous seated dumbbell press | Visible shoulder defect, fail | Pass | No | 3.62 |
| Pull-up with a later body jump | Hidden folds, information | Fail at frame 31 | No | 3.77 |

The pull-up control includes its actual behind comparison cameras at frames 1
and 19. Omitting that view would miss the visible 313 mm² elbow defect. Source
camera positions match the real comparison renderer within 0.0002 mm.

The later pull-up passes the skin check but has a 51 mm body jump on frame 31.
The existing movement rule catches it before rendering, with the exact same
failure row as the completed run.

Hidden folds remain informational; they are never relabeled as passes. Missing
camera metadata, interrupted scans and unreadable geometry block rendering with
the existing “not measured” rows. Source form and mechanical checks retain their
separate verdicts.

The run paths, exact rows, camera coverage and measured timings are retained in
[parity.json](parity.json). The timings describe these saved-scene checks.

Replay a completed run with:

```sh
blender --background RUN/scene.blend --python-exit-code 1 \
  --python studio/tests/blender/test_skin_preflight_scene.py -- --run RUN
```
