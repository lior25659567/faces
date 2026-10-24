# Exercise Studio

From the repository root, install JavaScript dependencies with `npm ci`. Set up the pose reader with
`cd studio && npm run setup`, then start Studio with `npm run dev` from `studio/`.
The full animation pipeline currently expects macOS Blender at `/Applications/Blender.app`, the MPFB
setup, and the mascot assets. Frontend builds and the pure Python/JavaScript tests can run on Linux.

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
pose are held throughout the rep to prevent drift while the arms move. They do not yet have
a dedicated bar or hand-contact solver. An existing non-squat motion clip is compared with the pictures
but is not corrected toward them. The reference check reports these limitations before a build; more
rerenders cannot remove them. Developing row-specific bar contact and motion constraints requires a
separate Blender validation pass with representative row images.

Successful image readings are cached per exercise under `runs/<exercise>/.pose-cache/`. Image contents,
reader code, model and dependency versions form the cache key. Phase/view changes rebuild targets from
cached detections; changing an image or the detector invalidates its reading. Failed readings are retried.
The cache may be deleted safely. It contains the same local pose data as the run measurements.

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
