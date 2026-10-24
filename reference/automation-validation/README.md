# Unattended batch validation

The real Studio backend processed all 14 exercises on 2026-10-24 using
`STUDIO_URL=http://127.0.0.1:5203 node studio/scripts/automate.mjs --all` and an isolated
`STUDIO_DATA` directory. No user references were changed. The reviewed catalog's
28 actual photographs (1,817,972 bytes) were downloaded and checksum verified.

`batch-result.json` records the actual outcomes and timings. The full batch took
about 5 minutes 47 seconds with already-imported references and cached detections.
The row rendered and completed its comparisons; 13 exercises were blocked with
specific reasons. The squat source lacks the required side/grip-width views;
other profiles lack validated equipment/motion support, and the plank/lunge source
pairs do not supply suitable movement endpoints. Completion does not mean approval.

The row's fitted mascot restored from the new cache in 0.38 seconds. Rendering
still took about 227 seconds, with another 86 seconds for comparison stills.
Its photo checks remained 10 pass / 6 fail. See the separate
[`exercise-row/validation`](../exercise-row/validation/README.md) for the movement's
rendered evidence and explanation of those differences.

Reopening this batch's actual saved scene and running
`studio/tests/blender/test_row_contact.py` passed all 81 frames, including rendered
files, loop continuity, foot planting, full-bar camera visibility, body clearance,
hand/skin contact, and the upward pull. Maximum hand-anchor error was 0.0043 mm;
torso hinge drift was zero. This verifies the cached rig through the real pipeline.

The cache also passed a real Blender library roundtrip and a separate cold/warm
mascot comparison of rest geometry, weights, bones, and grips. Pure tests cover
cache invalidation and corruption; batch tests cover retries, interruptions,
cancellation, input/code changes, artifact completeness, and concurrent manual work.

The batch above ran before the final reuse-fingerprint hardening. Regression tests
verify the additional FBX, selected-clip, actual-runtime and mid-batch code-change
invalidation. No historical run metadata was changed to make it reusable.

Final verification: 291 Studio tests passed across 27 suites, 355 Python tests
passed (7 explicitly skipped), and all 59 main-app tests passed. Both production
builds passed. Desktop/mobile browser checks against the real backend showed the
14 outcomes and source links; a reference-only row batch launched through the UI
completed without adding an animation run.

The broad legacy runs additionally encountered 47 JavaScript and 29 Python
failures tied to excluded private `3d-character/reference-squat` pictures or the
macOS-only `sips` tool. They are not counted as passing above. The focused Studio
run excludes `server/api.test.js`, `server/queue.test.js`, and `server/images.test.js`;
the Python run excludes `test_read_pose.py`. New API/queue/preflight tests and
actual source-photo/pipeline runs provide independent coverage of this change.
