# Expanded profile validation

These are real full-pipeline GIFs produced with the cached mascot, pinned source
photos, Blender 5.2.2/MPFB 2.0.18 and the Studio queue. The new previews use 12 fps
(32 frames), two 360 px cameras and four-sample EEVEE. Comparison stills remain
560 px. `results.json` retains actual run IDs, timings, measurements and failures.

| Exercise | Pass | Fail | Uncertain | Not measured | Actual GIF |
| --- | ---: | ---: | ---: | ---: | --- |
| Biceps curl | 12 | 0 | 10 | 0 | [Render](biceps-curl.gif) |
| Lateral raise | 13 | 0 | 10 | 0 | [Render](lateral-raise.gif) |
| Conventional deadlift | 14 | 1 | 10 | 0 | [Render](deadlift.gif) |
| Seated lat pulldown — rejected candidate | 13 | 0 | 10 | 0 | [Rejected render](lat-pulldown.gif) |

Every result needs review; the pulldown candidate is rejected. The ten angle measurements come from inferred photo
depth; they are never promoted into calibrated measurements. The deadlift still
fails a visible left collarbone skin-fold check (992 mm² against a 100 mm² limit).
Its neutral-neck adjustment did not eliminate that fold; the failure is retained.
Bounded clavicle-roll probes did not establish a correction and were not shipped.

The pulldown's clavicle elevation shares high-arm motion with the shoulder girdle
while preserving the measured world arm directions. Its former visible shoulder
fold no longer fails in the corrected candidate render, with the original fold limits.
Visual review nevertheless found an incomplete overhead start. The added movement
completion check rejects it before producing another GIF. Its historical candidate
is retained as evidence of why contact and skin checks alone cannot approve a lift.
The deadlift keeps its shaft outside the body and bounds the necessary reach
correction to 20 mm; this case needed at most 14.1 mm.

Independent saved-scene replays checked every frame's actual hand skin against
finite handles, body/floor penetration and the real GIF/still files. Maximum hand
gaps were below 0.08 mm and maximum handle penetration below 0.8 mm. Dumbbell-head
surface samples also passed for the curl and raise. These contact checks do not
approve the uncertain source angles or the deadlift skin fold.

An actual seated overhead-press attempt was rejected before rendering: the
handles finished 86 mm below the required overhead position and a weight head
overlapped the body by 7.8 mm. Measured Checks and the scene were saved; no GIF or
animation frames were produced. The final check also verifies that Studio keeps
the render step waiting when this happens. A grip alone cannot approve that pose.

All 14 source sets were rechecked unattended. Reference readiness is separate from
the physical and movement-completion gates. The final batch had four reference-ready
exercises and ten blocked. The squat needs genuine side and width photos;
the lunge lacks calibrated stepping positions. Other prototypes retain unresolved
head/reach, body-support or machine-path constraints, including the plank's raised
forearms and the leg press's incompatible sled path. They are not automatically
enabled merely because a prototype can create geometry.

Additional Commons/wger reference research returned proxy 403 responses. The
required hosts are saved in the environment draft; the draft does not activate
network access or publish the prepared environment.

Final portable verification: 294 Studio tests passed across 27 suites; 370 Python
tests passed with 9 explicit skips; all 59 main-app tests passed. Both production
builds passed. Real desktop/mobile browser checks confirmed that blocked physical
measurements remain visible in Checks and the automation panel works without
browser exceptions. The older private-photo/macOS suites retain the limitations
documented in the parent validation report.
