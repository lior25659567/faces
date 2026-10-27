# Automatic exercise references

`exercise-catalog.json` maps the 14 existing Studio exercise names to exact variants
in Free Exercise DB at commit `f00c92c7dcf1216a928a52c3706c7ce8e2f71ed5`.
The source repository declares its JSON and imagery public domain under the Unlicense;
its README credits `wrkout/exercises.json`, and does not name the photographers.
The pinned license URL is recorded in the manifest.

Unattended reference retrieval covers all 14 names. The preferred walking-lunge
source is the separate licensed video-frame bundle below; the pinned photo catalog
retains its earlier context pair. Existing manual references remain authoritative.
Finding exact source images and building an accurate movement are separate checks.

All 28 source photos and their instructions were inspected when preparing this map.
Each image records its actual view, movement position, context-only status and expected
SHA-256. Most photos are three-quarter views. They are never relabeled as side views to
satisfy a solver. The downloader accepts only the catalog's exact pinned URLs, rejects
redirects, checks image headers and byte limits, and verifies the expected content hash.
It returns image data and provenance; the automation owns importing it without replacing
existing user references.

The two walking-lunge photos show alternating lowered positions and remain context only.
The older [12-frame sequence](exercise-lunge/README.md) also retains its historical support-handoff blocker.
The preferred source is now the [reviewed one-step bundle](exercise-lunge/one-step/README.md):
26 licensed chronological video frames with conservative heel/toe observations. Its fitted 42-frame
animation passed the contact, range, floor, skin and continuity checks and an independent saved-scene
replay. It plays once and ends during the observed swing-through. The fixed frontal camera does not
calibrate human depth, so those comparisons remain uncertain.

Exact reviewed source selection is saved with each run. Unchanged automatic legacy references can
remain as context beside the new sequence without deleting or retagging them. Manual changes block
silent migration. Ordinary unreviewed two-photo stepping still cannot use the named sequence path.

The plank pair depicts a kneeling setup and a static forearm hold. Only the hold is
a usable pose; this pair must not become a kneel-to-plank animation cycle. The
static-hold reader consumes that one pose. The implemented static plank validates
forearm/toe support, whole-body floor clearance, body alignment, motion and visible
skin before rendering. Its original short-arm proportions can still fail the
physical checks; the explicit/automatic fitness variant preserves the source
model and face while providing fixed arm reach.

The back-squat pair shows the correct exercise but supplies only three-quarter views.
It cannot satisfy that solver's required side positions and front/rear grip-width
measurements. Its limitation is explicit in the catalog.
Automatic builds instead select the checked, pinned 3D squat preset with a declared fitted back-bar
grip. Photos are optional context in that mode. The separate photo-reconstruction mode retains its
side and front/rear requirements; the preset does not pretend the photo views changed.

The other implemented profiles build their required supports and finite equipment
contacts. The reference-driven builder holds isolation poses, checks overhead
completion separately from grip, aligns leg-press support to its measured rail,
and records the actual preview and source comparison cameras. Original, Fitness
and Automatic proportions are saved per run; the original GLB and fitted cache
are not changed by a fitness build.

Before GIF rendering, the same all-frame skin-fold and movement checks used by
final Checks run against the planned GIF views and the source comparison views
at their recorded pose times. Saved-scene scans took about 1.8–3.8 seconds,
excluding building/rendering. Visible defects and missing measurements block the
preview. Hidden folds remain informational. For profiles with only start/end
photographs, the five camera searches for an estimated middle are skipped,
reducing detector-rendered comparisons from 15 to 10 without dropping a source
photograph. See the [scan parity report](automation-validation/skin-preflight/README.md).

The [current validation](automation-validation/corrections/README.md) contains all 14 mapped examples
with no failed mechanical checks or missing measurements. The row retains two source comparisons
and pushdown retains four; inferred source depth still requires review. Their actual failure counts
remain visible. The pull-up uses 43 frames at 16 fps with one second of effort and a controlled
1.5-second lowering, along with full hang, chin clearance, stable grip and continuous elbows.
Current engine flags live in `studio/measure/exercise-kit.json`.

`autoBuild: true` means a source is reviewed for its declared movement mode; the engine's
capability checks and pose preflight still decide whether it can build the exercise.
Unknown or ambiguously named exercises have no automatic mapping. Adding another
exercise requires an exact source variant, inspection of its photos/instructions,
truthful positions and views, and pinned image hashes.

On proxy-managed Node 24 runtimes the downloader uses the provided environment proxy
and its `NO_PROXY` exclusions through Node's native `setGlobalProxyFromEnv` support.
Proxy configuration values are never included in errors or provenance. Environments
without a proxy keep their normal direct route.
