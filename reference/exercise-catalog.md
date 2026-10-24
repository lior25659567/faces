# Automatic exercise references

`exercise-catalog.json` maps the 14 existing Studio exercise names to exact variants
in Free Exercise DB at commit `f00c92c7dcf1216a928a52c3706c7ce8e2f71ed5`.
The source repository declares its JSON and imagery public domain under the Unlicense;
its README credits `wrkout/exercises.json`, and does not name the photographers.
The pinned license URL is recorded in the manifest.

All 28 source photos and their instructions were inspected when preparing this map.
Each image records its actual view, movement position, context-only status and expected
SHA-256. Most photos are three-quarter views. They are never relabeled as side views to
satisfy a solver. The downloader accepts only the catalog's exact pinned URLs, rejects
redirects, checks image headers and byte limits, and verifies the expected content hash.
It returns image data and provenance; the automation owns importing it without replacing
existing user references.

Walking-lunge references are deliberately ineligible for automatic animation:

- Walking lunge photos depict alternating lowered stances, without a standing start.
  Both are context-only references.

The plank pair depicts a kneeling setup and a static forearm hold. Only the hold is
a usable pose; this pair must not become a kneel-to-plank animation cycle. The
static-hold reader consumes that one pose. Automatic rendering remains disabled
until the mascot passes its actual forearm/toe support checks.

The back-squat pair shows the correct exercise but supplies only three-quarter views.
It cannot satisfy that solver's required side positions and front/rear grip-width
measurements. Its limitation is explicit in the catalog.

`autoBuild: true` means a source pair is suitable as movement endpoints; the engine's
capability checks and pose preflight still decide whether it can build the exercise.
Unknown or ambiguously named exercises have no automatic mapping. Adding another
exercise requires an exact source variant, inspection of its photos/instructions,
truthful positions and views, and pinned image hashes.

On proxy-managed Node 24 runtimes the downloader uses the provided environment proxy
and its `NO_PROXY` exclusions through Node's native `setGlobalProxyFromEnv` support.
Proxy configuration values are never included in errors or provenance. Environments
without a proxy keep their normal direct route.
