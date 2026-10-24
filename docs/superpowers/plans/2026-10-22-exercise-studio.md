# Exercise Studio Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A local web app where the user drops reference images of an exercise, watches the waiting time, and gets the cartoon character's animation as a GIF next to the reference with honest pass/fail checks and a run history.

**Architecture:** A React + Vite app in `studio/` with its own dev server (port 5199) whose Vite plugin is the API: it stores uploads and runs under `3d-character/studio/`, and a one-at-a-time queue runs each run as four subprocesses — read the reference (Python + MediaPipe), build/animate/render (Blender, the existing `animate.py --mpfb`), stills + our measurements (Blender, a Studio script on the saved scene), check (Python). The measurement definitions live in one numpy module used on both the coach's joints and our rig's joints, so "the same definitions" is literally the same code.

**Tech Stack:** React 19, Vite 8, Vitest 5 (all already in the root `node_modules`); Node 22; Python 3.10 venv with mediapipe 1.1.1, numpy, pillow, pytest; Blender 5.2.2 LTS (its Python 3.13 + numpy 2.3) with the MPFB extension; ffmpeg.

**Spec:** `docs/superpowers/specs/2026-10-22-exercise-studio-design.md` (read it first; "Calls made after the builder review" settles the measurement rules used below).

## Global Constraints

- **No git** in this repo: never run git. Before editing an existing file copy it to `<file>.bak-studio`. Record each finished task in `.sdd/progress.md`.
- **Leave the live face demo alone:** do not edit root `vite.config.js`, root `package.json`, `pipeline/frames-api.js`, `src/`, `input/`, or `pipeline/.venv`. The Studio runs on port **5199** with `strictPort: true` (never 5173). No new npm packages (the live demo uses the root `node_modules`).
- **Task 12 (the squat fix) owns** `pipeline/blender/mpfb_cage.py`, `animate.py`, `retarget.py`, `skin_wrap.py` until `.sdd/task-12-report.md` says DONE. Studio code may import them, never edit them. Only Task 9 edits `animate.py`, and only after task 12 is done.
- **One Blender at a time** (Mac with 8 GB): before starting Blender, `pgrep -fl "Blender.app/Contents/MacOS/Blender"`; if one is running, wait for it to finish.
- **Fast previews only:** EEVEE through `mpfb_cage.preview_set`, as `run_mpfb` uses it. No Cycles.
- **Reference images are internal QA only** (another company's screenshots): they live under `3d-character/studio/`, are served only on localhost, and are never copied into `faces-ios` or anything shipped.
- **Python:** Studio venv `studio/.venv` (Python 3.10, `mediapipe==1.1.1`, numpy, pillow, pytest). Never install into `pipeline/.venv`. Model `studio/models/pose_landmarker_heavy.task`, sha256 `64437af838a65d18e5ba7a0d39b465540069bc8aae8308de3e318aad31fcbc7b`.
- **Blender:** `/Applications/Blender.app/Contents/MacOS/Blender --background --python-exit-code 1 ...` (without `--python-exit-code 1` a crashing script exits 0).
- **Front-end conventions:** components PascalCase, files kebab-case, `const X = () => {}` components, vanilla CSS in one file per component.
- **Honesty:** every status comes from a measurement on the run's own output. A step that exits non-zero, or exits 0 without its required files, is failed — never done.
- **Numbers from the spec:** tolerance 5° for angles, 0.12 for width ratios, 0.10 torso lengths for hip-vs-knee; a measurement is "uncertain" when its two readings differ by more than twice the tolerance (10°, 0.24).
- **Plain words in the UI** (labels in Task 7); the user is not a rigging specialist.

## Measurement rules (from the spec's calls 2–3, used by Tasks 1–3 and 6)

| Metric | Side view | Front/behind view | Ours measured as |
|---|---|---|---|
| `knee_included`, `hip_included` | flat (image plane) is the value; 3D is the cross-check | not measured | projected on the side camera's plane |
| `torso_lean` | flat only | not measured | projected on the side camera's plane |
| `elbow_included`, `upperarm_vs_torso` | 3D only (flat collapses: the forearm hides behind the upper arm) | not measured | 3D |
| `grip_ratio`, `elbow_width_ratio` | not measured | flat is the value; 3D is the cross-check | projected on the behind camera's plane |
| `hip_below_knee` (end position) | flat, in torso lengths, + = hip lower | not measured | projected on the side camera's plane |

Three-quarter views and context-only images are shown but feed no target. Our side for one-sided metrics = the reference image's `camera_side`.

What these rules give on the user's squat (measured 2026-10-22, the tests pin them):

| image | knee flat / 3D | hip flat / 3D | elbow 3D | upper arm 3D | lean | other |
|---|---|---|---|---|---|---|
| 15.png (start) | 174.7 / 160.4 → uncertain | 170.8 / 167.1 | 90.5 | 41.3 | 6.7 | |
| 16.png (middle) | 104.7 / 87.9 → uncertain | 109.2 / 97.9 → uncertain | 99.9 | 50.5 | 18.4 | |
| 17.png (end) | 47.2 / 64.5 → uncertain | 69.3 / 77.3 | 82.3 | 52.0 | 22.4 | hip below knee −0.05 |
| behind-thumb.png | | | | | | grip 1.87 / 1.64; elbows 1.97 / 1.76 |

## Data layout and file formats (shared by every task)

```
3d-character/studio/
  README.md                         internal QA only — never ship these images
  added-exercises.json              [{"name", "equipment", "clip": "file.fbx"|null}]
  references/<slug>/references.json {"images": [ImageMeta]}
  references/<slug>/<id>.<ext>      uploaded images (never overwritten; delete = "deleted": true)
  runs/<slug>/<run-id>/
    run.json  references.json (snapshot)  logs/<process>.log
    measurements.json  targets.json  pipeline-targets.json (only when the solver supports the exercise)
    frames/ (frame_####.png, side/, retarget.json)  animation.gif  scene.blend
    stills/<image-id>.png  ours.json  checks.json
```

`ImageMeta`: `{"id": "img-1a2b3c4d", "file": "img-1a2b3c4d.png", "original": "15.png", "order": 1, "phase": "start"|"middle"|"end", "phaseSetBy": "guess"|"user", "view": "side"|"front"|"behind"|"three-quarter"|"frontal"|null, "viewSetBy": "detector"|"user"|null, "contextOnly": false, "deleted": false}`. `"frontal"` = the detector saw a front or back but cannot tell which (MediaPipe reads a back as a front); the user picks.

Run snapshot `references.json`: `{"exercise": "Barbell Back Squat", "slug": "barbell-back-squat", "images": [ImageMeta + "path": absolute path]}`, deleted images left out.

`run.json`: `{"id", "slug", "name", "clip": "squat.fbx"|null, "status": "queued"|"running"|"done"|"failed"|"interrupted", "createdAt", "startedAt", "finishedAt", "steps": [{"id", "status": "waiting"|"running"|"done"|"failed", "startedAt", "finishedAt", "seconds", "fraction", "note"}], "error": {"step", "message", "logTail": [string]}|null, "summary": {"pass", "fail", "limited", "uncertain", "info"}|null}`. Run id `YYYYMMDD-HHMMSS-xxxx` (4 hex).

Steps shown to the user, in order: `read` "Reading the reference", `build` "Building the character", `animate` "Animating", `render` "Rendering the GIF", `stills` "Taking the comparison stills", `check` "Checking against the reference".

## Review Focus

1. **A step that crashes or writes nothing** (Blender script error, missing GIF) → the run shows failed at that step with the last log lines; never done, never a pass. Test: Task 5, `fails a step that exits 0 without its outputs`.
2. **The Studio is stopped or restarted mid-run** → that run shows "interrupted", not running forever, and the Blender child is killed. Test: Task 5, `marks runs left running as interrupted`.
3. **Several images dropped at once, or edits while a run waits** → one queued run per exercise using the latest references, not one per file. Test: Task 5, `coalesces queued runs for the same exercise`.
4. **A video, PDF or iPhone HEIC is dropped** → video/PDF refused with a plain message, HEIC converted and accepted. Test: Task 4, `refuses videos with a plain message` / `converts heic with sips`.
5. **A step runs longer than its estimate** → the page says "taking longer than usual", never a negative or frozen "0 s left". Test: Task 5, `reports overdue instead of negative time`.

---

### Task 1: Studio Python environment and reading one reference image

**Files:**
- Create: `studio/setup.sh`, `studio/requirements.txt`, `studio/measure/metrics.py`, `studio/measure/read_pose.py`, `studio/tests/conftest.py`, `studio/tests/test_metrics.py`, `studio/tests/test_read_pose.py`, `studio/tests/fixtures/squat-measurements.json` (generated in Step 7)
- Copy (first step): `scratchpad/pose/behind-thumb.png` → `3d-character/reference-squat/behind-thumb.png` (the scratchpad is temporary; path `/private/tmp/claude-501/-Users-lywrbwm-Developer-faces/fb556b81-ba32-4179-9f78-dbf99c0dabb2/scratchpad/pose/behind-thumb.png`)

**Interfaces:**
- Produces `metrics.measure(joints: dict[str, np.ndarray], side: str, up: np.ndarray, view_normal: np.ndarray | None) -> dict[str, float]` — joints keyed `"shoulder.left"`, `"elbow.right"`, … for shoulder/elbow/wrist/hip/knee/ankle; `side` is `"left"`/`"right"`; with `view_normal` every point is first projected onto the plane perpendicular to it ("flat"), without it the 3D points are used. Returns `knee_included`, `hip_included`, `elbow_included`, `upperarm_vs_torso`, `torso_lean`, `hip_below_knee`, `grip_ratio`, `elbow_width_ratio` (definitions: `3d-character/reference-squat/targets.json`; `hip_below_knee` = (knee − hip)·up / |shoulder-mid − hip-mid|; ratios use the two sides' wrists/elbows over the two shoulders). Must run under Python 3.10 and Blender's 3.13 (numpy + math only).
- Produces `read_pose.read_pose(path: Path) -> dict` and `read_pose.NoBody(Exception)`; CLI `python studio/measure/read_pose.py IMG [IMG...]` prints one JSON object per line. Result: `{"image": "15.png", "size": [w, h], "view": "side"|"frontal"|"three-quarter", "shoulder_span": float, "camera_side": "left"|"right", "facing": "left"|"right"|null, "flat": {metric: value}, "3d": {metric: value}, "world": {joint: [x, y, z]}}` (`world` = MediaPipe world landmarks in metres, y down, for all 33 landmarks by MediaPipe name, kept for Task 9).

- [ ] **Step 1: Copy `behind-thumb.png` into `3d-character/reference-squat/` and write `studio/setup.sh`**

`setup.sh`: `uv venv --allow-existing --python 3.10 studio/.venv`, `uv pip install -r studio/requirements.txt` (`mediapipe==1.1.1`, `numpy`, `pillow`, `pytest`), download `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task` to `studio/models/` when missing, then refuse (exit 1, plain message) if its sha256 is not the one in Global Constraints. Run it: `bash studio/setup.sh` → ends with the model's hash line and exit 0.

- [ ] **Step 2: Write the failing metric tests** (`studio/tests/test_metrics.py`)

```python
def test_right_angle_knee_flat_and_3d():
    j = standing_joints()                      # helper: a straight figure, y up, facing -z
    j["ankle.left"] = j["knee.left"] + np.array([0, 0, -0.4])     # shin straight back
    assert measure(j, "left", UP, None)["knee_included"] == approx(90, abs=0.1)

def test_flat_projection_hides_out_of_plane_bend():
    j = standing_joints(); j["wrist.left"] = j["elbow.left"] + np.array([0, 0, -0.25])  # forearm toward camera
    assert measure(j, "left", UP, view_normal=np.array([0, 0, 1.0]))["elbow_included"] == approx(180, abs=0.5)
    assert measure(j, "left", UP, None)["elbow_included"] == approx(90, abs=0.5)

def test_torso_lean_and_hip_below_knee_signs():
    # torso tipped 20 degrees forward, hips 0.05 torso lengths below the knees
    assert m["torso_lean"] == approx(20, abs=0.2) and m["hip_below_knee"] == approx(0.05, abs=0.01)

def test_width_ratios():
    # wrists 0.6 apart, elbows 0.5 apart, shoulders 0.3 apart
    assert m["grip_ratio"] == approx(2.0) and m["elbow_width_ratio"] == approx(1.6667, abs=1e-3)
```

- [ ] **Step 3: Run** `studio/.venv/bin/python -m pytest studio/tests/test_metrics.py -q` → FAIL (`measure` not defined).

- [ ] **Step 4: Implement `measure` in `studio/measure/metrics.py`.** Run Step 3 again → PASS.

- [ ] **Step 5: Write the failing reference tests** (`studio/tests/test_read_pose.py`; `conftest.py` puts `studio/measure` on `sys.path` and defines `ROOT`, `REF = ROOT / "3d-character/reference-squat"`)

```python
@pytest.mark.parametrize("name,knee_f,knee_3,hip_f,hip_3,elbow_3,ua_3,lean", [
    ("15.png", 174.7, 160.4, 170.8, 167.1, 90.5, 41.3, 6.7),
    ("16.png", 104.7, 87.9, 109.2, 97.9, 99.9, 50.5, 18.4),
    ("17.png", 47.2, 64.5, 69.3, 77.3, 82.3, 52.0, 22.4)])
def test_side_views(name, knee_f, knee_3, hip_f, hip_3, elbow_3, ua_3, lean):
    r = read_pose(REF / name)
    assert (r["view"], r["camera_side"], r["facing"]) == ("side", "right", "right")
    assert r["flat"]["knee_included"] == approx(knee_f, abs=1.5)
    assert r["3d"]["knee_included"] == approx(knee_3, abs=1.5)
    assert r["flat"]["hip_included"] == approx(hip_f, abs=1.5)
    assert r["3d"]["hip_included"] == approx(hip_3, abs=1.5)
    assert r["3d"]["elbow_included"] == approx(elbow_3, abs=1.5)
    assert r["3d"]["upperarm_vs_torso"] == approx(ua_3, abs=1.5)
    assert r["flat"]["torso_lean"] == approx(lean, abs=1.0)

def test_bottom_is_parallel():
    assert read_pose(REF / "17.png")["flat"]["hip_below_knee"] == approx(-0.05, abs=0.05)

def test_behind_view_widths():
    r = read_pose(REF / "behind-thumb.png")
    assert r["view"] == "frontal"
    assert r["flat"]["grip_ratio"] == approx(1.87, abs=0.05)
    assert r["flat"]["elbow_width_ratio"] == approx(1.97, abs=0.05)
    assert r["3d"]["grip_ratio"] == approx(1.64, abs=0.08)

def test_no_body(tmp_path):
    Image.new("RGB", (400, 400), "white").save(tmp_path / "blank.png")
    with pytest.raises(NoBody):
        read_pose(tmp_path / "blank.png")
```

- [ ] **Step 6: Implement `read_pose` in `studio/measure/read_pose.py`.** MediaPipe Tasks API as in `scratchpad/pose/measure_ref.py` (load the model from `studio/models/` by absolute path — never rely on the working directory). Flat points = image pixels (y down, so `up = (0, -1, 0)`, `view_normal = (0, 0, 1)`); 3D = world landmarks (y down). `camera_side` = the side whose knee + elbow visibility is higher. `shoulder_span` = flat shoulder-to-shoulder / flat torso length: `< 0.35` side, `> 0.55` frontal, else three-quarter. `facing` (side views only) = the side of the shoulders the nose is on in the image. Run `studio/.venv/bin/python -m pytest studio/tests -q` → all PASS.

- [ ] **Step 7: Generate the fixture** `studio/tests/fixtures/squat-measurements.json` = `{"14.png": …, "15.png": …, "16.png": …, "17.png": …, "behind-thumb.png": …}` from the CLI (Tasks 2–3 test against it without MediaPipe). Note in `.sdd/progress.md`.

---

### Task 2: Targets from a run's references

**Files:**
- Create: `studio/measure/build_targets.py`, `studio/measure/run_read.py`, `studio/tests/test_build_targets.py`

**Interfaces:**
- Consumes: `read_pose.read_pose`, `NoBody` (Task 1); the run snapshot `references.json` (Data layout).
- Produces `build_targets.build_targets(exercise: str, images: list[ImageMeta], measured: dict[str, dict | Exception]) -> dict` (keyed by image id) returning:
  `{"exercise", "phases": {"start"|"middle"|"end": {metric: {"value", "method": "flat"|"3d", "cross": float|None, "uncertain": bool, "image": id}}, …, "middle"?: {"estimated": true, …}}, "widths": {"grip_ratio"|"elbow_width_ratio": {same shape}}, "tolerance": {"angles_deg": 5, "ratios": 0.12, "hip_below_knee": 0.10}, "images": {id: {"used": bool, "why": str|None, "view": str}}, "camera": {"side": "left"|"right", "facing": "left"|"right"}}`
- Produces `build_targets.TooFewPositions(Exception)` and `build_targets.to_pipeline_targets(targets: dict) -> dict | None` — task 12's format (`3d-character/reference-squat/targets.json`): phases `standing`/`mid`/`bottom` from start/middle/end with integer-rounded values for the five FORM_KEYS (`knee_included`, `hip_included`, `torso_lean`, `elbow_included`, `upperarm_vs_torso`), `arms_all_phases.grip_ratio`/`elbow_width_ratio` rounded to 2 decimals, `tolerance {"angles_deg": 5, "ratios": 0.12}`; `None` if any of those is missing.
- Produces CLI `studio/.venv/bin/python studio/measure/run_read.py RUN_DIR`: prints `image i/n` per image, writes `measurements.json` (`{id: result | {"error": "No body found in this image"}}`), `targets.json`, and `pipeline-targets.json` when `exercise in SOLVER_EXERCISES = ("Barbell Back Squat",)` and `to_pipeline_targets` is not `None`. Exit 0 on success; exit 3 with the plain reason as the last stderr line on `TooFewPositions`.

Rules: an image feeds targets only if it has a body, is not context-only, and its effective view (user's, else detector's) is side (angles) or front/behind/frontal (widths). Phase = the image's phase. First image in order wins when two side images share a phase; the other gets `why: "Another image already gives the <phase> position"`. Widths use the first front/behind image. Fewer than two side positions → `TooFewPositions("Need at least two key positions (start and end) from side-view images with a body found; have N.")`. Only start and end → middle is their mean, `"estimated": true`.

- [ ] **Step 1: Write the failing tests** (load `fixtures/squat-measurements.json`; images 14 context-only, 15 start, 16 middle, 17 end, behind-thumb `view: "behind"` set by user)

```python
def test_squat_targets():
    t = build_targets("Barbell Back Squat", IMAGES, MEASURED)
    knee = t["phases"]["start"]["knee_included"]
    assert (knee["method"], knee["uncertain"]) == ("flat", True)
    assert knee["value"] == approx(174.7, abs=1.5) and knee["cross"] == approx(160.4, abs=1.5)
    assert t["phases"]["start"]["hip_included"]["uncertain"] is False
    assert t["phases"]["middle"]["hip_included"]["uncertain"] is True
    assert t["phases"]["end"]["hip_included"]["uncertain"] is False
    elbow = t["phases"]["end"]["elbow_included"]
    assert (elbow["method"], elbow["cross"], elbow["uncertain"]) == ("3d", None, False)
    assert t["phases"]["end"]["hip_below_knee"]["value"] == approx(-0.05, abs=0.05)
    grip = t["widths"]["grip_ratio"]
    assert grip["value"] == approx(1.87, abs=0.05) and grip["uncertain"] is False
    assert t["images"]["img-14"] == {"used": False, "why": "Context only", "view": "side"}

def test_two_positions_estimate_the_middle(): ...   # drop 16 → middle["estimated"] is True, knee value = mean of 15 and 17
def test_one_position_is_too_few(): ...             # only 15 → raises TooFewPositions
def test_a_bad_image_fails_only_itself(): ...       # 16 measured as NoBody → images["img-16"]["why"] == "No body found in this image", middle estimated, no raise
def test_pipeline_format(): ...                     # to_pipeline_targets(t)["phases"]["bottom"]["knee_included"] == 47; ["arms_all_phases"]["grip_ratio"] == 1.87; missing widths → None
def test_run_read_cli(tmp_path): ...                # snapshot with the real 15/17 paths → exit 0, three files; with only 15 → exit 3 and "Need at least two key positions" on stderr
```

- [ ] **Step 2: Run** `studio/.venv/bin/python -m pytest studio/tests/test_build_targets.py -q` → FAIL.
- [ ] **Step 3: Implement `build_targets`, `to_pipeline_targets`, `run_read.py`.**
- [ ] **Step 4: Run the whole Python suite** → PASS. Record in `.sdd/progress.md`.

---

### Task 3: Checks — pass, fail, limited by proportions, uncertain

**Files:**
- Create: `studio/measure/feasibility.py`, `studio/measure/checks.py`, `studio/measure/run_check.py`, `studio/tests/test_feasibility.py`, `studio/tests/test_checks.py`, `studio/tests/fixtures/ours-squat.json`, `studio/tests/fixtures/retarget-squat.json` (from `scratchpad/t12/form5/retarget.json`, keys `contact`, `deformation`, `form`, `grip` only)

**Interfaces:**
- Consumes: `targets.json` (Task 2); `ours.json` (produced by Task 6, format below); `frames/retarget.json` (written by `animate.py --mpfb`: `contact.penetration_mm`, `contact.sides.{left,right}.sole_slide_mm`, `deformation.crack_max_mm`, `deformation.deepest.max_growth_pct`, and `form.contact` or `grip.contact` = `{"deepest_inside_bar_mm", "frames_touching_within_3mm", "frames"}`).
- `ours.json`: `{"first": 1, "last": 69, "camera_side": "right", "per_frame": [{"frame": 1, "flat": {metric: v}, "3d": {metric: v}}], "phase_frames": {"start": 68, "middle": 50, "end": 38}, "arm": {"shoulder_half_width": m, "upper_arm": m, "forearm": m}, "stills": [{"image": id, "phase", "view", "frame", "file": "stills/<id>.png", "facing": "left"|"right"|null}]}`.
- Produces `feasibility.closest_arm_pose(arm: dict, want: dict) -> dict` — `want` has `upperarm_vs_torso`, `elbow_included`, `grip_ratio`, `elbow_width_ratio`; returns the four values at the arm pose that best meets `want` with the bar-on-back constraints of `scratchpad/t12/armfeas.py` (forearm vertical seen from behind, wrist at shoulder height, torso upright; cost = squared misses in tolerance units). Port that search; the squat report's finding is that the cartoon tops out near 1.7× grip.
- Produces `checks.build_checks(targets: dict, ours: dict, report: dict, exercise: str) -> dict` → `{"summary": {"pass", "fail", "limited", "uncertain", "info"}, "rows": [{"group": "start"|"middle"|"end"|"all positions"|"body", "metric", "target", "ours", "unit": "deg"|"x"|"torso"|"mm"|"%", "status": "pass"|"fail"|"limited"|"uncertain"|"info", "frame": int|None, "note": str|None}]}`; CLI `run_check.py RUN_DIR` writes `checks.json`.

Status rules, in order: target uncertain → `uncertain` (note: "Two readings of the coach disagree: 175° flat, 160° in 3D"); phase estimated → `info`; within tolerance → `pass`; exercise in `ARM_FEASIBILITY = ("Barbell Back Squat",)`, metric one of the four arm metrics, closest achievable misses the target by more than the tolerance and ours is within tolerance of the achievable value → `limited` (note: "This character's arms reach at most 1.71×"); else `fail`. Width checks are judged at every key position (`all positions`, worst one reported). Body checks (`group: "body"`): cracks `crack_max_mm ≤ 1.0`; each foot's slide `≤ 5 mm`; feet in the floor `penetration_mm ≤ 5`; hands on the bar (when present) `deepest_inside_bar_mm ≤ 5` and touching in `≥ 80%` of frames; skin stretch `max_growth_pct` as `info` (spec call 6).

- [ ] **Step 1: Write the failing tests**

```python
CARTOON = {"shoulder_half_width": 0.117, "upper_arm": 0.105, "forearm": 0.124}
HUMAN = {"shoulder_half_width": 0.19, "upper_arm": 0.30, "forearm": 0.27}
WANT = {"upperarm_vs_torso": 52, "elbow_included": 82, "grip_ratio": 1.87, "elbow_width_ratio": 1.98}

def test_cartoon_cannot_reach_the_coaches_grip():
    assert closest_arm_pose(CARTOON, WANT)["grip_ratio"] < 1.87 - 0.12

def test_human_arm_can():
    assert closest_arm_pose(HUMAN, WANT)["grip_ratio"] == approx(1.87, abs=0.12)
```

If the cartoon test fails after a faithful port, that is a finding to report, not a number to tune.

```python
def test_statuses():
    c = build_checks(TARGETS, OURS, REPORT, "Barbell Back Squat")
    row = lambda g, m: next(r for r in c["rows"] if (r["group"], r["metric"]) == (g, m))
    assert row("start", "knee_included")["status"] == "uncertain"
    assert row("end", "hip_included")["status"] == "pass"            # ours 72.0 vs 69.3
    assert row("end", "torso_lean")["status"] == "fail"              # ours 30.0 vs 22.4
    assert row("all positions", "grip_ratio")["status"] == "limited" # ours 1.70, reachable ~1.7
    assert "×" in row("all positions", "grip_ratio")["note"]
    assert row("body", "feet_in_floor")["status"] == "fail"          # 28.91 mm in the fixture
    assert row("body", "hands_on_bar")["status"] == "pass"           # 1.92 mm, 61 of 69 frames
    assert row("body", "skin_stretch")["status"] == "info"
    assert sum(c["summary"].values()) == len(c["rows"])

def test_grip_far_below_reach_is_a_fail(): ...     # ours grip 1.40 → "fail"
def test_estimated_middle_is_info(): ...           # middle estimated → every middle row "info"
def test_run_check_cli(tmp_path): ...              # RUN_DIR with the three fixtures → checks.json written, exit 0; missing ours.json → exit 1, plain message
```

(`ours-squat.json` is hand-written to produce exactly the values in these comments.)

- [ ] **Step 2: Run** `studio/.venv/bin/python -m pytest studio/tests/test_feasibility.py studio/tests/test_checks.py -q` → FAIL.
- [ ] **Step 3: Implement `feasibility.py`, `checks.py`, `run_check.py`.**
- [ ] **Step 4: Run the whole Python suite** → PASS. Record in `.sdd/progress.md`.

---

### Task 4: Studio app shell, data store, uploads and API

**Files:**
- Create: `studio/package.json` (`"type": "module"`, scripts `dev` = `vite`, `build` = `vite build`, `test` = `vitest run`, `setup` = `bash setup.sh`; no dependencies — they resolve from the root `node_modules`), `studio/vite.config.js`, `studio/index.html`, `studio/src/main.jsx` (placeholder `App` until Task 7), `studio/server/paths.js`, `studio/server/store.js`, `studio/server/images.js`, `studio/server/api.js`, tests `studio/server/store.test.js`, `studio/server/images.test.js`, `studio/server/api.test.js`, `3d-character/studio/README.md`

**Interfaces:**
- `studio/vite.config.js`: `root` = `studio/`, `plugins: [react(), studioApi()]`, `server: {port: 5199, strictPort: true, host: 'localhost', watch: {ignored: ['**/.venv/**', '**/models/**']}}`, `test: {include: ['**/*.test.js'], environment: 'node'}`.
- `paths.js` exports `REPO`, `DATA` (`3d-character/studio`), `PYTHON` (`studio/.venv/bin/python`), `BLENDER`, `SKIN` (`3d-character/3d cartoon boy.glb`), `CLIPS_DIR` (`3d-character/mixamo`).
- `store.js` exports `createStore(dataRoot = DATA)` returning: `listExercises() -> [{slug, name, equipment, clip: string|null}]` (the 14 names in `reference/exercises/poses.json` plus `added-exercises.json`; clip from `CLIPS` = the table in `3d-character/mixamo/README.md`, e.g. `"Barbell Back Squat": "squat.fbx"`, only when the file exists), `addExercise({name, equipment, clip})`, `slugOf(name)` (lowercase words joined by hyphens), `listReferences(slug) -> ImageMeta[]` (not deleted, by order), `addReference(slug, bytes, originalName) -> ImageMeta`, `updateReference(slug, id, patch)` (phase, view, contextOnly; sets `phaseSetBy`/`viewSetBy` to `"user"`), `reorderReferences(slug, ids)`, `removeReference(slug, id)`, `applyDetectedViews(slug, measurements)` (only where `viewSetBy` is not `"user"`), `createRun(slug) -> run` (writes the snapshot), `readRun(slug, id)`, `writeRun(run)`, `listRuns(slug)` (newest first), `allRuns()`, `stepHistory(stepId) -> number[]` (seconds of that step in done runs, oldest first).
- Phase guess (re-applied after every add/reorder/remove/context change to images whose `phaseSetBy` is not `"user"`): non-context images in order → 1: start; 2: start, end; 3+: start, middle…, end.
- `images.js` exports `prepareUpload(bytes, name) -> Promise<{ext, bytes}>`, `UploadError` with `.message` shown to the user: video (`ftyp` at byte 4, or `.mov/.mp4/.m4v` name) → "Videos aren't supported yet. Upload still images of the key positions."; over 25 MB → "That picture is over 25 MB."; not jpg/png/webp → convert to JPEG with `sips` (HEIC), else "That file isn't a picture this Mac can open." Uses `imageKind` imported from `../../pipeline/frames-api.js`.
- `api.js` exports `studioApi({store, queue}) -> VitePlugin` with JSON routes: `GET /api/exercises`, `POST /api/exercises`, `GET /api/exercises/:slug` (`{exercise, references, runs, latest: {run, checks, ours, targets} | null}`), `POST /api/exercises/:slug/references` (raw body, header `x-filename`), `PATCH /api/exercises/:slug/references/:id`, `PUT /api/exercises/:slug/references/order`, `DELETE /api/exercises/:slug/references/:id`, `POST /api/exercises/:slug/runs`, `GET /api/runs/:slug/:id`, `GET /api/state`; `GET /files/<path under DATA>` streams the file with its content type, 404 for anything resolving outside `DATA`. Errors → `{error: message}` with 4xx/5xx. Task 5 supplies `queue`; until then `api.js` takes it as a parameter and the tests pass a stub.

- [ ] **Step 1: Write the failing tests** (each test uses a temp `dataRoot`)

```js
test('lists the 14 exercises with the squat clip', ...)        // 14 rows; squat.clip === 'squat.fbx'; deadlift.clip === null
test('guesses phases from order', ...)                          // 3 uploads → start, middle, end; mark #2 context → start, end
test('keeps phases the user set', ...)
test('refuses videos with a plain message', ...)                // bytes with 'ftyp' at offset 4 → UploadError, exact message above
test('converts heic with sips', ...)                            // a HEIC made with `sips -s format heic` from 15.png → ext 'jpg'
test('run snapshot leaves deleted images out and has absolute paths', ...)
test('serves files only inside the data folder', ...)           // GET /files/../../package.json → 404; an uploaded image → 200 image/png
test('applies detected views without overriding the user', ...)
```

- [ ] **Step 2: Run** `cd studio && npm test` → FAIL.
- [ ] **Step 3: Implement** `paths.js`, `store.js`, `images.js`, `api.js`, the shell files, and `3d-character/studio/README.md` (two lines: internal QA data; reference images belong to others and are never shipped).
- [ ] **Step 4: Run** `cd studio && npm test` → PASS; `cd studio && npm run dev` serves `http://localhost:5199/` and `curl -s localhost:5199/api/exercises | head -c 200` lists exercises. Confirm the live demo's port is untouched (`lsof -iTCP:5173 -sTCP:LISTEN` shows the same process as before). Stop the server. Record in `.sdd/progress.md`.

---

### Task 5: The run queue, steps, progress and waiting time

**Files:**
- Create: `studio/server/eta.js`, `studio/server/steps.js`, `studio/server/queue.js`, tests `studio/server/eta.test.js`, `studio/server/steps.test.js`, `studio/server/queue.test.js`, fixture `studio/server/fixtures/animate-log.txt` (copy of `scratchpad/t12/form5/log.txt`)
- Modify: `studio/server/api.js` (create the queue, recover on start, kill the child on server close)

**Interfaces:**
- `eta.js`: `DEFAULT_SECONDS = {read: 20, build: 150, animate: 90, render: 120, stills: 60, check: 5}`; `estimateStep(id, history: number[]) -> number` (median of the last five, else default); `timeLeft({steps, currentId, elapsed, fraction, estimates}) -> {seconds: number, overdue: boolean}` — current step left = `elapsed/fraction − elapsed` when `fraction > 0.05`, else `max(estimate − elapsed, 0)` with `overdue = elapsed > estimate`; plus the estimates of the steps after it.
- `steps.js`: `STEPS` (ids and labels from Data layout); `PROCESSES = [{id: 'read', steps: ['read']}, {id: 'animate', steps: ['build', 'animate', 'render']}, {id: 'stills', steps: ['stills']}, {id: 'check', steps: ['check']}]`; `commandFor(processId, run, runDir, exercise) -> {cmd, args, cwd} | {error: string}`; `requiredOutputs(processId, runDir, snapshot) -> string[]`; `readLine(processId, line, state) -> {step?: string, total?: number}`.
  - read: `PYTHON studio/measure/run_read.py RUN_DIR`; outputs `measurements.json`, `targets.json`.
  - animate: `BLENDER --background --python-exit-code 1 --python pipeline/blender/animate.py -- --mpfb --exercise NAME --wrap-skin SKIN --mixamo CLIPS_DIR/CLIP [--form-targets RUN_DIR/pipeline-targets.json when that file exists] --side-view --out RUN_DIR/frames --gif RUN_DIR/animation.gif --save RUN_DIR/scene.blend`, cwd `REPO`; outputs `animation.gif`, `frames/retarget.json`, `scene.blend`. No clip → `{error: "No motion for this exercise yet. Add a Mixamo clip named <file> to 3d-character/mixamo (its README says how)."}` (Task 9 replaces this). Log markers: `retarget: read <file>: N frames` → `total = 2N` frames (with the side view); `bake_to_skin:` → step `animate`; `retarget report ->` → step `render`. Render fraction = `frame_*.png` files in `frames/` and `frames/side/` over `total`, polled every 2 s.
  - stills: `BLENDER --background RUN_DIR/scene.blend --python-exit-code 1 --python studio/blender/inspect_run.py -- --run RUN_DIR`; outputs `ours.json` and every `stills[].file` it lists.
  - check: `PYTHON studio/measure/run_check.py RUN_DIR`; output `checks.json`.
- `queue.js`: `createQueue({store, spawnProcess = realSpawn, otherBlenderRunning = pgrepBlender, now = Date.now}) -> {enqueue(slug) -> run, state() -> {running: {slug, runId, step, label, elapsed, fraction, left, overdue} | null, queued: [{slug, runId}]}, recoverInterrupted(), stop()}`. One process at a time across all exercises. `enqueue` replaces a still-queued run of the same exercise (new snapshot, same id). Before a Blender process: while `otherBlenderRunning()` is true, the step note is "Waiting for another Blender job to finish" (checked every 5 s; time spent waiting is not counted in the step's seconds). After each process: required outputs must exist and be non-empty, else failed with "This step finished without producing <file>". On exit ≠ 0: failed with the last stderr line as the message and the last 20 log lines as `logTail`. After `read`: `store.applyDetectedViews`. After `check`: `run.summary` = `checks.json` summary. Each process's output goes to `RUN_DIR/logs/<process>.log`.

- [ ] **Step 1: Write the failing tests**

```js
test('median of the last five, default without history', ...)   // [10,20,30,40,50,1000] → 40; [] → DEFAULT_SECONDS.read
test('reports overdue instead of negative time', ...)            // estimate 60, elapsed 90, no fraction → {seconds: 0 + later steps, overdue: true}
test('uses the render fraction when there is one', ...)          // elapsed 60, fraction 0.5 → current step 60 s left
test('reads the animate log into steps', ...)                    // feed fixture lines: total 138; step 'animate' at the bake_to_skin line; 'render' at 'retarget report ->'
test('passes --form-targets only when the file exists', ...)
test('no clip gives a plain error', ...)
test('runs one process at a time', ...)                          // two exercises enqueued; fake spawn records max concurrency === 1
test('coalesces queued runs for the same exercise', ...)         // run A running, enqueue B twice → one queued B with the second snapshot
test('fails a step that exits 0 without its outputs', ...)       // fake read exits 0, writes nothing → run.status 'failed', step 'read' failed, message names targets.json
test('fails with the log tail on a non-zero exit', ...)
test('waits while another Blender runs', ...)                    // otherBlenderRunning true twice then false → note shown, then proceeds
test('marks runs left running as interrupted', ...)              // run.json status 'running' on disk → recoverInterrupted → 'interrupted' with message "The Studio was stopped while this ran."
test('copies the summary from checks.json', ...)
```

- [ ] **Step 2: Run** `cd studio && npm test` → the new tests FAIL.
- [ ] **Step 3: Implement** `eta.js`, `steps.js`, `queue.js`; wire into `api.js` (`recoverInterrupted()` on start, `stop()` kills the child on `server.httpServer` close, `POST …/runs` → `enqueue`, `GET /api/state` → `state()`; the exercise list carries each exercise's status: `running` if running or queued, `needs-reference` with no usable images, `passing` when the latest finished run is done with `fail === 0` and `pass > 0`, `failing` when it has fails or failed, `not-run` otherwise).
- [ ] **Step 4: Run** `cd studio && npm test` → PASS. Record in `.sdd/progress.md`.

---

### Task 6: Stills and our measurements from the saved scene

**Files:**
- Create: `studio/blender/inspect_run.py`, `studio/tests/blender/test_inspect.py`
- Uses (read-only): `pipeline/blender/mpfb_cage.py` (`preview_set`, `camera_on`, `distance_for`), `studio/measure/metrics.py`

**Interfaces:**
- Consumes: `RUN_DIR/scene.blend` (opened by Blender), `RUN_DIR/targets.json`, `RUN_DIR/references.json`.
- Produces `RUN_DIR/ours.json` (format in Task 3) and `RUN_DIR/stills/<image-id>.png`, 560×560.
- Functions: `find_rig_and_skin() -> (armature, mesh)` (the armature with a `mixamorig:Hips` bone; the mesh whose Armature modifier uses it); `joints_at(rig, frame) -> dict[str, np.ndarray]` (heads of `mixamorig:{Left,Right}{Arm,ForeArm,Hand,UpLeg,Leg,Foot}` → shoulder, elbow, wrist, hip, knee, ankle, world space; character's left is +X, facing −Y); `pick_phase_frames(per_frame, targets) -> dict[phase, int]` (put in `metrics.py` so it is testable outside Blender: the driving metric is the one with the largest |start − end| in tolerance units; start and end = frames nearest their targets, middle = the frame between them nearest the middle target).
- Camera azimuths (as `mpfb_cage.camera_on`): side 90, behind 0, front 180, three-quarter 35, frontal (front or behind not chosen) → behind; elevation 5 (10 for three-quarter); centred on the figure's box at that frame, distance from `distance_for(box height × 1.05)`. `flat` metrics use the side camera's view direction for angles and the behind camera's for widths. `facing` of a side still = the image side our nose is on, found by projecting the head bone through that camera.

- [ ] **Step 1: Write the failing tests.** `pick_phase_frames` in `studio/tests/test_metrics.py`:

```python
def test_pick_phase_frames_squat():
    # knee per frame: 175 at 1, falls to 50 at 30, back to 175 at 60; targets start 175, middle 105, end 47
    f = pick_phase_frames(PER_FRAME, TARGETS)
    assert f["start"] == 1 and f["end"] == 30 and 10 < f["middle"] < 30
```

and `studio/tests/blender/test_inspect.py` (run inside Blender): build an MPFB human with the mixamo rig (`HumanService.create_human()`, `HumanService.add_builtin_rig(body, "mixamo", import_weights=True)` from `bl_ext.user_default.mpfb.services.humanservice`), key frame 1 at rest and frame 20 with both knees bent so the side-projected knee angle is 90°, save to a temp `scene.blend`, write a matching `targets.json` (start knee 175 side, end knee 90 side) and `references.json` (one side image per phase), run `inspect_run.main(["--run", tmp])`, then assert `ours.json` `phase_frames["end"] == 20`, `per_frame[19]["flat"]["knee_included"] == approx(90, abs=2)`, two stills exist at 560×560 and are not one flat colour.

- [ ] **Step 2: Run** `studio/.venv/bin/python -m pytest studio/tests/test_metrics.py -q` and (no other Blender running) `/Applications/Blender.app/Contents/MacOS/Blender --background --python-exit-code 1 --python studio/tests/blender/test_inspect.py` → FAIL.
- [ ] **Step 3: Implement** `pick_phase_frames` and `inspect_run.py` (stills only for non-context key images; context and unused images get none).
- [ ] **Step 4: Run both** → PASS. Record in `.sdd/progress.md`.

---

### Task 7: The Workspace page

**Files:**
- Create: `studio/src/app.jsx` + `app.css` (`App`), `studio/src/components/exercise-list.jsx/.css` (`ExerciseList`), `exercise-page.jsx/.css` (`ExercisePage`), `upload-zone.jsx/.css` (`UploadZone`), `reference-strip.jsx/.css` (`ReferenceStrip`), `run-progress.jsx/.css` (`RunProgress`), `comparison.jsx/.css` (`Comparison`), `checks-table.jsx/.css` (`ChecksTable`), `run-history.jsx/.css` (`RunHistory`); `studio/src/lib/api.js`, `format.js`, `labels.js`, `use-poll.js`; tests `studio/src/lib/format.test.js`, `labels.test.js`
- Modify: `studio/src/main.jsx` (render `App`)

**Interfaces:**
- Consumes: the API routes (Task 4) and `GET /api/state` (Task 5); `checks.json`, `ours.json`, `targets.json` through `GET /api/exercises/:slug`.
- `format.js`: `formatDuration(seconds) -> string` ("45 s", "4 min 10 s", "1 h 2 min"); `formatLeft({seconds, overdue}) -> string` ("about 3 min left", "taking longer than usual"); `formatValue(value, unit) -> string` ("47°", "1.87×", "−0.05 torso", "28.9 mm", "13%").
- `labels.js`: `METRIC_LABELS` — `knee_included` "Knee angle", `hip_included` "Hip angle", `torso_lean` "Back lean from upright", `elbow_included` "Elbow angle", `upperarm_vs_torso` "Upper arm away from the body", `grip_ratio` "Grip width (× shoulder width)", `elbow_width_ratio` "Elbow width (× shoulder width)", `hip_below_knee` "Depth: hip below knee", `cracks` "Cracks in the skin", `feet_slide_left`/`feet_slide_right` "Left/Right foot sliding", `feet_in_floor` "Feet sinking into the floor", `hands_on_bar` "Hands on the bar", `skin_stretch` "Skin stretch at the deepest point"; `STATUS_LABELS` — pass "Pass", fail "Fail", limited "Limited by proportions", uncertain "Uncertain", info "Measured"; `PHASE_LABELS` start "Start", middle "Middle", end "End"; `label(metric)` falls back to the metric id.
- Layout A · Workspace (spec): left column `ExerciseList` (status dot + "7 of 18 pass" / "running · about 3 min left" / "needs a reference", and an "Add exercise" form: name + equipment select barbell/dumbbells/bench/cable/machine/bodyweight/other); right column `ExercisePage`: header (name, "N of M checks pass" counting pass + fail + limited, then limited/uncertain counts), `UploadZone` (drop or click; uploads files one by one, then calls `POST …/runs` itself — no run button in the normal flow; a "Run again" button only on failed/interrupted runs), `ReferenceStrip` (thumbnails in order, drag to reorder; per image: phase select, view select, "context only" toggle, delete, the image's `why` when unused; any change calls `POST …/runs`), `RunProgress` (each step: ✓/spinner/✗/waiting, time spent, the current step's left time and the whole run's; queue position when queued; the failure message and log tail when failed), `Comparison` (per key image: reference | our still, labelled by phase and view; our side still mirrored with CSS `scaleX(-1)` when its `facing` differs from the reference's; the GIF; context-only images in a "Context" row), `ChecksTable` (grouped Start/Middle/End/All positions/Body: check, coach, ours, status chip, note), `RunHistory` (runs newest first: time, "N of M pass", status; clicking one shows that run's comparison and checks).
- Polling: `usePoll(fn, ms)`; `GET /api/state` every 1 s while something runs, else every 5 s; refetch the open exercise when its run changes state.
- Styling: system font stack, neutral greys, status colours green/red/amber (limited)/blue (running)/grey (uncertain, info), CSS custom properties in `app.css`, light theme only (internal tool). Use the frontend-design skill's guidance for a clean, uncluttered look.

- [ ] **Step 1: Write the failing tests** for `format.js` and `labels.js` (each example string above is an assertion; `formatLeft({seconds: 0, overdue: true}) === 'taking longer than usual'`).
- [ ] **Step 2: Run** `cd studio && npm test` → FAIL.
- [ ] **Step 3: Implement** the libs and components.
- [ ] **Step 4: Run** `cd studio && npm test` → PASS and `cd studio && npm run build` → no errors.
- [ ] **Step 5: Smoke check** with fake data: copy `studio/tests/fixtures/*` into a hand-made `3d-character/studio/runs/barbell-back-squat/20261022-000000-test/` (run.json status done, checks.json from Task 3's fixture output, any PNG as stills/GIF), start `npm run dev`, open `http://localhost:5199/`, screenshot the squat page, confirm the list, comparison, checks and history render. Delete the fake run afterwards. Record in `.sdd/progress.md`.

---

### Task 8: The squat, end to end through the Studio (first milestone)

**Gate:** `.sdd/task-12-report.md` says DONE, and no other Blender is running.

**Files:**
- Create: `studio/scripts/squat-milestone.mjs`

**Interfaces:**
- Consumes: the running Studio (`http://localhost:5199`), images `3d-character/reference-squat/{14,15,16,17,behind-thumb}.png`.
- Script: removes the squat's existing references, uploads 14, 15, 16, 17, behind-thumb in that order, sets 14 context-only, 15 start/side, 16 middle/side, 17 end/side, behind-thumb start/behind, triggers one run, polls `GET /api/state` printing step, time spent and time left every 10 s, and when the run ends prints the status, the summary and every check row; exits 0 when the run finished (pass or fail), 1 when it failed or was interrupted.

- [ ] **Step 1: Start the Studio** (`cd studio && npm run dev`) **and run** `node studio/scripts/squat-milestone.mjs`. Expected: the run finishes with status `done`; `animation.gif`, `ours.json`, `checks.json` and four stills exist.
- [ ] **Step 2: Check the waiting time is honest:** each step's recorded seconds is within the real wall-clock time between its log markers (compare `run.json` with `logs/*.log` timestamps).
- [ ] **Step 3: Screenshot the squat page** (comparison + checks) into `3d-character/output/studio/squat-milestone.png` and save the run's GIF beside it.
- [ ] **Step 4: Report** — every check row with its status and note, what is limited by proportions and uncertain, and which fails remain. Do **not** edit any pipeline file to turn a fail into a pass (spec anti-goal); a fail is the report. Record in `.sdd/progress.md`.

---

### Task 9: Motion from the images when an exercise has no Mixamo clip

**Gate:** Task 8 done (task 12 has released `animate.py`).

**Files:**
- Create: `pipeline/blender/keypose_clip.py`, `studio/tests/blender/test_keypose.py`
- Modify: `pipeline/blender/animate.py` (back up first: `animate.py.bak-studio`) — add `--keyposes JSON` as the alternative to `--mixamo` in the `--mpfb` path; `studio/measure/run_read.py` — also write `keyposes.json`; `studio/server/steps.js` — no clip → `--keyposes RUN_DIR/keyposes.json` instead of the error.

**Interfaces:**
- `keyposes.json`: `{"phases": [{"phase": "start"|"middle"|"end", "world": {landmark: [x, y, z]}}]}` — MediaPipe world landmarks of each key side image (Task 1's `world`), in start, middle, end order.
- `keypose_clip.clip_from_keyposes(rest: retarget.Clip, keyposes: dict, fps: int = 30) -> retarget.Clip` — `rest` is the canonical Mixamo rest read from `3d-character/mixamo/squat.fbx` (every Mixamo FBX shares the Y Bot rest). Each key pose: landmarks turned into the character's frame (person's left → +X, up → +Z, facing −Y, using the hips' left-right line made level and MediaPipe's y-down as down), each bone's world rotation = the shortest turn taking its rest direction onto the landmark direction applied to its rest rotation (pelvis and chest take a full frame from hips/shoulders; spine bones share the hip-mid→shoulder-mid bend equally; head follows shoulders-mid→nose at the same 40% share as `NECK_SHARE`; hands keep the forearm's turn; feet take heel→toe). Timing (real-lift logic): start → middle → end over 1.5 s, hold 0.2 s, end → start over 1.0 s, eased in and out per segment (smoothstep), rotations slerped. The clip then goes through `retarget.BoneRetarget` unchanged, so grounding and foot lock are the existing ones.

- [ ] **Step 1: Write the failing Blender test:** keyposes built from a known synthetic standing pose and a known squat pose (knee 90°, hip 90°, lean 20°, written as world landmarks) → after `animate.py --mpfb --keyposes` on the cartoon skin (EEVEE, `--out` to a temp dir), `ours`-style metrics at the deepest frame are within 5° of 90/90/20, the descent has more frames than the ascent, both soles slide ≤ 5 mm and penetration ≤ 5 mm (from `retarget.json`).
- [ ] **Step 2: Run** `/Applications/Blender.app/Contents/MacOS/Blender --background --python-exit-code 1 --python studio/tests/blender/test_keypose.py` → FAIL.
- [ ] **Step 3: Implement** `keypose_clip.py`, the `animate.py` flag, `keyposes.json` in `run_read.py`, the steps change (and its Task 5 test: no clip → `--keyposes`).
- [ ] **Step 4: Run** the Blender test, the Python suite and `cd studio && npm test` → PASS.
- [ ] **Step 5: Run the squat once more with the clip hidden** (add the exercise "Back Squat from images" through the API, same five images and tags, no clip) and report its checks next to Task 8's. Record in `.sdd/progress.md`.

---

### Task 10: The "animate an exercise" project skill

**Files:**
- Create: `.claude/skills/animate-exercise/SKILL.md`

- [ ] **Step 1: Use superpowers:writing-skills.** Baseline first: a fresh subagent asked "the user dropped 3 images of a deadlift into the Studio and it says 2 checks fail — what now?" without the skill; note where it goes wrong (e.g. hand-editing the pipeline to make checks pass, treating "uncertain" as a fail, running Cycles, starting Blender while another runs).
- [ ] **Step 2: Write the skill:** when to use it; how to start the Studio (`cd studio && npm run dev`, `http://localhost:5199`) and its one-time setup (`npm run setup`); how to choose and tag images (side views for angles, a straight front or behind view for widths, start/middle/end, context only); what each status means and what to do (fail → report it with numbers, never hand-fix to pass; limited → the character's proportions, offer the user longer arms; uncertain → compare the stills by eye; needs a clip → `3d-character/mixamo/README.md`); the constraints that bite (one Blender at a time, EEVEE previews, reference images never shipped, live demo untouched).
- [ ] **Step 3: Re-run the baseline scenario with the skill** and confirm the failures from Step 1 no longer happen; close any loophole found. Record in `.sdd/progress.md`.

---

## Not in this plan (said out loud so nothing is silently dropped)

- Video input, the iOS app, cloth physics, the 876-exercise import, hosting, Cycles (spec non-goals).
- Equipment other than the back squat's barbell on the MPFB path (`animate.py --mpfb` mounts only that bar today); exercises needing dumbbells/bench/cable on this path get their props in a later plan.
- A front-view measuring rule for exercises whose key positions are only visible from the front (e.g. lateral raise): until then the read step fails them with its plain "need two side-view positions" message rather than guessing.
- Skin stretch at full depth is measured and shown, not fixed (spec call 6).
