"""Check a run: RUN_DIR/targets.json + ours.json + frames/retarget.json -> checks.json."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from checks import build_checks  # noqa: E402


def main(run_dir):
    run_dir = Path(run_dir)
    need = [run_dir / "targets.json", run_dir / "ours.json", run_dir / "frames" / "retarget.json"]
    missing = [str(p.relative_to(run_dir)) for p in need if not p.exists()]
    if missing:
        print("Cannot check yet, missing: " + ", ".join(missing), file=sys.stderr)
        return 1
    try:
        targets, ours, report = (json.loads(p.read_text()) for p in need)
        result = build_checks(targets, ours, report, targets["exercise"])
        text = json.dumps(result, indent=2, allow_nan=False)
    except Exception:
        print("Could not compare this run: its measurement files are incomplete or unreadable.", file=sys.stderr)
        return 1
    (run_dir / "checks.json").write_text(text)
    print(json.dumps(result["summary"]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
