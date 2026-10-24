"""Reuse successful reference readings while the image and pose reader are unchanged.

Tags (phase/view/context) deliberately are not part of this cache: they affect target
construction, which must run again, but not pose detection. Cache failures never prevent
reading a picture. NoBody and other failed detections are retried on the next run.
"""
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import tempfile


SCHEMA = 1
HERE = Path(__file__).resolve().parent
SOURCE_FILES = tuple(HERE / name for name in ("read_pose.py", "metrics.py", "posture.py", "pose_cache.py"))
MODEL = HERE.parent / "models" / "pose_landmarker_heavy.task"
PACKAGES = ("mediapipe", "numpy", "pillow")
_digests = {}


def _digest(path):
    """Hash each file once per stat revision, especially the large shared pose model."""
    path = Path(path).resolve()
    stat = path.stat()
    revision = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    cached = _digests.get(path)
    if cached and cached[0] == revision:
        return cached[1]
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    # A file edited during hashing must not be reused under its earlier stat revision.
    after = path.stat()
    if revision != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise OSError(f"{path.name} changed while being read")
    value = digest.hexdigest()
    _digests[path] = (revision, value)
    return value


def _versions():
    out = {}
    for package in PACKAGES:
        try:
            out[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            out[package] = None
    return out


def _key(path):
    content = {
        "schema": SCHEMA,
        "image": _digest(path),
        "reader": {p.name: _digest(p) for p in SOURCE_FILES},
        "model": _digest(MODEL),
        "packages": _versions(),
    }
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _finite(value):
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(isinstance(k, str) and _finite(v) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return all(_finite(v) for v in value)
    return value is None or isinstance(value, (str, int, bool))


def _valid(result):
    """Accept complete successful reader outputs, never error payloads or nonfinite data."""
    if not isinstance(result, dict) or "error" in result or not _finite(result):
        return False
    if result.get("view") not in ("side", "frontal", "three-quarter"):
        return False
    if result.get("camera_side") not in ("left", "right") or result.get("facing") not in (None, "left", "right"):
        return False
    size = result.get("size")
    if not isinstance(size, (list, tuple)) or len(size) != 2 or not all(_number(v) and v > 0 for v in size):
        return False
    for name in ("flat", "3d", "world", "pixels", "visibility"):
        if not isinstance(result.get(name), dict) or not result[name]:
            return False
    for name, dimensions in (("world", 3), ("pixels", 2)):
        if not all(isinstance(p, (list, tuple)) and len(p) == dimensions and all(_number(v) for v in p)
                   for p in result[name].values()):
            return False
    return all(_number(v) and 0 <= v <= 1 for v in result["visibility"].values())


def read_cached(path, reader, cache_dir):
    """Read through a content-addressed cache; return the same dict shape as ``reader``.

    The cache is optional. Missing model/source files, unreadable cache folders, invalid
    entries and failed writes all fall back to a fresh read. Reader exceptions propagate.
    """
    path = Path(path)
    if cache_dir is None:
        return reader(path)
    try:
        key = _key(path)
    except OSError:
        return reader(path)
    destination = Path(cache_dir) / f"{key}.json"
    try:
        entry = json.loads(destination.read_text())
        if entry.get("key") == key and entry.get("schema") == SCHEMA and _valid(entry.get("result")):
            return {**entry["result"], "image": path.name}
    except (OSError, ValueError, TypeError, AttributeError):
        pass

    result = reader(path)
    if not _valid(result):
        return result
    temporary = None
    try:
        # Upload replacement or a model/code edit during detection must not associate
        # this reading with the fingerprint captured before the edit.
        if _key(path) != key:
            return result
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf8", prefix=".pose-", suffix=".tmp",
                                         dir=destination.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump({"schema": SCHEMA, "key": key, "result": result}, stream, allow_nan=False)
        os.replace(temporary, destination)
    except (OSError, ValueError, TypeError):
        pass
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
    return result
