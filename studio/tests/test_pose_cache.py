import copy
import json
from pathlib import Path

import pytest

import pose_cache


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    image = tmp_path / "photo.png"
    image.write_bytes(b"image one")
    model = tmp_path / "model.task"
    model.write_bytes(b"model one")
    source = tmp_path / "reader.py"
    source.write_text("reader version one")
    monkeypatch.setattr(pose_cache, "MODEL", model)
    monkeypatch.setattr(pose_cache, "SOURCE_FILES", (source,))
    monkeypatch.setattr(pose_cache, "_versions", lambda: {"mediapipe": "test"})
    calls = []
    result = {
        "image": image.name, "size": [200, 300], "view": "side", "camera_side": "left", "facing": "right",
        "flat": {"knee_included": 90.0, "back_rounding": None}, "3d": {"knee_included": 92.0},
        "world": {"left_knee": [0.0, 0.1, 0.2]}, "pixels": {"left_knee": [100.0, 150.0]},
        "visibility": {"left_knee": 0.9},
    }

    def reader(path):
        calls.append(path)
        return {**copy.deepcopy(result), "image": path.name}

    return image, model, source, tmp_path / "cache", reader, calls, result


def test_unchanged_image_reuses_reading_and_identical_file_gets_own_name(fixture):
    image, _, _, cache, reader, calls, _ = fixture
    first = pose_cache.read_cached(image, reader, cache)
    second = pose_cache.read_cached(image, reader, cache)
    assert second == first
    renamed = image.with_name("another.png")
    renamed.write_bytes(image.read_bytes())
    assert pose_cache.read_cached(renamed, reader, cache)["image"] == renamed.name
    assert len(calls) == 1


@pytest.mark.parametrize("changed", ["image", "source", "model", "schema", "packages"])
def test_changed_inputs_invalidate(fixture, monkeypatch, changed):
    image, model, source, cache, reader, calls, _ = fixture
    pose_cache.read_cached(image, reader, cache)
    if changed in ("image", "source", "model"):
        {"image": image, "source": source, "model": model}[changed].write_bytes(b"changed content")
    elif changed == "schema":
        monkeypatch.setattr(pose_cache, "SCHEMA", pose_cache.SCHEMA + 1)
    else:
        monkeypatch.setattr(pose_cache, "_versions", lambda: {"mediapipe": "new"})
    pose_cache.read_cached(image, reader, cache)
    assert len(calls) == 2


@pytest.mark.parametrize("corrupt", ["{broken", "null", "[]", '{"schema": 1, "key": "wrong", "result": {}}'])
def test_corrupt_entries_recomputed(fixture, corrupt):
    image, _, _, cache, reader, calls, _ = fixture
    pose_cache.read_cached(image, reader, cache)
    entry = next(cache.glob("*.json"))
    entry.write_text(corrupt)
    pose_cache.read_cached(image, reader, cache)
    assert len(calls) == 2
    assert json.loads(entry.read_text())["result"]["flat"]["knee_included"] == 90


@pytest.mark.parametrize("invalid", [{"error": "No body"}, {"flat": {}}, {"shoulder_span": float("nan")}])
def test_errors_incomplete_and_nonfinite_results_not_cached(fixture, invalid):
    image, _, _, cache, reader, calls, result = fixture
    result.update(invalid)
    pose_cache.read_cached(image, reader, cache)
    pose_cache.read_cached(image, reader, cache)
    assert len(calls) == 2
    assert not cache.exists()


def test_detector_failure_retried_not_cached(fixture):
    image, _, _, cache, reader, calls, _ = fixture
    def fail(path):
        raise RuntimeError("No body found")
    with pytest.raises(RuntimeError, match="No body"):
        pose_cache.read_cached(image, fail, cache)
    assert pose_cache.read_cached(image, reader, cache)["view"] == "side"
    assert len(calls) == 1


def test_cache_write_failure_is_nonfatal_and_leaves_no_temporary_file(fixture, monkeypatch):
    image, _, _, cache, reader, calls, _ = fixture
    def denied(*args):
        raise PermissionError("read only")
    monkeypatch.setattr(pose_cache.os, "replace", denied)
    assert pose_cache.read_cached(image, reader, cache)["view"] == "side"
    assert len(calls) == 1
    assert list(cache.iterdir()) == []


def test_missing_model_falls_back_and_none_disables_cache(fixture):
    image, model, _, cache, reader, calls, _ = fixture
    model.unlink()
    pose_cache.read_cached(image, reader, cache)
    pose_cache.read_cached(image, reader, None)
    assert len(calls) == 2
    assert not cache.exists()


def test_model_bytes_hashed_only_once_until_stat_changes(fixture, monkeypatch):
    image, model, _, cache, reader, _, _ = fixture
    opens = []
    original = Path.open
    def track(path, *args, **kwargs):
        if path == model.resolve():
            opens.append(path)
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", track)
    pose_cache.read_cached(image, reader, cache)
    pose_cache.read_cached(image, reader, cache)
    assert len(opens) == 1


def test_image_change_during_detection_does_not_cache_under_old_fingerprint(fixture):
    image, _, _, cache, reader, _, _ = fixture
    def changing_reader(path):
        result = reader(path)
        path.write_bytes(b"replacement while reading")
        return result
    pose_cache.read_cached(image, changing_reader, cache)
    assert not cache.exists()
