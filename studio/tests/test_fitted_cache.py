"""Cache identity/integrity tests do not require Blender or external assets."""
from pathlib import Path
import json
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'pipeline/blender'))
import fitted_cache


@pytest.fixture
def inputs(tmp_path):
    skin = tmp_path / 'skin.gltf'; skin.write_text('{"asset": {"version": "2.0"}}')
    code = tmp_path / 'fitter.py'; code.write_text('code one')
    mpfb = tmp_path / 'mpfb'; mpfb.mkdir()
    (mpfb / 'target.json').write_text('asset one')
    return dict(skin_path=skin, sources=(code,), asset_root=mpfb,
                runtime={'blender': [5, 2, 2], 'numpy': 'test'}, options={'ages': [.3, .5]})


def test_key_is_content_based_and_repeatable(inputs):
    first = fitted_cache.make_key(**inputs)
    assert fitted_cache.make_key(**inputs) == first
    inputs['skin_path'].touch()
    assert fitted_cache.make_key(**inputs) == first


@pytest.mark.parametrize('change', ['skin', 'source', 'asset', 'added_asset', 'runtime', 'ages', 'schema'])
def test_changes_in_fitting_inputs_invalidate(inputs, monkeypatch, change):
    first = fitted_cache.make_key(**inputs)
    if change == 'skin': inputs['skin_path'].write_text('{"asset":{"version":"2.0"},"extras":"changed"}')
    elif change == 'source': inputs['sources'][0].write_text('code two')
    elif change == 'asset': (inputs['asset_root'] / 'target.json').write_text('asset two')
    elif change == 'added_asset': (inputs['asset_root'] / 'other.target').write_text('new target')
    elif change == 'runtime': inputs['runtime']['blender'] = [5, 2, 3]
    elif change == 'ages': inputs['options']['ages'] = [.5]
    else: monkeypatch.setattr(fitted_cache, 'SCHEMA', fitted_cache.SCHEMA + 1)
    assert fitted_cache.make_key(**inputs) != first


def test_runtime_cache_files_do_not_invalidate_fitting(inputs):
    first = fitted_cache.make_key(**inputs)
    generated = inputs['asset_root'] / '__pycache__'; generated.mkdir()
    (generated / 'module.pyc').write_bytes(b'cache')
    assert fitted_cache.make_key(**inputs) == first


def test_missing_assets_do_not_produce_reusable_identity(inputs):
    (inputs['asset_root'] / 'target.json').unlink()
    with pytest.raises(OSError, match='missing'):
        fitted_cache.make_key(**inputs)


def test_explicit_cache_disable_and_custom_directory(tmp_path, monkeypatch):
    for value in ('0', 'false', 'off', 'FALSE'):
        assert fitted_cache.cache_directory(value) is None
    assert fitted_cache.cache_directory(str(tmp_path)) == tmp_path
    monkeypatch.setenv('STUDIO_FIT_CACHE', '0')
    assert fitted_cache.cache_directory() is None


@pytest.fixture
def entry(tmp_path):
    (tmp_path / 'character.blend').write_bytes(b'complete snapshot')
    manifest = dict(schema=fitted_cache.SCHEMA, key='key',
                    sha256=fitted_cache.file_digest(tmp_path / 'character.blend'),
                    objects=['skin', 'rig', 'cage'], report={'bake': {'vertices': 123}})
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    return tmp_path, manifest


def test_complete_snapshot_metadata_is_accepted(entry):
    folder, manifest = entry
    assert fitted_cache.read_manifest(folder, 'key') == manifest


@pytest.mark.parametrize('damage', ['snapshot', 'key', 'schema', 'report', 'objects', 'missing_object', 'duplicate_object'])
def test_corrupt_or_incomplete_entries_are_rejected_before_blender_load(entry, damage):
    folder, manifest = entry
    if damage == 'snapshot': (folder / 'character.blend').write_bytes(b'truncated')
    elif damage == 'key': manifest['key'] = 'another'
    elif damage == 'schema': manifest['schema'] = -1
    elif damage == 'report': manifest['report'] = {}
    elif damage == 'objects': manifest['objects'] = [None, {}, []]
    elif damage == 'missing_object': manifest['objects'] = ['skin', 'rig']
    else: manifest['objects'] = ['skin', 'rig', 'rig']
    (folder / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises((ValueError, OSError)):
        fitted_cache.read_manifest(folder, 'key')


def test_external_model_texture_changes_invalidate(inputs):
    model = inputs['skin_path']
    texture = model.parent / 'skin.png'; texture.write_bytes(b'texture one')
    model.write_text(json.dumps({'images': [{'uri': 'skin.png'}]}))
    first = fitted_cache.make_key(**inputs)
    texture.write_bytes(b'texture two')
    assert fitted_cache.make_key(**inputs) != first


def test_remote_model_assets_disable_reuse(inputs):
    inputs['skin_path'].write_text(json.dumps({'images':[{'uri':'https://example.test/texture.png'}]}))
    with pytest.raises(ValueError, match='cached safely'):
        fitted_cache.make_key(**inputs)


def test_moved_model_gets_new_cache_to_preserve_relative_texture_provenance(inputs):
    first=fitted_cache.make_key(**inputs)
    moved=inputs['skin_path'].with_name('renamed.gltf')
    moved.write_bytes(inputs['skin_path'].read_bytes())
    inputs['skin_path']=moved
    assert fitted_cache.make_key(**inputs)!=first
