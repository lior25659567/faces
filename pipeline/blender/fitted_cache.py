"""Content-addressed rest-character snapshots. Importable without Blender for key tests.

Only skin, armature and cage datablocks are retained; animation, reference poses,
render settings and the current scene/file path are not cache inputs or outputs.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
from urllib.parse import unquote, urlsplit
import tempfile
import time

SCHEMA = 1
HERE = Path(__file__).resolve().parent
SOURCES = tuple(HERE / name for name in (
    'mpfb_cage.py', 'skin_wrap.py', 'hand_rig.py', 'fit_model.py',
    'make_character.py', 'character.py', 'equipment.py', 'fitted_cache.py',
)) + (HERE.parent / 'pose_model.py',)
_digests = {}


def file_digest(path):
    path = Path(path).resolve()
    stat = path.stat()
    revision = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    cached = _digests.get(path)
    if cached and cached[0] == revision:
        return cached[1]
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    after = path.stat()
    if revision != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise OSError(f'{path.name} changed while fingerprinting')
    value = digest.hexdigest()
    _digests[path] = (revision, value)
    return value


def cache_directory(value=None):
    value = os.environ.get('STUDIO_FIT_CACHE', '') if value is None else value
    if value.lower() in ('0', 'false', 'off'):
        return None
    return Path(value).expanduser().resolve() if value and value != '1' else HERE.parent / 'vendor' / 'fitted-cache'



def model_fingerprint(path):
    """Fingerprint external glTF buffers/textures as well as the selected model bytes."""
    path = Path(path)
    with path.open('rb') as stream:
        header = stream.read(12)
        if header[:4] == b'glTF':
            size, kind = struct.unpack('<II', stream.read(8))
            if kind != 0x4E4F534A or size > 16 * 1024 * 1024:
                raise ValueError('Model JSON chunk is unsupported for caching')
            document = json.loads(stream.read(size))
        elif path.suffix.lower() == '.gltf':
            document = json.loads(path.read_text())
        else:
            # Unknown formats may have external assets that cannot be safely fingerprinted.
            raise ValueError('Only glTF/GLB models can use the fitted cache')
    external = {}
    for row in document.get('buffers', []) + document.get('images', []):
        uri = row.get('uri')
        if not uri or uri.startswith('data:'):
            continue
        parsed = urlsplit(uri)
        if parsed.scheme or parsed.netloc or parsed.query or parsed.fragment:
            raise ValueError('Remote/external URI type cannot be cached safely')
        external[uri] = file_digest(path.parent / unquote(parsed.path))
    return {'path': str(path.resolve()), 'model': file_digest(path), 'external': external}


def make_key(skin_path, sources, asset_root, runtime, options):
    root = Path(asset_root).resolve()
    assets = {str(path.relative_to(root)): file_digest(path)
              for path in sorted(root.rglob('*'))
              if path.is_file() and '__pycache__' not in path.parts
              and '.git' not in path.parts and path.suffix not in ('.pyc', '.pyo')}
    if not assets:
        raise OSError('MPFB source/assets are missing')
    inputs = {'schema': SCHEMA, 'skin': model_fingerprint(skin_path),
              'sources': {str(Path(p).resolve()): file_digest(p) for p in sources},
              'mpfb': assets, 'runtime': runtime, 'options': options}
    return hashlib.sha256(json.dumps(inputs, sort_keys=True, allow_nan=False).encode()).hexdigest()


def read_manifest(folder, key):
    """Reject incomplete, altered or obsolete snapshots before loading any Blender data."""
    folder = Path(folder)
    entry = json.loads((folder / 'manifest.json').read_text())
    if entry.get('schema') != SCHEMA or entry.get('key') != key:
        raise ValueError('Fitted cache metadata is obsolete')
    if (not isinstance(entry.get('objects'), list) or len(entry['objects']) != 3
            or not all(isinstance(name, str) and name for name in entry['objects'])
            or len(set(entry['objects'])) != 3):
        raise ValueError('Fitted cache object list is incomplete')
    if not isinstance(entry.get('report'), dict) or not entry['report'].get('bake'):
        raise ValueError('Fitted cache has no completed bake report')
    if file_digest(folder / 'character.blend') != entry.get('sha256'):
        raise ValueError('Fitted cache snapshot checksum does not match')
    return entry


def _validate(objects):
    from mathutils import Matrix
    skin, rig, cage = objects
    if skin.type != 'MESH' or rig.type != 'ARMATURE' or cage.type != 'MESH':
        raise ValueError('Fitted cache has wrong object types')
    if not skin.data.vertices or not rig.data.bones or rig.get('rig_profile') != 'mpfb-mixamo':
        raise ValueError('Fitted cache has no complete mesh/rig')
    if skin.parent is not rig or not any(m.type == 'ARMATURE' and m.object is rig for m in skin.modifiers):
        raise ValueError('Cached skin is not bound to its rig')
    if not skin.get('skin_bake') or not rig.get('skin_fit'):
        raise ValueError('Cached fitting metadata is missing')
    if any(o.animation_data and (o.animation_data.action or o.animation_data.nla_tracks) for o in objects):
        raise ValueError('Animated characters cannot be fitted-cache entries')
    if any(max(abs(v) for row in pb.matrix_basis - Matrix.Identity(4) for v in row) > 1e-6 for pb in rig.pose.bones):
        raise ValueError('Posed characters cannot be fitted-cache entries')
    fit = json.loads(rig['skin_fit'])
    if any((fit.get('hands') or {}).get(side, {}).get('model') for side in ('left', 'right')):
        grips = json.loads(rig.get('hand_grip', '{}'))
        if not all(side in grips for side in ('left', 'right')):
            raise ValueError('Fitted hand grips are incomplete')


def _load(folder, entry):
    import bpy
    loaded = []
    try:
        with bpy.data.libraries.load(str(folder / 'character.blend'), link=False) as (source, target):
            if not all(name in source.objects for name in entry['objects']):
                raise ValueError('Fitted snapshot objects are missing')
            # Blender replaces this list's names with Object instances in place.
            target.objects = list(entry['objects'])
        loaded = list(target.objects)
        _validate(loaded)
        # Cold fitting also clears the imported clip's objects. The Clip itself is plain data.
        for obj in list(bpy.data.objects):
            if obj not in loaded:
                bpy.data.objects.remove(obj, do_unlink=True)
        for obj, name in zip(loaded, entry['objects']):
            obj.name = name
            bpy.context.scene.collection.objects.link(obj)
        bpy.context.view_layer.update()
        return (*loaded, entry['report'])
    except Exception:
        for obj in loaded:
            if obj and obj.name in bpy.data.objects:
                bpy.data.objects.remove(obj, do_unlink=True)
        raise


def _store(folder, key, objects, report):
    import bpy
    _validate(objects)
    folder.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.fitted-', dir=folder.parent))
    try:
        snapshot = temporary / 'character.blend'
        bpy.data.libraries.write(str(snapshot), set(objects), path_remap='ABSOLUTE', fake_user=True, compress=True)
        entry = {'schema': SCHEMA, 'key': key, 'sha256': file_digest(snapshot),
                 'objects': [o.name for o in objects], 'report': report}
        (temporary / 'manifest.json').write_text(json.dumps(entry, default=float, allow_nan=False))
        # A competing completed writer is equivalent. Never replace a partial directory in place.
        if folder.exists():
            try:
                read_manifest(folder, key)
                return
            except (OSError, ValueError, TypeError, KeyError):
                # Preserve an invalid entry for diagnosis; never overwrite files in place.
                abandoned = folder.with_name('.invalid-' + folder.name + '-' + temporary.name)
                try:
                    folder.rename(abandoned)
                except FileNotFoundError:
                    pass
        try:
            temporary.rename(folder)
        except FileExistsError:
            read_manifest(folder, key)  # another complete writer won the race
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def cached_wear(skin_path, scratch, ages, builder, log=print):
    """Return a validated cold fit or restore its unposed local snapshot."""
    import bpy
    import numpy as np
    directory = cache_directory()
    if directory is None:
        return builder(skin_path, scratch, ages=ages, log=log)
    started = time.monotonic()
    try:
        import bl_ext.user_default.mpfb as mpfb
        asset_root = Path(mpfb.__file__).resolve().parent
        runtime = {'blender': list(bpy.app.version), 'build': bpy.app.build_hash.decode(),
                   'numpy': np.__version__, 'mpfb': list(mpfb.VERSION)}
        # TargetService may resolve a customized target outside the extension tree.
        from bl_ext.user_default.mpfb.services.targetservice import TargetService
        target_names = {name + '-' + end for _, files in builder.__globals__['TARGETS']
                        for name in files for end in ('incr', 'decr')}
        runtime['resolved_targets'] = {name: file_digest(TargetService.target_full_path(name))
                                       for name in sorted(target_names)}
        options = {'ages': list(ages)}
        key = make_key(skin_path, SOURCES, asset_root, runtime, options)
        folder = directory / key
        try:
            entry = read_manifest(folder, key)
            result = _load(folder, entry)
            result[3]['fit_cache'] = {'status': 'hit', 'key': key, 'seconds': round(time.monotonic() - started, 3)}
            log(f"bake_to_skin: reused fitted character cache ({time.monotonic() - started:.2f}s)")
            return result
        except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
            if folder.exists():
                log(f'fit cache: cannot reuse snapshot ({error}); fitting again')
    except (OSError, ValueError, TypeError, AttributeError, ImportError, struct.error) as error:
        log(f'fit cache: unavailable ({error}); fitting normally')
        return builder(skin_path, scratch, ages=ages, log=log)
    result = builder(skin_path, scratch, ages=ages, log=log)
    try:
        if make_key(skin_path, SOURCES, asset_root, runtime, options) != key:
            raise ValueError('Fitting inputs changed during the build')
        _store(folder, key, result[:3], result[3])
        result[3]['fit_cache'] = {'status': 'stored', 'key': key}
        log('fit cache: saved complete rest character for future builds')
    except (OSError, ValueError, TypeError, RuntimeError) as error:
        log(f'fit cache: could not save ({error}); current fitted character remains usable')
    return result
