"""Print the actual fitted-character/toolchain identity without building a character. Run in Blender."""
import json
import sys
from pathlib import Path
import bpy
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'pipeline' / 'blender'))
import fitted_cache
import mpfb_cage
import bl_ext.user_default.mpfb as mpfb
from bl_ext.user_default.mpfb.services.targetservice import TargetService

runtime = {'blender': list(bpy.app.version), 'build': bpy.app.build_hash.decode(),
           'numpy': np.__version__, 'mpfb': list(mpfb.VERSION)}
names = {name + '-' + end for _, files in mpfb_cage.TARGETS for name in files for end in ('incr', 'decr')}
runtime['resolved_targets'] = {name: fitted_cache.file_digest(TargetService.target_full_path(name)) for name in sorted(names)}
key = fitted_cache.make_key(ROOT / '3d-character' / '3d cartoon boy.glb', fitted_cache.SOURCES,
                            Path(mpfb.__file__).resolve().parent, runtime, {'ages': list(mpfb_cage.AGES)})
print('FIT_RUNTIME_ID ' + key)
