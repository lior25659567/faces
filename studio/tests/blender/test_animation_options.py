"""CLI regression: --fast must reach the MPFB path with actual smaller dimensions."""
import sys
from pathlib import Path
try:
    import bpy
except ImportError:
    import pytest
    pytest.skip('Runs inside Blender', allow_module_level=True)
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'pipeline/blender'))
import animate
for flags, expected in (([],(560,560)),(['--fast'],(360,360)),(['--fast','--width','320','--height','240'],(320,240))):
    seen=[]
    animate.run_mpfb=lambda args: seen.append((args.width,args.height))
    sys.argv=['blender','--','--mpfb','--exercise','Barbell Bent-Over Row',*flags]
    animate.main()
    assert seen==[expected], (flags,seen,expected)
for mode in ('auto', 'original', 'fitness'):
    seen=[]
    animate.run_mpfb=lambda args: seen.append(args.character_fit)
    sys.argv=['blender','--','--mpfb','--exercise','Barbell Bent-Over Row','--character-fit',mode]
    animate.main()
    assert seen==[mode], (mode,seen)
print('test_animation_options: PASS')
