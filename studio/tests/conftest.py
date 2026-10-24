import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REF = ROOT / "3d-character/reference-squat"
sys.path.insert(0, str(ROOT / "studio/measure"))
