#!/usr/bin/env bash
# Sets up the Studio's own Python 3.10 environment and downloads the pose model.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL_URL="https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
MODEL="studio/models/pose_landmarker_heavy.task"
MODEL_SHA256="64437af838a65d18e5ba7a0d39b465540069bc8aae8308de3e318aad31fcbc7b"

uv venv --allow-existing --python 3.10 studio/.venv
uv pip install --python studio/.venv/bin/python -r studio/requirements.txt

mkdir -p studio/models
if [ ! -f "$MODEL" ]; then
  curl -fL --retry 3 -o "$MODEL" "$MODEL_URL"
fi

ACTUAL="$(shasum -a 256 "$MODEL" | cut -d' ' -f1)"
if [ "$ACTUAL" != "$MODEL_SHA256" ]; then
  echo "Model file $MODEL has the wrong sha256 ($ACTUAL, expected $MODEL_SHA256). Delete it and run setup again." >&2
  exit 1
fi
echo "model sha256 ok: $ACTUAL  $MODEL"
