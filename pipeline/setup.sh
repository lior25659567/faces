#!/usr/bin/env bash
# One-time setup for the local frame generator.
# Installs LivePortrait (pinned), a Python 3.10 env and the model weights, all inside pipeline/,
# and builds the small Swift program that finds faces with Apple's Vision framework.
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
repo="$here/vendor/LivePortrait"
commit="9b294b3d0536135442ea73cb01e6cb3ca7029dd3"
venv="$here/.venv"

command -v uv >/dev/null || { echo "uv is required: brew install uv" >&2; exit 1; }
xcrun --find swiftc >/dev/null 2>&1 || { echo "Xcode's command line tools are required: xcode-select --install" >&2; exit 1; }

# generate-frames.py calls LivePortrait internals, so the commit is pinned.
if [ ! -d "$repo/.git" ]; then
  git init --quiet "$repo"
  git -C "$repo" remote add origin https://github.com/KwaiVGI/LivePortrait.git
  git -C "$repo" fetch --quiet --depth 1 origin "$commit"
  git -C "$repo" checkout --quiet FETCH_HEAD
fi

uv venv --allow-existing --python 3.10 "$venv"
# requests is what restyle.py calls Cloudflare with.
uv pip install --python "$venv/bin/python" --index-strategy unsafe-best-match \
  -r "$repo/requirements_macOS.txt" huggingface_hub requests

# LivePortrait's human models only (about 630 MB). Its animal models are skipped, and so is the
# InsightFace face detector it ships with, which is for non-commercial use: Vision finds the faces.
"$venv/bin/python" - "$repo/pretrained_weights" <<'PY'
import sys
from huggingface_hub import snapshot_download

snapshot_download(
    "KlingTeam/LivePortrait",
    local_dir=sys.argv[1],
    allow_patterns=["liveportrait/*"],
)
PY

xcrun swiftc -O -o "$here/vendor/apple-vision" "$here/apple-vision.swift"

echo "Ready. Next: npm run frames -- input/<character>.png"
