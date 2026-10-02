#!/usr/bin/env bash
# Apply the project-owned cache-key fix to the pinned Musubi backend.
set -euo pipefail

QDE_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
QDE_BACKEND_DIR="${1:-$QDE_PROJECT_ROOT/third_party/musubi-tuner}"
QDE_PATCH_FILE="$QDE_PROJECT_ROOT/patches/musubi_unique_cache_keys.patch"
QDE_EXPECTED_HEAD="8ec957745626317e39f04f8547910e353553354b"

if [[ ! -d "$QDE_BACKEND_DIR/.git" ]]; then
  echo "Musubi backend is not a git checkout: $QDE_BACKEND_DIR" >&2
  exit 2
fi
if [[ ! -f "$QDE_PATCH_FILE" ]]; then
  echo "Required Musubi cache-key patch is missing: $QDE_PATCH_FILE" >&2
  exit 2
fi
if [[ "$(git -C "$QDE_BACKEND_DIR" rev-parse HEAD)" != "$QDE_EXPECTED_HEAD" ]]; then
  echo "Unexpected Musubi revision; expected $QDE_EXPECTED_HEAD" >&2
  exit 2
fi

if git -C "$QDE_BACKEND_DIR" apply --reverse --check "$QDE_PATCH_FILE" >/dev/null 2>&1; then
  echo "Musubi unique-cache-key patch is already applied."
elif git -C "$QDE_BACKEND_DIR" apply --check "$QDE_PATCH_FILE" >/dev/null 2>&1; then
  git -C "$QDE_BACKEND_DIR" apply "$QDE_PATCH_FILE"
  echo "Applied Musubi unique-cache-key patch."
else
  echo "Unable to apply or verify the Musubi unique-cache-key patch." >&2
  exit 2
fi
