#!/usr/bin/env bash
# Bootstrap the transferable Mac kit. This file, the GI bundle and the pnpm seed belong together.
set -eu
kit=$(cd -- "$(dirname -- "$0")" && pwd)
if [ "$(uname -s)" != Darwin ]; then
  echo "Run this kit on the Apple Silicon Mac." >&2
  exit 2
fi
if [ "$#" -ne 0 ] && { [ "$#" -ne 2 ] || [ "$1" != --chrome ]; }; then
  echo 'Usage: bash run-metal.sh [--chrome /path/to/Chrome]' >&2
  exit 2
fi
bundle="$kit/gi-radiance-cascades.bundle"
source_dir="$kit/gi-rc-metal"
store_dir="$kit/.rc-deps"
cd "$kit"
shasum -a 256 -c SHA256SUMS
if [ ! -d "$source_dir" ]; then
  git clone -b codex/radiance-cascades "$bundle" "$source_dir"
fi
expected=$(git bundle list-heads "$bundle" refs/heads/codex/radiance-cascades)
cd "$source_dir"
if [ "$(git rev-parse HEAD)" != "${expected%% *}" ] || ! git diff --quiet HEAD --; then
  echo "The checkout differs from the kit revision. Extract the kit into a new folder for a matched run." >&2
  exit 2
fi
mkdir -p "$store_dir"
tar -xzf "$kit/raytracer-pnpm-seed.tgz" -C "$store_dir"
if command -v pnpm >/dev/null && [ "$(pnpm --version)" = 10.23.0 ]; then
  pm=(pnpm)
elif command -v corepack >/dev/null; then
  pm=(corepack pnpm@10.23.0)
else
  pm=(npx --yes pnpm@10.23.0)
fi
"${pm[@]}" install --frozen-lockfile --prefer-offline --store-dir "$store_dir/pnpm-store"
name="rc-metal-results-$(date +%Y%m%d-%H%M%S)-$$"
status=0
node tools/gpu/pages/rc-polished-budget/metal.ts --out "$kit/$name" "$@" || status=$?
if [ -d "$kit/$name" ]; then
  tar -czf "$kit/$name.tar.gz" -C "$kit" "$name"
  echo "Return this results archive: $kit/$name.tar.gz"
fi
exit "$status"
