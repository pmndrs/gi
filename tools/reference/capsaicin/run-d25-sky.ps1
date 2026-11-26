# D25 (2026-11-19): GI-1.2 screen-probe brightness under a constant white sky. Run through the bench lock, from
# C:\Users\denni\Documents\dev\gi:
#   node tools/reference/bench-lock.ts run --no-cool-wait -- pwsh -File <this script>
# Needs: the flat-plane scene (python scripts\native\capsaicin\make_scenes.py flat-plane) and a scene_viewer built
# with tools/reference/capsaicin/debug-view.patch. Writes artifacts/native/capsaicin/runs/d25-*.
param([string]$Prefix='d25')
$ErrorActionPreference='Stop'
$r=(Resolve-Path (Join-Path $PSScriptRoot '../../..')).Path
$run=Join-Path $r 'tools/reference/capsaicin/run-capsaicin.ps1'
$scene=Join-Path $r 'artifacts/native/capsaicin/scenes/flat-plane.gltf'
$common=@{ExternalScene=$scene;Camera=1;EnvironmentMap=6}
# GI-1.2 as shipped: GlobalIllumination AOV, frames 1-16 each, then single frames 300 and 1000.
& $run @common -Renderer gi -DebugView GlobalIllumination -Frames 16 -ImageFirstFrame 0 -RunName "$Prefix-gi-early"
& $run @common -Renderer gi -DebugView GlobalIllumination -Frames 300 -RunName "$Prefix-gi-300"
& $run @common -Renderer gi -DebugView GlobalIllumination -Frames 1000 -RunName "$Prefix-gi-1000"
# Same with the specular layer compiled out, so GlobalIllumination = albedo/pi * E exactly.
$ns='gi1_disable_specular_materials=true'
& $run @common -Renderer gi -DebugView GlobalIllumination -RenderOptions $ns -Frames 16 -ImageFirstFrame 0 -RunName "$Prefix-gins-early"
& $run @common -Renderer gi -DebugView GlobalIllumination -RenderOptions $ns -Frames 300 -RunName "$Prefix-gins-300"
& $run @common -Renderer gi -DebugView GlobalIllumination -RenderOptions $ns -Frames 1000 -RunName "$Prefix-gins-1000"
# Reference path tracer (final colour, no post chain in that renderer), default material and diffuse-only.
& $run @common -Renderer pt -Frames 1000 -RunName "$Prefix-pt"
& $run @common -Renderer pt -RenderOptions 'reference_pt_disable_specular_materials=true' -Frames 1000 -RunName "$Prefix-ptns"
