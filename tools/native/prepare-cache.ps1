<#
.SYNOPSIS
  Builds the harness's scene cache for one or more scenes, one scene at a time, before anyone
  opens them interactively.

.DESCRIPTION
  The harness converts every texture on its main thread the first time it loads a scene and its
  window shows "Not Responding" until that is done (Sponza about 2 minutes, Bistro interior several
  minutes, Bistro exterior about 19 minutes). This script does that work up front in a minimized
  window, under the shared GPU lock (%TEMP%\ddgi-gpu-bench.lock), and prints how long each took.
  Scenes that already have a cache are skipped. Never run two of these at once: the Bistro scenes
  use most of the 6 GB card while loading.

.EXAMPLE
  ./tools/native/prepare-cache.ps1 -Scene bistro-interior,bistro-exterior
#>
param(
    [string[]]$Scene = @('two-room', 'cornell-box', 'sponza', 'bistro-interior', 'bistro-exterior'),
    [ValidateSet('vulkan', 'd3d12')][string]$Api = 'vulkan',
    [string]$Gpu = '3060'
)
$ErrorActionPreference = 'Stop'
. "$PSScriptRoot/native-lib.ps1"
$Scene = @($Scene | ForEach-Object { $_ -split ',' } | Where-Object { $_ })

# Hold the GPU lock for all the cache builds.
# The lock is not first come, first served: giving it back between steps would hand it to
# whichever job polls next. The Use-BenchLock calls inside reuse this hold.
$null = Enter-BenchLock -What "native cache build $($Scene -join ',')"
try {
    foreach ($sceneId in $Scene) {
        $manifest = Read-SceneManifest $sceneId
        $sceneFile = Resolve-SceneFile $manifest
        $seconds = Initialize-SceneCache $manifest $sceneFile $Api $Gpu
        if ($seconds -eq 0) { Write-Host "$sceneId : cache already exists, nothing to do." }
    }
} finally {
    Exit-BenchLock
}

