# Shared helpers for the native RTXGI test harness wrappers (play.ps1, bench.ps1).
# Dot-source this file: . "$PSScriptRoot/native-lib.ps1"
#
# The harness is a comparison tool only. Nothing from it is copied into the
# library packages; these scripts only write its .ini file and launch it.

Set-StrictMode -Version Latest

$script:Invariant = [System.Globalization.CultureInfo]::InvariantCulture

function Format-Number([double]$Value, [int]$Digits = 4) {
    return $Value.ToString("F$Digits", $script:Invariant)
}

function Get-RepoRoot {
    return (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path.Replace('\', '/')
}

# --- Harness executables -----------------------------------------------------

function Get-HarnessExe([ValidateSet('vulkan', 'd3d12')][string]$Api) {
    $root = Get-RepoRoot
    if ($Api -eq 'vulkan') {
        $exe = "$root/native/build/rtxgi/samples/bin/vulkan/Release/TestHarness-VK.exe"
    } else {
        $exe = "$root/native/build/rtxgi/samples/bin/d3d12/Release/TestHarness-D3D12.exe"
    }
    if (-not (Test-Path -LiteralPath $exe)) {
        throw "Harness not built: $exe`nSee tools/native/README.md, section 'Build'."
    }
    return $exe
}

# --- Scene manifest ----------------------------------------------------------

function Read-SceneManifest([string]$SceneId) {
    $path = "$(Get-RepoRoot)/scenes/manifests/$SceneId.json"
    if (-not (Test-Path -LiteralPath $path)) {
        $known = (Get-ChildItem "$(Get-RepoRoot)/scenes/manifests" -Filter *.json | ForEach-Object { $_.BaseName }) -join ', '
        throw "No manifest for scene '$SceneId'. Known scenes: $known"
    }
    return Get-Content -LiteralPath $path -Raw | ConvertFrom-Json
}

# A .glb file with textures cannot be loaded by the harness. Detect that by
# looking for an "images" list in the GLB's JSON chunk (only that chunk is read).
function Test-GlbHasTextures([string]$Path) {
    $stream = [System.IO.File]::OpenRead($Path)
    try {
        $header = New-Object byte[] 20
        [void]$stream.Read($header, 0, 20)
        $jsonLength = [BitConverter]::ToUInt32($header, 12)
        $json = New-Object byte[] $jsonLength
        [void]$stream.Read($json, 0, $jsonLength)
        $text = [System.Text.Encoding]::UTF8.GetString($json)
        return [regex]::IsMatch($text, '"images"\s*:\s*\[\s*\{')
    } finally {
        $stream.Dispose()
    }
}

# Finds the scene file for a manifest and returns what the .ini needs:
# an absolute folder (with a trailing slash), a file name, and a cache folder.
function Test-IdentityTransform($Transform) {
    return @(@($Transform.position) | Where-Object { $_ -ne 0 }).Count -eq 0 -and
           @(@($Transform.rotationDeg) | Where-Object { $_ -ne 0 }).Count -eq 0 -and
           $Transform.scale -eq 1
}

# The downloaded file of a model part. A clean re-fetch can still be running; the previous download
# is kept in downloads.bak, so that is used (with a warning) while assets/downloads has no copy.
function Resolve-ModelPartSource($Part) {
    $root = Get-RepoRoot
    $source = "$root/assets/downloads/$($Part.asset)/$($Part.path)"
    if (Test-Path -LiteralPath $source) { return $source }
    $backup = "$root/assets/downloads.bak/$($Part.asset)/$($Part.path)"
    if (Test-Path -LiteralPath $backup) {
        Write-Warning "Using $backup (not found in assets/downloads yet)"
        return $backup
    }
    throw "Missing $source. Run 'pnpm assets:fetch' (do not run it while another fetch is going)."
}

function Resolve-SceneFile($Manifest) {
    $root = Get-RepoRoot
    $parts = @($Manifest.parts)
    # The harness loads one glTF file and cannot place it. Scenes with several parts, or with one
    # model part that has a transform, are merged into one file first (see Resolve-MergedSceneFile).
    # A single part without a transform is loaded as it is, exactly as before.
    if ($parts.Count -gt 1 -or ($parts[0].kind -eq 'model' -and -not (Test-IdentityTransform $parts[0].transform))) {
        return Resolve-MergedSceneFile $Manifest
    }
    $part = $parts[0]

    if ($part.kind -eq 'procedural') {
        $source = "$root/assets/generated/$($Manifest.id).glb"
        if (-not (Test-Path -LiteralPath $source)) { throw "Missing $source. Run 'pnpm scenes:export' first." }
    } else {
        $source = Resolve-ModelPartSource $part
    }

    $cacheDir = "$root/native/build/cache/$($Manifest.id)/"
    New-Item -ItemType Directory -Force -Path $cacheDir | Out-Null

    $sourceItem = Get-Item -LiteralPath $source
    if ($sourceItem.Extension -eq '.glb' -and (Test-GlbHasTextures $source)) {
        # Split into .gltf + .bin + image files once, outside the asset folders.
        $stage = "$root/native/build/scenes/$($Manifest.id)"
        $staged = "$stage/$($Manifest.id).gltf"
        if (-not (Test-Path -LiteralPath $staged) -or (Get-Item $staged).LastWriteTime -lt $sourceItem.LastWriteTime) {
            Write-Host "Splitting $($sourceItem.Name) into glTF + images (the harness cannot read textures inside a .glb)..."
            & node "$PSScriptRoot/glb-to-gltf.mjs" $source $stage $Manifest.id | Out-Host
            if ($LASTEXITCODE -ne 0) { throw 'glb-to-gltf.mjs failed' }
            # The old cache was built from the old split; empty the folder so it is rebuilt.
            Clear-FolderContents $cacheDir
        }
        return @{ Directory = "$stage/"; File = "$($Manifest.id).gltf"; CacheDir = $cacheDir }
    }

    $fileName = $sourceItem.Name
    if ($fileName -notmatch '^[\w-]+\.(glb|gltf)$') {
        throw "The harness only accepts file names of letters, digits, '_' and '-': '$fileName'"
    }
    $directory = $sourceItem.DirectoryName.Replace('\', '/') + '/'
    return @{ Directory = $directory; File = $fileName; CacheDir = $cacheDir }
}

# Scenes with several model parts (sponza = Intel's base package + curtains), or one model part
# with a manifest transform: merge-gltf.mjs writes one native/build/scenes/<id>/<id>.gltf that
# loads every part with its transform, as the viewer and Cycles do. The merged file points at the
# parts' own buffers and images (nothing in assets/ is copied or changed). A .glb part is first
# split into .gltf + .bin + images (glb-to-gltf.mjs, the same step as for single .glb scenes),
# because a .gltf cannot refer into a .glb. The scene cache is <id>.cache and is deleted whenever
# the merged file changes.
function Resolve-MergedSceneFile($Manifest) {
    $root = Get-RepoRoot
    $id = $Manifest.id
    $stage = "$root/native/build/scenes/$id"
    $cacheDir = "$root/native/build/cache/$id/"
    New-Item -ItemType Directory -Force -Path $stage, $cacheDir | Out-Null

    $specParts = @()
    $index = 0
    foreach ($part in @($Manifest.parts)) {
        $index++
        if ($part.kind -ne 'model') {
            # pnpm scenes:export only writes assets/generated/<id>.glb for procedural-only scenes.
            throw "Scene '$id' mixes procedural and model parts. There is no exported file for its procedural part, so the native harness cannot load it."
        }
        $source = Resolve-ModelPartSource $part
        $sourceItem = Get-Item -LiteralPath $source
        if ($sourceItem.Extension -eq '.glb') {
            $partName = "part$index"
            $partStage = "$stage/$partName"
            $staged = "$partStage/$partName.gltf"
            if (-not (Test-Path -LiteralPath $staged) -or (Get-Item $staged).LastWriteTime -lt $sourceItem.LastWriteTime) {
                Write-Host "Splitting $($sourceItem.Name) into glTF + images for the merge..."
                & node "$PSScriptRoot/glb-to-gltf.mjs" $source $partStage $partName | Out-Host
                if ($LASTEXITCODE -ne 0) { throw 'glb-to-gltf.mjs failed' }
            }
            $source = $staged
        }
        $specParts += [ordered]@{
            path = $source.Replace('\', '/')
            label = "$($part.asset)"
            transform = [ordered]@{
                position = @($part.transform.position | ForEach-Object { [double]$_ })
                rotationDeg = @($part.transform.rotationDeg | ForEach-Object { [double]$_ })
                scale = [double]$part.transform.scale
            }
        }
    }

    $spec = [ordered]@{ output = $stage; name = $id; parts = $specParts }
    $specPath = "$stage/$id.merge.json"
    [System.IO.File]::WriteAllText($specPath, ($spec | ConvertTo-Json -Depth 6))
    Write-Host "Merging the $($specParts.Count) model parts of '$id' into one glTF for the harness:"
    & node "$PSScriptRoot/merge-gltf.mjs" $specPath | Out-Host
    switch ($LASTEXITCODE) {
        0 { }
        3 {
            # The merged file is new or changed: a cache built from the old one would load the old scene.
            $cacheFile = "$cacheDir$id.cache"
            if (Test-Path -LiteralPath $cacheFile) {
                Write-Host "The merged scene changed; deleting its old cache $cacheFile"
                Remove-Item -LiteralPath $cacheFile -Force
            }
        }
        default { throw "merge-gltf.mjs failed (exit code $LASTEXITCODE)" }
    }
    return @{ Directory = "$stage/"; File = "$id.gltf"; CacheDir = $cacheDir }
}

# Deletes everything inside a folder and keeps the folder. (-LiteralPath "$dir*" would look for a
# file literally named '*' and silently delete nothing.)
function Clear-FolderContents([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Container)) { return }
    Get-ChildItem -LiteralPath $Path -Force | Remove-Item -Recurse -Force
}

# --- Probe volume ------------------------------------------------------------

# Two layouts, both centred on volume.center (the harness's volume origin is the grid's centre):
#   'cells' (default): probes sit at the centres of equal cells that cover the manifest volume
#           exactly (the same layout as three.js LightProbeGrid): counts = ceil(size / spacing),
#           actual spacing = size / counts.
#   'ddgi'  : the grid of our DDGI technique (probeGrid() in packages/ddgi-core, see
#           docs/design/ddgi.md): counts = floor(size / spacing) + 1 per axis, spacing exactly as
#           given. Used for timing at matched probe counts and positions (T1.9).
function Get-VolumeGrid($Volume, [double]$ProbeSpacing = 0, [ValidateSet('cells', 'ddgi')][string]$GridRule = 'cells') {
    # $ProbeSpacing > 0 replaces the manifest's spacing (for timing sweeps over the probe count)
    $target = if ($ProbeSpacing -gt 0) { $ProbeSpacing } else { [double]$Volume.probeSpacing }
    $counts = @()
    $spacing = @()
    for ($axis = 0; $axis -lt 3; $axis++) {
        $size = [double]$Volume.size[$axis]
        if ($GridRule -eq 'ddgi') {
            # The same small tolerance as probeGrid(), so 3.0 / 0.5 does not floor to 5
            $counts += [int]([math]::Floor($size / $target + 1e-6) + 1)
            $spacing += $target
        } else {
            $n = [int][math]::Max(1, [math]::Ceiling($size / $target - 1e-4))
            $counts += $n
            $spacing += $size / $n
        }
    }
    return @{
        Center = @($Volume.center | ForEach-Object { [double]$_ })
        Counts = $counts
        Spacing = $spacing
        MinSpacing = ($spacing | Measure-Object -Minimum).Minimum
        Total = $counts[0] * $counts[1] * $counts[2]
    }
}

# --- Camera ------------------------------------------------------------------

# The harness (right-handed, y-up) looks along
#   (cos(pitch) sin(yaw), -sin(pitch), -cos(pitch) cos(yaw))
# with yaw and pitch in degrees; positive pitch looks down.
function ConvertTo-YawPitch($Position, $Target) {
    $dx = [double]$Target[0] - [double]$Position[0]
    $dy = [double]$Target[1] - [double]$Position[1]
    $dz = [double]$Target[2] - [double]$Position[2]
    $length = [math]::Sqrt($dx * $dx + $dy * $dy + $dz * $dz)
    $pitch = -[math]::Asin($dy / $length) * 180 / [math]::PI
    $yaw = [math]::Atan2($dx, -$dz) * 180 / [math]::PI
    if ($yaw -lt 0) { $yaw += 360 }
    return @{ Yaw = $yaw; Pitch = $pitch }
}

# --- The .ini ----------------------------------------------------------------

function ConvertTo-Vec([object[]]$Values, [int]$Digits = 4) {
    return (@($Values) | ForEach-Object { Format-Number ([double]$_) $Digits }) -join ' '
}

# Builds the harness config text for a scene.
#   Clean       : no probe spheres, no ray traced AO, no dither (for comparing images and timing the GI alone)
#   Camera      : the bookmark to start on, 1 = first bookmark in the manifest
#   Rays        : rays per probe
#   HideUi      : start with the settings panel hidden ('u' toggles it in the harness)
#   AutoBench   : run the F4 benchmark by itself after 300 frames and exit
#   UpdateEveryFrame : update the probes every frame even after the volume has converged. Upstream
#                 (and the default here) stops tracing and blending a volume once its measured
#                 variability is below the threshold, so a benchmark would mostly time frames in
#                 which no probe work happens at all. Timing runs set this (fork ini option
#                 ddgi.volume.0.probeVariability.skipConverged=0).
#   ProbeSpacing: > 0 replaces the manifest probe spacing (timing sweeps over the probe count)
#   GridRule    : 'cells' (default) or 'ddgi' (our DDGI's floor(size / spacing) + 1 grid), see Get-VolumeGrid
#   SkyTexture  : (fork, T2.7) the shared sky file (scenes/sky/<id>.exr) for every missed ray, instead of
#                 its flat equivalent. Empty (default): the flat sky, as before. Vulkan only.
#   KeepLights  : labels of lights to keep even at intensity 0 (a motion's target: Sponza's arcade lamp)
#   AppLines    : extra app.* ini lines (fork options: the motion and recording keys, see Motion.h in the fork)
function New-HarnessIni {
    param(
        $Manifest,
        [hashtable]$SceneFile,
        [string]$OutputDir,
        [string]$Gpu = '',
        [int]$Width = 1600,
        [int]$Height = 900,
        [int]$Vsync = 1,
        [switch]$Clean,
        [int]$Camera = 1,
        [int]$Rays = 256,
        [switch]$AutoBench,
        [switch]$HideUi,
        [int]$WarmupFrames = 300,
        [switch]$UpdateEveryFrame,
        [double]$ProbeSpacing = 0,
        [ValidateSet('cells', 'ddgi')][string]$GridRule = 'cells',
        [string]$SkyTexture = '',
        [string[]]$KeepLights = @(),
        [string[]]$AppLines = @()
    )
    $root = Get-RepoRoot
    $harness = "$root/native/rtxgi-ddgi/samples/test-harness"
    $grid = Get-VolumeGrid $Manifest.volume $ProbeSpacing $GridRule
    $s = $grid.MinSpacing
    $cameras = @($Manifest.cameras)
    if ($Camera -lt 1 -or $Camera -gt $cameras.Count) {
        throw "Scene '$($Manifest.id)' has $($cameras.Count) camera bookmarks; -Camera must be 1 to $($cameras.Count)."
    }

    $env = $Manifest.environment
    switch ($env.type) {
        'none' { $skyColor = '0 0 0'; $skyIntensity = 1 }
        'color' { $skyColor = ConvertTo-Vec $env.color; $skyIntensity = if ($null -ne $env.intensity) { $env.intensity } else { 1 } }
        # The harness's shaders take one flat sky radiance for every ray that misses
        # (app.skyRadiance), not an image, so a procedural sky becomes the shared sky
        # file's flat equivalent: the uniform radiance that gives a flat open ground the
        # same irradiance as the file, per channel (scenes/sky/<id>.json, written by
        # tools/sky/generate.ts; environment.intensity is already in it). Unlike the file,
        # the harness also returns it below the horizon.
        'procedural' {
            $sidecarPath = "$root/scenes/sky/$($Manifest.id).json"
            if (-not (Test-Path $sidecarPath)) { throw "Sky sidecar $sidecarPath is missing; run 'node tools/sky/generate.ts'." }
            $radiance = @((Get-Content -Raw $sidecarPath | ConvertFrom-Json).flatEquivalentRadianceRgb)
            $skyIntensity = ($radiance | Measure-Object -Maximum).Maximum
            $skyColor = ($radiance | ForEach-Object { Format-Number ($_ / $skyIntensity) 6 }) -join ' '
        }
    }

    $showProbes = if ($Clean) { 0 } else { 1 }
    $rtao = if ($Clean) { 0 } else { 1 }
    $dither = if ($Clean) { 0 } else { 1 }
    $screenshots = "$OutputDir/screenshots"
    $movementSpeed = if (($Manifest.volume.size | Measure-Object -Maximum).Maximum -gt 50) { 20 } else { 5 }

    $ini = New-Object System.Collections.Generic.List[string]
    $ini.Add("# Written by tools/native/native-lib.ps1 from scenes/manifests/$($Manifest.id).json. Do not edit; rerun the script.")
    $ini.Add("# Positions and directions are right-handed, y-up, in metres.")
    $ini.Add('')
    $ini.Add('# application')
    $ini.Add("app.width=$Width")
    $ini.Add("app.height=$Height")
    $ini.Add("app.vsync=$Vsync")
    $ini.Add('app.fullscreen=0')
    $ini.Add('app.renderMode=1')
    $ini.Add("app.showUI=$(if ($HideUi) { 0 } else { 1 })")
    $ini.Add("app.root=$harness/")
    $ini.Add("app.rtxgiSDK=$root/native/rtxgi-ddgi/rtxgi-sdk/")
    $ini.Add("app.title=RTXGI Test Harness - $($Manifest.id)")
    if ($Gpu) { $ini.Add("app.gpu=$Gpu") }
    # Unattended runs (benchmarks, captures, cache builds) ignore the keyboard and mouse: GLFW gives
    # a new window the keyboard focus even when it starts minimized, so a key typed into another
    # window (M switches to the path tracer) would otherwise change the run. Fork option.
    if ($AutoBench -or $HideUi) { $ini.Add('app.ignoreInput=1') }
    if ($AutoBench) {
        $ini.Add('app.autoBenchmark=1')
        $ini.Add("app.benchmarkWarmupFrames=$WarmupFrames")
    }
    foreach ($line in $AppLines) { $ini.Add($line) }
    $ini.Add('')
    $ini.Add('# shader compilation')
    $ini.Add('shaders.warningsAsErrors=1')
    $ini.Add('shaders.disableOptimizations=0')
    $ini.Add('shaders.disableValidation=0')
    $ini.Add('shaders.shaderSymbols=0')
    $ini.Add('shaders.lifetimeMarkers=0')
    $ini.Add('')
    $ini.Add('# scene')
    $ini.Add("scene.name=$($Manifest.title -replace '[#%=]', '')")
    $ini.Add("scene.path=$($SceneFile.Directory)")
    $ini.Add("scene.file=$($SceneFile.File)")
    $ini.Add("scene.cacheDir=$($SceneFile.CacheDir)")
    $ini.Add("scene.screenshotPath=$screenshots")
    $ini.Add("scene.activeCamera=$($Camera - 1)")
    $ini.Add("scene.skyColor=$skyColor")
    $ini.Add("scene.skyIntensity=$(Format-Number $skyIntensity 6)")
    # Fork option (T2.7): the sky file itself for every missed ray. skyColor stays as the fallback (D3D12).
    if ($SkyTexture) { $ini.Add("scene.skyTexture=$SkyTexture") }
    $ini.Add('')
    $ini.Add('# scene lights (directional: lux as power; point and spot: candela as power, range as radius)')

    # A light at intensity 0 is off until a manifest motion plays it (Sponza's arcade lamp, T2.1):
    # left out, as the viewer leaves it out, so an off lamp costs the harness's shading nothing.
    # A motion's target is kept (KeepLights), since the motion turns it on.
    $lights = @($Manifest.lights | Where-Object { $_.intensity -gt 0 -or ($_.PSObject.Properties['label'] -and $KeepLights -contains $_.label) })
    for ($i = 0; $i -lt $lights.Count; $i++) {
        $light = $lights[$i]
        $label = if ($light.PSObject.Properties['label']) { $light.label } else { "$($light.type)-$i" }
        $ini.Add("scene.lights.$i.name=$label")
        $ini.Add("scene.lights.$i.color=$(ConvertTo-Vec $light.color)")
        $ini.Add("scene.lights.$i.power=$(Format-Number $light.intensity)")
        switch ($light.type) {
            'directional' {
                $ini.Add("scene.lights.$i.type=0")
                $ini.Add("scene.lights.$i.direction=$(ConvertTo-Vec $light.direction)")
            }
            'spot' {
                $ini.Add("scene.lights.$i.type=1")
                $ini.Add("scene.lights.$i.position=$(ConvertTo-Vec $light.position)")
                $ini.Add("scene.lights.$i.direction=$(ConvertTo-Vec $light.direction)")
                $ini.Add("scene.lights.$i.radius=$(Format-Number $light.range)")
                $ini.Add("scene.lights.$i.umbraAngle=$(Format-Number ($light.outerConeDeg * [math]::PI / 180) 6)")
                $ini.Add("scene.lights.$i.penumbraAngle=$(Format-Number ($light.innerConeDeg * [math]::PI / 180) 6)")
            }
            'point' {
                $ini.Add("scene.lights.$i.type=2")
                $ini.Add("scene.lights.$i.position=$(ConvertTo-Vec $light.position)")
                $ini.Add("scene.lights.$i.radius=$(Format-Number $light.range)")
            }
        }
    }

    $ini.Add('')
    $ini.Add('# scene cameras (the manifest bookmarks, in order)')
    for ($i = 0; $i -lt $cameras.Count; $i++) {
        $c = $cameras[$i]
        $angles = ConvertTo-YawPitch $c.position $c.target
        $ini.Add("scene.cameras.$i.name=$($c.name)")
        $ini.Add("scene.cameras.$i.fov=$(Format-Number $c.fovYDeg 2)")
        $ini.Add("scene.cameras.$i.aspect=$(Format-Number ($Width / $Height) 6)")
        $ini.Add("scene.cameras.$i.yaw=$(Format-Number $angles.Yaw 3)")
        $ini.Add("scene.cameras.$i.pitch=$(Format-Number $angles.Pitch 3)")
        $ini.Add("scene.cameras.$i.position=$(ConvertTo-Vec $c.position)")
    }

    $ini.Add('')
    $ini.Add('# input')
    $ini.Add("input.movementSpeed=$($movementSpeed).f")
    $ini.Add('input.rotationSpeed=3.f')
    $ini.Add('input.invertPan=1')
    $ini.Add('')
    $ini.Add('# path tracer (M switches to it)')
    $ini.Add('pt.rayViewBias=0.0001')
    $ini.Add('pt.rayNormalBias=0.008')
    $ini.Add('pt.numBounces=10')
    $ini.Add('pt.samplesPerPixel=1')
    $ini.Add('pt.antialiasing=1')
    $ini.Add('')
    $ini.Add("# ddgi volume: $($grid.Counts -join ' x ') = $($grid.Total) probes, $Rays rays per probe")
    $ini.Add("# biases scale with the smallest probe spacing ($(Format-Number $s 3) m)")
    $ini.Add('ddgi.volume.0.name=Scene-Volume')
    $ini.Add('ddgi.volume.0.probeRelocation.enabled=1')
    $ini.Add("ddgi.volume.0.probeRelocation.minFrontfaceDistance=$(Format-Number (0.3 * $s))")
    $ini.Add('ddgi.volume.0.probeClassification.enabled=1')
    $ini.Add('ddgi.volume.0.probeVariability.enabled=1')
    $ini.Add('ddgi.volume.0.probeVariability.threshold=0.4')
    if ($UpdateEveryFrame) { $ini.Add('ddgi.volume.0.probeVariability.skipConverged=0') }
    $ini.Add('ddgi.volume.0.infiniteScrolling.enabled=0')
    $ini.Add('ddgi.volume.0.textures.rayData.format=5')
    $ini.Add('ddgi.volume.0.textures.irradiance.format=0')
    $ini.Add('ddgi.volume.0.textures.distance.format=2')
    $ini.Add('ddgi.volume.0.textures.data.format=3')
    $ini.Add('ddgi.volume.0.textures.variability.format=1')
    $ini.Add("ddgi.volume.0.origin=$(ConvertTo-Vec $grid.Center)")
    $ini.Add("ddgi.volume.0.probeCounts=$($grid.Counts -join ' ')")
    $ini.Add("ddgi.volume.0.probeSpacing=$(ConvertTo-Vec $grid.Spacing 5)")
    $ini.Add("ddgi.volume.0.probeNumRays=$Rays")
    $ini.Add('ddgi.volume.0.probeNumIrradianceTexels=8')
    $ini.Add('ddgi.volume.0.probeNumDistanceTexels=16')
    $ini.Add('ddgi.volume.0.probeHysteresis=0.97')
    $ini.Add("ddgi.volume.0.probeNormalBias=$(Format-Number (0.1 * $s))")
    $ini.Add("ddgi.volume.0.probeViewBias=$(Format-Number (0.3 * $s))")
    $ini.Add('ddgi.volume.0.probeMaxRayDistance=10000')
    $ini.Add('ddgi.volume.0.probeIrradianceThreshold=0.2')
    $ini.Add('ddgi.volume.0.probeBrightnessThreshold=1.0')
    $ini.Add('ddgi.volume.0.vis.probeVisType=1')
    $ini.Add("ddgi.volume.0.vis.probeRadius=$(Format-Number (0.1 * $s))")
    $ini.Add('ddgi.volume.0.vis.probeDistanceDivisor=200')
    $ini.Add("ddgi.volume.0.vis.showProbes=$showProbes")
    $ini.Add('ddgi.volume.0.vis.texture.irradianceScale=0.36')
    $ini.Add('ddgi.volume.0.vis.texture.distanceScale=0.18')
    $ini.Add('ddgi.volume.0.vis.texture.probeDataScale=2.88')
    $ini.Add('ddgi.volume.0.vis.texture.rayDataScale=0.247')
    $ini.Add('ddgi.volume.0.vis.texture.probeVariabilityScale=0.479')
    $ini.Add('')
    $ini.Add('# ray traced ambient occlusion')
    $ini.Add("rtao.enable=$rtao")
    $ini.Add('rtao.rayLength=0.25')
    $ini.Add('rtao.rayNormalBias=0.001')
    $ini.Add('rtao.rayViewBias=0.001')
    $ini.Add('rtao.powerLog=5.0')
    $ini.Add('rtao.filterDistanceSigma=3.0')
    $ini.Add('rtao.filterDepthSigma=0.1')
    $ini.Add('')
    $ini.Add('# post process')
    $ini.Add('pp.enable=1')
    $ini.Add('pp.exposure.enable=1')
    $ini.Add('pp.exposure.fstops=1.0')
    $ini.Add('pp.tonemap.enable=1')
    $ini.Add("pp.dither.enable=$dither")
    $ini.Add('pp.gamma.enable=1')

    # The harness skips a last line that has no line break, so end with one.
    return @{ Text = ($ini -join "`n") + "`n"; Grid = $grid; Rays = $Rays; Screenshots = $screenshots }
}

# --- Launching ---------------------------------------------------------------

# Starts the harness with a config. Extra environment variables are set only for the child.
function Start-Harness {
    param(
        [string]$Exe,
        [string]$IniPath,
        [string]$WorkingDirectory,
        [hashtable]$Environment = @{},
        [string]$WindowStyle = 'Normal'
    )
    $names = @('DDGI_CAPTURE_FRAMES', 'DDGI_CAPTURE_CAMERA', 'DDGI_CAPTURE_MODE', 'DDGI_CAPTURE_LINEAR', 'DDGI_CAPTURE_FIELD') + @($Environment.Keys)
    # Remove-Item is the reliable way to unset a variable; setting it to $null leaves an empty
    # value behind, and the harness treats "set but empty" as "set".
    $previous = @{}
    foreach ($name in $names) {
        $previous[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
        Remove-Item -LiteralPath "Env:$name" -ErrorAction SilentlyContinue
    }
    foreach ($key in $Environment.Keys) { Set-Item -LiteralPath "Env:$key" -Value ([string]$Environment[$key]) }
    try {
        return Start-Process -FilePath $Exe -ArgumentList ('"' + $IniPath + '"') -WorkingDirectory $WorkingDirectory -WindowStyle $WindowStyle -PassThru
    } finally {
        foreach ($name in $names) {
            Remove-Item -LiteralPath "Env:$name" -ErrorAction SilentlyContinue
            if ($null -ne $previous[$name]) { Set-Item -LiteralPath "Env:$name" -Value $previous[$name] }
        }
    }
}

# Waits for the harness to exit. The harness shows a modal "Error!" box when it fails (for example
# when the GPU resets) and then waits for a click, so a plain WaitForExit would hang. Returns
# 'exited', 'error-dialog' (process killed) or 'timeout' (process killed).
function Wait-Harness([System.Diagnostics.Process]$Process, [int]$TimeoutMinutes = 30) {
    $deadline = (Get-Date).AddMinutes($TimeoutMinutes)
    while (-not $Process.WaitForExit(2000)) {
        $Process.Refresh()
        if ($Process.MainWindowTitle -eq 'Error!') { $Process.Kill(); return 'error-dialog' }
        if ((Get-Date) -gt $deadline) { $Process.Kill(); return 'timeout' }
    }
    return 'exited'
}

# Closes a harness this script started, if it is still running. Used in finally blocks: on Ctrl-C
# the harness would otherwise keep using the GPU after the lock has been given back. Only the given
# process is touched (never by name). Asks the window to close first, then kills after 5 s.
function Stop-StartedHarness([System.Diagnostics.Process]$Process) {
    if ($null -eq $Process -or $Process.HasExited) { return }
    Write-Host "Closing the harness we started (PID $($Process.Id))."
    $null = $Process.CloseMainWindow()
    if (-not $Process.WaitForExit(5000)) { $Process.Kill() }
}

function Get-HarnessRunDir([string]$SceneId, [string]$Api, [string]$Tag = '') {
    $dir = "$(Get-RepoRoot)/native/build/runs/$SceneId-$Api$Tag"
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    return $dir
}

# A run only counts if the harness got to the end of its own log ("Done.") and wrote the files we
# expect after the launch time. Anything else throws, with the last lines of log.txt.
function Assert-HarnessCompleted([string]$RunDir, [datetime]$LaunchTime, [string[]]$ExpectedFiles = @(), [string]$What = 'harness run') {
    $logPath = "$RunDir/log.txt"
    $problems = @()
    $tail = @()
    if (-not (Test-Path -LiteralPath $logPath)) {
        $problems += 'there is no log.txt'
    } else {
        $tail = Get-Content -LiteralPath $logPath -Tail 8
        if (-not ($tail | Where-Object { $_ -match '^Done\.' })) { $problems += 'log.txt does not end with "Done." (the harness stopped or was closed early)' }
    }
    foreach ($file in $ExpectedFiles) {
        if (-not (Test-Path -LiteralPath $file)) { $problems += "missing $file" }
        elseif ((Get-Item -LiteralPath $file).LastWriteTime -lt $LaunchTime) { $problems += "$file is older than this launch (stale)" }
    }
    if ($problems.Count -gt 0) {
        throw "FAILED: $What. $($problems -join '; ').`nLast lines of ${logPath}:`n$($tail -join "`n")"
    }
}

# --- Scene cache ---------------------------------------------------------------

# Measured first-load times on the RTX 3060 laptop (texture conversion on the CPU, one thread, no progress):
# Sponza about 100 s (base package alone; 112 s with the curtains merged in, 330 MB cache), Bistro exterior about 18.5 minutes (277 textures, 1.4 GB cache).
$script:CacheBuildEstimate = @{
    'sponza' = 'about 2 minutes'
    'bistro-interior' = 'about 8 to 10 minutes'
    'bistro-exterior' = 'about 19 minutes'
}

# The harness converts every texture on its main thread the first time it loads a scene, so its
# window shows "Not Responding" for that whole time. This builds that cache first, in a minimized
# window and under the GPU lock, so an interactive window never has to do it. Returns the seconds
# it took (0 if the cache already existed).
function Initialize-SceneCache($Manifest, [hashtable]$SceneFile, [string]$Api = 'vulkan', [string]$Gpu = '', [int]$TimeoutMinutes = 60) {
    $cacheFile = Join-Path $SceneFile.CacheDir (([System.IO.Path]::GetFileNameWithoutExtension($SceneFile.File)) + '.cache')
    if (Test-Path -LiteralPath $cacheFile) { return 0 }

    $estimate = if ($script:CacheBuildEstimate.ContainsKey($Manifest.id)) { $script:CacheBuildEstimate[$Manifest.id] } else { 'a minute or less for small scenes' }
    Write-Host "No scene cache for '$($Manifest.id)' yet. Building it first, $estimate."
    Write-Host 'The harness is started MINIMIZED for this (no window in your way). If you restore it, it shows "Not Responding" until the cache is done. That is normal. Do not close it.'
    $exe = Get-HarnessExe $Api
    $runDir = Get-HarnessRunDir $Manifest.id $Api '-cache'
    $config = New-HarnessIni -Manifest $Manifest -SceneFile $SceneFile -OutputDir $runDir -Gpu $Gpu -Width 640 -Height 360 -Vsync 0 -Clean -Rays 32 -HideUi
    $iniPath = "$runDir/$($Manifest.id).ini"
    [System.IO.File]::WriteAllText($iniPath, $config.Text)

    $watch = [System.Diagnostics.Stopwatch]::StartNew()
    $launched = Get-Date
    $result = 'exited'
    Use-BenchLock -What "native cache build $($Manifest.id)" -Action {
        $watch.Restart()
        $process = $null
        try {
            $process = Start-Harness -Exe $exe -IniPath $iniPath -WorkingDirectory $runDir -Environment @{ DDGI_CAPTURE_FRAMES = 1 } -WindowStyle Minimized
            $script:cacheBuildResult = Wait-Harness $process $TimeoutMinutes
        } finally {
            # Ctrl-C: close the harness we started before Use-BenchLock gives the lock back.
            # (A half-written cache file is not kept: the harness writes it at the end of the load.)
            Stop-StartedHarness $process
        }
    }
    $watch.Stop()
    Assert-HarnessCompleted $runDir $launched @($cacheFile) "scene cache build for $($Manifest.id) ($script:cacheBuildResult)"
    Write-Host ("Cache for '{0}' built in {1:N0} s." -f $Manifest.id, $watch.Elapsed.TotalSeconds)
    return [math]::Round($watch.Elapsed.TotalSeconds)
}

# --- The shared GPU timing lock ---------------------------------------------

# Only one machine-heavy GPU job at a time (browser or native timing, an interactive harness
# window, a scene cache build). The protocol lives in tools/reference/bench-lock.ts; these
# functions call its command line, so every script on the machine follows the same rules:
# the lock folder gets an owner file with this PowerShell's PID, a waiter is told who holds it,
# and a lock whose owner process has died is taken over automatically.

$script:BenchLockDepth = 0
# True when the lock is held for us by a parent process (see Test-BenchLockHeldByParent); then this
# script never takes or releases it.
$script:BenchLockInherited = $false

function Invoke-BenchLockCli([string[]]$Arguments) {
    $cli = "$(Get-RepoRoot)/tools/reference/bench-lock.ts"
    # Waiting messages go to stderr; let them through to the console as they come.
    & node $cli @Arguments 2>&1 | ForEach-Object {
        if ($_ -is [System.Management.Automation.ErrorRecord]) { Write-Host $_.Exception.Message } else { Write-Host $_ }
    }
    return $LASTEXITCODE
}

# Takes the lock for this PowerShell process. With -NoWait it does not wait: if someone else has
# the lock it prints who and returns $false. Otherwise it waits (printing who holds it about once
# a minute) and returns $true once the lock is ours.
# `node tools/reference/bench-lock.ts run -- pwsh tools/native/bench.ps1` already holds the lock, and
# says so in DDGI_BENCH_LOCK_HELD=<owner pid>. Taking it again would wait forever on our own parent.
# The variable only counts if that pid really owns the lock right now (bench-lock.ts checks: owner
# file, this host, process running), so a leftover variable never skips the lock.
function Test-BenchLockHeldByParent {
    if (-not $env:DDGI_BENCH_LOCK_HELD) { return $false }
    $cli = "$(Get-RepoRoot)/tools/reference/bench-lock.ts"
    $json = & node $cli status --json 2>$null
    try { $status = ($json -join "`n") | ConvertFrom-Json } catch { return $false }
    $held = $status.PSObject.Properties['heldByParent']
    if ($null -eq $held -or $null -eq $held.Value) {
        Write-Host "DDGI_BENCH_LOCK_HELD=$env:DDGI_BENCH_LOCK_HELD is set, but that process does not hold the GPU lock; taking it normally."
        return $false
    }
    Write-Host "The GPU lock is already held for this script by its parent (pid $($held.Value), bench-lock run); not taking it again."
    return $true
}

function Enter-BenchLock([string]$What, [switch]$NoWait, [int]$MaxWaitMinutes = 180) {
    if ($script:BenchLockDepth -gt 0) { $script:BenchLockDepth++; return $true }
    if (Test-BenchLockHeldByParent) {
        $script:BenchLockInherited = $true
        $script:BenchLockDepth = 1
        return $true
    }
    $cli = "$(Get-RepoRoot)/tools/reference/bench-lock.ts"
    $null = & node $cli status 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host ''
        Write-Host 'Another GPU job is using the graphics card right now (a benchmark, a reference render or another harness window):'
        $null = Invoke-BenchLockCli @('status')
        if ($NoWait) {
            Write-Host 'Not waiting (-NoWait). Try again when that job has finished.'
            return $false
        }
        Write-Host 'Waiting for it to finish; this starts by itself afterwards. Press Ctrl-C to give up.'
    }
    $arguments = @('acquire', '--pid', "$PID", '--what', $What, '--timeout-minutes', "$MaxWaitMinutes", '--command', "tools/native ($What)")
    if ($NoWait) { $arguments[6] = '0' }
    $code = Invoke-BenchLockCli $arguments
    if ($code -eq 5 -and $NoWait) {
        Write-Host 'Someone took the GPU lock just now. Not waiting (-NoWait).'
        return $false
    }
    if ($code -ne 0) { throw "Could not take the GPU bench lock (bench-lock.ts exit code $code)." }
    $script:BenchLockDepth = 1
    return $true
}

function Exit-BenchLock {
    if ($script:BenchLockDepth -le 0) { return }
    $script:BenchLockDepth--
    if ($script:BenchLockDepth -gt 0) { return }
    if ($script:BenchLockInherited) {
        # The parent took it and gives it back; nothing to do here.
        $script:BenchLockInherited = $false
        return
    }
    $code = Invoke-BenchLockCli @('release', '--pid', "$PID")
    if ($code -ne 0) { Write-Warning "Releasing the GPU bench lock returned exit code $code." }
}

# Holds the lock while $Action runs and always gives it back, also on an error or Ctrl-C
# (PowerShell runs finally blocks when a script is stopped). Nested calls reuse the lock the
# caller already holds, so play.ps1 can hold it for a whole session that includes a cache build.
# Usage: Use-BenchLock -What 'native bench sponza' -Action { ...timed run... }
function Use-BenchLock([scriptblock]$Action, [string]$What = 'native harness', [int]$MaxWaitMinutes = 180) {
    $null = Enter-BenchLock -What $What -MaxWaitMinutes $MaxWaitMinutes
    try {
        & $Action
    } finally {
        Exit-BenchLock
    }
}
