<#
.SYNOPSIS
  Opens one of our test scenes in NVIDIA's RTXGI test harness, with DDGI on.

.DESCRIPTION
  Writes the harness .ini from scenes/manifests/<Scene>.json (model file, camera
  bookmarks, lights, probe volume) and starts the harness. The window stays open
  until you close it. Keys inside the harness: F2 dumps DDGI textures, F4 runs
  the 1024-frame benchmark, J toggles the performance overlay, R reloads shaders
  and config, M switches between DDGI and the path tracer.

.PARAMETER Scene
  A scene id from scenes/manifests (sponza, two-room, cornell-box, bistro-exterior, ...).

.PARAMETER Clean
  No probe spheres, no ray traced ambient occlusion and no dither, so only the
  GI shows. This is the mode to use for comparing images.

.PARAMETER Camera
  Camera bookmark to open on, 1 = the first bookmark in the manifest (default 1).

.PARAMETER Api
  vulkan (default) or d3d12.

.PARAMETER Gpu
  Part of the GPU name to use, any case, for example "3060". Default: the first GPU with ray tracing.
  The environment variable RTXGI_GPU does the same.

.PARAMETER CaptureFrames
  Render this many frames, save the back buffer as a PNG and exit. The PNG path is printed.
  For checking a scene without a person at the keyboard.

.PARAMETER CaptureField
  With -CaptureFrames: also write the linear direct and indirect images, the probe field and the
  G-buffer next to the PNG (tools/native/sky-check.ts reads them). Vulkan only.

.PARAMETER Sky
  flat (default): a procedural sky is the sky file's flat equivalent, one colour for every
  missed ray, as before. texture: the shared sky file itself (scenes/sky/<scene>.exr), read by
  the harness's miss shaders (fork option scene.skyTexture; Vulkan only).

.PARAMETER Motion
  Plays a motion: a manifest motion id (light-sweep, light-step, ...), or time-of-day for a
  sunrise-to-sunset day on a scene with a sun and a procedural sky (T2.4d's day: elevation
  5 -> 75 -> 5 degrees, azimuth through 160 degrees, -DaySeconds long; the sky follows the
  sun, so -Sky texture is implied). The motion's times are the manifest's seconds. In a window
  you watch, the motion runs on the wall clock; with -Record it advances a fixed 1/60 s a frame.
  Probes are updated every frame while a motion plays (the harness would otherwise stop
  updating a converged volume and never see the light move).

.PARAMETER Record
  Records the motion without a person at the keyboard (minimized window, settings panel hidden):
  frames.csv (one row a frame: the motion's time, the moving light, the GI-only mean of the
  view) and every -RecordImageEvery-th frame as a PNG, then exits. The folder is printed
  (default native/build/recordings/<scene>-<motion>[-<Tag>]). See tools/native/README.md,
  "Motions and recordings".

.PARAMETER Log
  Plays the motion in a visible window on the wall clock, as a person watching would see it,
  and writes frames.csv (the motion's time and the moving light, every frame; no images, no GI
  mean, so nothing slows the window down) for -RecordSeconds, then closes. The played positions
  are then checked against the manifest at the logged times.

.PARAMETER GridRule
  cells (default): the harness's own probe grid, ceil(size / spacing) cell centres. ddgi: our
  DDGI's grid (floor(size / spacing) + 1 probes at the manifest spacing), as bench.ps1 -GridRule
  ddgi, for comparing with our response at the same probes.

.PARAMETER RecordSeconds
  How much of the motion to record (default: one pass of the motion, its last key's time).

.PARAMETER WarmupFrames
  Frames rendered at the motion's start before its clock starts, so the probes have settled
  (default 600 with -Record, 60 otherwise).

.PARAMETER View
  full (default): the normal image. indirect: the GI alone (the harness's DDGI indirect view).

.PARAMETER SettleKey
  With -Motion: hold the light at this key of the motion (0 = the first) instead of playing it.
  Recorded after a long -WarmupFrames, it is the field converged in that state: the settled
  level tools/native/response.ts holds a step's response against.

.PARAMETER NoWait
  The script holds the shared GPU lock (%TEMP%\ddgi-gpu-bench.lock) for as long as the harness
  is open, so no benchmark runs while you look at a scene. If another GPU job holds the lock, the
  script says who and waits; with -NoWait it exits instead.

.EXAMPLE
  ./tools/native/play.ps1 -Scene sponza -Clean
  ./tools/native/play.ps1 -Scene two-room -Camera 2 -Api d3d12
  ./tools/native/play.ps1 -Scene two-room -Clean -Motion light-sweep
  ./tools/native/play.ps1 -Scene two-room -Clean -Motion light-step -Record -RecordSeconds 12
  ./tools/native/play.ps1 -Scene sponza -Clean -Motion time-of-day
#>
param(
    [Parameter(Mandatory)][string]$Scene,
    [switch]$Clean,
    [int]$Camera = 1,
    [ValidateSet('vulkan', 'd3d12')][string]$Api = 'vulkan',
    [string]$Gpu = '',
    [int]$Width = 1600,
    [int]$Height = 900,
    [int]$Rays = 256,
    [int]$CaptureFrames = 0,
    [switch]$CaptureField,
    [ValidateSet('flat', 'texture')][string]$Sky = 'flat',
    [string]$Motion = '',
    [double]$DaySeconds = 10,
    [switch]$Record,
    [switch]$Log,
    [ValidateSet('cells', 'ddgi')][string]$GridRule = 'cells',
    [string]$RecordDir = '',
    [double]$RecordSeconds = 0,
    [int]$RecordImageEvery = 4,
    [int]$WarmupFrames = -1,
    [ValidateSet('full', 'indirect')][string]$View = 'full',
    [string]$Tag = '',
    [int]$SettleKey = -1,
    [switch]$NoVsync,
    [switch]$NoWait
)
$ErrorActionPreference = 'Stop'
. "$PSScriptRoot/native-lib.ps1"

$exe = Get-HarnessExe $Api
$manifest = Read-SceneManifest $Scene
$sceneFile = Resolve-SceneFile $manifest
$root = Get-RepoRoot

if ($Record -and $Log) { throw 'Use -Record (fixed clock, unattended) or -Log (wall clock, a visible window), not both.' }
# Both write frames.csv; -Record on the fixed 1/60 s clock in a minimized window, -Log on the wall clock in a visible one
$recording = $Record -or $Log
if ($recording -and -not $Motion) { throw '-Record and -Log need -Motion.' }
if ($recording -and $CaptureFrames -gt 0) { throw 'Use -Record, -Log or -CaptureFrames, not two of them.' }
$timeOfDay = $Motion -eq 'time-of-day'
if ($timeOfDay) { $Sky = 'texture' }
if (($Sky -eq 'texture' -or $recording) -and $Api -ne 'vulkan') { throw 'The texture sky and recording are built for the Vulkan harness only.' }
$unattended = $CaptureFrames -gt 0 -or $Record

$skyTexture = ''
if ($Sky -eq 'texture') {
    if ($manifest.environment.type -ne 'procedural') { throw "Scene '$Scene' has no procedural sky (its environment is '$($manifest.environment.type)'), so it has no sky file." }
    $skyTexture = "$root/scenes/sky/$Scene.exr"
    if (-not (Test-Path -LiteralPath $skyTexture)) { throw "Sky file $skyTexture is missing; run 'node tools/sky/generate.ts'." }
}

# Hold the GPU lock for the whole session (cache build included), so no benchmark runs on the
# card while the harness is open. Released in the finally block: on a normal exit, on an error
# and on Ctrl-C.
if (-not (Enter-BenchLock -What "native play $Scene ($Api)$(if ($Motion) { ", $Motion" })" -NoWait:$NoWait)) { exit 2 }
$process = $null
try {
    $null = Initialize-SceneCache $manifest $sceneFile $Api $Gpu
    $tagParts = @()
    if ($Clean) { $tagParts += 'clean' }
    if ($Motion) { $tagParts += $Motion }
    if ($Sky -eq 'texture') { $tagParts += 'sky' }
    if ($GridRule -ne 'cells') { $tagParts += "grid-$GridRule" }
    $runDir = Get-HarnessRunDir $Scene $Api $(if ($tagParts.Count) { '-' + ($tagParts -join '-') } else { '' })

    # The motion (fork options, see include/Motion.h in the fork): a motion file written from the manifest
    $appLines = @()
    $keepLights = @()
    $motionSeconds = 0
    $warmup = 0
    $recordFolder = ''
    $recordFrames = 0
    $motionFile = ''
    if ($Motion) {
        $motionFile = "$runDir/$Motion.motion"
        if ($timeOfDay) {
            & node "$root/tools/native/motion.ts" write --scene $Scene --day --seconds $DaySeconds --out $motionFile
            $motionSeconds = $DaySeconds
        } else {
            $definition = @($manifest.motions | Where-Object { $_.id -eq $Motion })
            if ($definition.Count -ne 1) { throw "Scene '$Scene' has no motion '$Motion' (its motions: $(@($manifest.motions | ForEach-Object { $_.id }) -join ', '); or time-of-day)." }
            if ($SettleKey -ge 0) { & node "$root/tools/native/motion.ts" write --scene $Scene --motion $Motion --settle-key $SettleKey --out $motionFile }
            else { & node "$root/tools/native/motion.ts" write --scene $Scene --motion $Motion --out $motionFile }
            $motionSeconds = [double](@($definition[0].keyframes)[-1].t)
            $keepLights += ($definition[0].target -replace '^light:', '')
        }
        if ($LASTEXITCODE -ne 0) { throw "Could not write the motion file for '$Motion'." }
        $warmup = if ($WarmupFrames -ge 0) { $WarmupFrames } elseif ($Record) { 600 } else { 60 }
        $appLines += "app.motionFile=$motionFile"
        # Watched: the motion runs in real time. Recorded: frame k is the motion at exactly k/60 s (DDGI's
        # clock on the web side), however long recording the frame takes.
        $appLines += "app.motionClock=$(if ($Record) { 'fixed' } else { 'wall' })"
        $appLines += 'app.motionFrameRate=60'
        $appLines += "app.motionWarmupFrames=$warmup"
        if ($recording) {
            $seconds = if ($RecordSeconds -gt 0) { $RecordSeconds } else { $motionSeconds }
            $recordFrames = [int][math]::Round($seconds * 60) + 1
            $recordFolder = if ($RecordDir) { $RecordDir.Replace('\', '/') } else { "$root/native/build/recordings/$Scene-$Motion$(if ($Log) { '-log' })$(if ($Tag) { "-$Tag" })" }
            if (Test-Path -LiteralPath $recordFolder) { Clear-FolderContents $recordFolder }
            New-Item -ItemType Directory -Force -Path $recordFolder | Out-Null
            $appLines += "app.recordDir=$recordFolder"
            $appLines += "app.recordFrames=$recordFrames"
            if ($Log) {
                # Nothing that waits for the GPU, so the window plays at its own (vsync) rate
                $RecordImageEvery = 0
                $appLines += 'app.recordGiMean=0'
            }
            $appLines += "app.recordImageEvery=$RecordImageEvery"
        }
    }

    $vsync = if ($NoVsync -or $Record) { 0 } else { 1 }
    $config = New-HarnessIni -Manifest $manifest -SceneFile $sceneFile -OutputDir $runDir -Gpu $Gpu `
        -Width $Width -Height $Height -Vsync $vsync -Clean:$Clean -Camera $Camera -Rays $Rays -HideUi:($unattended -or $Log) `
        -UpdateEveryFrame:([bool]$Motion) -SkyTexture $skyTexture -KeepLights $keepLights -AppLines $appLines -GridRule $GridRule
    $iniPath = "$runDir/$Scene.ini"
    [System.IO.File]::WriteAllText($iniPath, $config.Text)

    $environment = @{}
    if ($CaptureFrames -gt 0) { $environment['DDGI_CAPTURE_FRAMES'] = $CaptureFrames }
    # With -CaptureField the capture also writes the linear images, the probe field and the G-buffer (fork capture options; sky-check.ts reads them)
    if ($CaptureFrames -gt 0 -and $CaptureField) { $environment['DDGI_CAPTURE_LINEAR'] = 1; $environment['DDGI_CAPTURE_FIELD'] = 1 }
    if ($View -eq 'indirect') { $environment['DDGI_CAPTURE_MODE'] = 'indirect' }

    $cameraName = @($manifest.cameras)[$Camera - 1].name
    Write-Host "Scene       : $($manifest.title) ($Scene)"
    Write-Host "Camera      : bookmark $Camera, $cameraName"
    Write-Host "Probe volume: $($config.Grid.Counts -join ' x ') = $($config.Grid.Total) probes, $Rays rays per probe"
    Write-Host "API         : $Api$(if ($Clean) { ', clean mode' })"
    Write-Host "Sky         : $(if ($skyTexture) { "texture, $skyTexture" } else { 'flat (the sky file''s flat equivalent, or the manifest colour)' })"
    if ($Motion) { Write-Host "Motion      : $Motion, $motionSeconds s$(if ($Record) { ", recording $recordFrames frames at a fixed 1/60 s after $warmup warm-up frames" } elseif ($Log) { ", wall clock, logging $recordFrames frames" } else { ', wall clock' }), probes updated every frame" }
    Write-Host "Config      : $iniPath"

    $launched = Get-Date
    $windowStyle = if ($unattended) { 'Minimized' } else { 'Normal' }
    if ($unattended) { Write-Host 'Unattended run: the harness is started MINIMIZED and closes itself.' }
    else { Write-Host 'Opening a visible harness window now. This console keeps the GPU lock until you close the window (Ctrl-C here closes the harness too).' }
    $process = Start-Harness -Exe $exe -IniPath $iniPath -WorkingDirectory $runDir -Environment $environment -WindowStyle $windowStyle
    Write-Host "Started the harness (PID $($process.Id))."

    if ($CaptureFrames -gt 0) {
        $result = Wait-Harness $process 30
        $screenshot = "$($config.Screenshots)/R-BackBuffer.png"
        Assert-HarnessCompleted $runDir $launched @($screenshot) "capture run of $Scene ($result)"
        $harnessLog = Get-Content -LiteralPath "$runDir/log.txt" -Raw
        if ($harnessLog -match 'device: (.+)') { Write-Host "Device      : $($Matches[1].Trim())" }
        Write-Host "Screenshot  : $((Resolve-Path $screenshot).Path)"
    }
    elseif ($recording) {
        $result = Wait-Harness $process 60
        $framesCsv = "$recordFolder/frames.csv"
        Assert-HarnessCompleted $runDir $launched @($framesCsv) "recording of $Scene $Motion ($result)"
        $harnessLog = Get-Content -LiteralPath "$runDir/log.txt" -Raw
        if ($harnessLog -notmatch 'Recording complete') { throw "The recording of $Scene $Motion stopped early; see $runDir/log.txt." }
        $device = if ($harnessLog -match 'device: (.+)') { $Matches[1].Trim() } else { '' }
        Copy-Item -LiteralPath $motionFile -Destination "$recordFolder/$Motion.motion"
        if ($timeOfDay) { Copy-Item -LiteralPath "$runDir/$Motion-sky" -Destination "$recordFolder/$Motion-sky" -Recurse -Force }
        $fork = "$(git -C "$root/native/rtxgi-ddgi" rev-parse --short HEAD)".Trim()
        if ("$(git -C "$root/native/rtxgi-ddgi" status --porcelain)".Trim()) { $fork += '-dirty' }
        $images = @(Get-ChildItem -LiteralPath "$recordFolder/frames" -Filter 'frame-*.png' -ErrorAction SilentlyContinue | Sort-Object Name)
        $description = [ordered]@{
            format = 'webgpu-ddgi native motion recording, version 1 (tools/native/README.md, "Motions and recordings")'
            scene = $Scene
            motion = $Motion
            motionSeconds = $motionSeconds
            camera = $cameraName
            cameraBookmark = $Camera
            width = $Width
            height = $Height
            view = $View
            clean = [bool]$Clean
            sky = $Sky
            clock = $(if ($Log) { 'wall' } else { 'fixed' })
            frameSeconds = $(if ($Log) { $null } else { 1 / 60 })
            frameRate = 60
            gridRule = $GridRule
            warmupFrames = $warmup
            frames = $recordFrames
            imageEvery = $RecordImageEvery
            settleKey = $SettleKey
            images = $images.Count
            framesCsv = 'frames.csv'
            motionFile = "$Motion.motion"
            probeUpdate = 'every probe every frame (probeVariability.skipConverged=0)'
            probes = $config.Grid.Total
            probeCounts = $config.Grid.Counts
            raysPerProbe = $Rays
            device = $device
            harnessCommit = $fork
            recorded = (Get-Date).ToString('s')
        }
        [System.IO.File]::WriteAllText("$recordFolder/recording.json", ($description | ConvertTo-Json) + "`n")

        # A video of the saved images at their own rate (60 / imageEvery a second), when ffmpeg is on PATH.
        # Image n of the video is motion frame n x imageEvery; the concat list keeps that whatever the names.
        $ffmpeg = Get-Command ffmpeg -ErrorAction SilentlyContinue
        if ($ffmpeg -and $images.Count -gt 1) {
            $step = (Format-Number ([math]::Max(1, $RecordImageEvery) / 60) 6)
            $list = ($images | ForEach-Object { "file 'frames/$($_.Name)'`nduration $step" }) -join "`n"
            [System.IO.File]::WriteAllText("$recordFolder/video-frames.txt", $list + "`n")
            Push-Location $recordFolder
            try { & $ffmpeg.Source -hide_banner -loglevel error -y -f concat -safe 0 -i video-frames.txt -vsync cfr -r (60 / [math]::Max(1, $RecordImageEvery)) -c:v libx264 -pix_fmt yuv420p -crf 18 video.mp4 2>&1 | Out-Null }
            finally { Pop-Location }
            if (-not (Test-Path -LiteralPath "$recordFolder/video.mp4")) { Write-Host 'Video       : ffmpeg could not make video.mp4 (the PNG frames are complete).' }
            else { Write-Host "Video       : $recordFolder/video.mp4" }
        }

        # The played motion against the manifest's, at the same times (motion.ts reads recording.json's clock: on the
        # fixed clock the time is the frame number / 60, worked out there, not the harness's logged time)
        if ($timeOfDay) { & node "$root/tools/native/motion.ts" check --scene $Scene --day --seconds $DaySeconds --log $framesCsv }
        else {
            $firstKey = @(@($manifest.motions | Where-Object { $_.id -eq $Motion })[0].keyframes)[0]
            if ($SettleKey -lt 0 -and $firstKey.PSObject.Properties['position']) { & node "$root/tools/native/motion.ts" check --scene $Scene --motion $Motion --log $framesCsv }
        }
        if ($device) { Write-Host "Device      : $device" }
        Write-Host "Recording   : $recordFolder ($recordFrames frames, $($images.Count) images)"
    }
    else {
        # Interactive: keep the lock until the person closes the window.
        while (-not $process.WaitForExit(1000)) { }
        Write-Host "The harness has closed (exit code $($process.ExitCode))."
    }
} finally {
    # On Ctrl-C the harness we started would keep using the GPU without the lock, so close it
    # (only this PID), then give the lock back.
    Stop-StartedHarness $process
    Exit-BenchLock
    Write-Host 'Released the GPU lock.'
}
