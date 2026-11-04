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

.PARAMETER NoWait
  The script holds the shared GPU lock (%TEMP%\ddgi-gpu-bench.lock) for as long as the harness
  is open, so no benchmark runs while you look at a scene. If another GPU job holds the lock, the
  script says who and waits; with -NoWait it exits instead.

.EXAMPLE
  ./tools/native/play.ps1 -Scene sponza -Clean
  ./tools/native/play.ps1 -Scene two-room -Camera 2 -Api d3d12
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
    [switch]$NoVsync,
    [switch]$NoWait
)
$ErrorActionPreference = 'Stop'
. "$PSScriptRoot/native-lib.ps1"

$exe = Get-HarnessExe $Api
$manifest = Read-SceneManifest $Scene
$sceneFile = Resolve-SceneFile $manifest

# Hold the GPU lock for the whole session (cache build included), so no benchmark runs on the
# card while the harness is open. Released in the finally block: on a normal exit, on an error
# and on Ctrl-C.
if (-not (Enter-BenchLock -What "native play $Scene ($Api)" -NoWait:$NoWait)) { exit 2 }
$process = $null
try {
    $null = Initialize-SceneCache $manifest $sceneFile $Api $Gpu
    $runDir = Get-HarnessRunDir $Scene $Api $(if ($Clean) { '-clean' } else { '' })

    $vsync = if ($NoVsync) { 0 } else { 1 }
    $config = New-HarnessIni -Manifest $manifest -SceneFile $sceneFile -OutputDir $runDir -Gpu $Gpu `
        -Width $Width -Height $Height -Vsync $vsync -Clean:$Clean -Camera $Camera -Rays $Rays -HideUi:($CaptureFrames -gt 0)
    $iniPath = "$runDir/$Scene.ini"
    [System.IO.File]::WriteAllText($iniPath, $config.Text)

    $environment = @{}
    if ($CaptureFrames -gt 0) { $environment['DDGI_CAPTURE_FRAMES'] = $CaptureFrames }

    $cameraName = @($manifest.cameras)[$Camera - 1].name
    Write-Host "Scene       : $($manifest.title) ($Scene)"
    Write-Host "Camera      : bookmark $Camera, $cameraName"
    Write-Host "Probe volume: $($config.Grid.Counts -join ' x ') = $($config.Grid.Total) probes, $Rays rays per probe"
    Write-Host "API         : $Api$(if ($Clean) { ', clean mode' })"
    Write-Host "Config      : $iniPath"

    $launched = Get-Date
    $windowStyle = if ($CaptureFrames -gt 0) { 'Minimized' } else { 'Normal' }
    if ($CaptureFrames -gt 0) { Write-Host 'Capture run: the harness is started MINIMIZED and closes itself.' }
    else { Write-Host 'Opening a visible harness window now. This console keeps the GPU lock until you close the window (Ctrl-C here closes the harness too).' }
    $process = Start-Harness -Exe $exe -IniPath $iniPath -WorkingDirectory $runDir -Environment $environment -WindowStyle $windowStyle
    Write-Host "Started the harness (PID $($process.Id))."

    if ($CaptureFrames -gt 0) {
        $result = Wait-Harness $process 30
        $screenshot = "$($config.Screenshots)/R-BackBuffer.png"
        Assert-HarnessCompleted $runDir $launched @($screenshot) "capture run of $Scene ($result)"
        $log = Get-Content -LiteralPath "$runDir/log.txt" -Raw
        if ($log -match 'device: (.+)') { Write-Host "Device      : $($Matches[1].Trim())" }
        Write-Host "Screenshot  : $((Resolve-Path $screenshot).Path)"
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
