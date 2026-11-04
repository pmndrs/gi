<#
.SYNOPSIS
  Runs the harness's own 1024-frame GPU benchmark (the F4 benchmark) on our scenes and
  appends the per-pass GPU times to bench/results/native.csv.

.DESCRIPTION
  For each scene:
    1. Writes the .ini from the manifest in clean mode (no probe spheres, no RTAO, no dither),
       vsync off, 1920x1080, 256 rays per probe, probe grid from the manifest volume.
    2. Loads the scene once without timing so the texture cache exists (no lock needed).
    3. Takes the shared GPU timing lock (%TEMP%\ddgi-gpu-bench.lock), waits 300 frames for the
       probes to settle, records 1024 frames, exits, and gives the lock back. The harness is
       started MINIMIZED.
    4. Reads benchmarkGpu.csv, benchmarkCpu.csv, the probe-state dump and the log, and appends
       rows to the CSV.

  Probes are updated EVERY frame (fork ini option probeVariability.skipConverged=0). Upstream
  stops tracing and blending a volume once it has converged; with that, about 16 of the 1024
  frames contained probe work and the means were mostly empty frames. -SkipConverged restores
  the upstream behaviour (for showing that difference only; its rows say so).

  What each row means (all per frame):
    probes_active / probes_inactive   probe states after the run (classification); inactive
                                      probes trace only the 32 fixed rays and are not blended
    rays_dispatched_per_frame         probe_count x rays_per_probe (ray generation threads launched)
    rays_traced_per_frame             active x rays + inactive x min(32, rays)
    frames_probes_updated             frames (of 1024) in which the probe trace ran
    gpu_ms_<pass>, _median            GPU timestamps around that pass, mean and median
    cpu_ms_frame, _median             CPU time of the whole frame loop iteration
    mrays_per_s                       rays_traced_per_frame / median probe trace time
    scene_parts                       manifest parts loaded (multi-part scenes are merged into one glTF)

  Never run this while a browser GPU benchmark is running. The lock enforces that for scripts
  that also use it; if you time something by hand, take the lock yourself.

.PARAMETER Scene
  Scene ids. Default: the four scenes of T0.6 (bistro-exterior and bistro-interior both).

.PARAMETER Gpu
  Part of the GPU name. Default "3060" (the RTX 3060 Laptop GPU).

.PARAMETER ProbeSpacing
  Replaces the manifest probe spacing (metres), for sweeps over the probe count.

.PARAMETER Tag
  Added to the run folder name, so sweep runs do not overwrite each other.

.PARAMETER SkipConverged
  Upstream behaviour: stop updating the volume once it has converged. Not for real timing.

.PARAMETER HideUi
  Start with the harness settings panel hidden (no ImGui windows are built each frame).

.PARAMETER NoAppend
  Run and print, but leave bench/results/native.csv alone.

.EXAMPLE
  ./tools/native/bench.ps1 -Scene two-room,sponza
  ./tools/native/bench.ps1 -Scene sponza -Api d3d12
#>
param(
    [string[]]$Scene = @('two-room', 'cornell-box', 'sponza', 'bistro-exterior', 'bistro-interior'),
    [ValidateSet('vulkan', 'd3d12')][string]$Api = 'vulkan',
    [string]$Gpu = '3060',
    [int]$Rays = 256,
    [int]$Width = 1920,
    [int]$Height = 1080,
    [int]$Camera = 1,
    [int]$WarmupFrames = 300,
    [int]$TimeoutMinutes = 30,
    [int]$Attempts = 3,
    [string]$Csv = '',
    [double]$ProbeSpacing = 0,
    [string]$Tag = '',
    [switch]$SkipConverged,
    [switch]$HideUi,
    [switch]$NoAppend
)
$ErrorActionPreference = 'Stop'
. "$PSScriptRoot/native-lib.ps1"

# pwsh -File passes 'a,b' as one string
$Scene = @($Scene | ForEach-Object { $_ -split ',' } | Where-Object { $_ })
$root = Get-RepoRoot
if (-not $Csv) { $Csv = "$root/bench/results/native.csv" }
$exe = Get-HarnessExe $Api

# The commit column is the FORK's commit (the harness that was measured). '-dirty' when the fork has
# uncommitted changes, or when tools/native has (these scripts write the .ini, so they shape the run
# too; untracked files there count).
function Get-HarnessCommit {
    $hash = (& git -C "$root/native/rtxgi-ddgi" rev-parse --short HEAD).Trim()
    $forkDirty = & git -C "$root/native/rtxgi-ddgi" status --porcelain --untracked-files=no
    $scriptsDirty = & git -C $root status --porcelain --untracked-files=all -- tools/native
    if ($forkDirty -or $scriptsDirty) { $hash += '-dirty' }
    return $hash
}

# Appends one scene's rows straight away, so a later scene that fails cannot lose them.
function Add-NativeCsvRows([string[]]$Rows) {
    if ($NoAppend -or $Rows.Count -eq 0) { return }
    if (-not (Test-Path -LiteralPath $Csv)) { Set-Content -LiteralPath $Csv -Value 'date,commit,device,browser,scene,metric,value' }
    Add-Content -LiteralPath $Csv -Value $Rows
    Write-Host "Appended $($Rows.Count) rows for this scene to $Csv"
}

# "NVIDIA GeForce RTX 3060 Laptop GPU" -> "RTX 3060 Laptop"
function Get-ShortDeviceName([string]$Name) {
    return ($Name.Trim() -replace '^NVIDIA GeForce ', '' -replace ' GPU$', '').Trim()
}

function Read-BenchmarkCsv([string]$Path) {
    $lines = Get-Content -LiteralPath $Path
    $averages = ($lines[0] -split ',')
    $names = ($lines[1] -split ',')
    $stats = [ordered]@{}
    for ($i = 1; $i -lt $names.Count; $i++) {
        if (-not $names[$i]) { continue }
        $values = New-Object System.Collections.Generic.List[double]
        foreach ($row in $lines[2..($lines.Count - 1)]) {
            $cells = $row -split ','
            if ($cells.Count -gt $i -and $cells[$i] -ne '') { $values.Add([double]::Parse($cells[$i], $script:Invariant)) }
        }
        $sorted = $values | Sort-Object
        $median = if ($sorted.Count) { @($sorted)[[int][math]::Floor(($sorted.Count - 1) / 2)] } else { 0 }
        $stats[$names[$i].Trim()] = @{
            Mean = [double]::Parse($averages[$i], $script:Invariant)
            Median = $median
            Frames = $values.Count
        }
    }
    return $stats
}

# Probe states from the probe-data texture the benchmark dumps when it ends (one PNG per array
# layer; the alpha channel is the state: 0 = active, 1 = inactive, so 0 or 255 in the PNG).
function Get-ProbeStateCounts([string]$ScreenshotDir) {
    $files = @(Get-ChildItem -LiteralPath $ScreenshotDir -File | Where-Object { $_.Name -match 'Probe-?Data-Layer-\d+\.png$' })
    if ($files.Count -eq 0) { return $null }
    $active = 0
    $inactive = 0
    foreach ($file in $files) {
        $histogram = & magick $file.FullName -alpha extract -format '%c' histogram:info:-
        foreach ($line in $histogram) {
            if ($line -match '^\s*(\d+):\s*\((\d+)') {
                if ([int]$Matches[2] -lt 128) { $active += [int]$Matches[1] } else { $inactive += [int]$Matches[1] }
            }
        }
    }
    return @{ Active = $active; Inactive = $inactive }
}

function ConvertTo-MetricName([string]$StatName) {
    return 'gpu_ms_' + (($StatName.ToLower() -replace '[^a-z0-9]+', '_').Trim('_'))
}

$commit = Get-HarnessCommit
$date = Get-Date -Format 'yyyy-MM-dd'
$browserColumn = if ($Api -eq 'vulkan') { 'native' } else { 'native-d3d12' }
$appendedRows = 0
# Runs with a non-default ray count or probe spacing are sweeps: they get a `sweep,1` row so a
# filter can keep them apart from the default runs.
$isSweep = ($Rays -ne 256) -or ($ProbeSpacing -gt 0)
$summary = New-Object System.Collections.Generic.List[object]
$failedScenes = @()

# Hold the GPU lock for the whole run: every cache build and every timed scene.
# The lock is not first come, first served: giving it back between steps would hand it to
# whichever job polls next. The Use-BenchLock calls inside reuse this hold.
$null = Enter-BenchLock -What "native bench $($Scene -join ',') ($Api)"
try {
    foreach ($sceneId in $Scene) {
        Write-Host "=== $sceneId ($Api) ==="
        try {
            $manifest = Read-SceneManifest $sceneId
            $sceneFile = Resolve-SceneFile $manifest
            $runDir = Get-HarnessRunDir $sceneId $Api "-bench$Tag"
            $config = New-HarnessIni -Manifest $manifest -SceneFile $sceneFile -OutputDir $runDir -Gpu $Gpu `
                -Width $Width -Height $Height -Vsync 0 -Clean -Camera $Camera -Rays $Rays -AutoBench -WarmupFrames $WarmupFrames `
                -UpdateEveryFrame:(-not $SkipConverged) -ProbeSpacing $ProbeSpacing -HideUi:$HideUi
            Write-Host "Probe grid: $($config.Grid.Counts -join ' x ') = $($config.Grid.Total) probes, $Rays rays per probe, probes updated $(if ($SkipConverged) { 'until converged (upstream)' } else { 'every frame' })"
            $iniPath = "$runDir/$sceneId.ini"
            [System.IO.File]::WriteAllText($iniPath, $config.Text)

            # 2. Untimed cache build, only if the cache is missing (minimized, under the lock)
            $null = Initialize-SceneCache $manifest $sceneFile $Api $Gpu $TimeoutMinutes

            # 3. Timed run under the lock
            # Old outputs must not be mistaken for this run's
            Get-ChildItem -LiteralPath $config.Screenshots -File -ErrorAction SilentlyContinue | Remove-Item
            # The harness can lose the GPU on the big Bistro scenes (see README); retry a few times.
            $csvPath = "$($config.Screenshots)/benchmarkGpu.csv"
            for ($attempt = 1; $attempt -le $Attempts -and -not (Test-Path -LiteralPath $csvPath); $attempt++) {
                Use-BenchLock -What "native bench $sceneId ($Api)" -Action {
                    Write-Host "The harness is started MINIMIZED for timing (no window in your way)."
                    Write-Host "Timing $sceneId, attempt $attempt ($WarmupFrames warm-up frames, then 1024 recorded)..."
                    $script:launched = Get-Date
                    $run = $null
                    try {
                        $run = Start-Harness -Exe $exe -IniPath $iniPath -WorkingDirectory $runDir -WindowStyle Minimized
                        $result = Wait-Harness $run $TimeoutMinutes
                        if ($result -ne 'exited') { Write-Warning "Benchmark for $sceneId ended with: $result" }
                    } finally {
                        # On Ctrl-C the harness would keep using the GPU after the lock is given back:
                        # close the one we started (this PID only) before the lock is released.
                        Stop-StartedHarness $run
                    }
                }
            }
            $csvPath = "$($config.Screenshots)/benchmarkGpu.csv"
            try {
                Assert-HarnessCompleted $runDir $script:launched @($csvPath) "timing run of $sceneId after $Attempts attempts"
            } catch {
                # One known case still counts: the benchmark finished and wrote its CSVs, and the harness
                # failed only afterwards, while writing its texture dumps (D3D12 on bistro-exterior fails
                # on the probe-variability-average texture). The timings are complete then; the
                # probe-data dump that the active-probe count needs is written before that texture.
                $logText = if (Test-Path -LiteralPath "$runDir/log.txt") { Get-Content -LiteralPath "$runDir/log.txt" -Raw } else { '' }
                $benchmarkDone = $logText -match 'Wrote benchmark results to csv' -and (Test-Path -LiteralPath $csvPath) -and
                    (Get-Item -LiteralPath $csvPath).LastWriteTime -ge $script:launched
                if (-not $benchmarkDone) {
                    Write-Warning $_.Exception.Message
                    $failedScenes += $sceneId
                    continue
                }
                Write-Warning "$sceneId ($Api): the 1024-frame benchmark finished and wrote its CSVs, but the harness failed afterwards while writing its texture dumps. Recording the timings; the failure is: $($_.Exception.Message)"
            }

            # 4. Parse and collect
            $log = Get-Content -LiteralPath "$runDir/log.txt" -Raw
            $deviceName = if ($log -match 'device: (.+)') { Get-ShortDeviceName $Matches[1] } else { 'unknown' }
            $stats = Read-BenchmarkCsv $csvPath
            $cpuStats = Read-BenchmarkCsv "$($config.Screenshots)/benchmarkCpu.csv"
            $rows = New-Object System.Collections.Generic.List[object]
            $grid = $config.Grid

            # Frames in which the probe trace actually ran (a timestamp pair around an empty pass reads
            # about 0.001 ms). The column is found by its name, "Probe Trace". Without it the check below
            # would count nothing and pass by accident, so a missing column stops the scene.
            $traceColumn = [array]::IndexOf(@((Get-Content -LiteralPath $csvPath -TotalCount 2)[1] -split ',' | ForEach-Object { $_.Trim() }), 'Probe Trace')
            if ($traceColumn -lt 1) {
                Write-Warning "$sceneId : benchmarkGpu.csv has no 'Probe Trace' column, so it cannot be checked that the probes were updated every frame. Not recorded."
                $failedScenes += $sceneId
                continue
            }
            $traceValues = @(Get-Content -LiteralPath $csvPath | Select-Object -Skip 2 | ForEach-Object {
                    $cells = $_ -split ','
                    if ($cells.Count -gt $traceColumn -and $cells[$traceColumn] -ne '') { [double]::Parse($cells[$traceColumn], $script:Invariant) }
                })
            $framesUpdated = @($traceValues | Where-Object { $_ -gt 0.005 }).Count
            if ($traceValues.Count -eq 0) {
                Write-Warning "$sceneId : the 'Probe Trace' column of benchmarkGpu.csv has no values. Not recorded."
                $failedScenes += $sceneId
                continue
            }
            if (-not $SkipConverged -and $framesUpdated -lt 0.99 * $traceValues.Count) {
                Write-Warning "$sceneId : the probe trace ran in only $framesUpdated of $($traceValues.Count) frames although every frame should update. Not recorded."
                $failedScenes += $sceneId
                continue
            }

            $states = Get-ProbeStateCounts $config.Screenshots
            if ($null -eq $states) { throw "$sceneId : no probe-data dump in $($config.Screenshots), so the active probe count is unknown." }
            $raysTraced = $states.Active * $Rays + $states.Inactive * [math]::Min(32, $Rays)
            $traceMedian = if ($stats.Contains('Probe Trace')) { $stats['Probe Trace'].Median } else { 0 }

            $fixed = [ordered]@{
                probes_x = $grid.Counts[0]; probes_y = $grid.Counts[1]; probes_z = $grid.Counts[2]
                probe_count = $grid.Total
                probes_active = $states.Active
                probes_inactive = $states.Inactive
                probe_spacing_min_m = [math]::Round($grid.MinSpacing, 4)
                rays_per_probe = $Rays
                rays_dispatched_per_frame = $grid.Total * $Rays
                rays_traced_per_frame = $raysTraced
                probe_update_every_frame = $(if ($SkipConverged) { 0 } else { 1 })
                frames_probes_updated = $framesUpdated
                resolution_width = $Width; resolution_height = $Height
                benchmark_frames = $traceValues.Count; warmup_frames = $WarmupFrames
                camera_bookmark = $Camera
                # How many manifest parts the harness loaded (several are merged into one glTF).
                # Sponza rows without this row are older runs of Intel's base package alone, before
                # the curtains part was added to the manifest.
                scene_parts = @($manifest.parts).Count
            }
            if ($isSweep) { $fixed['sweep'] = 1 }
            foreach ($key in $fixed.Keys) { $rows.Add(@($key, $fixed[$key])) }
            # Passes that clean mode switches off are left out (always 0)
            $skipped = @('Path Tracing', 'DDGI Probe Vis', 'DDGI Texture Vis', 'RTAO')
            foreach ($name in $stats.Keys) {
                if ($skipped -contains $name) { continue }
                $metric = ConvertTo-MetricName $name
                $rows.Add(@($metric, [math]::Round($stats[$name].Mean, 4)))
                $rows.Add(@("${metric}_median", [math]::Round($stats[$name].Median, 4)))
            }
            if ($cpuStats.Contains('Frame')) {
                $rows.Add(@('cpu_ms_frame', [math]::Round($cpuStats['Frame'].Mean, 3)))
                $rows.Add(@('cpu_ms_frame_median', [math]::Round($cpuStats['Frame'].Median, 3)))
            }
            if ($traceMedian -gt 0) { $rows.Add(@('mrays_per_s', [math]::Round($raysTraced / $traceMedian / 1000, 0))) }
            $sceneRows = foreach ($row in $rows) {
                $value = if ($row[1] -is [double]) { $row[1].ToString('0.####', $script:Invariant) } else { [string]$row[1] }
                "$date,$commit,$deviceName,$browserColumn,$sceneId,$($row[0]),$value"
            }
            Add-NativeCsvRows @($sceneRows)
            $appendedRows += @($sceneRows).Count

            $pick = { param($n) if ($stats.Contains($n)) { [math]::Round($stats[$n].Mean, 3) } else { $null } }
            $summary.Add([pscustomobject]@{
                scene = $sceneId; api = $Api; device = $deviceName
                probes = "$($grid.Counts -join 'x') = $($grid.Total)"
                active = $states.Active
                rays = $Rays
                updated = $framesUpdated
                trace_ms = & $pick 'Probe Trace'; blend_ms = & $pick 'Blend'; relocate_ms = & $pick 'Relocate'
                classify_ms = & $pick 'Classify'; lighting_ms = & $pick 'Lighting'; variability_ms = & $pick 'Variability'
                ddgi_ms = & $pick 'DDGI'; frame_ms = & $pick 'Frame'
                trace_median = [math]::Round($traceMedian, 3)
                cpu_frame_ms = $(if ($cpuStats.Contains('Frame')) { [math]::Round($cpuStats['Frame'].Median, 1) } else { $null })
            })
        } catch {
            # Anything unexpected in one scene (a missing file, a parse error) fails that scene only;
            # the scenes before it are already in the CSV and the ones after it still run.
            Write-Warning "$sceneId ($Api) failed: $($_.Exception.Message)"
            $failedScenes += $sceneId
        }
    }
} finally {
    Exit-BenchLock
}

if ($appendedRows -gt 0) { Write-Host "Appended $appendedRows rows in total to $Csv" }
$summary | Format-Table -AutoSize
if ($failedScenes.Count -gt 0) {
    # Write-Host, not Write-Error: with $ErrorActionPreference = 'Stop' Write-Error would throw
    # before the exit code is set.
    Write-Host "Timing FAILED for: $($failedScenes -join ', '). Nothing was recorded for them; the other scenes' rows are in the CSV." -ForegroundColor Red
    exit 1
}
