param(
 [ValidateSet('gi','pt')][string]$Renderer='gi',
 [ValidateRange(0,4)][int]$Scene=4,
 [string]$ExternalScene='',
 [ValidateRange(-1,64)][int]$Camera=0,
 [double[]]$CameraPosition,
 [double[]]$CameraLookAt,
 [ValidateRange(0,7)][int]$EnvironmentMap=2,
 [ValidateRange(64,7680)][int]$Width=1280,
 [ValidateRange(64,4320)][int]$Height=720,
 [switch]$Interactive,
 [ValidateRange(1,100000)][int]$Frames=120,
 [ValidateRange(0,100000)][int]$TimingFirstFrame=0,
 [string]$RunName='',
 [ValidateRange(1,1800)][int]$TimeoutSeconds=300,
 # Extra Capsaicin render options, e.g. 'taa_enable=false'. Passed through --render-options.
 [string]$RenderOptions='',
 # Debug view or AOV to save instead of the final colour, e.g. 'GlobalIllumination'. Needs the local
 # scene_viewer patch tools/reference/capsaicin/debug-view.patch (adds --debug-view).
 [string]$DebugView='',
 # First frame (0-based) whose image is saved; every frame from here to the end is dumped. Default: last frame only.
 [ValidateRange(-1,100000)][int]$ImageFirstFrame=-1
)
$ErrorActionPreference='Stop'
# Capsaicin v1.3 (GI-1.2 renderer index 0, Reference Path Tracer index 1). Scenes are resolved relative to the CWD,
# so the process runs from the Capsaicin source root. Run only one instance at a time.
$taskRoot=(Resolve-Path (Join-Path $PSScriptRoot '../../..')).Path
$taskSource=Join-Path $taskRoot 'native/capsaicin'
$taskExe=Join-Path $taskRoot 'native/build/capsaicin/bin/Release/scene_viewer.exe'
if(-not (Test-Path -LiteralPath $taskExe)){throw "Build first: $taskExe not found (see research/gi-1.0/native-capsaicin.md)"}
if($Renderer -eq 'pt'){$taskRendererIndex=1}else{$taskRendererIndex=0}
if([string]::IsNullOrEmpty($RunName)){$RunName=(Get-Date -Format 'yyyyMMdd-HHmmss')+'-'+$Renderer}
if($RunName -notmatch '^[a-zA-Z0-9_-]+$'){throw 'RunName must be alphanumeric, dash or underscore'}
$taskOutput=Join-Path $taskRoot "artifacts/native/capsaicin/runs/$RunName"
if(Test-Path -LiteralPath $taskOutput){throw 'Use a new RunName'}
New-Item -ItemType Directory -Path $taskOutput -Force | Out-Null
$taskArgs=@('--width',$Width,'--height',$Height,'--start-renderer-index',$taskRendererIndex,'--start-environment-map-index',$EnvironmentMap)
if($ExternalScene){
 $taskScenePath=(Resolve-Path -LiteralPath $ExternalScene).Path
 $taskArgs+=@('--load-external-scene',$taskScenePath)
}else{$taskArgs+=@('--start-scene-index',$Scene)}
# Camera 0 is the built-in "User" camera (90 deg vertical FOV, no CLI flag to change it). Cameras 1.. are the glTF
# cameras of the loaded scene and carry their own yfov, which is how we match our manifest FOV exactly
# (see research/gi-1.0/native-capsaicin.md, "Our scenes in Capsaicin"). CameraPosition/LookAt only work with camera 0.
if($Camera -ge 0){$taskArgs+=@('--start-camera-index',$Camera)}
if($RenderOptions){$taskArgs+=@('--render-options')+($RenderOptions -split '\s+'|Where-Object{$_})}
if($DebugView){$taskArgs+=@('--debug-view',$DebugView)}
if($CameraPosition){if($CameraPosition.Count -ne 3){throw 'CameraPosition needs 3 values'};$taskArgs+=@('--user-camera-position')+($CameraPosition|ForEach-Object{[string]$_})}
if($CameraLookAt){if($CameraLookAt.Count -ne 3){throw 'CameraLookAt needs 3 values'};$taskArgs+=@('--user-camera-lookat')+($CameraLookAt|ForEach-Object{[string]$_})}
if(-not $Interactive){
 $taskArgs+=@('--benchmark-mode','--benchmark-frames',$Frames,'--benchmark-first-frame',$TimingFirstFrame)
 if($ImageFirstFrame -ge 0){$taskArgs+=@($ImageFirstFrame)}
 $taskArgs+=@(
  '--benchmark-capture-frames','true','--benchmark-capture-timings','true','--benchmark-suffix',$Renderer,
  '--dump-folder',($taskOutput.Replace('\','/')+'/'))
}
$taskArgs=$taskArgs|ForEach-Object{[string]$_}
# The exe asks DXGI for the high-performance adapter, which is the RTX 3060 on this laptop. Confirm by watching
# nvidia-smi utilisation while it runs; the app's own "Created Direct3D12 device" line only shows on a parent console.
$taskBaseMem=[int]((& nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>$null|Select-Object -First 1).Trim())
$taskStart=Get-Date
$taskProc=Start-Process -FilePath $taskExe -ArgumentList $taskArgs -WorkingDirectory $taskSource -PassThru
$taskMaxUtil=0
$taskMaxMem=$taskBaseMem
$taskDeadline=(Get-Date).AddSeconds($TimeoutSeconds)
while(-not $taskProc.HasExited){
 if((Get-Date) -gt $taskDeadline){Stop-Process -Id $taskProc.Id -Force;throw "Timed out after $TimeoutSeconds s"}
 Start-Sleep -Milliseconds 500
 # One query for both counters. Memory is whole-GPU used memory, so subtract the idle baseline for the app's share.
 $taskSmi=(& nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>$null|Select-Object -First 1)
 if($taskSmi -match '^\s*(\d+)\s*,\s*(\d+)'){$taskMaxUtil=[Math]::Max($taskMaxUtil,[int]$Matches[1]);$taskMaxMem=[Math]::Max($taskMaxMem,[int]$Matches[2])}
 if($Interactive){$taskDeadline=(Get-Date).AddSeconds($TimeoutSeconds)}
}
$taskMeta=[ordered]@{run=$RunName;renderer=$Renderer;scene=$(if($ExternalScene){$ExternalScene}else{$Scene});camera=$Camera;width=$Width;height=$Height;
 frames=$(if($Interactive){$null}else{$Frames});exitCode=$taskProc.ExitCode;wallSeconds=[Math]::Round(((Get-Date)-$taskStart).TotalSeconds,1);
 nvidiaMaxUtilizationPercent=$taskMaxUtil;nvidiaBaselineMiB=$taskBaseMem;nvidiaPeakMiB=$taskMaxMem;nvidiaPeakDeltaMiB=($taskMaxMem-$taskBaseMem);args=($taskArgs -join ' ')}
$taskMeta|ConvertTo-Json|Set-Content -LiteralPath (Join-Path $taskOutput 'run.json')
if($taskMaxUtil -eq 0 -and -not $Interactive){Write-Warning 'NVIDIA GPU showed 0% utilisation; run may have used another adapter'}
Write-Output "Output: $taskOutput (exit $($taskProc.ExitCode), $($taskMeta.wallSeconds) s, NVIDIA peak $taskMaxUtil%, VRAM +$($taskMaxMem-$taskBaseMem) MiB)"
