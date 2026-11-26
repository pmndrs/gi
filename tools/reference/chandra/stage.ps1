param([Parameter(Mandatory)][string] $SourceCheckout, [Parameter(Mandatory)][string] $BinDirectory)
$ErrorActionPreference = 'Stop'
$gi = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../../..'))
$runtime = Join-Path $gi 'bench/screenshots/chandra-local'
foreach ($folder in 'bin','assets','scenes') {
    [void][IO.Directory]::CreateDirectory((Join-Path $runtime $folder))
}
foreach ($name in 'capture.exe','chandra.exe','luisa_nvrtc.exe') {
    Copy-Item -LiteralPath (Join-Path $BinDirectory $name) -Destination (Join-Path $runtime "bin/$name")
}
Get-ChildItem -LiteralPath $BinDirectory -Filter '*.dll' -File | ForEach-Object {
    Copy-Item -LiteralPath $_.FullName -Destination (Join-Path $runtime "bin/$($_.Name)")
}
Get-ChildItem -LiteralPath $BinDirectory -Filter '*.data' -Directory -Force | ForEach-Object {
    Copy-Item -LiteralPath $_.FullName -Destination (Join-Path $runtime 'bin') -Recurse -Force
}
foreach ($asset in 'assets/sponza.obj','assets/sponza.mtl','scenes/sponza.json') {
    Copy-Item -LiteralPath (Join-Path $SourceCheckout $asset) -Destination (Join-Path $runtime $asset)
}
# The unmodified interactive binary chooses this filename inside main(). Give its local runtime
# the same Sponza geometry, camera, lights and black environment used by the headless capture.
$scene = Get-Content -Raw -LiteralPath (Join-Path $runtime 'scenes/sponza.json') | ConvertFrom-Json
$scene.world.envmap = 'assets/black.png'
$scene.world.envmap_scale = 0
$scene.world.sun_color = @(0, 0, 0)
$utf8 = New-Object Text.UTF8Encoding($false)
[IO.File]::WriteAllText((Join-Path $runtime 'scenes/sanmiguel_alt.json'), ($scene | ConvertTo-Json -Depth 12), $utf8)
$blackPng = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGNgYGAAAAAEAAH2FzhVAAAAAElFTkSuQmCC'
[IO.File]::WriteAllBytes((Join-Path $runtime 'assets/black.png'), [Convert]::FromBase64String($blackPng))
Write-Output "Local Chandra runtime staged in $runtime"
