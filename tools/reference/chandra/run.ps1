param([string] $BuildRoot, [string] $OutputDirectory, [switch] $Interactive)
$ErrorActionPreference = 'Stop'
$gi = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../../..'))
if ($BuildRoot -and !$Interactive) {
    $source = Join-Path $BuildRoot 'chandra'
    $bin = Join-Path $BuildRoot 'target/release'
    $cudaBin = Join-Path $BuildRoot 'cuda12/bin'
} else {
    $source = Join-Path $gi 'bench/screenshots/chandra-local'
    $bin = Join-Path $source 'bin'
    $cudaBin = $null
}
if (!$OutputDirectory) { $OutputDirectory = Join-Path $gi 'bench/references/chandra-sponza' }
$OutputDirectory = [IO.Path]::GetFullPath($OutputDirectory)
$capture = Join-Path $bin 'capture.exe'
if (!(Test-Path -LiteralPath $capture)) { throw 'Run build.ps1 and supply its -Root as -BuildRoot, or use a staged local runtime.' }
$previousPath = $env:PATH
$env:PATH = $bin + ';' + $env:PATH
if ($cudaBin) { $env:PATH = $cudaBin + ';' + $env:PATH }
Push-Location $source
try {
    if ($Interactive) {
        & node (Join-Path $gi 'tools/reference/bench-lock.ts') run --what chandra-interactive --no-cool-wait -- (Join-Path $bin 'chandra.exe')
    } else {
        & node (Join-Path $gi 'tools/reference/bench-lock.ts') run --what chandra-capture --no-cool-wait -- $capture $OutputDirectory
    }
    if ($LASTEXITCODE -ne 0) { throw "Chandra capture failed with exit code $LASTEXITCODE" }
} finally {
    Pop-Location
    $env:PATH = $previousPath
}
if ($Interactive) { return }
$env:GI_CHANDRA_ROOT = $source
$env:GI_CHANDRA_LINEAR = Join-Path $OutputDirectory 'diffuse.rgb32f'
$env:GI_CHANDRA_SPECULAR = Join-Path $OutputDirectory 'specular.rgb32f'
$env:GI_CHANDRA_PRIMARY = Join-Path $OutputDirectory 'primary.u8'
Write-Output "Reference images and the alignment mask are in $OutputDirectory."
