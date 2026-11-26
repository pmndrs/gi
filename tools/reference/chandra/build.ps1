param(
    [string] $Root = (Join-Path ([IO.Path]::GetTempPath()) 'gi-chandra-reference'),
    [string] $VisualStudio,
    [string] $Rustup,
    [switch] $ChildBuild
)
$ErrorActionPreference = 'Stop'
if (!$ChildBuild) {
    # VS DevShell sets dozens of compiler variables. Keep the caller's environment intact.
    $shell = if (Test-Path (Join-Path $PSHOME 'pwsh.exe')) { 'pwsh.exe' } else { 'powershell.exe' }
    $parameters = @('-NoProfile', '-File', $PSCommandPath, '-Root', $Root, '-ChildBuild')
    if ($VisualStudio) { $parameters += @('-VisualStudio', $VisualStudio) }
    if ($Rustup) { $parameters += @('-Rustup', $Rustup) }
    & (Join-Path $PSHOME $shell) @parameters
    if ($LASTEXITCODE -ne 0) { throw "Isolated build failed with exit code $LASTEXITCODE" }
    return
}
$Root = [IO.Path]::GetFullPath($Root)
[void] [IO.Directory]::CreateDirectory($Root)

function Run([string] $Program, [string[]] $Arguments) {
    & $Program @Arguments | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "$Program failed with exit code $LASTEXITCODE" }
}

function Checkout([string] $Project, [string] $Commit) {
    $folder = Join-Path $Root $Project
    if (!(Test-Path -LiteralPath $folder)) {
        Run git @('clone', '--no-checkout', "https://github.com/entropylost/$Project.git", $folder)
        Run git @('-C', $folder, 'checkout', '--detach', $Commit)
    }
    $actual = & git -C $folder rev-parse HEAD
    if ($LASTEXITCODE -ne 0 -or $actual -ne $Commit) { throw "Expected $Project at $Commit in $folder" }
    return $folder
}

$source = Checkout 'chandra' '9af00e8de135a4126cb137d1ec62b451bd6b8e37'
$null = Checkout 'sefirot' '571a9461897b7c8d52df0caa1d533d6c5e44ad9c'
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'capture.rs') -Destination (Join-Path $source 'src/bin/capture.rs')

if (!$Rustup) { $Rustup = Join-Path $env:USERPROFILE '.cargo/bin/rustup.exe' }
if (!(Test-Path -LiteralPath $Rustup)) { throw 'Set -Rustup to your rustup.exe.' }
$env:RUSTUP_HOME = Join-Path $Root 'rustup'
$env:CARGO_HOME = Join-Path $Root 'cargo'
$env:CARGO_TARGET_DIR = Join-Path $Root 'target'
$env:RUSTUP_TOOLCHAIN = 'nightly-2026-11-19'
Run $Rustup @('toolchain','install',$env:RUSTUP_TOOLCHAIN,'--profile','minimal','--no-self-update')

if (!$VisualStudio) {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
    $VisualStudio = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
}
Import-Module (Join-Path $VisualStudio 'Common7/Tools/Microsoft.VisualStudio.DevShell.dll')
Enter-VsDevShell -VsInstallPath $VisualStudio -SkipAutomaticLocation -DevCmdArguments '-arch=x64 -host_arch=x64'
$cmakeTools = Join-Path $VisualStudio 'Common7/IDE/CommonExtensions/Microsoft/CMake'
$env:CMAKE = Join-Path $cmakeTools 'CMake/bin/cmake.exe'
Remove-Item Env:CMAKE_TOOLCHAIN_FILE -ErrorAction SilentlyContinue

# The original CUDA backend requires CUDA >=12.1. Only expand developer archives under Root.
$cuda = Join-Path $Root 'cuda12'
[void] [IO.Directory]::CreateDirectory($cuda)
$base = 'https://developer.download.nvidia.com/compute/cuda/redist/'
$manifestPath = Join-Path $cuda 'redistrib_12.4.1.json'
Run curl.exe @('-fsSL', ($base + 'redistrib_12.4.1.json'), '-o', $manifestPath)
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
foreach ($name in 'cuda_nvcc', 'cuda_cudart', 'cuda_nvrtc', 'cuda_cccl', 'cuda_nvtx', 'libnvjitlink') {
    $entry = $manifest.$name.'windows-x86_64'
    $zip = Join-Path $cuda ($name + '.zip')
    if (!(Test-Path -LiteralPath $zip)) { Run curl.exe @('-fsSL', ($base + $entry.relative_path), '-o', $zip) }
    if ((Get-FileHash -LiteralPath $zip -Algorithm SHA256).Hash -ne $entry.sha256) { throw "SHA-256 mismatch for $name" }
    Run tar.exe @('-xf', $zip, '--strip-components', '1', '-C', $cuda)
}
$env:CUDA_PATH = $cuda
$env:CUDAToolkit_ROOT = $cuda
$env:PATH = ((Join-Path $cuda 'bin'), (Split-Path $Rustup), (Split-Path $env:CMAKE), (Join-Path $cmakeTools 'Ninja'), $env:PATH) -join ';'
$env:GIT_CONFIG_COUNT = '1'
$env:GIT_CONFIG_KEY_0 = 'core.longpaths'
$env:GIT_CONFIG_VALUE_0 = 'true'
$env:CARGO_NET_GIT_FETCH_WITH_CLI = 'true'
$env:CMAKE_BUILD_PARALLEL_LEVEL = '4'
$env:CARGO_BUILD_JOBS = '4'
Push-Location $source
try {
    Run (Join-Path (Split-Path $Rustup) 'cargo.exe') @('build','--locked','--release','--bin','capture','--bin','chandra')
} finally {
    Pop-Location
}
& (Join-Path $PSScriptRoot 'build-nvrtc.ps1') -CargoHome $env:CARGO_HOME -Cuda $cuda -BinDirectory (Join-Path $env:CARGO_TARGET_DIR 'release')
& (Join-Path $PSScriptRoot 'stage.ps1') -SourceCheckout $source -BinDirectory (Join-Path $env:CARGO_TARGET_DIR 'release')
