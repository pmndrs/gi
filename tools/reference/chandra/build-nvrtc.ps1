param([string] $CargoHome, [string] $Cuda, [string] $BinDirectory)
$ErrorActionPreference = 'Stop'
$sources = @(Get-ChildItem -LiteralPath (Join-Path $CargoHome 'git/checkouts') -Directory -Filter 'luisa-compute-rs-*')
if ($sources.Count -ne 1) { throw 'Expected one pinned Luisa checkout in the isolated Cargo home.' }
$original = Join-Path $sources[0].FullName '374e0b3/luisa_compute_sys/LuisaCompute/src/backends/cuda/cuda_nvrtc_compiler.cpp'
$text = [IO.File]::ReadAllText($original)
$needle = 'int nvrtc_argc = argc - 1;'
if (!$text.Contains($needle)) { throw "Unexpected Luisa NVRTC helper source at $original" }
# CUDA 12.4's Windows NVRTC passes this optional scheduling flag to llc, which rejects it.
# All shader source, target options, OptiX entry points, and optimization flags pass through.
$text = $text.Replace($needle, @'
int nvrtc_argc = 0;
    for (int n = 1; n < argc; n++) {
        if (strcmp(argv[n], "-split-compile=0") != 0) argv[++nvrtc_argc] = argv[n];
    }
'@)
# The helper protocol carries NUL-delimited source and binary OptiX IR over the pipes.
$text = "#include <fcntl.h>`n#include <io.h>`n" + $text.Replace('    // read filename and source code', @'
    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);
    // read filename and source code
'@)
$scratch = Join-Path (Split-Path $BinDirectory) 'chandra-nvrtc'
[void][IO.Directory]::CreateDirectory($scratch)
$patched = Join-Path $scratch 'nvrtc-windows.cpp'
[IO.File]::WriteAllText($patched, $text)
$output = Join-Path $BinDirectory 'luisa_nvrtc.exe'
$libraries = Join-Path $Cuda 'lib/x64'
Push-Location $scratch
try {
    & cl.exe /nologo /MD /O2 /DNDEBUG $patched "/Fe:$output" /link "/LIBPATH:$libraries" nvrtc_static.lib nvrtc-builtins_static.lib nvptxcompiler_static.lib cuda.lib nvJitLink_static.lib Ws2_32.lib kernel32.lib user32.lib gdi32.lib winspool.lib shell32.lib ole32.lib oleaut32.lib uuid.lib comdlg32.lib advapi32.lib
    if ($LASTEXITCODE -ne 0) { throw "NVRTC helper build failed with exit code $LASTEXITCODE" }
} finally {
    Pop-Location
}
