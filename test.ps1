param([switch]$CpuOnly)
$ErrorActionPreference = 'Stop'
$savedHome = $env:CAMEL_HOME
$savedPackages = $env:CAMEL_PACKAGES
$savedOut = $env:DDPM_OUT
$savedStdlib = $env:CAMEL_STD_LIB
Push-Location $PSScriptRoot
try {
    $env:CAMEL_HOME = Join-Path $PSScriptRoot '.camel'
    $env:CAMEL_STD_LIB = Join-Path $env:CAMEL_HOME 'stdlib'
    $env:CAMEL_PACKAGES = Join-Path $PSScriptRoot 'src'
    $env:DDPM_OUT = (Join-Path $PSScriptRoot 'runs\checks').Replace('\','/')
    $camel = Join-Path $env:CAMEL_HOME 'bin\camel.exe'
    & $camel tests/checks.cml std::macro std::fvm
    if ($LASTEXITCODE -ne 0) { throw 'Numerical checks failed' }
    if (-not $CpuOnly) {
        & $camel tests/gpu_checks.cml std::macro std::fvm
        if ($LASTEXITCODE -ne 0) { throw 'GPU comparison failed' }
    }
} finally {
    Pop-Location
    $env:CAMEL_HOME = $savedHome
    $env:CAMEL_PACKAGES = $savedPackages
    $env:DDPM_OUT = $savedOut
    $env:CAMEL_STD_LIB = $savedStdlib
}
