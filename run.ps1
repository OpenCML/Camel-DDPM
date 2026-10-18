param(
    [string]$Camel = (Join-Path $PSScriptRoot '.camel\bin\camel.exe'),
    [string]$Out = '',
    [int]$Steps = 10000, [int]$Batch = 32, [int]$Width = 256,
    [int]$Timesteps = 100, [int]$Samples = 64, [int]$Threads = 1,
    [double]$Rate = 0.001, [int]$SampleSeed = 12345,
    [int]$DelayMs = 10,
    [ValidateSet('std::nvm', 'std::fvm', 'std::jit')][string]$Backend = 'std::fvm',
    [ValidateSet('gpu','cpu')][string]$Device = 'gpu',
    [ValidateSet('epsilon','patch-x0')][string]$Model = 'epsilon',
    [switch]$SampleOnly, [switch]$SkipPrepare
)
$ErrorActionPreference = 'Stop'
if ($Steps -lt 1 -or $Batch -lt 1 -or $Batch -gt 60000 -or $Width -lt 1 -or $Timesteps -lt 2 -or $Samples -lt 1 -or $Threads -lt 1 -or $Rate -le 0 -or $DelayMs -lt 0) { throw 'Invalid sizes/rate' }
if ($Model -ne 'epsilon' -and $Device -eq 'cpu') { throw 'The patch model currently requires the explicit GPU path' }
if ($Model -eq 'patch-x0') {
    if ($PSBoundParameters.ContainsKey('Width') -and $Width -ne 64) { throw 'patch-x0 uses a fixed hidden width of 64' }
    $Width = 64
}
$Camel = (Resolve-Path -LiteralPath $Camel).Path
if ($Out -eq '') { $Out = Join-Path $PSScriptRoot ('runs\' + (Get-Date -Format 'yyyyMMdd-HHmmss')) }
$Out = [System.IO.Path]::GetFullPath($Out)
if (-not $SampleOnly -and (Test-Path -LiteralPath (Join-Path $Out 'config.json'))) { throw 'Run exists; choose a fresh output directory' }
New-Item -ItemType Directory -Path $Out -Force | Out-Null
$savedHome = $env:CAMEL_HOME
$savedOut = $env:DDPM_OUT
$savedStdlib = $env:CAMEL_STD_LIB
$savedPackages = $env:CAMEL_PACKAGES
$savedPriority = [Diagnostics.Process]::GetCurrentProcess().PriorityClass
[Diagnostics.Process]::GetCurrentProcess().PriorityClass = 'BelowNormal'
$env:CAMEL_HOME = Split-Path (Split-Path $Camel -Parent) -Parent
$env:CAMEL_STD_LIB = Join-Path $env:CAMEL_HOME 'stdlib'
$env:CAMEL_PACKAGES = Join-Path $PSScriptRoot 'src'
$env:DDPM_OUT = $Out.Replace('\', '/')
Push-Location $PSScriptRoot
try {
    if (-not $SampleOnly) {
        if (-not $SkipPrepare) { & "$PSScriptRoot\tools\prepare.ps1" }
        $config = @{ steps=$Steps; batch=$Batch; width=$Width; timesteps=$Timesteps; samples=$Samples; threads=$Threads; rate=$Rate; delay_ms=$DelayMs; sample_seed=$SampleSeed; backend=$Backend; device=$Device; model=$Model; camel=$Camel; version=(& $Camel --version | Out-String).Trim(); created=(Get-Date -Format o) }
        $config | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $Out 'config.json') -Encoding utf8
        $clock = [Diagnostics.Stopwatch]::StartNew()
        $trainSource = if ($Device -eq 'gpu') { 'src/train_gpu.cml' } else { 'src/train.cml' }
        $deviceArgs = if ($Device -eq 'gpu') { @('--arg', "delay_ms=$DelayMs", '--arg', "model_kind=$Model") } else { @() }
        & $Camel -I src --arg "steps=$Steps" --arg "batch=$Batch" --arg "width=$Width" --arg "timesteps=$Timesteps" --arg 'report=100' --arg "threads=$Threads" --arg "rate=$Rate" @deviceArgs $trainSource std::macro $Backend 2>&1 | Tee-Object -FilePath (Join-Path $Out 'train.log')
        if ($LASTEXITCODE -ne 0) { throw "Training failed: $LASTEXITCODE" }
        $clock.Stop()
        @{ training_seconds=$clock.Elapsed.TotalSeconds } | ConvertTo-Json | Set-Content (Join-Path $Out 'timing.json')
    } else {
        $config = Get-Content -LiteralPath (Join-Path $Out 'config.json') -Raw | ConvertFrom-Json
        $Timesteps = $config.timesteps
        $Device = $config.device
        $Model = if ($config.model) { $config.model } else { 'epsilon' }
    }
    $sampleSource = if ($Device -eq 'gpu') { 'src/sample_gpu.cml' } else { 'src/sample.cml' }
    $sampleArgs = if ($Device -eq 'gpu') { @('--arg', "model_kind=$Model") } else { @() }
    & $Camel -I src --arg "timesteps=$Timesteps" --arg "samples=$Samples" --arg "sample_seed=$SampleSeed" --arg "threads=$Threads" @sampleArgs $sampleSource std::macro $Backend 2>&1 | Tee-Object -FilePath (Join-Path $Out 'sample.log')
    if ($LASTEXITCODE -ne 0) { throw "Sampling failed: $LASTEXITCODE" }
    Write-Host "Results: $Out"
} finally {
    Pop-Location
    $env:CAMEL_HOME = $savedHome
    $env:DDPM_OUT = $savedOut
    $env:CAMEL_STD_LIB = $savedStdlib
    $env:CAMEL_PACKAGES = $savedPackages
    [Diagnostics.Process]::GetCurrentProcess().PriorityClass = $savedPriority
}
