param([Parameter(Mandatory=$true)][string]$Distribution, [switch]$Update)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
$target = Join-Path $projectRoot '.camel'
if ((Test-Path -LiteralPath $target) -and -not $Update) { throw "Runtime already exists: $target (use -Update for an explicit reinstall)" }
$source = (Resolve-Path -LiteralPath $Distribution).Path
if (-not (Test-Path -LiteralPath (Join-Path $source 'bin\camel.exe'))) { throw 'Not a Camel distribution' }
# Install the complete collected distribution; never link back to a developer checkout.
New-Item -ItemType Directory -Path $target -Force | Out-Null
foreach ($part in @('bin','libs','stdlib','include')) {
    Copy-Item -LiteralPath (Join-Path $source $part) -Destination $target -Recurse -Force
}
$binary = Join-Path $target 'bin\camel.exe'
@{ installed=(Get-Date -Format o); source=$source; version=(& $binary --version | Out-String).Trim(); sha256=(Get-FileHash -LiteralPath $binary -Algorithm SHA256).Hash } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $target 'installation.json')
Write-Host "Installed project runtime: $target"
