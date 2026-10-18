[CmdletBinding()]
param(
    [switch]$Sample,
    [ValidatePattern('^[a-zA-Z0-9_-]+$')][string]$SshHost = '88zjwei',
    [ValidatePattern('^/[a-zA-Z0-9_./-]+$')][string]$ServerRoot = '/mnt/data3/zjwei/Camel-DDPM-data/wit/metadata'
)
$ErrorActionPreference = 'Stop'
$project = Split-Path -Parent $PSScriptRoot
$cache = Join-Path $project 'data\wit\metadata'
New-Item -ItemType Directory -Force $cache | Out-Null
$names = if ($Sample) { @('wit_v1.train.all-1percent_sample.tsv.gz') } else {
    @(0..9 | ForEach-Object { 'wit_v1.train.all-{0:D5}-of-00010.tsv.gz' -f $_ })
}
& ssh $SshHost "mkdir -p $ServerRoot"
if ($LASTEXITCODE -ne 0) { throw 'Cannot initialize server destination' }
foreach ($name in $names) {
    $url = "https://storage.googleapis.com/gresearch/wit/$name"
    $headers = (& curl.exe --http1.1 --fail --silent --show-error --head --retry 3 --retry-all-errors --connect-timeout 10 --max-time 60 $url) -join "`n"
    if ($LASTEXITCODE -ne 0) { throw "Cannot read official source metadata: $name" }
    $etagMatch = [regex]::Match($headers, '(?im)^etag:\s*"([a-f0-9]{32})"')
    $sizeMatch = [regex]::Match($headers, '(?im)^content-length:\s*(\d+)')
    if (!$etagMatch.Success -or !$sizeMatch.Success) { throw 'Source lacks expected MD5/length' }
    $expectedHash = $etagMatch.Groups[1].Value.ToLowerInvariant()
    $expectedSize = [long]$sizeMatch.Groups[1].Value
    if ($expectedSize -gt 6GB) { throw 'Unexpected source size; review before downloading' }
    $destination = Join-Path $cache $name
    $partial = "$destination.part"
    if (!(Test-Path -LiteralPath $destination)) {
        $drive = Get-PSDrive -Name ([IO.Path]::GetPathRoot($cache).TrimEnd('\').TrimEnd(':'))
        if ($drive.Free -lt $expectedSize + 10GB) { throw 'Keep at least 10GB local free space' }
        Write-Output "DOWNLOAD $name expected_bytes=$expectedSize"
        & curl.exe --http1.1 --fail --silent --show-error --location --proto '=https' --connect-timeout 10 --max-time 14400 --retry 2 --retry-all-errors --continue-at - --output $partial $url
        if ($LASTEXITCODE -ne 0) { throw 'Download incomplete; rerun to continue the partial file' }
        if ((Get-Item -LiteralPath $partial).Length -ne $expectedSize -or (Get-FileHash -LiteralPath $partial -Algorithm MD5).Hash.ToLowerInvariant() -ne $expectedHash) {
            throw "Source checksum/length mismatch: $name"
        }
        Move-Item -LiteralPath $partial -Destination $destination
    }
    if ((Get-Item -LiteralPath $destination).Length -ne $expectedSize -or (Get-FileHash -LiteralPath $destination -Algorithm MD5).Hash.ToLowerInvariant() -ne $expectedHash) {
        throw "Cached source checksum/length mismatch: $name"
    }
    # A prior server .part from another transport attempt is never trusted.
    $remoteHash = (& ssh $SshHost "if test -f $ServerRoot/$name; then md5sum $ServerRoot/$name; fi") -join "`n"
    if ($LASTEXITCODE -ne 0) { throw 'Cannot inspect server copy' }
    if (!$remoteHash.StartsWith($expectedHash)) {
        Write-Output "TRANSFER $name"
        & scp $destination "${SshHost}:$ServerRoot/$name.incoming"
        if ($LASTEXITCODE -ne 0) { throw 'Transfer incomplete; no server file published' }
        $remoteHash = (& ssh $SshHost "md5sum $ServerRoot/$name.incoming") -join "`n"
        if ($LASTEXITCODE -ne 0 -or !$remoteHash.StartsWith($expectedHash)) { throw 'Server copy checksum mismatch' }
        & ssh $SshHost "mv $ServerRoot/$name.incoming $ServerRoot/$name"
        if ($LASTEXITCODE -ne 0) { throw 'Cannot publish verified server metadata' }
    }
    Write-Output "VERIFIED $name bytes=$expectedSize md5=$expectedHash"
}
Write-Output 'TRANSPORT COMPLETE: metadata only; images have not been admitted for training.'
