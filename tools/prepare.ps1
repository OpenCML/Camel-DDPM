# Only transport and gzip decompression use the host. IDX decoding is Camel.
param([string]$DataDirectory = (Join-Path $PSScriptRoot '..\data'))
$ErrorActionPreference = 'Stop'
New-Item -ItemType Directory -Path $DataDirectory -Force | Out-Null
$files = @(
    @{ Name = 'train-images-idx3-ubyte'; MD5 = 'f68b3c2dcbeaaa9fbdd348bbdeb94873' },
    @{ Name = 't10k-images-idx3-ubyte'; MD5 = '9fb629c4189551a2d022fa330f9573f3' }
)
foreach ($file in $files) {
    $archive = Join-Path $DataDirectory ($file.Name + '.gz')
    $destination = Join-Path $DataDirectory $file.Name
    if (-not (Test-Path -LiteralPath $archive)) {
        Invoke-WebRequest -Uri ('https://storage.googleapis.com/cvdf-datasets/mnist/' + $file.Name + '.gz') -OutFile $archive
    }
    if ((Get-FileHash -LiteralPath $archive -Algorithm MD5).Hash.ToLowerInvariant() -ne $file.MD5) {
        throw "MNIST checksum mismatch: $archive"
    }
    $inputStream = [System.IO.File]::OpenRead($archive)
    try {
        $gzipStream = [System.IO.Compression.GzipStream]::new($inputStream, [System.IO.Compression.CompressionMode]::Decompress)
        try {
            $outputStream = [System.IO.File]::Create($destination)
            try { $gzipStream.CopyTo($outputStream) } finally { $outputStream.Dispose() }
        } finally { $gzipStream.Dispose() }
    } finally { $inputStream.Dispose() }
    Write-Host "Ready: $destination"
}
