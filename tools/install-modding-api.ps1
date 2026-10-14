[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$GameRoot,
    [Parameter(Mandatory)][string]$ArchivePath,
    [Parameter(Mandatory)][string]$BackupRoot,
    [string]$ExpectedArchiveSha256 = "bc9f0db3d0916b05cd5a2420bb602fb1b239ce3ff6c289fd84bffb682fb8f1d6"
)

$ErrorActionPreference = "Stop"

function Resolve-RequiredPath {
    param([Parameter(Mandatory)][string]$Path, [Parameter(Mandatory)][string]$Label)
    if (-not (Test-Path -LiteralPath $Path)) { throw "$Label not found: $Path" }
    return (Resolve-Path -LiteralPath $Path).Path
}

function Get-FileRecord {
    param([Parameter(Mandatory)][string]$Path, [Parameter(Mandatory)][string]$RelativePath)
    $item = Get-Item -LiteralPath $Path
    return [ordered]@{
        relative_path = $RelativePath
        length = $item.Length
        sha256 = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
    }
}

$resolvedGameRoot = Resolve-RequiredPath -Path $GameRoot -Label "Game root"
$resolvedArchive = Resolve-RequiredPath -Path $ArchivePath -Label "API archive"
$resolvedBackupRoot = [IO.Path]::GetFullPath($BackupRoot).TrimEnd('\')
$gamePathRoot = [IO.Path]::GetPathRoot($resolvedGameRoot).TrimEnd('\')

if ($resolvedGameRoot.TrimEnd('\') -eq $gamePathRoot) {
    throw "Refusing to operate on a drive root: $resolvedGameRoot"
}
if ($resolvedBackupRoot.StartsWith($resolvedGameRoot.TrimEnd('\') + '\', [StringComparison]::OrdinalIgnoreCase)) {
    throw "Backup root must not be inside the game installation"
}
if (Test-Path -LiteralPath $resolvedBackupRoot) {
    throw "Backup root already exists: $resolvedBackupRoot"
}

$executableCandidates = @("Hollow Knight.exe", "hollow_knight.exe")
$executable = $null
foreach ($candidate in $executableCandidates) {
    $path = Join-Path $resolvedGameRoot $candidate
    if (Test-Path -LiteralPath $path) { $executable = $path; break }
}
if ($null -eq $executable) { throw "Hollow Knight executable was not found" }

$dataCandidates = @("Hollow Knight_Data", "hollow_knight_Data")
$dataRoot = $null
foreach ($candidate in $dataCandidates) {
    $path = Join-Path $resolvedGameRoot $candidate
    if (Test-Path -LiteralPath $path) { $dataRoot = $path; break }
}
if ($null -eq $dataRoot) { throw "Hollow Knight data directory was not found" }

$managedRoot = Resolve-RequiredPath -Path (Join-Path $dataRoot "Managed") -Label "Managed directory"
if (-not $managedRoot.StartsWith($resolvedGameRoot.TrimEnd('\') + '\', [StringComparison]::OrdinalIgnoreCase)) {
    throw "Managed directory resolved outside the game root"
}

$running = @(
    Get-Process -ErrorAction SilentlyContinue | Where-Object {
        try {
            -not [string]::IsNullOrWhiteSpace($_.Path) -and
                $_.Path.StartsWith($resolvedGameRoot.TrimEnd('\') + '\', [StringComparison]::OrdinalIgnoreCase)
        }
        catch { $false }
    }
)
if ($running.Count -gt 0) {
    throw "Game installation has running processes: $($running.Name -join ', ')"
}

$actualArchiveHash = (Get-FileHash -LiteralPath $resolvedArchive -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actualArchiveHash -ne $ExpectedArchiveSha256.ToLowerInvariant()) {
    throw "API archive SHA-256 mismatch"
}

Add-Type -AssemblyName System.IO.Compression.FileSystem
$zip = [System.IO.Compression.ZipFile]::OpenRead($resolvedArchive)
try {
    $apiFiles = @(
        $zip.Entries |
            Where-Object { -not [string]::IsNullOrWhiteSpace($_.Name) -and $_.FullName -ne "README.md" } |
            ForEach-Object { $_.FullName }
    )
}
finally {
    $zip.Dispose()
}
if ($apiFiles.Count -eq 0) { throw "API archive contains no installable files" }
if ($apiFiles | Where-Object { $_ -match '[\\/]' -or $_ -eq '.' -or $_ -eq '..' }) {
    throw "API archive contains unexpected nested paths"
}

New-Item -ItemType Directory -Path $resolvedBackupRoot | Out-Null
$coreBackup = Join-Path $resolvedBackupRoot "managed-before"
$packageRoot = Join-Path $resolvedBackupRoot "api-v77-package"
New-Item -ItemType Directory -Path $coreBackup | Out-Null
New-Item -ItemType Directory -Path $packageRoot | Out-Null

$before = @()
foreach ($fileName in $apiFiles) {
    $target = Join-Path $managedRoot $fileName
    if (Test-Path -LiteralPath $target) {
        $record = Get-FileRecord -Path $target -RelativePath $fileName
        $before += $record
        Copy-Item -LiteralPath $target -Destination (Join-Path $coreBackup $fileName)
        $backupHash = (Get-FileHash -LiteralPath (Join-Path $coreBackup $fileName) -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($backupHash -ne $record.sha256) { throw "Backup verification failed for $fileName" }
    }
}

$modsRoot = Join-Path $managedRoot "Mods"
$legacyMods = Join-Path $resolvedBackupRoot "legacy-mods"
$legacyModRecords = @()
if (Test-Path -LiteralPath $modsRoot) {
    $legacyModRecords = @(
        Get-ChildItem -LiteralPath $modsRoot -Recurse -File | ForEach-Object {
            Get-FileRecord -Path $_.FullName -RelativePath ([IO.Path]::GetRelativePath($modsRoot, $_.FullName))
        }
    )
    Move-Item -LiteralPath $modsRoot -Destination $legacyMods
}

Expand-Archive -LiteralPath $resolvedArchive -DestinationPath $packageRoot
foreach ($fileName in $apiFiles) {
    $source = Join-Path $packageRoot $fileName
    $target = Join-Path $managedRoot $fileName
    if (-not (Test-Path -LiteralPath $source)) { throw "Extracted API file missing: $fileName" }
    Copy-Item -LiteralPath $source -Destination $target -Force
}
New-Item -ItemType Directory -Path $modsRoot | Out-Null

$installed = @()
foreach ($fileName in $apiFiles) {
    $source = Join-Path $packageRoot $fileName
    $target = Join-Path $managedRoot $fileName
    $sourceHash = (Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash.ToLowerInvariant()
    $targetRecord = Get-FileRecord -Path $target -RelativePath $fileName
    if ($targetRecord.sha256 -ne $sourceHash) { throw "Installed API verification failed for $fileName" }
    $installed += $targetRecord
}

$manifest = [ordered]@{
    schema_version = 1
    installed_utc = [DateTime]::UtcNow.ToString("o")
    game_root = $resolvedGameRoot
    executable = Get-FileRecord -Path $executable -RelativePath ([IO.Path]::GetFileName($executable))
    managed_root = $managedRoot
    api = [ordered]@{
        version = "1.5.78.11833-77"
        archive = $resolvedArchive
        archive_sha256 = $actualArchiveHash
        installed_files = $installed
    }
    backup = [ordered]@{
        root = $resolvedBackupRoot
        overwritten_files = $before
        legacy_mods_path = if (Test-Path -LiteralPath $legacyMods) { $legacyMods } else { $null }
        legacy_mods = $legacyModRecords
    }
}
$manifestPath = Join-Path $resolvedBackupRoot "install-manifest.json"
$manifest | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $manifestPath -Encoding utf8

[pscustomobject]@{
    game_root = $resolvedGameRoot
    api_version = "1.5.78.11833-77"
    installed_file_count = $installed.Count
    archived_mod_file_count = $legacyModRecords.Count
    backup_root = $resolvedBackupRoot
    manifest = $manifestPath
} | ConvertTo-Json
