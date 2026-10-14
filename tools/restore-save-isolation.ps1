[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$ManifestPath,
    [Parameter(Mandatory)][string]$BackupRoot
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
Import-Module (Join-Path $PSScriptRoot "HkSaveIsolation.psm1") -Force

$manifestFile = Get-HkCanonicalPath $ManifestPath
$backupPath = Get-HkCanonicalPath $BackupRoot
if (-not (Test-Path -LiteralPath $manifestFile -PathType Leaf)) {
    throw "Manifest not found: $manifestFile"
}
if (-not (Test-HkPathNested -Candidate $manifestFile -Parent $backupPath)) {
    throw "Manifest must be inside the declared backup root"
}

New-Item -ItemType Directory -Path $backupPath -Force | Out-Null
$lockPath = Join-Path $backupPath ".save-isolation.lock"
$lock = $null
try {
    try {
        $lock = [System.IO.File]::Open(
            $lockPath,
            [System.IO.FileMode]::OpenOrCreate,
            [System.IO.FileAccess]::ReadWrite,
            [System.IO.FileShare]::None
        )
    } catch [System.IO.IOException] {
        throw "Another save-isolation session is active (lock: $lockPath)"
    }

    $manifest = Get-Content -LiteralPath $manifestFile -Raw | ConvertFrom-Json
    if ([int]$manifest.schema_version -notin @(1, 2)) { throw "Unsupported manifest schema" }
    Assert-HkManifestPaths $manifest $manifestFile $backupPath
    if ($manifest.status -eq "COMPLETED") {
        Write-Output "Transaction is already complete: $manifestFile"
        exit 0
    }
    $guardExecutable = if ([int]$manifest.schema_version -ge 2) {
        [string]$manifest.guard_process_executable
    } else {
        [string]$manifest.game_executable
    }
    if (Test-HkExecutableProcessRunning $guardExecutable) {
        throw "The game process is running; recovery is forbidden until it exits"
    }
    Restore-HkSaveTransaction $manifest $manifestFile
    Write-Output "Personal saves restored and hash-verified: $($manifest.personal_root)"
} finally {
    if ($null -ne $lock) { $lock.Dispose() }
}
