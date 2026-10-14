[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$GameExecutable,
    [Parameter(Mandatory)][string]$PersonalRoot,
    [Parameter(Mandatory)][string]$ExperimentalRoot,
    [Parameter(Mandatory)][string]$BackupRoot,
    [string]$GuardProcessExecutable,
    [string[]]$GameArguments = @(),
    [ValidateRange(0, 3600)][int]$AutoCloseAfterSeconds = 0
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
Import-Module (Join-Path $PSScriptRoot "HkSaveIsolation.psm1") -Force

$gamePath = Get-HkCanonicalPath $GameExecutable
$personalPath = Get-HkCanonicalPath $PersonalRoot
$experimentalPath = Get-HkCanonicalPath $ExperimentalRoot
$backupPath = Get-HkCanonicalPath $BackupRoot
Assert-HkSaveRoots $personalPath $experimentalPath $backupPath
Assert-HkPathChainNoReparsePoints $personalPath "PersonalRoot"
Assert-HkPathChainNoReparsePoints $experimentalPath "ExperimentalRoot"
Assert-HkPathChainNoReparsePoints $backupPath "BackupRoot"
if (Test-Path -LiteralPath $personalPath) { Assert-HkNoReparsePoints $personalPath }
if (Test-Path -LiteralPath $experimentalPath) { Assert-HkNoReparsePoints $experimentalPath }
if (Test-Path -LiteralPath $backupPath) { Assert-HkNoReparsePoints $backupPath }

if (-not (Test-Path -LiteralPath $gamePath -PathType Leaf)) {
    throw "Game executable not found: $gamePath"
}
$guardPath = if ([string]::IsNullOrWhiteSpace($GuardProcessExecutable)) {
    $gamePath
} else {
    Get-HkCanonicalPath $GuardProcessExecutable
}
if (-not (Test-Path -LiteralPath $guardPath -PathType Leaf)) {
    throw "Guard process executable not found: $guardPath"
}
$processName = [System.IO.Path]::GetFileNameWithoutExtension($guardPath)
if (Test-HkExecutableProcessRunning $guardPath) {
    throw "The guarded executable is already running. Close it before isolating saves: $guardPath"
}

New-Item -ItemType Directory -Path $backupPath -Force | Out-Null
Assert-HkPathChainNoReparsePoints $backupPath "BackupRoot"
$lockPath = Join-Path $backupPath ".save-isolation.lock"
$lock = $null
$manifest = $null
$manifestPath = $null
$launchedProcess = $null

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

    $incomplete = @(
        Get-ChildItem -LiteralPath $backupPath -Filter "manifest.json" -File -Recurse |
            ForEach-Object {
                $candidate = Get-Content -LiteralPath $_.FullName -Raw | ConvertFrom-Json
                if ($candidate.status -ne "COMPLETED") { $_.FullName }
            }
    )
    if ($incomplete.Count -gt 0) {
        throw "Incomplete save transaction found. Recover it before launching: $($incomplete -join ', ')"
    }

    New-Item -ItemType Directory -Path $experimentalPath -Force | Out-Null
    Assert-HkPathChainNoReparsePoints $experimentalPath "ExperimentalRoot"
    Assert-HkNoReparsePoints $experimentalPath
    $transactionId = "hk-save-{0}-{1}" -f ([DateTime]::UtcNow.ToString("yyyyMMddTHHmmssZ")), ([Guid]::NewGuid().ToString("N"))
    $transactionRoot = Join-Path $backupPath $transactionId
    $validatedBackup = Join-Path $transactionRoot "validated-backup"
    $preservedRoot = Join-Path ([System.IO.Path]::GetDirectoryName($personalPath)) (".{0}.preserved" -f $transactionId)
    New-Item -ItemType Directory -Path $transactionRoot | Out-Null
    $manifestPath = Join-Path $transactionRoot "manifest.json"
    $personalExisted = Test-Path -LiteralPath $personalPath -PathType Container
    if ($personalExisted) { Assert-HkNoReparsePoints $personalPath }
    $inventory = @(Get-HkFileInventory $personalPath)
    $manifest = [ordered]@{
        schema_version = 2
        transaction_id = $transactionId
        status = "PREPARING"
        created_utc = [DateTime]::UtcNow.ToString("o")
        updated_utc = $null
        game_executable = $gamePath
        guard_process_executable = $guardPath
        guard_process_name = $processName
        personal_root = $personalPath
        experimental_root = $experimentalPath
        backup_root = $backupPath
        preserved_personal_root = $preservedRoot
        validated_backup_root = $validatedBackup
        personal_root_existed = $personalExisted
        personal_inventory = $inventory
        game_pid = $null
        game_exit_code = $null
    }
    Write-HkManifestAtomic $manifestPath $manifest

    if ($personalExisted) {
        New-Item -ItemType Directory -Path $validatedBackup | Out-Null
        Get-ChildItem -LiteralPath $personalPath -Force |
            Copy-Item -Destination $validatedBackup -Recurse -Force
        $backupInventory = @(Get-HkFileInventory $validatedBackup)
        if (-not (Test-HkInventoriesEqual $inventory $backupInventory)) {
            throw "Backup validation failed; the personal save directory was not moved"
        }
        $manifest.status = "BACKED_UP"
        Write-HkManifestAtomic $manifestPath $manifest
        Move-Item -LiteralPath $personalPath -Destination $preservedRoot
        $manifest.status = "PERSONAL_PRESERVED"
        Write-HkManifestAtomic $manifestPath $manifest
    } elseif (Test-Path -LiteralPath $personalPath) {
        throw "PersonalRoot exists but is not a directory: $personalPath"
    }

    New-Item -ItemType Junction -Path $personalPath -Target $experimentalPath | Out-Null
    if (-not (Test-HkJunctionTarget $personalPath $experimentalPath)) {
        throw "Failed to verify the experimental save junction"
    }
    $manifest.status = "ACTIVE"
    Write-HkManifestAtomic $manifestPath $manifest

    if (Test-HkExecutableProcessRunning $guardPath) {
        throw "The guarded executable started during isolation setup. Refusing to launch: $guardPath"
    }

    $startParameters = @{
        FilePath = $gamePath
        WorkingDirectory = [System.IO.Path]::GetDirectoryName($gamePath)
        PassThru = $true
    }
    if ($GameArguments.Count -gt 0) { $startParameters.ArgumentList = $GameArguments }
    $launchedProcess = Start-Process @startParameters
    $manifest.game_pid = $launchedProcess.Id
    Write-HkManifestAtomic $manifestPath $manifest
    if ($AutoCloseAfterSeconds -gt 0) {
        $exitedNaturally = $launchedProcess.WaitForExit($AutoCloseAfterSeconds * 1000)
        if (-not $exitedNaturally) {
            $closeRequested = $launchedProcess.CloseMainWindow()
            if (-not $closeRequested -or -not $launchedProcess.WaitForExit(10000)) {
                throw "The game did not close normally. Saves remain isolated; close it, then run recovery."
            }
        }
    } else {
        $launchedProcess.WaitForExit()
    }
    $manifest.game_exit_code = $launchedProcess.ExitCode
    $manifest.status = "RESTORING"
    Write-HkManifestAtomic $manifestPath $manifest
} finally {
    try {
        if ($null -ne $manifest -and $null -ne $manifestPath -and $manifest.status -ne "COMPLETED") {
            if ($null -ne $launchedProcess -and -not $launchedProcess.HasExited) {
                $manifest.status = "GAME_STILL_RUNNING"
                Write-HkManifestAtomic $manifestPath $manifest
                throw "The launched process is still running. Saves remain isolated; close it, then run recovery."
            }
            if (Test-HkExecutableProcessRunning $guardPath) {
                $manifest.status = "ANOTHER_GAME_PROCESS_RUNNING"
                Write-HkManifestAtomic $manifestPath $manifest
                throw "The guarded executable is running. Saves remain isolated; close it, then run recovery."
            }
            Restore-HkSaveTransaction $manifest $manifestPath
        }
    } finally {
        if ($null -ne $lock) { $lock.Dispose() }
    }
}

[pscustomobject]@{
    status = "COMPLETED"
    manifest = $manifestPath
    game_exit_code = $manifest.game_exit_code
}
