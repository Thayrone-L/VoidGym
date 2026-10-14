Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-HkCanonicalPath {
    param([Parameter(Mandatory)][string]$Path)

    return [System.IO.Path]::GetFullPath($Path).TrimEnd(
        [System.IO.Path]::DirectorySeparatorChar,
        [System.IO.Path]::AltDirectorySeparatorChar
    )
}

function Test-HkPathNested {
    param(
        [Parameter(Mandatory)][string]$Candidate,
        [Parameter(Mandatory)][string]$Parent
    )

    $candidatePath = Get-HkCanonicalPath $Candidate
    $parentPath = Get-HkCanonicalPath $Parent
    $prefix = $parentPath + [System.IO.Path]::DirectorySeparatorChar
    return $candidatePath.StartsWith($prefix, [System.StringComparison]::OrdinalIgnoreCase)
}

function Test-HkExecutableProcessRunning {
    param([Parameter(Mandatory)][string]$ExecutablePath)

    $expectedPath = Get-HkCanonicalPath $ExecutablePath
    $expectedName = [System.IO.Path]::GetFileName($expectedPath)
    $processes = @(Get-CimInstance -ClassName Win32_Process -ErrorAction Stop)
    foreach ($process in $processes) {
        $candidateName = [string]$process.Name
        $candidate = [string]$process.ExecutablePath
        if ([string]::IsNullOrWhiteSpace($candidate)) {
            if ($candidateName.Equals(
                    $expectedName,
                    [System.StringComparison]::OrdinalIgnoreCase)) {
                throw "Unable to verify executable path for a same-name running process: $candidateName"
            }
            continue
        }
        try {
            $candidatePath = Get-HkCanonicalPath $candidate
        } catch {
            if ($candidateName.Equals(
                    $expectedName,
                    [System.StringComparison]::OrdinalIgnoreCase)) {
                throw "Unable to verify executable path for a same-name running process: $candidateName"
            }
            continue
        }
        if ($candidatePath.Equals($expectedPath, [System.StringComparison]::OrdinalIgnoreCase)) {
            return $true
        }
    }
    return $false
}

function Assert-HkSaveRoots {
    param(
        [Parameter(Mandatory)][string]$PersonalRoot,
        [Parameter(Mandatory)][string]$ExperimentalRoot,
        [Parameter(Mandatory)][string]$BackupRoot
    )

    $roots = @(
        (Get-HkCanonicalPath $PersonalRoot),
        (Get-HkCanonicalPath $ExperimentalRoot),
        (Get-HkCanonicalPath $BackupRoot)
    )
    for ($left = 0; $left -lt $roots.Count; $left++) {
        for ($right = $left + 1; $right -lt $roots.Count; $right++) {
            if ($roots[$left].Equals($roots[$right], [System.StringComparison]::OrdinalIgnoreCase)) {
                throw "Save isolation roots must be distinct: $($roots[$left])"
            }
            if ((Test-HkPathNested $roots[$left] $roots[$right]) -or
                (Test-HkPathNested $roots[$right] $roots[$left])) {
                throw "Save isolation roots must not contain one another: $($roots[$left]); $($roots[$right])"
            }
        }
    }
}

function Get-HkFileInventory {
    param([Parameter(Mandatory)][string]$Root)

    $canonicalRoot = Get-HkCanonicalPath $Root
    if (-not (Test-Path -LiteralPath $canonicalRoot -PathType Container)) {
        return @()
    }

    return @(
        Get-ChildItem -LiteralPath $canonicalRoot -File -Recurse -Force |
            Sort-Object FullName |
            ForEach-Object {
                $stream = [System.IO.File]::OpenRead($_.FullName)
                try {
                    $hasher = [System.Security.Cryptography.SHA256]::Create()
                    try {
                        $hashBytes = $hasher.ComputeHash($stream)
                    } finally {
                        $hasher.Dispose()
                    }
                } finally {
                    $stream.Dispose()
                }
                $hash = [System.BitConverter]::ToString($hashBytes).Replace("-", "").ToLowerInvariant()
                [ordered]@{
                    relative_path = $_.FullName.Substring($canonicalRoot.Length).TrimStart('\', '/')
                    size = $_.Length
                    sha256 = $hash
                }
            }
    )
}

function Assert-HkNoReparsePoints {
    param([Parameter(Mandatory)][string]$Root)

    if (-not (Test-Path -LiteralPath $Root)) { return }
    $rootItem = Get-Item -LiteralPath $Root -Force
    if ($rootItem.Attributes -band [System.IO.FileAttributes]::ReparsePoint) {
        throw "Refusing a personal save root that is already a reparse point: $Root"
    }
    $nestedLinks = @(Get-ChildItem -LiteralPath $Root -Force -Recurse -Attributes ReparsePoint)
    if ($nestedLinks.Count -gt 0) {
        throw "Refusing personal saves containing a reparse point: $($nestedLinks[0].FullName)"
    }
}

function Assert-HkPathChainNoReparsePoints {
    param(
        [Parameter(Mandatory)][string]$Path,
        [Parameter(Mandatory)][string]$Label
    )

    $cursor = Get-HkCanonicalPath $Path
    while (-not [string]::IsNullOrEmpty($cursor)) {
        if (Test-Path -LiteralPath $cursor) {
            $item = Get-Item -LiteralPath $cursor -Force
            if ($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) {
                throw "Refusing $Label because it or an ancestor is a reparse point: $cursor"
            }
        }
        $parent = [System.IO.Path]::GetDirectoryName($cursor)
        if ([string]::IsNullOrEmpty($parent) -or
            $parent.Equals($cursor, [System.StringComparison]::OrdinalIgnoreCase)) {
            break
        }
        $cursor = $parent
    }
}

function Test-HkInventoriesEqual {
    param(
        [Parameter(Mandatory)]$Expected,
        [Parameter(Mandatory)]$Actual
    )

    return (ConvertTo-Json @($Expected) -Depth 5 -Compress) -ceq
        (ConvertTo-Json @($Actual) -Depth 5 -Compress)
}

function Set-HkManifestField {
    param(
        [Parameter(Mandatory)]$Manifest,
        [Parameter(Mandatory)][string]$Name,
        $Value
    )

    if ($Manifest -is [System.Collections.IDictionary]) {
        $Manifest[$Name] = $Value
    } else {
        $Manifest | Add-Member -NotePropertyName $Name -NotePropertyValue $Value -Force
    }
}

function Assert-HkManifestPaths {
    param(
        [Parameter(Mandatory)]$Manifest,
        [Parameter(Mandatory)][string]$ManifestPath,
        [Parameter(Mandatory)][string]$BackupRoot
    )

    $backupPath = Get-HkCanonicalPath $BackupRoot
    if ([int]$Manifest.schema_version -ge 2) {
        $guardExecutable = [string]$Manifest.guard_process_executable
        $guardName = [string]$Manifest.guard_process_name
        if ([string]::IsNullOrWhiteSpace($guardExecutable) -or
            -not (Test-Path -LiteralPath $guardExecutable -PathType Leaf)) {
            throw "Manifest guard process executable is missing or unavailable"
        }
        $expectedGuardName = [System.IO.Path]::GetFileNameWithoutExtension($guardExecutable)
        if ([string]::IsNullOrWhiteSpace($guardName) -or
            -not $guardName.Equals($expectedGuardName, [System.StringComparison]::OrdinalIgnoreCase)) {
            throw "Manifest guard process name does not match its executable"
        }
    }
    $manifestBackup = Get-HkCanonicalPath ([string]$Manifest.backup_root)
    if (-not $backupPath.Equals($manifestBackup, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Manifest backup root does not match the requested backup root"
    }
    $transactionId = [string]$Manifest.transaction_id
    if ($transactionId -notmatch '^hk-save-[A-Za-z0-9-]+$') {
        throw "Invalid save transaction identifier"
    }
    $transactionRoot = Join-Path $backupPath $transactionId
    $expectedManifest = Join-Path $transactionRoot "manifest.json"
    if (-not (Get-HkCanonicalPath $ManifestPath).Equals(
            (Get-HkCanonicalPath $expectedManifest),
            [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Manifest path does not match its transaction identifier"
    }
    $expectedBackup = Join-Path $transactionRoot "validated-backup"
    if (-not (Get-HkCanonicalPath ([string]$Manifest.validated_backup_root)).Equals(
            (Get-HkCanonicalPath $expectedBackup),
            [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Validated backup path does not match the transaction"
    }
    $personalRoot = Get-HkCanonicalPath ([string]$Manifest.personal_root)
    $expectedPreserved = Join-Path (
        [System.IO.Path]::GetDirectoryName($personalRoot)
    ) (".{0}.preserved" -f $transactionId)
    if (-not (Get-HkCanonicalPath ([string]$Manifest.preserved_personal_root)).Equals(
            (Get-HkCanonicalPath $expectedPreserved),
            [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Preserved personal-save path does not match the transaction"
    }
}

function Write-HkManifestAtomic {
    param(
        [Parameter(Mandatory)][string]$Path,
        [Parameter(Mandatory)]$Manifest
    )

    $temporaryPath = "$Path.tmp"
    Set-HkManifestField $Manifest "updated_utc" ([DateTime]::UtcNow.ToString("o"))
    $Manifest | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $temporaryPath -Encoding UTF8
    Move-Item -LiteralPath $temporaryPath -Destination $Path -Force
}

function Test-HkJunctionTarget {
    param(
        [Parameter(Mandatory)][string]$Path,
        [Parameter(Mandatory)][string]$ExpectedTarget
    )

    $item = Get-Item -LiteralPath $Path -Force -ErrorAction SilentlyContinue
    if ($null -eq $item -or -not $item.PSIsContainer -or
        -not ($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint)) {
        return $false
    }

    $targets = @($item.Target)
    return $targets.Count -eq 1 -and
        (Get-HkCanonicalPath ([string]$targets[0])).Equals(
            (Get-HkCanonicalPath $ExpectedTarget),
            [System.StringComparison]::OrdinalIgnoreCase
        )
}

function Remove-HkExpectedJunction {
    param(
        [Parameter(Mandatory)][string]$Path,
        [Parameter(Mandatory)][string]$ExpectedTarget
    )

    if (-not (Test-HkJunctionTarget $Path $ExpectedTarget)) {
        throw "Refusing to remove an unexpected path at the personal save location: $Path"
    }
    [System.IO.Directory]::Delete((Get-HkCanonicalPath $Path), $false)
}

function Restore-HkSaveTransaction {
    param(
        [Parameter(Mandatory)]$Manifest,
        [Parameter(Mandatory)][string]$ManifestPath
    )

    $personalRoot = Get-HkCanonicalPath ([string]$Manifest.personal_root)
    $experimentalRoot = Get-HkCanonicalPath ([string]$Manifest.experimental_root)
    $preservedRoot = Get-HkCanonicalPath ([string]$Manifest.preserved_personal_root)
    $validatedBackup = Get-HkCanonicalPath ([string]$Manifest.validated_backup_root)

    if (Test-Path -LiteralPath $personalRoot) {
        if (Test-HkJunctionTarget $personalRoot $experimentalRoot) {
            Remove-HkExpectedJunction $personalRoot $experimentalRoot
        } else {
            if (Test-Path -LiteralPath $preservedRoot) {
                throw "Recovery stopped: both personal and preserved paths exist; nothing was overwritten"
            }
            $currentInventory = Get-HkFileInventory $personalRoot
            if (-not (Test-HkInventoriesEqual $Manifest.personal_inventory $currentInventory)) {
                throw "Recovery stopped: the existing personal save path differs from the manifest"
            }
            $Manifest.status = "COMPLETED"
            Set-HkManifestField $Manifest "restored_utc" ([DateTime]::UtcNow.ToString("o"))
            Write-HkManifestAtomic $ManifestPath $Manifest
            return
        }
    }

    if ([bool]$Manifest.personal_root_existed) {
        if (-not (Test-Path -LiteralPath $preservedRoot -PathType Container)) {
            $backupInventory = Get-HkFileInventory $validatedBackup
            if (-not (Test-HkInventoriesEqual $Manifest.personal_inventory $backupInventory)) {
                throw "Recovery stopped: both preserved saves and a valid backup are unavailable"
            }
            New-Item -ItemType Directory -Path $personalRoot | Out-Null
            Get-ChildItem -LiteralPath $validatedBackup -Force |
                Copy-Item -Destination $personalRoot -Recurse -Force
        } else {
            if (Test-Path -LiteralPath $personalRoot) {
                throw "Recovery stopped rather than overwriting the personal save root: $personalRoot"
            }
            Move-Item -LiteralPath $preservedRoot -Destination $personalRoot
        }
        $restoredInventory = Get-HkFileInventory $personalRoot
        if (-not (Test-HkInventoriesEqual $Manifest.personal_inventory $restoredInventory)) {
            throw "CRITICAL: personal saves were restored, but their content hashes differ from the manifest"
        }
    }

    $Manifest.status = "COMPLETED"
    Set-HkManifestField $Manifest "restored_utc" ([DateTime]::UtcNow.ToString("o"))
    Write-HkManifestAtomic $ManifestPath $Manifest
}

Export-ModuleMember -Function @(
    "Assert-HkSaveRoots",
    "Assert-HkNoReparsePoints",
    "Assert-HkPathChainNoReparsePoints",
    "Assert-HkManifestPaths",
    "Get-HkCanonicalPath",
    "Get-HkFileInventory",
    "Restore-HkSaveTransaction",
    "Test-HkPathNested",
    "Test-HkInventoriesEqual",
    "Test-HkJunctionTarget",
    "Test-HkExecutableProcessRunning",
    "Write-HkManifestAtomic"
)
