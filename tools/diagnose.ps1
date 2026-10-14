[CmdletBinding()]
param(
    [string]$OutputDirectory = "artifacts/diagnostics",
    [string[]]$GameRoots = @(),
    [switch]$IncludeSaveHashes
)

$ErrorActionPreference = "Stop"

function Get-CommandInfo {
    param([Parameter(Mandatory)][string]$Name)

    $command = Get-Command $Name -ErrorAction SilentlyContinue
    if ($null -eq $command) {
        return [ordered]@{ found = $false; path = $null; version = $null }
    }

    $version = $null
    try {
        $version = $command.Version.ToString()
    }
    catch {
        $version = $null
    }

    return [ordered]@{
        found = $true
        path = $command.Source
        version = $version
    }
}

function Get-SteamLibraries {
    $steamRoot = $null
    foreach ($registryPath in @(
        "HKCU:\Software\Valve\Steam",
        "HKLM:\SOFTWARE\WOW6432Node\Valve\Steam",
        "HKLM:\SOFTWARE\Valve\Steam"
    )) {
        if (-not (Test-Path -LiteralPath $registryPath)) { continue }
        $properties = Get-ItemProperty -LiteralPath $registryPath
        $candidate = $properties.SteamPath
        if ([string]::IsNullOrWhiteSpace($candidate)) { $candidate = $properties.InstallPath }
        if (-not [string]::IsNullOrWhiteSpace($candidate)) {
            $steamRoot = $candidate
            break
        }
    }

    if ([string]::IsNullOrWhiteSpace($steamRoot)) {
        return @()
    }

    $libraryFile = Join-Path $steamRoot "steamapps\libraryfolders.vdf"
    $roots = [System.Collections.Generic.List[string]]::new()
    $roots.Add($steamRoot)

    if (Test-Path -LiteralPath $libraryFile) {
        $vdf = Get-Content -Raw -LiteralPath $libraryFile
        foreach ($match in [regex]::Matches($vdf, '"path"\s+"([^"]+)"')) {
            $path = $match.Groups[1].Value -replace '\\\\', '\'
            if (-not $roots.Contains($path)) { $roots.Add($path) }
        }
    }

    return @($roots)
}

function Get-SafeSaveInventory {
    param(
        [Parameter(Mandatory)][string]$Root,
        [Parameter(Mandatory)][bool]$HashFiles
    )

    if (-not (Test-Path -LiteralPath $Root)) {
        return [ordered]@{ exists = $false; accessible = $false; files = @() }
    }

    try {
        $files = @(
            Get-ChildItem -LiteralPath $Root -File | ForEach-Object {
                $entry = [ordered]@{
                    name = $_.Name
                    length = $_.Length
                    last_write_utc = $_.LastWriteTimeUtc.ToString("o")
                }
                if ($HashFiles) {
                    $entry.sha256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash
                }
                [pscustomobject]$entry
            }
        )
        return [ordered]@{ exists = $true; accessible = $true; files = $files }
    }
    catch {
        return [ordered]@{
            exists = $true
            accessible = $false
            error = $_.Exception.Message
            files = @()
        }
    }
}

$resolvedOutput = Join-Path (Get-Location) $OutputDirectory
New-Item -ItemType Directory -Path $resolvedOutput -Force | Out-Null

$os = Get-CimInstance Win32_OperatingSystem
$cpu = Get-CimInstance Win32_Processor | Select-Object -First 1
$computer = Get-CimInstance Win32_ComputerSystem
$gpus = @(Get-CimInstance Win32_VideoController | Select-Object Name, DriverVersion, AdapterRAM)
$drives = @(
    Get-PSDrive -PSProvider FileSystem | ForEach-Object {
        [ordered]@{
            name = $_.Name
            free_gb = [math]::Round($_.Free / 1GB, 2)
            used_gb = [math]::Round($_.Used / 1GB, 2)
        }
    }
)

$nvidia = $null
if ((Get-CommandInfo -Name "nvidia-smi").found) {
    try {
        $nvidia = (& nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>$null) -join "`n"
    }
    catch {
        $nvidia = "ERROR: $($_.Exception.Message)"
    }
}

$steamLibraries = @(Get-SteamLibraries)
$steamGameCandidates = @(
    $steamLibraries | ForEach-Object {
        $exe = Join-Path $_ "steamapps\common\Hollow Knight\hollow_knight.exe"
        [ordered]@{
            library = $_
            executable = $exe
            exists = Test-Path -LiteralPath $exe
        }
    }
)

$standaloneGameCandidates = @(
    $GameRoots | ForEach-Object {
        $resolvedRoot = [IO.Path]::GetFullPath($_)
        $exe = Join-Path $resolvedRoot "hollow_knight.exe"
        $entry = [ordered]@{
            root = $resolvedRoot
            executable = $exe
            exists = Test-Path -LiteralPath $exe
            sha256 = $null
            file_version = $null
        }
        if ($entry.exists) {
            $entry.sha256 = (Get-FileHash -LiteralPath $exe -Algorithm SHA256).Hash
            $entry.file_version = (Get-Item -LiteralPath $exe).VersionInfo.FileVersion
        }
        [pscustomobject]$entry
    }
)

$saveRoot = Join-Path $env:USERPROFILE "AppData\LocalLow\Team Cherry\Hollow Knight"
$report = [ordered]@{
    schema_version = 1
    generated_utc = [DateTime]::UtcNow.ToString("o")
    machine = [ordered]@{
        windows = $os.Caption
        version = $os.Version
        build = $os.BuildNumber
        cpu = $cpu.Name.Trim()
        logical_processors = $cpu.NumberOfLogicalProcessors
        ram_gb = [math]::Round($computer.TotalPhysicalMemory / 1GB, 2)
        gpus = $gpus
        nvidia_smi = $nvidia
        drives = $drives
    }
    commands = [ordered]@{
        python = Get-CommandInfo -Name "python"
        py = Get-CommandInfo -Name "py"
        git = Get-CommandInfo -Name "git"
        dotnet = Get-CommandInfo -Name "dotnet"
        docker = Get-CommandInfo -Name "docker"
        wsl = Get-CommandInfo -Name "wsl"
    }
    steam = [ordered]@{
        libraries = $steamLibraries
        hollow_knight_candidates = $steamGameCandidates
        optional = $true
    }
    standalone = [ordered]@{
        supplied_roots = $GameRoots
        hollow_knight_candidates = $standaloneGameCandidates
    }
    saves = [ordered]@{
        root = $saveRoot
        inventory = Get-SafeSaveInventory -Root $saveRoot -HashFiles $IncludeSaveHashes.IsPresent
    }
}

$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$outputPath = Join-Path $resolvedOutput "diagnostics-$timestamp.json"
$report | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $outputPath -Encoding utf8
Write-Output $outputPath
