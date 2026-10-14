[CmdletBinding()]
param(
    [string]$ConfigPath = "config.local.toml"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $ConfigPath)) {
    throw "Configuration not found: $ConfigPath"
}

$raw = Get-Content -Raw -LiteralPath $ConfigPath

function Get-TomlString {
    param([Parameter(Mandatory)][string]$Section, [Parameter(Mandatory)][string]$Key)
    $sectionPattern = "(?ms)^\[$([regex]::Escape($Section))\]\s*(.*?)(?=^\[|\z)"
    $sectionMatch = [regex]::Match($raw, $sectionPattern)
    if (-not $sectionMatch.Success) { return $null }
    $keyPattern = "(?m)^$([regex]::Escape($Key))\s*=\s*`"([^`"]*)`"\s*$"
    $keyMatch = [regex]::Match($sectionMatch.Groups[1].Value, $keyPattern)
    if (-not $keyMatch.Success) { return $null }
    return $keyMatch.Groups[1].Value
}

$installPath = Get-TomlString -Section "game" -Key "install_path"
$personalRoot = Get-TomlString -Section "saves" -Key "personal_root"
$experimentalRoot = Get-TomlString -Section "saves" -Key "experimental_root"
$backupRoot = Get-TomlString -Section "saves" -Key "backup_root"

if ([string]::IsNullOrWhiteSpace($installPath)) {
    throw "game.install_path is empty"
}
if (-not (Test-Path -LiteralPath $installPath)) {
    throw "game.install_path does not exist: $installPath"
}
if ([string]::IsNullOrWhiteSpace($personalRoot)) {
    throw "saves.personal_root is empty"
}
if ([string]::IsNullOrWhiteSpace($experimentalRoot)) {
    throw "saves.experimental_root is empty"
}
if ([string]::IsNullOrWhiteSpace($backupRoot)) {
    throw "saves.backup_root is empty"
}

$personalFull = [IO.Path]::GetFullPath($personalRoot).TrimEnd('\')
$experimentalFull = [IO.Path]::GetFullPath($experimentalRoot).TrimEnd('\')
$backupFull = [IO.Path]::GetFullPath($backupRoot).TrimEnd('\')

if ($personalFull -eq $experimentalFull) {
    throw "Personal and experimental save roots must differ"
}
if ($personalFull -eq $backupFull) {
    throw "The backup root must differ from the active personal save root"
}
if ($experimentalFull.StartsWith($personalFull + '\', [StringComparison]::OrdinalIgnoreCase)) {
    throw "The experimental save root must not be nested under the personal save root"
}

Write-Output "Configuration paths passed the initial safety checks."
