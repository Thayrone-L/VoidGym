param(
    [string]$Python = "py",
    [string]$VenvPath = ".venv-cuda"
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location -LiteralPath $root
$venv = Join-Path $root $VenvPath
$lock = Join-Path $root "requirements-training-cuda.lock"

if (-not (Test-Path -LiteralPath $venv)) {
    & $Python -3.12 -m venv $venv
    if ($LASTEXITCODE -ne 0) { throw "Não foi possível criar o venv CUDA." }
}
$pip = Join-Path $venv "Scripts\python.exe"
& $pip -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "Falha ao atualizar pip no venv CUDA." }
& $pip -m pip install -r $lock
if ($LASTEXITCODE -ne 0) { throw "Falha ao instalar requirements-training-cuda.lock." }
Write-Output "Ambiente CUDA preparado em $venv"
