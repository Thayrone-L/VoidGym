param(
    [string]$Python = "",
    [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "../..")).Path
)
$ErrorActionPreference = "Stop"
$PythonArgs = @()
if (-not $Python) {
    $Python = (Get-Command py -ErrorAction SilentlyContinue).Source
    if ($Python) { $PythonArgs = @("-3.12") }
    else { $Python = (Get-Command python -ErrorAction Stop).Source; $PythonArgs = @() }
}
$venv = Join-Path $ProjectRoot ".venv-cuda"
$lock = Join-Path $ProjectRoot "requirements-training-cuda.lock"
if (-not (Test-Path -LiteralPath $venv)) {
    & $Python @PythonArgs -m venv $venv
    if ($LASTEXITCODE -ne 0) { throw "Falha ao criar o venv CUDA." }
}
$pip = Join-Path $venv "Scripts/python.exe"
& $pip -m pip install -r $lock
if ($LASTEXITCODE -ne 0) { throw "Falha ao instalar requirements-training-cuda.lock." }
Write-Host "Venv CUDA separado preparado em $venv. O .venv da v1 não foi alterado pelo script."
