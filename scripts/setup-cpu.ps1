$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$projectPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $projectPython)) {
    python -m venv (Join-Path $projectRoot '.venv')
    if ($LASTEXITCODE -ne 0) { throw 'Creating Python environment failed; use Python 3.12.' }
}
& $projectPython -m pip install --index-url https://download.pytorch.org/whl/cpu -r (Join-Path $projectRoot 'requirements-cpu-lock.txt')
if ($LASTEXITCODE -ne 0) { throw 'Installing CPU dependencies failed.' }
& $projectPython -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Installed dependencies have conflicts.' }
