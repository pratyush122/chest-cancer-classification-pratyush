param(
    [switch]$Train
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path ".venv")) {
    & powershell -ExecutionPolicy Bypass -File ".\scripts\setup.ps1"
}

if (-not (Test-Path ".env") -and (Test-Path ".env.example")) {
    Copy-Item ".env.example" ".env"
}

if ($Train) {
    & .\.venv\Scripts\python.exe main.py
}

& .\.venv\Scripts\python.exe app.py
