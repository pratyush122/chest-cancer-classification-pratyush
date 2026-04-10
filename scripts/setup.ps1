param(
    [string]$PythonPath = "python"
)

$ErrorActionPreference = "Stop"

Write-Host "Using Python interpreter: $PythonPath"

if (-not (Test-Path ".env") -and (Test-Path ".env.example")) {
    Copy-Item ".env.example" ".env"
    Write-Host "Created .env from .env.example"
}

& $PythonPath -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip "setuptools<81" wheel
& .\.venv\Scripts\python.exe -m pip install -r requirements-pipeline.txt

Write-Host ""
Write-Host "Setup completed."
Write-Host "Activate environment: .\\.venv\\Scripts\\Activate.ps1"
Write-Host "Run web app: python app.py"
Write-Host "Run training pipeline: python main.py"
