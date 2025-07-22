# PowerShell script to start the Cryptocurrency Image Analysis Backend Server
Write-Host "Starting Cryptocurrency Image Analysis Backend Server" -ForegroundColor Green
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$backendPath = Join-Path $scriptPath "backend"
Set-Location $backendPath

# Check if Python is installed
try {
    python --version | Out-Null
}
catch {
    Write-Host "Python is not installed or not in PATH. Please install Python." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit
}

# Check if virtual environment exists
$venvActivate = Join-Path $backendPath "venv\Scripts\Activate.ps1"
if (Test-Path $venvActivate) {
    Write-Host "Using existing virtual environment..." -ForegroundColor Cyan
    & $venvActivate
}
else {
    Write-Host "Creating new virtual environment..." -ForegroundColor Yellow
    python -m venv venv
    & $venvActivate
    Write-Host "Installing required packages..." -ForegroundColor Yellow
    pip install fastapi uvicorn pillow numpy python-multipart matplotlib
}

# Generate the survey JSON file if it doesn't exist
python examples\generate_survey_json.py

# Start the FastAPI server
Write-Host "`nStarting API server..." -ForegroundColor Green
python app.py

# Deactivate virtual environment when done
deactivate

Read-Host "Press Enter to exit"
