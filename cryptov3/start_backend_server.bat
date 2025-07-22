@echo off
echo Starting Cryptocurrency Image Analysis Backend Server
cd %~dp0
cd ..\backend

rem Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed or not in PATH. Please install Python.
    goto :end
)

rem Check if virtual environment exists
if exist venv\Scripts\activate.bat (
    echo Using existing virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo Creating new virtual environment...
    python -m venv venv
    call venv\Scripts\activate.bat
    echo Installing required packages...
    pip install fastapi uvicorn pillow numpy python-multipart matplotlib
)

rem Generate the survey JSON file if it doesn't exist
python examples\generate_survey_json.py

rem Start the FastAPI server
echo.
echo Starting API server...
python app.py

rem Deactivate virtual environment when done
call venv\Scripts\deactivate.bat

:end
pause
