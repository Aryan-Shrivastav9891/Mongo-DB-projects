@echo off
echo Starting the Unified API Server...
echo.

REM Check if Python is installed
python --version > nul 2>&1
if errorlevel 1 (
    echo Python is not installed or not in the PATH.
    echo Please install Python and try again.
    exit /b 1
)

REM Check if the unified server file exists
if not exist unified_server.py (
    echo unified_server.py does not exist in the current directory.
    echo Please make sure you are in the correct directory.
    exit /b 1
)

REM Start the unified server
echo Starting the unified server...
python unified_server.py
