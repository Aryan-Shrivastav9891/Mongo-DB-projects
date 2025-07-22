@echo off
echo Running simplified demo for candlestick chart prediction algorithms
echo.
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
    pip install pandas numpy matplotlib statsmodels scikit-learn
)

rem Run the simplified demo
echo.
echo Running simplified demo...
python backend\examples\demo_simplified.py
echo.
echo Demo completed.

rem Deactivate virtual environment
call venv\Scripts\deactivate.bat

:end
pause
