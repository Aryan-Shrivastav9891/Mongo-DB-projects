@echo off
echo Starting Candlestick Chart Prediction Platform development environment...

echo.
echo Starting backend server...
start cmd /k "cd backend && python -m venv venv && venv\Scripts\activate && pip install -r requirements.txt && uvicorn main:app --reload --host 0.0.0.0 --port 8000"

echo.
echo Starting frontend development server...
start cmd /k "npm run dev"

echo.
echo Development environment started!
echo - Frontend: http://localhost:3000
echo - Backend API: http://localhost:8000
echo.
