@echo off
echo ========================================
echo    Trading Bot Project Startup Script
echo ========================================
echo.

REM Set colors for better output
color 0A

REM Check if we're in the right directory
if not exist "backend" (
    echo ERROR: backend folder not found!
    echo Please run this script from the project root directory.
    pause
    exit /b 1
)

if not exist "frontend" (
    echo ERROR: frontend folder not found!
    echo Please run this script from the project root directory.
    pause
    exit /b 1
)

echo [1/6] Checking project structure...
echo ✅ Backend folder found
echo ✅ Frontend folder found
echo.

echo [2/6] Installing frontend dependencies...
cd frontend
if not exist "node_modules" (
    echo Installing npm packages...
    call npm install
    if errorlevel 1 (
        echo ERROR: Failed to install npm packages
        pause
        exit /b 1
    )
    echo ✅ Frontend dependencies installed
) else (
    echo ✅ Frontend dependencies already installed
)
cd ..
echo.

echo [3/6] Checking backend Python dependencies...
cd backend
python -c "import fastapi, uvicorn, pandas, numpy" 2>nul
if errorlevel 1 (
    echo Installing Python dependencies...
    pip install fastapi uvicorn pandas numpy scipy requests
    if errorlevel 1 (
        echo ERROR: Failed to install Python dependencies
        pause
        exit /b 1
    )
    echo ✅ Python dependencies installed
) else (
    echo ✅ Python dependencies already installed
)
cd ..
echo.

echo [4/6] Starting backend server...
cd backend
start "Backend Server" cmd /k "python src/bybit_integrated_api.py"
echo ✅ Backend server starting...
cd ..
echo.

echo [5/6] Waiting for backend to start...
timeout /t 5 /nobreak >nul
echo ✅ Backend should be running on http://localhost:8000
echo.

echo [6/6] Starting frontend development server...
cd frontend
start "Frontend Server" cmd /k "npm run dev"
echo ✅ Frontend server starting...
cd ..
echo.

echo ========================================
echo    Project Started Successfully!
echo ========================================
echo.
echo Backend API:  http://localhost:8000
echo Frontend:     http://localhost:3000 (or http://localhost:5173)
echo API Docs:     http://localhost:8000/docs
echo.
echo Backtest Endpoint: POST http://localhost:8000/backtest/run
echo.
echo Press any key to open the frontend in your browser...
pause >nul

REM Try to open the frontend in browser
start http://localhost:3000 2>nul
if errorlevel 1 (
    start http://localhost:5173 2>nul
)

echo.
echo ========================================
echo    Project is now running!
echo ========================================
echo.
echo To stop the servers:
echo 1. Close the "Backend Server" window
echo 2. Close the "Frontend Server" window
echo.
echo Or press Ctrl+C in each window to stop them.
echo.
pause
