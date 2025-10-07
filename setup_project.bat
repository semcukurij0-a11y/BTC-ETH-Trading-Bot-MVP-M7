@echo off
echo ========================================
echo    Trading Bot Project Setup Script
echo ========================================
echo.

REM Set colors
color 0B

echo [1/4] Checking system requirements...

REM Check if Node.js is installed
node --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Node.js not found! Please install Node.js from https://nodejs.org/
    pause
    exit /b 1
) else (
    echo ✅ Node.js found
)

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found! Please install Python from https://python.org/
    pause
    exit /b 1
) else (
    echo ✅ Python found
)

REM Check if pip is available
pip --version >nul 2>&1
if errorlevel 1 (
    echo ❌ pip not found! Please install pip
    pause
    exit /b 1
) else (
    echo ✅ pip found
)
echo.

echo [2/4] Installing Python dependencies...
cd backend
pip install fastapi uvicorn pandas numpy scipy requests python-multipart
if errorlevel 1 (
    echo ❌ Failed to install Python dependencies
    pause
    exit /b 1
)
echo ✅ Python dependencies installed
cd ..
echo.

echo [3/4] Installing frontend dependencies...
cd frontend
if exist "package.json" (
    npm install
    if errorlevel 1 (
        echo ❌ Failed to install frontend dependencies
        pause
        exit /b 1
    )
    echo ✅ Frontend dependencies installed
) else (
    echo ❌ package.json not found in frontend folder!
    echo Please make sure the frontend folder contains a valid Node.js project.
    pause
    exit /b 1
)
cd ..
echo.

echo [4/4] Verifying project structure...
if not exist "backend\src\bybit_integrated_api.py" (
    echo ❌ Backend API file not found!
    echo Expected: backend\src\bybit_integrated_api.py
    pause
    exit /b 1
) else (
    echo ✅ Backend API file found
)

if not exist "backend\backtester.py" (
    echo ❌ Backtester file not found!
    echo Expected: backend\backtester.py
    pause
    exit /b 1
) else (
    echo ✅ Backtester file found
)

if not exist "backend\data" (
    echo ⚠️  Warning: backend\data folder not found
    echo This is needed for parquet files (BTCUSDT, ETHUSDT)
) else (
    echo ✅ Data folder found
)
echo.

echo ========================================
echo    Setup Completed Successfully!
echo ========================================
echo.
echo Next steps:
echo 1. Run 'run_project.bat' to start the project
echo 2. Open http://localhost:3000 in your browser
echo 3. Use the Backtest page to test the integration
echo.
echo Backtest API endpoint: POST http://localhost:8000/backtest/run
echo API Documentation: http://localhost:8000/docs
echo.
pause
