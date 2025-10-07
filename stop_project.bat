@echo off
echo ========================================
echo    Stopping Trading Bot Project
echo ========================================
echo.

REM Set colors
color 0C

echo [1/3] Stopping backend server...
taskkill /f /im python.exe 2>nul
if errorlevel 1 (
    echo ✅ No Python processes found
) else (
    echo ✅ Backend server stopped
)
echo.

echo [2/3] Stopping frontend server...
taskkill /f /im node.exe 2>nul
if errorlevel 1 (
    echo ✅ No Node processes found
) else (
    echo ✅ Frontend server stopped
)
echo.

echo [3/3] Stopping any remaining processes...
taskkill /f /im cmd.exe /fi "WINDOWTITLE eq Backend Server*" 2>nul
taskkill /f /im cmd.exe /fi "WINDOWTITLE eq Frontend Server*" 2>nul
echo ✅ All project processes stopped
echo.

echo ========================================
echo    Project Stopped Successfully!
echo ========================================
echo.
pause
