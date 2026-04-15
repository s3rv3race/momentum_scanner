@echo off
echo Starting Momentum Scanner in background...
start "" /B pythonw "%~dp0flask_app.py" > "%~dp0scanner.log" 2>&1
timeout /t 2 /nobreak >nul
echo Server running at http://localhost:5000
echo Log: %~dp0scanner.log
start "" http://localhost:5000
echo.
echo To stop the scanner, run stop.bat
