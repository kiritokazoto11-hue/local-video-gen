@echo off
rem ============================================================
rem  VideoAI - one-click launcher
rem  Creates the environment on first run, then starts the app.
rem ============================================================
cd /d "%~dp0"

set "PY=%LOCALAPPDATA%\miniconda3\python.exe"
if not exist "%PY%" set "PY=python"

if not exist ".venv\Scripts\python.exe" (
  echo [1/2] Creating virtual environment...
  "%PY%" -m venv .venv
  echo [2/2] Installing dependencies (first run only, takes a few minutes)...
  .venv\Scripts\python.exe -m pip install --upgrade pip
  .venv\Scripts\python.exe -m pip install -r requirements.txt
)

echo Starting VideoAI on http://127.0.0.1:8000 ...
start "" http://127.0.0.1:8000
.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
pause
