@echo off
setlocal
cd /d "%~dp0"

rem bundled portable python first
set "PY=%~dp0python\python.exe"
if exist "%PY%" goto :launch

set "PY=python"
"%PY%" -c "import sys" >nul 2>&1
if errorlevel 1 (
  echo [ERROR] Python not found. Install Python 3.9+ ^(check "Add to PATH"^).
  pause
  exit /b 1
)
"%PY%" -c "import PySide6, numpy, PIL, cv2, psd_tools" >nul 2>&1
if errorlevel 1 (
  echo [Setup] Installing dependencies, this may take a few minutes...
  "%PY%" -m pip install -r requirements.txt
  if errorlevel 1 (
    echo [ERROR] pip install failed. Run manually: pip install -r requirements.txt
    pause
    exit /b 1
  )
)

:launch
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
chcp 65001 >nul
"%PY%" app.py
if errorlevel 1 pause
