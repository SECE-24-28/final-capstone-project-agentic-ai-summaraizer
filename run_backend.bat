@echo off
echo ==============================================
echo Starting Webpage Summariser Backend Setup...
echo ==============================================

:: Find the best Python executable (preferring standard Windows Python over MSYS2/Cygwin)
set "PYTHON_EXE=python"
if exist "%USERPROFILE%\AppData\Local\Programs\Python\Python310\python.exe" (
    set "PYTHON_EXE=%USERPROFILE%\AppData\Local\Programs\Python\Python310\python.exe"
)

:: If a .venv created by MSYS2 (with 'bin' folder) exists, remove it first
if exist .venv\bin (
    echo Detected incompatible MSYS2 virtual environment layout. Recreating...
    rmdir /s /q .venv
)

:: Create virtual environment if it doesn't exist
if not exist .venv (
    echo Creating virtual environment in .venv using: %PYTHON_EXE%
    "%PYTHON_EXE%" -m venv .venv
    if errorlevel 1 (
        echo Failed to create virtual environment.
        pause
        exit /b 1
    )
)

:: Detect binary folder
set "VENV_BIN=.venv\Scripts"
if not exist %VENV_BIN% (
    set "VENV_BIN=.venv\bin"
)

:: Install/Upgrade dependencies using the virtual environment's pip
echo Installing dependencies using %VENV_BIN%\pip.exe...
%VENV_BIN%\pip.exe install -r backend\requirements.txt
if errorlevel 1 (
    echo Dependency installation failed.
    pause
    exit /b 1
)

:: Start the FastAPI server using the virtual environment's uvicorn
echo Starting FastAPI server on http://127.0.0.1:8000...
%VENV_BIN%\uvicorn.exe backend.main:app --reload --port 8000


