@echo off
setlocal EnableExtensions
cd /d "%~dp0"
set "PY=%CD%\.venv\Scripts\python.exe"

if /I "%~1"=="setup" goto install
if not exist "%PY%" goto install
if /I "%~1"=="explorer" goto explorer
if /I "%~1"=="dashboard" goto dashboard
if "%~1"=="" goto menu

echo Unknown parameter: %~1
echo Usage: START.bat [setup^|dashboard^|explorer]
pause
exit /b 2

:install
echo [1/4] Checking Python 3.12...
where py >nul 2>nul
if errorlevel 1 (
    echo Python Launcher was not found. Install Python 3.12 x64 first.
    pause
    exit /b 1
)
py -3.12 --version
if errorlevel 1 (
    echo Python 3.12 is required for this project.
    pause
    exit /b 1
)
if not exist "%PY%" (
    echo [2/4] Creating virtual environment...
    py -3.12 -m venv .venv
    if errorlevel 1 goto failed
)
echo [3/4] Installing packages from requirements.txt...
"%PY%" -m pip install --upgrade pip
if errorlevel 1 goto failed
"%PY%" -m pip install -r requirements.txt
if errorlevel 1 goto failed
echo [4/4] Installing the project in editable mode...
"%PY%" -m pip install --no-deps -e .
if errorlevel 1 goto failed
if not exist ".env" if exist ".env.example" copy /Y ".env.example" ".env" >nul
echo Installation completed successfully.
if /I "%~1"=="setup" (
    pause
    exit /b 0
)
goto menu

:menu
cls
echo ========================================================
echo                 SCIENTIFIC CLUSTERING
echo ========================================================
echo 1. Main analytics dashboard
echo 2. Topic communities explorer
echo 3. Install or repair Python packages
echo 4. Exit
echo.
choice /c 1234 /n /m "Select an option: "
if errorlevel 4 exit /b 0
if errorlevel 3 goto install
if errorlevel 2 goto explorer
if errorlevel 1 goto dashboard

:dashboard
if not exist "dashboard\app.py" (
    echo Dashboard file was not found: dashboard\app.py
    pause
    exit /b 1
)
echo Starting the main analytics dashboard...
"%PY%" -m streamlit run dashboard\app.py --server.address 127.0.0.1
if errorlevel 1 goto failed
pause
exit /b 0

:explorer
if not exist "src\sciintel\semantic\streamlit_topic_explorer.py" (
    echo Topic explorer file was not found.
    pause
    exit /b 1
)
set "SIL_PROJECT_ROOT=%CD%"
echo Starting the topic communities explorer...
"%PY%" -m streamlit run src\sciintel\semantic\streamlit_topic_explorer.py --server.address 127.0.0.1
if errorlevel 1 goto failed
pause
exit /b 0

:failed
echo.
echo The command failed. Check the error message above.
pause
exit /b 1
