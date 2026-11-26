@echo off
chcp 65001 >nul
setlocal EnableExtensions
cd /d "%~dp0"
set "PY=%CD%\.venv\Scripts\python.exe"

if /I "%~1"=="setup" goto install
if not exist "%PY%" goto install
if /I "%~1"=="explorer" goto explorer
if /I "%~1"=="dashboard" goto dashboard
if "%~1"=="" goto menu

echo Неизвестный параметр: %~1
echo Использование: START.bat [setup^|dashboard^|explorer]
pause
exit /b 2

:install
echo [1/4] Проверка Python 3.12...
where py >nul 2>nul
if errorlevel 1 (
    echo Не найден Python Launcher. Сначала установите Python 3.12 x64.
    pause
    exit /b 1
)
py -3.12 --version
if errorlevel 1 (
    echo Для проекта требуется Python 3.12.
    pause
    exit /b 1
)
if not exist "%PY%" (
    echo [2/4] Создание виртуального окружения...
    py -3.12 -m venv .venv
    if errorlevel 1 goto failed
)
echo [3/4] Установка библиотек из requirements.txt...
"%PY%" -m pip install --upgrade pip
if errorlevel 1 goto failed
"%PY%" -m pip install -r requirements.txt
if errorlevel 1 goto failed
echo [4/4] Установка проекта в режиме разработки...
"%PY%" -m pip install --no-deps -e .
if errorlevel 1 goto failed
if not exist ".env" copy ".env.example" ".env" >nul
echo Установка завершена.
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
echo 1. Основная аналитическая панель
echo 2. Карта тематических сообществ
echo 3. Установить или восстановить библиотеки
echo 4. Выход
echo.
choice /c 1234 /n /m "Выберите действие: "
if errorlevel 4 exit /b 0
if errorlevel 3 goto install
if errorlevel 2 goto explorer
if errorlevel 1 goto dashboard

:dashboard
echo Запуск основной панели...
"%PY%" -m streamlit run dashboard\app.py
if errorlevel 1 goto failed
pause
exit /b 0

:explorer
set "SIL_PROJECT_ROOT=%CD%"
echo Запуск карты тематических сообществ...
"%PY%" -m streamlit run src\sciintel\semantic\streamlit_topic_explorer.py --server.address 127.0.0.1
if errorlevel 1 goto failed
pause
exit /b 0

:failed
echo.
echo Не удалось выполнить действие. Проверьте сообщение об ошибке выше.
pause
exit /b 1
