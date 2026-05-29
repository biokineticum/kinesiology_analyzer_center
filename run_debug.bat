@echo off
cd /d "%~dp0"
echo ===================================================
echo   KINESIOLOGY ANALYZER CENTER - DEBUG MODE
echo ===================================================
echo Uruchamianie aplikacji w konsoli, aby zobaczyc bledy...
echo.
python -u standalone\launch_desktop.py
echo.
echo.
if %errorlevel% neq 0 (
    echo ===================================================
    echo  [BLAD] Aplikacja zakonczyla sie bledem!
    echo  Kod bledu: %errorlevel%
    echo ===================================================
) else (
    echo Aplikacja zakonczyla sie pomyslnie.
)
pause
