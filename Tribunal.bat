@echo off
setlocal

cd /d "%~dp0"

if exist ".venv\Scripts\activate.bat" (
    echo Activating virtual environment: .venv
    call ".venv\Scripts\activate.bat"
) else if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment: venv
    call "venv\Scripts\activate.bat"
) else (
    echo No virtual environment found - using global Python.
)

echo Launching Tribunal Streamlit dashboard...
python -m streamlit run app.py
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
    echo.
    echo Streamlit exited with an error (code %EXIT_CODE%).
)

echo.
pause
endlocal
