@echo off
setlocal enabledelayedexpansion

title Multi-Digit Handwritten Number Recognition AI - Launcher

echo ================================================================
echo    MULTI-DIGIT HANDWRITTEN NUMBER RECOGNITION AI - LAUNCHER
echo ================================================================
echo.

REM 1. Check Python compatibility
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python was not found in PATH.
    echo Please install Python (recommended: 3.11 or 3.12 64-bit) from python.org.
    pause
    exit /b 1
)

for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PY_VER=%%v
echo [OK] Detected Python version: %PY_VER%

REM 2. Create virtual environment if not present
if not exist "venv\Scripts\activate.bat" (
    echo [INFO] Creating Python virtual environment in 'venv'...
    python -m venv venv
    if %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created.
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM 3. Install requirements
echo [INFO] Verifying and installing dependencies...
python -m pip install --upgrade pip --quiet
python -m pip install -r requirements.txt --quiet
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to install dependencies from requirements.txt.
    pause
    exit /b 1
)
echo [OK] Dependencies ready.

REM 4 & 5. Check model and download dataset / train if model is missing
if not exist "models\multidigit_crnn.keras" (
    echo [INFO] Trained model not found. Initiating multi-digit dataset generation and CRNN model training...
    python train.py --epochs 6 --batch-size 64 --num-train 8000 --num-test 1500
    if %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Model training failed.
        pause
        exit /b 1
    )
    echo [OK] Model trained and verified successfully.
) else (
    echo [OK] Existing trained CRNN model found at models\multidigit_crnn.keras.
)

REM 6. Start NiceGUI Application
echo.
echo ================================================================
echo  Starting NiceGUI Web Studio at http://127.0.0.1:8080 ...
echo ================================================================
echo.

python app.py

pause

