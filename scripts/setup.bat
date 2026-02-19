@echo off
setlocal EnableDelayedExpansion

:: ============================================================
:: AI Face Recognition & Face Swap - Windows Setup Script
:: ============================================================
:: This script will:
::   1. Check Python 3.10+ is installed
::   2. Create a virtual environment (.venv)
::   3. Upgrade pip
::   4. Install all requirements
::   5. Copy .env.example to .env (if not already present)
::   6. Create required runtime directories
::   7. Run the model downloader (minimum models)
:: ============================================================

title AI Face Recognition - Setup

:: ── Colours (Windows 10+) ────────────────────────────────────
set "GREEN=[92m"
set "YELLOW=[93m"
set "RED=[91m"
set "CYAN=[96m"
set "RESET=[0m"
set "BOLD=[1m"

echo.
echo %CYAN%============================================================%RESET%
echo %BOLD%%CYAN%   AI Face Recognition ^& Face Swap - Environment Setup%RESET%
echo %CYAN%============================================================%RESET%
echo.

:: ── Step 0: Move to project root (parent of scripts\) ────────
cd /d "%~dp0.."
echo %CYAN%[INFO]%RESET%  Working directory: %CD%
echo.

:: ============================================================
:: STEP 1 — Check Python
:: ============================================================
echo %CYAN%[1/7]%RESET%  Checking Python installation...

where python >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo %RED%[ERROR]%RESET% Python not found in PATH.
    echo         Please install Python 3.10+ from https://www.python.org/downloads/
    echo         Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

:: Get Python version
for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo %GREEN%[OK]%RESET%    Python %PYVER% found.

:: Check minimum version (3.10)
for /f "tokens=1,2 delims=." %%a in ("%PYVER%") do (
    set PY_MAJOR=%%a
    set PY_MINOR=%%b
)

if %PY_MAJOR% LSS 3 (
    echo %RED%[ERROR]%RESET% Python 3.10+ is required. Found: %PYVER%
    pause
    exit /b 1
)
if %PY_MAJOR% EQU 3 (
    if %PY_MINOR% LSS 10 (
        echo %RED%[ERROR]%RESET% Python 3.10+ is required. Found: %PYVER%
        echo         Download from: https://www.python.org/downloads/
        pause
        exit /b 1
    )
)
echo.

:: ============================================================
:: STEP 2 — Create virtual environment
:: ============================================================
echo %CYAN%[2/7]%RESET%  Setting up virtual environment (.venv)...

if exist ".venv\" (
    echo %YELLOW%[SKIP]%RESET%  Virtual environment already exists at .venv\
) else (
    python -m venv .venv
    if %ERRORLEVEL% NEQ 0 (
        echo %RED%[ERROR]%RESET% Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo %GREEN%[OK]%RESET%    Virtual environment created at .venv\
)
echo.

:: ── Activate venv ────────────────────────────────────────────
call ".venv\Scripts\activate.bat"
if %ERRORLEVEL% NEQ 0 (
    echo %RED%[ERROR]%RESET% Failed to activate virtual environment.
    pause
    exit /b 1
)
echo %GREEN%[OK]%RESET%    Virtual environment activated.
echo.

:: ============================================================
:: STEP 3 — Upgrade pip & core build tools
:: ============================================================
echo %CYAN%[3/7]%RESET%  Upgrading pip, setuptools, and wheel...

python -m pip install --upgrade pip setuptools wheel --quiet
if %ERRORLEVEL% NEQ 0 (
    echo %RED%[ERROR]%RESET% Failed to upgrade pip.
    pause
    exit /b 1
)
echo %GREEN%[OK]%RESET%    pip, setuptools, and wheel are up to date.
echo.

:: ============================================================
:: STEP 4 — Install requirements
:: ============================================================
echo %CYAN%[4/7]%RESET%  Installing dependencies from requirements.txt...
echo         This may take several minutes on first run.
echo.

:: Check for CUDA-capable GPU via nvidia-smi
set CUDA_AVAILABLE=0
where nvidia-smi >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    nvidia-smi >nul 2>&1
    if !ERRORLEVEL! EQU 0 (
        set CUDA_AVAILABLE=1
    )
)

if %CUDA_AVAILABLE% EQU 1 (
    echo %GREEN%[GPU]%RESET%    NVIDIA GPU detected — installing GPU-accelerated packages.
    echo.
    :: Install PyTorch with CUDA 12.1 support first (must precede requirements.txt)
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --quiet
    if !ERRORLEVEL! NEQ 0 (
        echo %YELLOW%[WARN]%RESET%  GPU PyTorch install failed, falling back to CPU build.
        pip install torch torchvision --quiet
    )
) else (
    echo %YELLOW%[CPU]%RESET%    No NVIDIA GPU detected — installing CPU-only packages.
    echo         For GPU support, install CUDA 12.x and re-run this script.
    echo.
    :: Replace onnxruntime-gpu with cpu variant in requirements
    pip install onnxruntime --quiet
    pip install torch torchvision --quiet
)

:: Install remaining requirements (skip torch/onnxruntime if already installed)
pip install -r requirements.txt --quiet
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo %YELLOW%[WARN]%RESET%  Some packages may have failed. Retrying with verbose output...
    pip install -r requirements.txt
    if !ERRORLEVEL! NEQ 0 (
        echo %RED%[ERROR]%RESET% Dependency installation failed.
        echo         Check the error above and install manually if needed.
        pause
        exit /b 1
    )
)
echo.
echo %GREEN%[OK]%RESET%    All dependencies installed successfully.
echo.

:: ============================================================
:: STEP 5 — Environment file
:: ============================================================
echo %CYAN%[5/7]%RESET%  Configuring environment file...

if exist ".env" (
    echo %YELLOW%[SKIP]%RESET%  .env already exists — not overwriting.
) else (
    if exist ".env.example" (
        copy ".env.example" ".env" >nul
        echo %GREEN%[OK]%RESET%    .env created from .env.example
        echo %YELLOW%[NOTE]%RESET%  Please review and edit .env with your settings.
        if %CUDA_AVAILABLE% EQU 1 (
            echo %CYAN%[TIP]%RESET%   Set EXECUTION_PROVIDER=cuda in .env for GPU acceleration.
        ) else (
            echo %CYAN%[TIP]%RESET%   Set EXECUTION_PROVIDER=cpu in .env ^(no GPU detected^).
        )
    ) else (
        echo %YELLOW%[WARN]%RESET%  .env.example not found — skipping .env creation.
    )
)
echo.

:: ============================================================
:: STEP 6 — Create runtime directories
:: ============================================================
echo %CYAN%[6/7]%RESET%  Creating runtime directories...

set DIRS=models uploads output cache logs tmp

for %%d in (%DIRS%) do (
    if not exist "%%d\" (
        mkdir "%%d"
        echo %GREEN%[OK]%RESET%    Created: %%d\
    ) else (
        echo %YELLOW%[SKIP]%RESET%  Already exists: %%d\
    )
)
echo.

:: ============================================================
:: STEP 7 — Download minimum models
:: ============================================================
echo %CYAN%[7/7]%RESET%  Downloading minimum required AI models...
echo         ^(YOLOv8n-face, InsightFace buffalo_l, inswapper_128^)
echo.

set /p DOWNLOAD_MODELS="Download models now? [Y/n]: "
if /i "%DOWNLOAD_MODELS%"=="" set DOWNLOAD_MODELS=Y
if /i "%DOWNLOAD_MODELS%"=="Y" (
    python utils\download_models.py --minimum
    if %ERRORLEVEL% NEQ 0 (
        echo.
        echo %YELLOW%[WARN]%RESET%  Some models failed to download.
        echo         You can retry later with:
        echo           python utils\download_models.py --minimum
        echo         Or download all models with:
        echo           python utils\download_models.py --all
    ) else (
        echo %GREEN%[OK]%RESET%    Minimum models downloaded successfully.
    )
) else (
    echo %YELLOW%[SKIP]%RESET%  Model download skipped.
    echo         Run later with: python utils\download_models.py --minimum
)
echo.

:: ============================================================
:: DONE
:: ============================================================
echo %GREEN%============================================================%RESET%
echo %BOLD%%GREEN%   Setup Complete!%RESET%
echo %GREEN%============================================================%RESET%
echo.
echo  Next steps:
echo.
echo   1. Activate the virtual environment:
echo      %CYAN%.venv\Scripts\activate%RESET%
echo.
echo   2. Edit your configuration (if not done yet):
echo      %CYAN%notepad .env%RESET%
echo.
echo   3. Start the API server:
echo      %CYAN%python api\main.py%RESET%
echo.
echo   4. Start the Streamlit UI (in a new terminal):
echo      %CYAN%streamlit run ui\app.py%RESET%
echo.
echo   5. Download all models (including enhancers):
echo      %CYAN%python utils\download_models.py --all%RESET%
echo.
if %CUDA_AVAILABLE% EQU 0 (
    echo  %YELLOW%[TIP]%RESET% Install CUDA 12.x for GPU acceleration:
    echo        https://developer.nvidia.com/cuda-downloads
    echo.
)
echo %GREEN%============================================================%RESET%
echo.

pause
endlocal
