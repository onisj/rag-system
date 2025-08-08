@echo off
REM Streamlit Frontend Launcher for Windows
REM This script starts the Streamlit web interface for the RAG system

echo ========================================
echo     RAG System - Streamlit Frontend
echo ========================================
echo.

REM Try to find conda installation
set CONDA_FOUND=0

REM Check common conda installation paths
if exist "%USERPROFILE%\miniconda3\Scripts\conda.exe" (
    set CONDA_PATH="%USERPROFILE%\miniconda3\Scripts\conda.exe"
    set CONDA_FOUND=1
    goto :conda_found
)

if exist "%USERPROFILE%\anaconda3\Scripts\conda.exe" (
    set CONDA_PATH="%USERPROFILE%\anaconda3\Scripts\conda.exe"
    set CONDA_FOUND=1
    goto :conda_found
)

if exist "C:\miniconda3\Scripts\conda.exe" (
    set CONDA_PATH="C:\miniconda3\Scripts\conda.exe"
    set CONDA_FOUND=1
    goto :conda_found
)

if exist "C:\anaconda3\Scripts\conda.exe" (
    set CONDA_PATH="C:\anaconda3\Scripts\conda.exe"
    set CONDA_FOUND=1
    goto :conda_found
)

REM Check if conda is in PATH
conda --version >nul 2>&1
if not errorlevel 1 (
    set CONDA_PATH=conda
    set CONDA_FOUND=1
    goto :conda_found
)

echo ERROR: Conda not found in common locations or PATH
echo Please install Miniconda/Anaconda and try again
pause
exit /b 1

:conda_found
echo Found conda at: %CONDA_PATH%

REM Activate conda environment
echo Activating rag-gpu conda environment...
call %CONDA_PATH% activate rag-gpu
if errorlevel 1 (
    echo ERROR: Failed to activate conda environment
    echo Trying alternative activation method...
    if exist "%USERPROFILE%\miniconda3\Scripts\activate.bat" (
        call "%USERPROFILE%\miniconda3\Scripts\activate.bat" rag-gpu
    ) else if exist "%USERPROFILE%\anaconda3\Scripts\activate.bat" (
        call "%USERPROFILE%\anaconda3\Scripts\activate.bat" rag-gpu
    ) else (
        echo ERROR: Failed to activate conda environment
        pause
        exit /b 1
    )
)

REM Install Streamlit if not already installed
echo Checking Streamlit installation...
python -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo Installing Streamlit and dependencies...
    pip install -r requirements_streamlit.txt
    if errorlevel 1 (
        echo ERROR: Failed to install Streamlit dependencies
        pause
        exit /b 1
    )
)

echo.
echo Starting Streamlit web interface...
echo.
echo The web interface will open in your browser at:
echo   http://localhost:8501
echo.
echo Press Ctrl+C to stop the server
echo.

REM Start Streamlit
streamlit run streamlit_app.py --server.port 8501 --server.address localhost

pause 