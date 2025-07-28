@echo off
REM RAG System Demo Script for Windows
REM This script runs the complete RAG pipeline and answers a sample query

echo ========================================
echo     RAG System Demo - Windows
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
echo Common installation paths:
echo   - %%USERPROFILE%%\miniconda3\
echo   - %%USERPROFILE%%\anaconda3\
echo   - C:\miniconda3\
echo   - C:\anaconda3\
pause
exit /b 1

:conda_found
echo Found conda at: %CONDA_PATH%

REM Initialize conda
%CONDA_PATH% init cmd.exe >nul 2>&1

REM Check if conda is available
%CONDA_PATH% --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Conda is not installed or not in PATH
    echo Please install Miniconda/Anaconda and try again
    pause
    exit /b 1
)

REM Check conda version
for /f "tokens=2" %%i in ('%CONDA_PATH% --version 2^>^&1') do set conda_version=%%i
echo Conda version: %conda_version%

REM Check if rag-gpu environment exists
%CONDA_PATH% env list | findstr "rag-gpu" >nul 2>&1
if errorlevel 1 (
    echo Creating rag-gpu conda environment...
    %CONDA_PATH% env create -f environment.yml
    if errorlevel 1 (
        echo ERROR: Failed to create conda environment
        pause
        exit /b 1
    )
)

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

echo.
echo Step 1: Checking RAG system components...
call python -c "import openvino; import transformers; import sentence_transformers; print('All core packages imported successfully')"
if errorlevel 1 (
    echo ERROR: Core packages not available
    pause
    exit /b 1
)

echo.
echo Step 2: Running demo query...
call python rag_cli.py --query "What is Procyon and what are its main features?" --k 5 --max-tokens 100 --temperature 0.1
if errorlevel 1 (
    echo ERROR: Demo query failed
    pause
    exit /b 1
)

echo.
echo ========================================
echo     Demo completed successfully!
echo ========================================
echo.
echo You can now run interactive queries with:
echo   python rag_cli.py --query "Your question"
echo.
echo Or run interactive mode with:
echo   python rag_cli.py
echo.
pause 