@echo off
REM Environment setup script for AV-PINO Motor Fault Diagnosis System (Windows)

echo Setting up AV-PINO Motor Fault Diagnosis environment...

REM Check if conda is available
where conda >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    echo Using conda for environment setup...
    
    REM Create conda environment
    conda env create -f environment.yml
    
    echo Activating environment...
    call conda activate av-pino-motor-fault-diagnosis
    
    REM Install package in development mode
    pip install -e .
    
    echo Environment setup complete! Activate with: conda activate av-pino-motor-fault-diagnosis
    
) else (
    REM Check if python is available
    where python >nul 2>nul
    if %ERRORLEVEL% EQU 0 (
        echo Using pip for environment setup...
        
        REM Create virtual environment
        python -m venv venv
        call venv\Scripts\activate.bat
        
        REM Upgrade pip
        python -m pip install --upgrade pip
        
        REM Install requirements
        pip install -r requirements.txt
        
        REM Install package in development mode
        pip install -e .
        
        echo Environment setup complete! Activate with: venv\Scripts\activate.bat
        
    ) else (
        echo Error: Neither conda nor python found. Please install Python 3.8+ or Anaconda/Miniconda.
        exit /b 1
    )
)

REM Verify installation
echo Verifying installation...
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import numpy; print(f'NumPy version: {numpy.__version__}')"
python -c "import matplotlib; print(f'Matplotlib version: {matplotlib.__version__}')"

echo Setup verification complete!
pause