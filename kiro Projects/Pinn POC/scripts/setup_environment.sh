#!/bin/bash
# Environment setup script for AV-PINO Motor Fault Diagnosis System

set -e  # Exit on any error

echo "Setting up AV-PINO Motor Fault Diagnosis environment..."

# Check if conda is available
if command -v conda &> /dev/null; then
    echo "Using conda for environment setup..."
    
    # Create conda environment
    conda env create -f environment.yml
    
    echo "Activating environment..."
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate av-pino-motor-fault-diagnosis
    
    # Install package in development mode
    pip install -e .
    
    echo "Environment setup complete! Activate with: conda activate av-pino-motor-fault-diagnosis"
    
elif command -v python3 &> /dev/null; then
    echo "Using pip for environment setup..."
    
    # Create virtual environment
    python3 -m venv venv
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install requirements
    pip install -r requirements.txt
    
    # Install package in development mode
    pip install -e .
    
    echo "Environment setup complete! Activate with: source venv/bin/activate"
    
else
    echo "Error: Neither conda nor python3 found. Please install Python 3.8+ or Anaconda/Miniconda."
    exit 1
fi

# Verify installation
echo "Verifying installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import numpy; print(f'NumPy version: {numpy.__version__}')"
python -c "import matplotlib; print(f'Matplotlib version: {matplotlib.__version__}')"

echo "Setup verification complete!"