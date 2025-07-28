#!/bin/bash
# RAG System Script for Unix/Linux
# This script runs the complete RAG pipeline and answers a sample query

set -e  # Exit on any error

echo "========================================"
echo "    RAG System Demo - Unix/Linux"
echo "========================================"
echo

# Try to find conda installation
CONDA_FOUND=0

# Check common conda installation paths
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    CONDA_FOUND=1
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
    CONDA_FOUND=1
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    source "/opt/conda/etc/profile.d/conda.sh"
    CONDA_FOUND=1
elif [ -f "/usr/local/conda/etc/profile.d/conda.sh" ]; then
    source "/usr/local/conda/etc/profile.d/conda.sh"
    CONDA_FOUND=1
elif [ -f "/opt/miniconda3/etc/profile.d/conda.sh" ]; then
    source "/opt/miniconda3/etc/profile.d/conda.sh"
    CONDA_FOUND=1
elif [ -f "/opt/anaconda3/etc/profile.d/conda.sh" ]; then
    source "/opt/anaconda3/etc/profile.d/conda.sh"
    CONDA_FOUND=1
fi

# Check if conda is available after sourcing
if command -v conda &> /dev/null; then
    CONDA_FOUND=1
fi

if [ $CONDA_FOUND -eq 0 ]; then
    echo "ERROR: Conda not found in common locations"
    echo "Please install Miniconda/Anaconda and try again"
    echo "Common installation paths:"
    echo "  - \$HOME/miniconda3/"
    echo "  - \$HOME/anaconda3/"
    echo "  - /opt/conda/"
    echo "  - /opt/miniconda3/"
    echo "  - /opt/anaconda3/"
    exit 1
fi

echo "Found conda installation"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "ERROR: Conda is not installed or not in PATH"
    echo "Please install Miniconda/Anaconda and try again"
    exit 1
fi

# Check conda version
conda_version=$(conda --version)
echo "Conda version: $conda_version"

# Check if rag-gpu environment exists
if ! conda env list | grep -q "rag-gpu"; then
    echo "Creating rag-gpu conda environment..."
    conda env create -f environment.yml
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to create conda environment"
        exit 1
    fi
fi

# Activate conda environment
echo "Activating rag-gpu conda environment..."
conda activate rag-gpu
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate conda environment"
    exit 1
fi

echo
echo "Step 1: Checking RAG system components..."
python -c "import openvino; import transformers; import sentence_transformers; print('All core packages imported successfully')"
if [ $? -ne 0 ]; then
    echo "ERROR: Core packages not available"
    exit 1
fi

echo
echo "Step 2: Running demo query..."
python rag_cli.py --query "What is Procyon and what are its main features?" --k 5 --max-tokens 100 --temperature 0.1
if [ $? -ne 0 ]; then
    echo "ERROR: Demo query failed"
    exit 1
fi

echo
echo "========================================"
echo "    Demo completed successfully!"
echo "========================================"
echo
echo "You can now run interactive queries with:"
echo "  python rag_cli.py --query 'Your question'"
echo
echo "Or run interactive mode with:"
echo "  python rag_cli.py"
echo 