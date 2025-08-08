#!/bin/bash
# Streamlit Frontend Launcher for Unix/Linux
# This script starts the Streamlit web interface for the RAG system

set -e  # Exit on any error

echo "========================================"
echo "    RAG System - Streamlit Frontend"
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

# Activate conda environment
echo "Activating rag-gpu conda environment..."
conda activate rag-gpu
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate conda environment"
    exit 1
fi

# Install Streamlit if not already installed
echo "Checking Streamlit installation..."
if ! python -c "import streamlit" &> /dev/null; then
    echo "Installing Streamlit and dependencies..."
    pip install -r requirements_streamlit.txt
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to install Streamlit dependencies"
        exit 1
    fi
fi

echo
echo "Starting Streamlit web interface..."
echo
echo "The web interface will open in your browser at:"
echo "  http://localhost:8501"
echo
echo "Press Ctrl+C to stop the server"
echo

# Start Streamlit
streamlit run streamlit_app.py --server.port 8501 --server.address localhost 