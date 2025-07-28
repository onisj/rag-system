# RAG System - Llama-3.1 8B Instruct with Vector Search

A complete **Retrieval-Augmented Generation (RAG)** system that answers questions about the Procyon Guide using Llama-3.1 8B Instruct quantized to INT4 and vector-based search.

## Objective

This system implements a production-ready RAG pipeline that:

- Downloads and converts **Llama-3.1 8B Instruct** to **INT4** using **OpenVINO**
- Processes the **Procyon Guide PDF** into semantic chunks
- Creates embeddings and stores them in a **FAISS** vector database
- Provides a **command-line interface** for querying with streaming responses
- Runs efficiently on **Windows x86-64** with CPU/GPU support

## Toolkit & Platform

**Platform**: Windows x86-64  
**Toolkit**: OpenVINO<sup>TM</sup> for quantized inference on CPU/GPU  
**Model**: Llama-3.1 8B Instruct (INT4 quantized)  
**Vector Store**: FAISS-GPU with sentence-transformers  
**Environment**: Conda with pinned dependencies for reproducibility

## System Requirements

### Hardware

- **RAM**: Minimum 16GB, Recommended 32GB
- **Storage**: 20GB free space for models and data
- **GPU**: NVIDIA GPU with OpenVINO support

### Software

- **OS**: Windows 10/11 (64-bit)
- **Python**: 3.11 (recommended)
- **Git**: For cloning the repository

## Quick Start

### Step 1: Environment Setup (Recommended: Conda)

#### Prerequisites: Install Conda

**Option A: Install Miniconda (Recommended)**

```bash
# Download Miniconda for your platform
# Windows: https://docs.conda.io/en/latest/miniconda.html
# Linux/macOS: https://docs.conda.io/en/latest/miniconda.html

# After installation, restart your terminal/command prompt
```

**Option B: Install Anaconda**

```bash
# Download Anaconda from: https://www.anaconda.com/download
# Follow installation instructions for your platform
```

#### Verify Conda Installation

```bash
# Check conda version
conda --version

# Expected output: conda 25.x.x or similar
```

#### Create and Activate Environment

```bash
# Create the conda environment (recommended for GPU/FAISS-GPU/OpenVINO)
conda env create -f environment.yml
conda activate rag-gpu
```

**Note**: This project uses a Conda environment with FAISS-GPU for optimal performance. The `environment.yml` file contains all pinned dependencies for complete reproducibility.

#### Troubleshooting Conda Path Issues

If you encounter conda path issues in the demo scripts, you may need to update the paths in:

**Windows (`run_demo.bat`):**

```batch
REM Update this line with your conda installation path
call "C:\Users\YOUR_USERNAME\miniconda3\Scripts\conda.exe" init cmd.exe >nul 2>&1
```

**Unix/Linux (`run_demo.sh`):**

```bash
# Update these paths with your conda installation location
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    source "/opt/conda/etc/profile.d/conda.sh"
fi
```

**Common Conda Installation Paths:**

- **Windows**: `C:\Users\USERNAME\miniconda3\` or `C:\Users\USERNAME\anaconda3\`
- **Linux**: `$HOME/miniconda3/` or `$HOME/anaconda3/` or `/opt/conda/`
- **macOS**: `$HOME/miniconda3/` or `$HOME/anaconda3/` or `/opt/conda/`

### Step 2: Configure Hugging Face Token

Create a `.env` file in the project root with your Hugging Face token:

```bash
# Create .env file
echo HUGGINGFACE_TOKEN=your_huggingface_token_here > .env
```

**Get your token from**: <https://huggingface.co/settings/tokens>

**Note**: The Llama-3.1 8B Instruct model requires special access. If you don't have access:

- Request access at: <https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct>

**Alternative**: You can also create the `.env` file manually with any text editor:

```bash
# .env file content:
HUGGINGFACE_TOKEN=your_huggingface_token_here
```

### Step 3: One-Command Demo

Run the complete pipeline with a single command:

```bash
# Windows (Command Prompt)
run_demo.bat

# Unix/Linux/macOS (Bash)
bash run_demo.sh
```

**Prerequisites for Demo Scripts:**

- Conda installed and in PATH
- `environment.yml` file present
- `rag_cli.py` executable
- Proper conda paths configured (see troubleshooting section above)

**What the Demo Scripts Do:**

1. Check conda installation and version
2. Create `rag-gpu` conda environment (if needed)
3. Activate conda environment
4. Verify core packages (OpenVINO, Transformers, Sentence-Transformers)
5. Initialize RAG system (GPU detection, model loading, vector store)
6. Execute demo query with optimized parameters
7. Display success message with usage instructions

**Expected Demo Output:**

```
========================================
    RAG System Demo - Windows/Unix/Linux
========================================

Conda version: 25.5.1
Activating rag-gpu conda environment...

Step 1: Checking RAG system components...
All core packages imported successfully

Step 2: Running demo query...
Initializing RAG system...
[System initialization details...]
Retrieved 5 relevant chunks in 9.53s
[Query processing details...]
Response: [Generated response]

========================================
    Demo completed successfully!
========================================

You can now run interactive queries with:
  python rag_cli.py --query "Your question"

Or run interactive mode with:
  python rag_cli.py
```

**Troubleshooting Demo Scripts:**

If the demo scripts fail, check:

1. **Conda Installation**: `conda --version` should work
2. **Conda Paths**: Update paths in scripts if conda is installed elsewhere
3. **Environment**: Ensure `environment.yml` exists and is valid
4. **Permissions**: Ensure scripts are executable (`chmod +x run_demo.sh` on Unix/Linux)

## Complete Step-by-Step Setup Guide

This guide will walk you through setting up the entire RAG system from scratch, including model downloading, conversion, and testing.

### Prerequisites

1. **Install Conda** (if not already installed)
2. **Get Hugging Face Token** from <https://huggingface.co/settings/tokens>
3. **Request Llama-3.1 Access** at <https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct>

### Step 1: Environment Setup

```bash
# Clone the repository (if not already done)
git clone https://github.com/onisj/ai-dev-coding
cd rag_system

# Create and activate the conda environment
conda env create -f environment.yml
conda activate rag-gpu

# Verify installation
python -c "import openvino, torch, faiss, transformers; print('All dependencies installed successfully!')"
```

### Step 2: Configure Environment Variables

```bash
# Create .env file with your Hugging Face token
echo HUGGINGFACE_TOKEN=replace_this_with_your_huggingface_token > .env

# Verify the file was created
cat .env
```

### Step 3: Prepare Data Directory

```bash
# Create data directory structure
mkdir -p data/raw data/processed_data

# Copy your procyon_guide.pdf to the data/raw/ directory
# The system expects: data/raw/procyon_guide.pdf
```

### Step 4: Download and Convert Model (INT4)

```bash
# This will download the model and convert it to INT4
python rag_cli.py convert-model

# Expected output:
# Hardware compatibility check passed
# Model downloaded successfully
# Model converted to INT4
# Model saved to: ./models/llama-3.1-8b-int4/
```

**Time**: ~30-60 minutes (depending on internet speed and hardware)

### Step 5: Process PDF and Create Vector Store

```bash
# Process the PDF and create embeddings
python rag_cli.py setup

# Expected output:
# PDF processed: 89 chunks created
# Vector store built: FAISS index with embeddings
# Setup complete!
# Data saved to: ./data/processed_data/
```

**Time**: ~5-10 minutes

### Step 6: Test the Complete System

```bash
# Test with a simple query
python rag_cli.py --query "What is Procyon?"

# Expected output:
# RAG system initialized
# Retrieved relevant chunks
# Generated response with context
```

### Step 7: Run Interactive Mode

```bash
# Start interactive querying
python rag_cli.py --query

# You can now ask questions interactively:
# > What are the main features of Procyon?
# > How does Procyon handle data processing?
# > Exit
```

### Step 8: Advanced Usage

```bash
# Query with custom parameters
python rag_cli.py --query "Explain Procyon's architecture in detail" \
  --k 10 \
  --max-tokens 1024 \
  --temperature 0.8 \
  --device GPU

# Monitor performance
python rag_cli.py performance --duration 60

# Check hardware information
python rag_cli.py hardware --detailed
```

### Step 9: Run Tests (Optional)

```bash
# Run all tests to verify everything works
python -m pytest tests/ -v

# Expected: 103 tests passed
```

## Troubleshooting Common Issues

### Issue 1: Model Download Fails

```bash
# Check your token
cat .env

# Test token manually
python -c "from huggingface_hub import HfApi; api = HfApi(); print('Token valid:', api.whoami())"
```

### Issue 2: Memory Issues

```bash
# Reduce chunk size for processing
python rag_cli.py setup --chunk-size 256

# Use CPU instead of GPU
python rag_cli.py --query "test" --device CPU
```

### Issue 3: GPU Not Detected

```bash
# Check GPU availability
python rag_cli.py hardware --detailed

# Fall back to CPU
python rag_cli.py --query "test" --device CPU
```

### Issue 4: Slow Performance

```bash
# Reduce number of retrieved chunks
python rag_cli.py --query "test" --k 3

# Use GPU acceleration
python rag_cli.py --query "test" --device GPU
```

### Issue 5: Demo Scripts Fail

**Windows (`run_demo.bat` fails):**

```batch
# Check if conda is in PATH
conda --version

# If not, update the path in run_demo.bat:
# Change: call "C:\Users\pierc\miniconda3\Scripts\conda.exe"
# To: call "C:\Users\YOUR_USERNAME\miniconda3\Scripts\conda.exe"
```

**Unix/Linux (`run_demo.sh` fails):**

```bash
# Check if conda is available
conda --version

# If not, update the conda paths in run_demo.sh:
# Add your conda installation path to the source commands
# Common paths: $HOME/miniconda3/, $HOME/anaconda3/, /opt/conda/
```

**General Demo Script Issues:**

```bash
# Make shell script executable (Unix/Linux)
chmod +x run_demo.sh

# Check if environment.yml exists
ls -la environment.yml

# Verify rag_cli.py is present
ls -la rag_cli.py
```

## Expected Directory Structure After Setup

```
rag_system/
├── data/
│   ├── raw/
│   │   └── procyon_guide.pdf     # Your input PDF
│   └── processed_data/           # Processed data and vector store
│       ├── chunks.json           # Text chunks
│       └── vector_store/         # FAISS index and embeddings
│           ├── index.faiss       # Vector index
│           ├── documents.json    # Document metadata
│           └── embeddings.npy    # Embeddings
├── models/
│   └── llama-3.1-8b-int4/        # Downloaded and converted model
│       ├── model.xml             # OpenVINO INT4 model
│       ├── model.bin             # Model weights
│       └── tokenizer.json        # Tokenizer
├── .env                          # Environment variables
└── ... (other files)
```

## Performance Expectations

- **Model Download**: 30-60 minutes (depending on internet)
- **Model Conversion**: 10-30 minutes (depending on hardware)
- **PDF Processing**: 5-10 minutes
- **Query Response**: 2-5 seconds (first query may be slower)
- **Memory Usage**: 8-12GB RAM during operation

## Success Indicators

- **Environment**: `conda activate rag-gpu` works  
- **Model**: `./models/llama-3.1-8b-int4/model.xml` exists  
- **Vector Store**: `./data/processed_data/vector_store/index.faiss` exists  
- **Query**: `python rag_cli.py --query "test"` returns a response  
- **Tests**: `python -m pytest tests/` shows 103 passed  

You now have a fully functional RAG system!

## Project Structure

```
rag_system/
├── data/
│   ├── raw/
│   │   └── procyon_guide.pdf     # Source document
│   └── processed_data/           # Processed data and vector store
│       ├── chunks.json           # Text chunks
│       └── vector_store/         # FAISS index
├── src/
│   ├── __init__.py               # Package initialization
│   ├── model_converter.py        # Model conversion (INT4)
│   ├── document_processor.py     # PDF processing & chunking
│   ├── vector_store.py           # FAISS vector database
│   ├── rag_pipeline.py           # Core RAG engine 
│   └── cli.py                    # Command-line interface
├── scripts/
│   ├── setup_model.py            # Model setup script
│   └── ingest_pdf.py             # PDF ingestion script
├── tests/
│   └── test_files                # For Unit tests
├── models/                       # Model storage
│   └── llama-3.1-8b-int4/        # Converted INT4 model
├── run_demo.sh                   # Demo script (Unix/Linux/macOS)
├── run_demo.bat                  # Demo script (Windows)
├── environment.yml               # Conda environment (pinned dependencies)
├── .env                          # Environment variables
└── README.md                     # This file
```

## Configuration

### Model Settings

- **Model**: Llama-3.1 8B Instruct
- **Quantization**: INT4 (4-bit)
- **Framework**: OpenVINO
- **Device**: CPU/GPU (auto-detected)

### Vector Store Settings

- **Embedding Model**: all-MiniLM-L6-v2
- **Dimension**: 384
- **Index Type**: FAISS-GPU IndexFlatIP
- **Similarity**: Cosine similarity
- **GPU Acceleration**: Automatic (when available)

### Chunking Settings

- **Chunk Size**: 512 characters
- **Overlap**: 50 characters
- **Strategy**: Sentence-aware splitting

## Usage Examples

### Basic Query

```bash
python rag_cli.py query "What is Procyon?"
```

### Advanced Query with Parameters

```bash
python rag_cli.py --query "Explain the main features of Procyon in detail" \
  --k 10 \
  --max-tokens 1024 \
  --temperature 0.8
```

### Interactive Mode

```bash
python rag_cli.py --query
```

### GPU Acceleration

```bash
python rag_cli.py --query "Your question here" --device GPU
```

## Development

### Running Individual Components

#### Model Conversion

```bash
python src/model_converter.py --model-name meta-llama/Llama-3.1-8B-Instruct
```

#### PDF Processing

```bash
python rag_cli.py setup --pdf-path data/raw/procyon_guide.pdf
```

#### Vector Store Operations

```bash
# Build index
python src/vector_store.py --chunks chunks.json --save-dir vector_store

# Test search
python src/vector_store.py --load-dir vector_store --query "test query"
```

#### RAG Engine Testing

```bash
python rag_cli.py --query "What is Procyon?" \
  --model-path models/llama-3.1-8b-int4/model.xml \
  --vector-store-path vector_store
```

### Testing

```bash
# Run all tests (103 tests)
python -m pytest tests/

# Run specific test categories
python -m pytest tests/test_rag_system.py -v
python -m pytest tests/test_integration.py -v
python -m pytest tests/test_cli.py -v

# Run tests with no warnings
python -m pytest tests/ --disable-warnings
```

## Performance Metrics

### Expected Performance (Windows x86-64)

- **Model Loading**: ~30-60 seconds
- **Vector Store Loading**: ~5-10 seconds
- **Query Response**: ~2-5 seconds (depending on complexity)
- **Memory Usage**: ~8-12GB RAM

### Hardware Recommendations

- **CPU**: Intel i7/i9 or AMD Ryzen 7/9
- **RAM**: 32GB DDR4
- **Storage**: NVMe SSD
- **GPU**: Intel GPU or NVIDIA GPU with OpenVINO support (optional)

## Performance Optimization

1. **GPU Usage**: Ensure OpenVINO GPU support is properly configured
2. **FAISS-GPU**: Automatically uses GPU acceleration for vector operations
3. **Memory**: Adjust chunk size based on available RAM
4. **Batch Size**: Modify embedding batch size in `vector_store.py`
5. **Index Type**: Consider using FAISS IndexIVF for large datasets

## Security Considerations

- Models are downloaded from Hugging Face (verified source)
- No external API calls during inference
- All processing happens locally
- No sensitive data is transmitted

## Self-Test Script

The `run_demo.sh` (Unix/Linux/macOS) or `run_demo.bat` (Windows) script provides a complete end-to-end test:

```bash
# Linux/macOS
./run_demo.sh

# Windows
run_demo.bat
```

This script:

1. Sets up the environment
2. Processes the PDF
3. Converts the model to INT4
4. Runs a sample query
5. Displays expected output

## Environment Variables

The system uses environment variables for configuration. Create a `.env` file in the project root:

### Required Variables

- **`HUGGINGFACE_TOKEN`**: Your Hugging Face authentication token
  - Get it from: <https://huggingface.co/settings/tokens>
  - Required for downloading Llama-3.1 8B Instruct model

### Example `.env` file

```bash
# Hugging Face Configuration
HUGGINGFACE_TOKEN=your_token_here

# Optional: Other API keys can be added here
# OPENAI_API_KEY=your_openai_key_here
# ANTHROPIC_API_KEY=your_anthropic_key_here
```

### Security Notes

- The `.env` file is automatically ignored by Git (listed in `.gitignore`)
- Never commit your actual tokens to version control
- Use different tokens for development and production environments

## Dependencies

All dependencies are pinned in `environment.yml` for complete reproducibility:

- **openvino==2025.2.0** - Intel's OpenVINO toolkit
- **transformers==4.53.3** - Hugging Face transformers
- **pytorch==2.2.2** - PyTorch with CUDA 12.1 support
- **faiss-gpu==1.8.0** - FAISS-GPU vector database with CUDA acceleration
- **sentence-transformers==5.0.0** - Sentence embeddings
- **python-dotenv==1.1.1** - Environment variable management
- **pytest==8.4.1** - Testing framework
- **rich==14.0.0** - CLI interface
- **click==8.2.1** - Command-line interface

## License

This project is for educational and evaluation purposes.

## Contributing

1. Follow PEP 8 style guidelines
2. Add proper error handling
3. Include docstrings for all functions
4. Test with different hardware configurations
5. Update documentation for any changes

## Support

For issues or questions:

1. Check the troubleshooting section
2. Review error logs
3. Ensure all dependencies are properly installed
4. Verify hardware compatibility

## Project Structure

```
rag_system/
├── data/                          # Input documents
│   └── procyon_guide.pdf         # Source PDF document
├── models/                        # Downloaded and converted models
│   └── llama-3.1-8b-int4/        # INT4 quantized model
├── vector_store/                  # FAISS index and embeddings
├── src/                          # Source code
│   ├── cli.py                    # Command-line interface
│   ├── rag_pipeline.py           # Core RAG engine
│   ├── model_converter.py        # Model conversion utilities
│   ├── document_processor.py     # PDF processing
│   ├── vector_store.py           # Vector database operations
│   ├── performance_monitor.py    # Performance monitoring
│   └── warning_config.py         # Warning management
├── tests/                        # Test suite (103 tests)
├── scripts/                      # Utility scripts
├── environment.yml               # Conda environment (pinned dependencies)
├── Dockerfile                    # Container configuration
├── rag_cli.py                    # Main CLI entry point
└── README.md                     # This file
```

## Reproducibility & Environment

This project uses a Conda environment with FAISS-GPU for optimal performance and complete reproducibility:

```bash
# Create environment with all pinned dependencies
conda env create -f environment.yml
conda activate rag-gpu

# Update environment (if dependencies change)
conda env export -n rag-gpu > environment.yml
```

**Key Features:**

- **FAISS-GPU**: Automatic GPU acceleration for vector operations
- **OpenVINO**: INT4 quantization for efficient inference
- **Pinned Dependencies**: Complete reproducibility across environments
- **Clean Structure**: Separated data, models, and code directories
- **Comprehensive Testing**: 103 tests covering all functionality
