# RAG System Dockerfile - Python-based for GPU/FAISS/OpenVINO
#
# This Dockerfile builds a fully reproducible environment using Python and pip
# for the Llama-3.1 8B Instruct RAG system.

FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data models processed_data

# Default command
CMD ["python", "rag_cli.py", "--help"] 