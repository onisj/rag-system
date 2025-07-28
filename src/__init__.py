"""
RAG System - Retrieval-Augmented Generation with Llama-3.1 8B Instruct

A complete implementation of a Retrieval-Augmented Generation (RAG) system designed
for efficient document querying and intelligent response generation. The system
combines vector search capabilities with large language model inference to provide
accurate, context-aware answers to user queries.

Core Components:
    - document_processor: PDF text extraction and semantic chunking
    - vector_store: FAISS-based vector indexing and similarity search
    - model_converter: OpenVINO model conversion and optimization
    - rag_pipeline: Main RAG orchestration and inference pipeline
    - performance_monitor: Real-time system performance tracking
    - cli: Command-line interface for system interaction
    - warning_config: Warning suppression and configuration management

Key Features:
    - INT4 quantization using OpenVINO for efficient inference
    - Vector search with FAISS for fast similarity matching
    - Streaming responses for real-time user interaction
    - Modular architecture for easy maintenance and extension
    - GPU acceleration support (Intel UHD Graphics for OpenVINO, NVIDIA for embeddings)
    - Comprehensive error handling and logging
    - Performance monitoring and optimization

Architecture:
    The system follows a modular design pattern where each component has a specific
    responsibility. The document processor handles text extraction and chunking,
    the vector store manages embeddings and similarity search, the model converter
    optimizes models for deployment, and the RAG pipeline orchestrates the complete
    workflow from query to response.

Usage:
    The system can be used through the command-line interface (rag_cli.py) or
    programmatically by importing the individual components. The CLI provides
    convenient commands for setup, document processing, and querying.

Dependencies:
    - OpenVINO: For model inference and optimization
    - FAISS: For vector similarity search
    - SentenceTransformers: For text embedding generation
    - PyPDF2: For PDF text extraction
    - Rich: For console output and user interface
    - PyTorch: For deep learning operations

Author: Segun Oni
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Segun Oni"
__description__ = "Complete RAG system for Procyon Guide queries" 