"""
Scripts Package - Utility Scripts for RAG System

This package contains utility scripts for setting up and managing the RAG system.
These scripts provide convenient command-line interfaces for common tasks.

Available Scripts:
    - ingest_pdf.py: Complete PDF processing pipeline (text extraction, chunking, vectorization)
    - setup_model.py: Model download and conversion pipeline (Llama-3.1 8B Instruct to INT4)

Usage:
    # Process a PDF document
    python scripts/ingest_pdf.py data/raw/document.pdf
    
    # Setup the model (download and convert)
    python scripts/setup_model.py

Dependencies:
    - src.document_processor: For PDF text extraction and chunking
    - src.vector_store: For embedding generation and vector storage
    - src.model_converter: For model download and OpenVINO conversion
    - rich: For console output and user interface
    - argparse: For command-line argument parsing

Author: Segun Oni
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Segun Oni"
__description__ = "Utility scripts for RAG system setup and management"
