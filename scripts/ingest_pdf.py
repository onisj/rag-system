"""
PDF Ingestion Script - Complete Document Processing Pipeline

This script provides a comprehensive pipeline for processing PDF documents into a searchable
vector database. It handles the complete workflow from raw PDF to indexed vectors.

Pipeline Steps:
1. PDF Text Extraction: Extract and clean text from PDF documents
2. Semantic Chunking: Split text into meaningful chunks with overlap
3. Embedding Generation: Convert text chunks to vector embeddings using Sentence Transformers
4. Vector Index Building: Create FAISS index for efficient similarity search
5. Data Persistence: Save processed data for later use

Key Features:
- Configurable chunk size and overlap for optimal text processing
- GPU-accelerated embedding generation (NVIDIA CUDA support)
- FAISS vector index with GPU acceleration when available
- Rich console interface with progress tracking
- Comprehensive error handling and validation
- Flexible output directory configuration

Example Usage:
    # Basic usage with default settings
    python scripts/ingest_pdf.py data/raw/document.pdf
    
    # Custom configuration
    python scripts/ingest_pdf.py data/raw/document.pdf \
        --output-dir ./custom_output \
        --chunk-size 256 \
        --chunk-overlap 25 \
        --embedding-model all-MiniLM-L6-v2

Dependencies:
    - src.document_processor: PDF text extraction and chunking
    - src.vector_store: Embedding generation and vector indexing
    - rich: Console output and user interface
    - argparse: Command-line argument parsing
    - pathlib: Cross-platform path handling

Author: Segun Oni
Version: 1.0.0
"""

import sys
from pathlib import Path
import argparse

# Import core RAG system components
from src.document_processor import DocumentProcessor
from src.vector_store import VectorStore

# Import UI components
from rich.console import Console
from rich.panel import Panel

# Initialize console for rich output
console = Console()

def main():
    """
    Main ingestion function that orchestrates the complete PDF processing pipeline.
    
    This function:
    1. Parses command-line arguments
    2. Validates input parameters
    3. Creates output directory structure
    4. Processes PDF into text chunks
    5. Generates embeddings and builds vector index
    6. Saves processed data
    7. Displays completion statistics
    
    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    # Set up command-line argument parser with comprehensive options
    parser = argparse.ArgumentParser(
        description="Ingest PDF document into vector store for RAG system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            %(prog)s data/raw/document.pdf
            %(prog)s data/raw/document.pdf --chunk-size 256 --chunk-overlap 25
            %(prog)s data/raw/document.pdf --output-dir ./custom_output --embedding-model all-MiniLM-L6-v2
        """
    )
    
    # Required arguments
    parser.add_argument(
        "pdf_path", 
        help="Path to the PDF file to process (should be in ./data/raw/)"
    )
    
    # Optional configuration arguments
    parser.add_argument(
        "--output-dir", 
        default="./data/processed_data", 
        help="Output directory for processed data (default: ./data/processed_data)"
    )
    parser.add_argument(
        "--chunk-size", 
        type=int, 
        default=512, 
        help="Size of text chunks in characters (default: 512)"
    )
    parser.add_argument(
        "--chunk-overlap", 
        type=int, 
        default=50, 
        help="Overlap between chunks in characters (default: 50)"
    )
    parser.add_argument(
        "--embedding-model", 
        default="all-MiniLM-L6-v2", 
        help="Sentence Transformer model for embeddings (default: all-MiniLM-L6-v2)"
    )
    
    # Parse command-line arguments
    args = parser.parse_args()
    
    # Display configuration summary to user
    console.print(Panel(
        f"PDF Ingestion Pipeline Configuration\n\n"
        f"PDF File: {args.pdf_path}\n"
        f"Output Directory: {args.output_dir}\n"
        f"  Chunk Size: {args.chunk_size} characters\n"
        f"  Chunk Overlap: {args.chunk_overlap} characters\n"
        f"Embedding Model: {args.embedding_model}",
        title="Ingestion Setup",
        border_style="blue"
    ))
    
    try:
        # Step 1: Create and validate output directory structure
        console.print("\nStep 1: Setting up output directory...", style="blue")
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        console.print(f"Output directory ready: {output_path}", style="green")
        
        # Step 2: Process PDF document into text chunks
        console.print("\nStep 2: Processing PDF document...", style="blue")
        processor = DocumentProcessor(
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap
        )
        
        # Extract text and create semantic chunks
        chunks = processor.process_document(
            pdf_path=args.pdf_path,
            output_path=str(output_path / "chunks.json")
        )
        console.print(f"PDF processed: {len(chunks)} chunks created", style="green")
        
        # Step 3: Create vector store with embeddings
        console.print("\nStep 3: Creating vector store...", style="blue")
        vector_store = VectorStore(model_name=args.embedding_model)
        
        # Convert chunks to dictionary format and build index
        chunk_dicts = [chunk.to_dict() for chunk in chunks]
        vector_store.build_index(chunk_dicts)
        
        # Save vector store to disk
        vector_store.save(str(output_path / "vector_store"))
        console.print("Vector store created and saved", style="green")
        
        # Step 4: Display completion statistics
        stats = vector_store.get_stats()
        console.print(Panel(
            f"Ingestion completed successfully!\n\n"
            f"Statistics:\n"
            f"  • Text Chunks: {len(chunks)}\n"
            f"  • Vector Embeddings: {stats['total_vectors']}\n"
            f"  • Embedding Model: {stats['model_name']}\n"
            f"  • Index Type: {stats['index_type']}\n"
            f"  • Output Directory: {output_path}\n\n"
            f"Your document is now ready for RAG queries!",
            title="Ingestion Complete",
            border_style="green"
        ))
        
        return 0
        
    except FileNotFoundError as e:
        console.print(f"File not found: {e}", style="red")
        console.print("Make sure the PDF file exists and the path is correct.", style="yellow")
        return 1
    except Exception as e:
        console.print(f"Ingestion failed: {e}", style="red")
        console.print("\nTroubleshooting tips:", style="yellow")
        console.print("  • Check if the PDF file is valid and not corrupted", style="dim")
        console.print("  • Ensure you have sufficient disk space", style="dim")
        console.print("  • Verify all dependencies are installed", style="dim")
        console.print("  • Check if the embedding model is available", style="dim")
        return 1

if __name__ == "__main__":
    # Run the main function and exit with appropriate code
    sys.exit(main()) 