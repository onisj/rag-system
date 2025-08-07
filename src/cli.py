"""
RAG CLI - Retrieval-Augmented Generation Command Line Interface

This module provides a comprehensive command-line interface for the RAG system
using OpenVINO GenAI for optimized inference on Intel hardware.
"""

import os
import sys
import time
import json
import logging
import argparse
from pathlib import Path
from typing import Optional, Dict, Any, List

# Add src to path for imports
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from genai_pipeline import GenAIRAGEngine

# Configure logging for the CLI module
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize rich console for enhanced output
console = Console()

@click.group(invoke_without_command=True)
@click.version_option(version="1.0.0")
@click.option('--model-path', default=lambda: os.getenv('MODEL_PATH', './models/Llama-3.1-8B-Instruct-int4-genai/original_model'), 
              help='Path to the GenAI model directory')
@click.option('--vector-store-path', default=lambda: os.getenv('VECTOR_STORE_PATH', './data/processed_data/vector_store'), 
              help='Path to the vector store')
@click.option('--device', default=lambda: os.getenv('DEVICE', 'AUTO'), 
              help='Device to run inference on (CPU/GPU/AUTO)')
@click.option('--k', default=lambda: int(os.getenv('K_CHUNKS', '5')), 
              help='Number of chunks to retrieve')
@click.option('--max-tokens', default=lambda: int(os.getenv('MAX_TOKENS', '512')), 
              help='Maximum tokens to generate')
@click.option('--temperature', default=lambda: float(os.getenv('TEMPERATURE', '0.7')), 
              help='Sampling temperature')
@click.option('--no-stream', is_flag=True, 
              help='Disable streaming output')
@click.option('--query', help='Question to ask about the Procyon Guide')
@click.pass_context
def cli(ctx, model_path, vector_store_path, device, k, max_tokens, temperature, no_stream, query):
    """
    RAG CLI - Retrieval-Augmented Generation Command Line Interface using OpenVINO GenAI
    
    A powerful tool for querying documents using Llama-3.1 8B Instruct with vector search
    and OpenVINO GenAI optimization. Provides comprehensive functionality for document 
    processing, model management, and interactive querying with performance monitoring.
    
    Examples:
        python rag_cli.py --query "What is Procyon?"
        python rag_cli.py setup
        python rag_cli.py convert-model
    """
    # If no subcommand is provided and --query is given, run the query
    if ctx.invoked_subcommand is None and query:
        _run_query(model_path, vector_store_path, device, k, max_tokens, temperature, no_stream, query)
    elif ctx.invoked_subcommand is None and not query:
        # Show help if no subcommand and no query
        click.echo(ctx.get_help())

def _run_query(model_path, vector_store_path, device, k, max_tokens, temperature, no_stream, query):
    """
    Run a query against the RAG system using OpenVINO GenAI.
    
    Args:
        model_path: Path to the optimized GenAI INT4 model file
        vector_store_path: Path to the vector store containing document embeddings
        device: Inference device (CPU/GPU/AUTO)
        k: Number of document chunks to retrieve for context
        max_tokens: Maximum number of tokens to generate in response
        temperature: Sampling temperature for response generation
        no_stream: Flag to disable streaming output
        query: Question to ask about the Procyon Guide
    """
    try:
        # Validate that the model exists
        if not os.path.exists(model_path):
            console.print(f"GenAI model not found: {model_path}", style="red")
            console.print("Run 'python rag_cli.py convert-model' to convert the model first.", style="yellow")
            sys.exit(1)
        
        # Validate that the vector store exists before proceeding
        if not os.path.exists(vector_store_path):
            console.print(f"Vector store not found: {vector_store_path}", style="red")
            console.print("Run 'python rag_cli.py setup' to create the vector store first.", style="yellow")
            sys.exit(1)
        
        # Initialize the GenAI RAG engine with the specified configuration
        console.print("Initializing OpenVINO GenAI RAG system...", style="bold blue")
        rag_engine = GenAIRAGEngine(
            model_path=model_path,
            vector_store_path=vector_store_path,
            device=device,
            enable_monitoring=True
        )
        
        # Run the query
        console.print(f"Query: {query}", style="bold cyan")
        console.print("=" * 80, style="dim")
        
        start_time = time.time()
        
        if no_stream:
            # Non-streaming mode
            response = rag_engine.query(
                question=query,
                k=k,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=False
            )
            console.print(f"Response: {response}", style="green")
        else:
            # Streaming mode
            response = rag_engine.query(
                question=query,
                k=k,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True
            )
        
        end_time = time.time()
        
        console.print("=" * 80, style="dim")
        console.print(f"Query completed in {end_time - start_time:.2f} seconds", style="dim")
        
    except Exception as e:
        console.print(f"Query failed: {e}", style="red")
        sys.exit(1)

def _run_interactive_mode(model_path, vector_store_path, device, k, max_tokens, temperature, no_stream):
    """Run interactive mode for continuous querying"""
    try:
        console.print("Interactive mode - Type 'quit' to exit", style="bold blue")
        
        while True:
            try:
                query = input("\nEnter your question: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    console.print("Goodbye!", style="green")
                    break
                
                if not query:
                    continue
                
                _run_query(model_path, vector_store_path, device, k, max_tokens, temperature, no_stream, query)
                
            except KeyboardInterrupt:
                console.print("\nGoodbye!", style="green")
                break
            except Exception as e:
                console.print(f"Error: {e}", style="red")
                
    except Exception as e:
        console.print(f"Interactive mode failed: {e}", style="red")
        sys.exit(1)

@cli.command()
@click.option('--model-path', default=lambda: os.getenv('MODEL_PATH', './models/Llama-3.1-8B-Instruct-int4-genai/original_model'), 
              help='Path to the GenAI model directory')
@click.option('--vector-store-path', default=lambda: os.getenv('VECTOR_STORE_PATH', './data/processed_data/vector_store'), 
              help='Path to the vector store')
@click.option('--device', default=lambda: os.getenv('DEVICE', 'AUTO'), 
              help='Device to run inference on (CPU/GPU/AUTO)')
@click.option('--duration', default=lambda: int(os.getenv('MONITORING_DURATION', '60')), 
              help='Monitoring duration in seconds')
@click.option('--benchmark', is_flag=True, 
              help='Run performance benchmark')
def performance(model_path, vector_store_path, device, duration, benchmark):
    """
    Monitor system performance and run benchmarks.
    
    This command provides real-time performance monitoring and benchmarking
    capabilities for the RAG system using OpenVINO GenAI.
    """
    try:
        from performance_monitor import PerformanceMonitor
        
        monitor = PerformanceMonitor()
        
        if benchmark:
            console.print("Running performance benchmark...", style="bold blue")
            
            # Run benchmark tests
            benchmark_results = {
                "cpu_usage": monitor.get_cpu_usage(),
                "memory_usage": monitor.get_memory_usage(),
                "gpu_usage": monitor.get_gpu_usage() if monitor.has_gpu() else None
            }
            
            # Display results
            table = Table(title="Performance Benchmark Results")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            
            for metric, value in benchmark_results.items():
                if value is not None:
                    table.add_row(metric.replace("_", " ").title(), f"{value:.2f}%")
            
            console.print(table)
            
        else:
            console.print(f"Monitoring performance for {duration} seconds...", style="bold blue")
            
            # Start monitoring
            monitor.start_monitoring()
            time.sleep(duration)
            monitor.stop_monitoring()
            
            # Display results
            stats = monitor.get_stats()
            console.print("Performance monitoring completed", style="green")
            
            # Display summary
            table = Table(title="Performance Summary")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            
            for metric, value in stats.items():
                table.add_row(metric.replace("_", " ").title(), str(value))
            
            console.print(table)
            
    except Exception as e:
        console.print(f"Performance monitoring failed: {e}", style="red")
        sys.exit(1)

@cli.command()
@click.option('--detailed', is_flag=True, help='Show detailed hardware information')
def hardware(detailed):
    """
    Display hardware information and compatibility.
    
    This command shows detailed information about the available hardware
    and its compatibility with OpenVINO GenAI.
    """
    try:
        console.print("Hardware Information", style="bold blue")
        console.print("=" * 50, style="dim")
        
        # Basic hardware info
        import psutil
        import torch
        
        # CPU Information
        cpu_info = {
            "CPU Cores": psutil.cpu_count(),
            "CPU Cores (Physical)": psutil.cpu_count(logical=False),
            "CPU Usage": f"{psutil.cpu_percent()}%",
            "Memory Total": f"{psutil.virtual_memory().total / (1024**3):.2f} GB",
            "Memory Available": f"{psutil.virtual_memory().available / (1024**3):.2f} GB"
        }
        
        # GPU Information
        gpu_info = {}
        if torch.cuda.is_available():
            gpu_info = {
                "GPU Count": torch.cuda.device_count(),
                "Current GPU": torch.cuda.current_device(),
                "GPU Name": torch.cuda.get_device_name(0),
                "GPU Memory": f"{torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB"
            }
        
        # Display information
        table = Table(title="Hardware Information")
        table.add_column("Component", style="cyan")
        table.add_column("Value", style="green")
        
        for component, value in cpu_info.items():
            table.add_row(component, str(value))
        
        for component, value in gpu_info.items():
            table.add_row(component, str(value))
        
        console.print(table)
        
        if detailed:
            console.print("\nDetailed Information:", style="bold blue")
            
            # OpenVINO GenAI compatibility
            try:
                import openvino_genai as ov_genai
                console.print("OpenVINO GenAI: Available", style="green")
            except ImportError:
                console.print("OpenVINO GenAI: Not available", style="red")
            
            # PyTorch information
            console.print(f"PyTorch Version: {torch.__version__}", style="dim")
            console.print(f"CUDA Available: {torch.cuda.is_available()}", style="dim")
            
            if torch.cuda.is_available():
                console.print(f"CUDA Version: {torch.version.cuda}", style="dim")
        
    except Exception as e:
        console.print(f"Hardware detection failed: {e}", style="red")
        sys.exit(1)

def _detect_additional_gpus() -> List[Dict[str, Any]]:
    """Detect additional GPU information"""
    gpus = []
    
    try:
        # Try to detect Intel GPUs
        import subprocess
        result = subprocess.run(['nvidia-smi', '--list-gpus'], capture_output=True, text=True)
        
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if line.strip():
                    gpus.append({"type": "NVIDIA", "info": line.strip()})
        
    except FileNotFoundError:
        pass
    
    try:
        # Try to detect Intel GPUs
        import subprocess
        result = subprocess.run(['lspci'], capture_output=True, text=True)
        
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if 'VGA' in line and ('Intel' in line or 'UHD' in line):
                    gpus.append({"type": "Intel", "info": line.strip()})
        
    except FileNotFoundError:
        pass
    
    return gpus

@cli.command()
@click.option('--pdf-path', default=lambda: os.getenv('PDF_PATH', './data/raw/procyon_guide.pdf'), 
              help='Path to the Procyon Guide PDF')
@click.option('--chunk-size', default=lambda: int(os.getenv('CHUNK_SIZE', '512')), 
              help='Chunk size in characters')
@click.option('--chunk-overlap', default=lambda: int(os.getenv('CHUNK_OVERLAP', '50')), 
              help='Chunk overlap in characters')
@click.option('--output-dir', default=lambda: os.getenv('OUTPUT_DIR', './data/processed_data'), 
              help='Output directory for processed data')
def setup(pdf_path, chunk_size, chunk_overlap, output_dir):
    """
    Set up the RAG system by processing documents and creating vector store.
    
    This command processes the Procyon Guide PDF, creates text chunks,
    generates embeddings, and builds a vector store for efficient retrieval.
    """
    try:
        from document_processor import DocumentProcessor
        from vector_store import VectorStore
        
        console.print("Setting up RAG system...", style="bold blue")
        
        # Process PDF
        console.print(f"Processing PDF: {pdf_path}", style="blue")
        processor = DocumentProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = processor.process_document(pdf_path)
        
        # Save chunks
        chunks_path = Path(output_dir) / "chunks.json"
        processor.save_chunks(chunks, chunks_path)
        console.print(f"Chunks saved to: {chunks_path}", style="green")
        
        # Create vector store
        console.print("Creating vector store...", style="blue")
        store = VectorStore()
        
        # Convert chunks to documents
        documents = [{"text": chunk.text} for chunk in chunks]
        store.build_index(documents)
        
        # Save vector store
        vector_store_path = Path(output_dir) / "vector_store"
        store.save(vector_store_path)
        console.print(f"Vector store saved to: {vector_store_path}", style="green")
        
        console.print("Setup completed successfully!", style="bold green")
        
    except Exception as e:
        console.print(f"Setup failed: {e}", style="red")
        sys.exit(1)

@cli.command()
@click.option('--model-name', default=lambda: os.getenv('MODEL_NAME', 'meta-llama/Llama-3.1-8B-Instruct'),
              help='Model name to download and convert')
@click.option('--skip-download', is_flag=True,
              help='Skip downloading if model already exists')
def convert_model(model_name, skip_download):
    """
    Download and convert models to OpenVINO GenAI INT4 format.
    """
    try:
        console.print("Converting model to OpenVINO GenAI INT4 format...", style="bold blue")
        
        # Import GenAI model converter
        from genai_model_converter import GenAIModelConverter
        
        converter = GenAIModelConverter(model_name)
        converted_path = converter.run_conversion_pipeline(skip_download, skip_validation=False)
        
        console.print(f"Model conversion completed!", style="bold green")
        console.print(f"Converted model location: {converted_path}", style="green")
        console.print("Ready to use with OpenVINO GenAI!", style="green")
        
    except Exception as e:
        console.print(f"Model conversion failed: {e}", style="red")
        sys.exit(1)

@cli.command()
def demo():
    """
    Run a demonstration of the RAG system with OpenVINO GenAI.
    """
    try:
        console.print("RAG System Demo with OpenVINO GenAI", style="bold blue")
        console.print("=" * 80, style="dim")
        
        # Check if model exists
        model_path = './models/llama-3.1-8b-int4-genai/model.xml'
        if not os.path.exists(model_path):
            console.print("GenAI model not found!", style="red")
            console.print("Please run: python rag_cli.py convert-model", style="yellow")
            sys.exit(1)
        
        # Check if vector store exists
        vector_store_path = './data/processed_data/vector_store'
        if not os.path.exists(vector_store_path):
            console.print("Vector store not found!", style="red")
            console.print("Please run: python rag_cli.py setup", style="yellow")
            sys.exit(1)
        
        # Initialize RAG engine
        console.print("Initializing OpenVINO GenAI RAG system...", style="blue")
        rag_engine = GenAIRAGEngine(
            model_path=model_path,
            vector_store_path=vector_store_path,
            device="AUTO",
            enable_monitoring=True
        )
        
        # Demo queries
        demo_queries = [
            "What is Procyon?",
            "What are the main features of Procyon?",
            "How does Procyon handle data processing?"
        ]
        
        for i, query in enumerate(demo_queries, 1):
            console.print(f"Demo Query {i}: {query}", style="cyan")
            console.print("-" * 60, style="dim")
            
            start_time = time.time()
            response = rag_engine.query(query, k=3, max_tokens=256, stream=False)
            end_time = time.time()
            
            console.print(f"Response: {response}", style="green")
            console.print(f"Time: {end_time - start_time:.2f}s", style="dim")
        
        console.print("Demo completed successfully!", style="bold green")
        console.print("Try: python rag_cli.py --query 'Your question here'", style="yellow")
        
    except Exception as e:
        console.print(f"Demo failed: {e}", style="red")
        sys.exit(1)

def show_help():
    """Show help information"""
    console.print("""
RAG System with OpenVINO GenAI - Help

Available Commands:
  query <question>     - Ask a question about the Procyon Guide
  setup               - Process PDF and create vector store
  convert-model       - Download and convert model to GenAI INT4
  performance         - Monitor system performance
  hardware            - Show hardware information
  demo                - Run demonstration queries

Examples:
  python rag_cli.py --query "What is Procyon?"
  python rag_cli.py setup
  python rag_cli.py convert-model
  python rag_cli.py performance --benchmark
  python rag_cli.py hardware --detailed

For more information, visit: https://github.com/intel/openvino-genai
    """, style="cyan")

if __name__ == "__main__":
    cli()