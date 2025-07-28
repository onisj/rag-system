"""
RAG CLI - Command-line interface for the RAG system

This module provides a comprehensive command-line interface for the Retrieval-Augmented 
Generation (RAG) system, enabling users to interact with the system through various 
commands and options.

Core Functionality:
- Query documents using natural language questions
- Set up and initialize the RAG system
- Convert and optimize models for inference
- Monitor system performance and run benchmarks
- Display hardware information and compatibility
- Run interactive query sessions

Key Features:
- Interactive query mode with command history
- Performance monitoring and benchmarking
- Hardware detection and compatibility checking
- Model conversion pipeline management
- Vector store setup and management
- Rich console output with progress tracking

Available Commands:
- query: Query the RAG system with questions
- setup: Initialize the system by processing PDFs and creating vector stores
- convert-model: Download and convert models to optimized formats
- performance: Monitor system performance and run benchmarks
- hardware: Display detailed hardware information
- demo: Run demonstration queries

Dependencies:
- click: Command-line interface creation
- rich: Enhanced console output and formatting
- openvino: Model inference and optimization
- psutil: System information and monitoring
- dotenv: Environment variable management

Usage Examples:
    python rag_cli.py query --query "What is Procyon?"
    python rag_cli.py setup --pdf-path ./data/raw/procyon_guide.pdf
    python rag_cli.py convert-model --model-name meta-llama/Llama-3.1-8B-Instruct
    python rag_cli.py performance --benchmark
    python rag_cli.py hardware --detailed

Author: Segun Oni
Version: 1.0.0
"""

import click
import os
import sys
import subprocess
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich import box
import logging
from dotenv import load_dotenv
import time
import openvino as ov
import psutil

# Load environment variables from .env file
load_dotenv()

from rag_pipeline import RAGEngine

# Configure logging for the CLI module
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize rich console for enhanced output
console = Console()

@click.group(invoke_without_command=True)
@click.version_option(version="1.0.0")
@click.option('--model-path', default='./models/llama-3.1-8b-int4/model.xml', 
              help='Path to the converted INT4 model')
@click.option('--vector-store-path', default='./data/processed_data/vector_store', 
              help='Path to the vector store')
@click.option('--device', default='AUTO', 
              help='Device to run inference on (CPU/GPU/AUTO)')
@click.option('--k', default=5, 
              help='Number of chunks to retrieve')
@click.option('--max-tokens', default=512, 
              help='Maximum tokens to generate')
@click.option('--temperature', default=0.7, 
              help='Sampling temperature')
@click.option('--no-stream', is_flag=True, 
              help='Disable streaming output')
@click.option('--query', help='Question to ask about the Procyon Guide')
@click.pass_context
def cli(ctx, model_path, vector_store_path, device, k, max_tokens, temperature, no_stream, query):
    """
    RAG CLI - Retrieval-Augmented Generation Command Line Interface
    
    A powerful tool for querying documents using Llama-3.1 8B Instruct with vector search.
    Provides comprehensive functionality for document processing, model management,
    and interactive querying with performance monitoring capabilities.
    
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
    Run a query against the RAG system.
    
    Args:
        model_path: Path to the optimized INT4 model file
        vector_store_path: Path to the vector store containing document embeddings
        device: Inference device (CPU/GPU/AUTO)
        k: Number of document chunks to retrieve for context
        max_tokens: Maximum number of tokens to generate in response
        temperature: Sampling temperature for response generation
        no_stream: Flag to disable streaming output
        query: Question to ask about the Procyon Guide
    """
    try:
        # Validate that the model file exists before proceeding
        if not os.path.exists(model_path):
            console.print(f"Model not found: {model_path}!!!", style="red")
            console.print("Run 'python rag_cli.py convert-model' to convert the model first.", style="yellow")
            sys.exit(1)
        
        # Validate that the vector store exists before proceeding
        if not os.path.exists(vector_store_path):
            console.print(f"Vector store not found: {vector_store_path}", style="red")
            console.print("Run 'python rag_cli.py setup' to create the vector store first.", style="yellow")
            sys.exit(1)
        
        # Initialize the RAG engine with the specified configuration
        console.print("Initializing RAG system...", style="bold blue")
        rag_engine = RAGEngine(
            model_path=model_path,
            vector_store_path=vector_store_path,
            device=device,
            enable_monitoring=True
        )
        
        # Get system statistics for display
        stats = rag_engine.get_stats()
        console.print(f"RAG system ready on device: {device}", style="green")
        
        # Single query mode - process one question and exit
        response = rag_engine.query(
            question=query,
            k=k,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=not no_stream
        )
        
        # Display the response
        if not no_stream:
            console.print("\n[bold green]Response:[/bold green]")
            console.print(response)
        else:
            console.print(f"\n[bold green]Response:[/bold green]\n{response}")
            
    except Exception as e:
        console.print(f"Error during query: {str(e)}", style="red")
        logger.error(f"Query error: {str(e)}")
        sys.exit(1)

def _run_interactive_mode(model_path, vector_store_path, device, k, max_tokens, temperature, no_stream):
    """
    Run the RAG system in interactive mode.
    
    Args:
        model_path: Path to the optimized INT4 model file
        vector_store_path: Path to the vector store containing document embeddings
        device: Inference device (CPU/GPU/AUTO)
        k: Number of document chunks to retrieve for context
        max_tokens: Maximum number of tokens to generate in response
        temperature: Sampling temperature for response generation
        no_stream: Flag to disable streaming output
    """
    try:
        # Validate that the model file exists before proceeding
        if not os.path.exists(model_path):
            console.print(f"Model not found: {model_path}!!!", style="red")
            console.print("Run 'python rag_cli.py convert-model' to convert the model first.", style="yellow")
            sys.exit(1)
        
        # Validate that the vector store exists before proceeding
        if not os.path.exists(vector_store_path):
            console.print(f"Vector store not found: {vector_store_path}", style="red")
            console.print("Run 'python rag_cli.py setup' to create the vector store first.", style="yellow")
            sys.exit(1)
        
        # Initialize the RAG engine with the specified configuration
        console.print("Initializing RAG system...", style="bold blue")
        rag_engine = RAGEngine(
            model_path=model_path,
            vector_store_path=vector_store_path,
            device=device,
            enable_monitoring=True
        )
        
        # Get system statistics for display
        stats = rag_engine.get_stats()
        console.print(f"RAG system ready on device: {device}", style="green")
        
        # Interactive mode - continuous question-answer session
        console.print("\nInteractive RAG Query Mode", style="bold blue")
        console.print("Type 'quit' or 'exit' to stop, 'help' for commands\n", style="dim")
        
        while True:
            try:
                # Get user input for the next question
                user_query = Prompt.ask("\nYour question")
                
                # Handle exit commands
                if user_query.lower() in ['quit', 'exit', 'q']:
                    console.print("Goodbye!", style="green")
                    break
                
                # Handle help command
                if user_query.lower() == 'help':
                    show_help()
                    continue
                
                # Handle statistics command
                if user_query.lower() == 'stats':
                    show_stats(stats)
                    continue
                
                # Handle performance monitoring command
                if user_query.lower() == 'performance':
                    rag_engine.display_performance_stats()
                    continue
                
                # Skip empty queries
                if not user_query.strip():
                    continue
                
                # Process the user's question through the RAG system
                response = rag_engine.query(
                    question=user_query,
                    k=k,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=not no_stream
                )
                
                # Display response based on streaming preference
                if no_stream:
                    console.print(f"\nAnswer: {response}", style="bold green")
                
            except KeyboardInterrupt:
                # Handle Ctrl+C gracefully
                console.print("\nGoodbye!", style="green")
                break
            except Exception as e:
                # Handle any errors during query processing
                console.print(f"Error: {e}!!!", style="red")
    
    except Exception as e:
        # Handle fatal errors during initialization
        console.print(f"Fatal error: {e}!!!", style="red")
        sys.exit(1)

@cli.command()
@click.option('--model-path', default='./models/llama-3.1-8b-int4/model.xml', 
              help='Path to the converted INT4 model')
@click.option('--vector-store-path', default='./data/processed_data/vector_store', 
              help='Path to the vector store')
@click.option('--device', default='AUTO', 
              help='Device to run inference on (CPU/GPU/AUTO)')
@click.option('--duration', default=60, 
              help='Monitoring duration in seconds')
@click.option('--benchmark', is_flag=True, 
              help='Run performance benchmark')
def performance(model_path, vector_store_path, device, duration, benchmark):
    """
    Performance Testing and Monitoring
    
    This command provides comprehensive performance monitoring and benchmarking
    capabilities for the RAG system. It can run live monitoring sessions or
    execute predefined benchmark tests to evaluate system performance.
    
    Args:
        model_path: Path to the optimized INT4 model file
        vector_store_path: Path to the vector store containing document embeddings
        device: Inference device (CPU/GPU/AUTO)
        duration: Duration for live monitoring in seconds
        benchmark: Flag to run performance benchmark instead of live monitoring
    """
    try:
        # Validate that the model file exists before proceeding
        if not os.path.exists(model_path):
            console.print(f"Model not found: {model_path}", style="red")
            console.print("Run 'python rag_cli.py convert-model' to convert the model first.", style="yellow")
            sys.exit(1)
        
        # Validate that the vector store exists before proceeding
        if not os.path.exists(vector_store_path):
            console.print(f"Vector store not found: {vector_store_path}", style="red")
            console.print("Run 'python rag_cli.py setup' to create the vector store first.", style="yellow")
            sys.exit(1)
        
        # Initialize RAG engine with performance monitoring enabled
        console.print("Initializing RAG system with performance monitoring...", style="bold blue")
        rag_engine = RAGEngine(
            model_path=model_path,
            vector_store_path=vector_store_path,
            device=device,
            enable_monitoring=True
        )
        
        if benchmark:
            # Run comprehensive performance benchmark with predefined test queries
            console.print("Running Performance Benchmark...", style="bold green")
            _run_performance_benchmark(rag_engine)
        else:
            # Start live performance monitoring for the specified duration
            console.print(f"Starting live performance monitoring for {duration} seconds...", style="bold green")
            console.print("Press Ctrl+C to stop early", style="yellow")
            rag_engine.start_performance_monitor(duration)
            
    except KeyboardInterrupt:
        # Handle graceful interruption of monitoring
        console.print("\nPerformance monitoring stopped by user", style="yellow")
    except Exception as e:
        # Handle errors during performance testing
        console.print(f"Performance testing failed: {e}", style="red")
        sys.exit(1)

@cli.command()
@click.option('--detailed', is_flag=True, help='Show detailed hardware information')
def hardware(detailed):
    """
    Display Hardware Information
    
    This command provides comprehensive information about the system hardware
    and OpenVINO compatibility. It detects available GPUs, their memory,
    and provides recommendations for optimal performance.
    
    Args:
        detailed: Flag to show detailed hardware information including memory detection
    """
    try:
        console.print("Hardware Information", style="bold blue")
        
        # Display basic system information
        console.print("\n[bold]System Information:[/bold]")
        console.print(f"CPU: {psutil.cpu_count()} cores")
        console.print(f"Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB")
        
        # Display OpenVINO version and available devices
        console.print("\n[bold]OpenVINO Information:[/bold]")
        core = ov.Core()
        console.print(f"OpenVINO Version: {ov.__version__}")
        console.print(f"Available Devices: {core.available_devices}")
        
        # Initialize list to store GPU information
        gpu_info = []
        gpu_devices = [d for d in core.available_devices if d.startswith("GPU")]
        
        if gpu_devices:
            console.print("\n[bold]GPU Information:[/bold]")
            
            # Process each detected GPU device
            for i, gpu_device in enumerate(gpu_devices):
                try:
                    # Get the full device name for identification
                    gpu_name = core.get_property(gpu_device, "FULL_DEVICE_NAME")
                    console.print(f"GPU {i+1} ({gpu_device}): {gpu_name}")
                    
                    # Determine GPU compatibility based on manufacturer
                    is_intel = any(keyword in gpu_name.lower() for keyword in ["intel", "uhd", "iris"])
                    is_nvidia = any(keyword in gpu_name.lower() for keyword in ["nvidia", "rtx", "geforce"])
                    is_compatible = is_intel or is_nvidia  # Support both NVIDIA and Intel GPUs
                    
                    # Initialize memory detection variables
                    total_memory = 0
                    memory_detected = False
                    
                    if detailed:
                        # Method 1: Try OpenVINO GPU memory statistics
                        try:
                            gpu_memory = core.get_property(gpu_device, "GPU_MEMORY_STATISTICS")
                            if gpu_memory:
                                total_memory = gpu_memory.get("GPU_TOTAL_MEM_SIZE", 0) / (1024**3)
                                if total_memory > 0:
                                    memory_detected = True
                                    console.print(f"  Memory: {total_memory:.1f} GB (via OpenVINO)", style="dim")
                        except Exception as e:
                            console.print(f"  OpenVINO memory detection failed: {e}", style="dim")
                        
                        # Method 2: Try alternative OpenVINO properties for memory detection
                        if not memory_detected:
                            for prop in ["GPU_MEMORY_SIZE", "DEVICE_MEMORY_SIZE", "MEMORY_SIZE"]:
                                try:
                                    memory_size = core.get_property(gpu_device, prop)
                                    if isinstance(memory_size, (int, float)) and memory_size > 0:
                                        total_memory = memory_size / (1024**3)
                                        memory_detected = True
                                        console.print(f"  Memory: {total_memory:.1f} GB (via {prop})", style="dim")
                                        break
                                except:
                                    continue
                        
                        # Method 3: Fallback to nvidia-smi for NVIDIA GPUs
                        if not memory_detected and is_nvidia:
                            try:
                                result = subprocess.run(
                                    ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                                    capture_output=True, text=True, check=True
                                )
                                memory_mb = int(result.stdout.strip().split('\n')[0])
                                total_memory = memory_mb / 1024
                                memory_detected = True
                                console.print(f"  Memory: {total_memory:.1f} GB (via nvidia-smi)", style="dim")
                            except Exception as e:
                                console.print(f"  nvidia-smi memory detection failed: {e}", style="dim")
                        
                        # Method 4: Conservative estimate if all memory detection methods fail
                        if not memory_detected:
                            total_memory = 8.0
                            console.print(f"  Memory: {total_memory:.1f} GB (conservative estimate)", style="dim")
                    
                    # Store GPU information for later analysis
                    gpu_info.append({
                        'device': gpu_device,
                        'name': gpu_name,
                        'memory': total_memory,
                        'compatible': is_compatible
                    })
                    
                    # Display compatibility status
                    if is_compatible:
                        console.print("NVIDIA GPU detected - OpenVINO support available", style="green")
                        console.print("Hardware acceleration enabled", style="green")
                    else:
                        console.print("  Non-NVIDIA GPU (e.g., Intel UHD Graphics) - not supported", style="red")
                
                except Exception as e:
                    # Handle errors during GPU information retrieval
                    console.print(f"Could not get detailed GPU info for {gpu_device}: {e}", style="yellow")
        
        # Fallback detection for additional GPUs not detected by OpenVINO
        if not gpu_info or not any(gpu['compatible'] for gpu in gpu_info):
            console.print("\nChecking for additional GPUs via nvidia-smi...", style="dim")
            additional_gpus = _detect_additional_gpus()
            for i, gpu in enumerate(additional_gpus):
                gpu_device = f"GPU.{i}"
                console.print(f"GPU {len(gpu_info)+i+1} ({gpu_device}): {gpu['name']}")
                console.print(f"Memory: {gpu['memory_gb']:.1f} GB (via nvidia-smi)", style="dim")
                console.print("NVIDIA GPU detected - OpenVINO support available", style="green")
                console.print("Hardware acceleration enabled", style="green")
                gpu_info.append({
                    'device': gpu_device,
                    'name': gpu['name'],
                    'memory': gpu['memory_gb'],
                    'compatible': True
                })
        
        # Show detailed GPU selection analysis if requested
        if detailed and gpu_info:
            console.print("\n[bold]GPU Selection Analysis:[/bold]")
            compatible_gpus = [gpu for gpu in gpu_info if gpu['compatible']]
            
            if compatible_gpus:
                # Sort compatible GPUs by memory size (highest first)
                compatible_gpus.sort(key=lambda x: x['memory'], reverse=True)
                
                console.print("Compatible GPUs ranked by memory:")
                for i, gpu in enumerate(compatible_gpus):
                    status = "SELECTED" if i == 0 else f"#{i+1}"
                    console.print(f"  {status}: {gpu['device']} - {gpu['memory']:.1f} GB ({gpu['name']})")
                
                # Provide recommendations based on the best GPU
                best_gpu = compatible_gpus[0]
                if best_gpu['memory'] >= 8:
                    console.print(f"\nWill use: {best_gpu['device']} ({best_gpu['memory']:.1f} GB) for optimal performance", style="bold green")
                else:
                    console.print(f"\n{best_gpu['device']} has limited memory ({best_gpu['memory']:.1f} GB) - may fall back to CPU", style="yellow")
            else:
                console.print("No compatible NVIDIA GPUs found - will use CPU", style="red")
        
        # Display performance recommendations
        console.print("\n[bold]Performance Recommendations:[/bold]")
        console.print("• Use device='AUTO' for automatic optimization")
        console.print("• Enable performance monitoring for real-time stats")
        console.print("• Monitor GPU memory usage during inference")
        console.print("• Consider batch processing for multiple queries")
        
    except Exception as e:
        # Handle errors during hardware information retrieval
        console.print(f"Hardware info failed: {e}", style="red")
        sys.exit(1)

def _detect_additional_gpus() -> list[dict]:
    """
    Detect additional GPUs using system commands.
    
    This function uses system-specific commands (nvidia-smi on Linux/Windows,
    wmic on Windows) to detect GPUs that might not be detected by OpenVINO.
    
    Returns:
        list[dict]: List of additional GPU information dictionaries containing
                   name, memory_gb, and source information.
    """
    additional_gpus = []
    try:
        # Try nvidia-smi for NVIDIA GPU detection
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True
        )
        # Parse nvidia-smi output for each GPU
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                parts = line.split(',')
                if len(parts) >= 2:
                    gpu_name = parts[0].strip()
                    memory_mb = int(parts[1].strip())
                    memory_gb = memory_mb / 1024
                    # Only include NVIDIA GPUs
                    if any(keyword in gpu_name for keyword in ["NVIDIA", "RTX", "GeForce"]):
                        additional_gpus.append({
                            'name': gpu_name,
                            'memory_gb': memory_gb,
                            'source': 'nvidia-smi'
                        })
    except Exception as e:
        console.print(f"nvidia-smi detection failed: {e}", style="dim")
    
    # Fallback to Windows Management Instrumentation (WMI) on Windows
    if not additional_gpus and os.name == "nt":
        try:
            result = subprocess.run(
                ["wmic", "path", "win32_VideoController", "get", "name,AdapterRAM", "/format:csv"],
                capture_output=True, text=True, check=True
            )
            # Parse WMI output for GPU information
            for line in result.stdout.strip().split('\n'):
                if line.strip() and ',' in line and 'Name' not in line:
                    parts = line.split(',')
                    if len(parts) >= 2:
                        gpu_name = parts[1].strip()
                        adapter_ram = parts[2].strip()
                        # Only include NVIDIA GPUs
                        if any(keyword in gpu_name for keyword in ["NVIDIA", "RTX", "GeForce"]):
                            try:
                                memory_bytes = int(adapter_ram)
                                memory_gb = memory_bytes / (1024**3)
                                additional_gpus.append({
                                    'name': gpu_name,
                                    'memory_gb': memory_gb,
                                    'source': 'wmic'
                                })
                            except ValueError:
                                pass
        except Exception as e:
            console.print(f"WMI GPU detection failed: {e}", style="dim")
    
    return additional_gpus

@cli.command()
@click.option('--pdf-path', default='./data/raw/procyon_guide.pdf', 
              help='Path to the Procyon Guide PDF')
@click.option('--chunk-size', default=512, 
              help='Chunk size in characters')
@click.option('--chunk-overlap', default=50, 
              help='Chunk overlap in characters')
@click.option('--output-dir', default='./data/processed_data', 
              help='Output directory for processed data')
def setup(pdf_path, chunk_size, chunk_overlap, output_dir):
    """
    Set up the RAG system by processing the PDF and creating the vector store.
    
    This command performs the complete setup process for the RAG system:
    1. Processes the PDF document into semantic chunks
    2. Creates embeddings for each chunk
    3. Builds a vector store for efficient similarity search
    
    Args:
        pdf_path: Path to the PDF document to process
        chunk_size: Size of text chunks in characters
        chunk_overlap: Overlap between consecutive chunks in characters
        output_dir: Directory to save processed data and vector store
    """
    try:
        from document_processor import DocumentProcessor
        from vector_store import VectorStore
        
        console.print("Setting up RAG system...", style="bold blue")
        
        # Validate that the PDF file exists
        if not os.path.exists(pdf_path):
            console.print(f"PDF not found: {pdf_path}!!!", style="red")
            sys.exit(1)
        
        # Create output directory structure
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Process PDF into semantic chunks
        console.print("\nStep 1: Processing PDF...", style="blue")
        processor = DocumentProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = processor.process_document(
            pdf_path=pdf_path,
            output_path=str(output_path / "chunks.json")
        )
        
        # Step 2: Create vector store with embeddings
        console.print("\nStep 2: Creating vector store...", style="blue")
        vector_store = VectorStore()
        vector_store.build_index([chunk.to_dict() for chunk in chunks])
        vector_store.save(str(output_path / "vector_store"))
        
        # Display completion message with statistics
        console.print(f"\nSetup complete! Data saved to: {output_path}", style="bold green")
        console.print(f"Processed {len(chunks)} chunks", style="green")
        
        # Provide guidance for next steps
        console.print("\nNext steps:", style="bold")
        console.print("1. Convert the model: python rag_cli.py convert-model (already done, if you follow README instructions)", style="dim")
        console.print("2. Run queries: python rag_cli.py --query 'Your question'", style="dim")
        
    except Exception as e:
        # Handle errors during setup process
        console.print(f"Setup failed: {e}!!!", style="red")
        sys.exit(1)

@cli.command()
@click.option('--model-name', default='meta-llama/Llama-3.1-8B-Instruct',
              help='Model name to download')
@click.option('--skip-download', is_flag=True,
              help='Skip downloading if model already exists')
def convert_model(model_name, skip_download):
    """
    Download and convert Llama-3.1 8B Instruct to INT4 using OpenVINO.
    
    This command downloads the specified model from Hugging Face and converts
    it to an optimized INT4 format using OpenVINO for improved performance
    and reduced memory usage.
    
    Args:
        model_name: Name of the model to download from Hugging Face
        skip_download: Flag to skip download if model already exists locally
    """
    try:
        from model_converter import ModelConverter
        
        # Create converter instance with the specified model
        converter = ModelConverter(model_name=model_name)
        
        # Run the complete conversion pipeline
        converter.run_conversion_pipeline(skip_download=skip_download)
        
    except Exception as e:
        # Handle errors during model conversion
        console.print(f"Model conversion failed: {e}!!!", style="red")
        sys.exit(1) # type: ignore[reportAttributeAccessIssue]

@cli.command()
def demo():
    """
    Run a demo query to test the RAG system.
    
    This command runs a series of predefined demonstration queries to
    test the RAG system functionality and provide examples of the
    system's capabilities.
    """
    try:
        # Define paths for model and vector store
        model_path = './models/llama-3.1-8b-int4/model.xml'
        vector_store_path = './data/processed_data/vector_store'
        
        # Validate that the model file exists
        if not os.path.exists(model_path):
            console.print("Model not found. Run 'python rag_cli.py convert-model' first.", style="red")
            sys.exit(1)
        
        # Validate that the vector store exists
        if not os.path.exists(vector_store_path):
            console.print("Vector store not found. Run 'python rag_cli.py setup' first.", style="red")
            sys.exit(1)
        
        # Initialize RAG engine for demo
        console.print("Starting RAG demo...", style="bold blue")
        rag_engine = RAGEngine(
            model_path=model_path,
            vector_store_path=vector_store_path,
            device="AUTO"
        )
        
        # Define demonstration questions to showcase system capabilities
        demo_questions = [
            "What is Procyon?",
            "What are the main features of Procyon?",
            "How does Procyon handle data processing?"
        ]
        
        # Process each demo question
        for i, question in enumerate(demo_questions, 1):
            console.print(f"\n{'='*60}", style="dim")
            console.print(f"Demo Question {i}: {question}", style="bold blue")
            console.print(f"{'='*60}", style="dim")
            
            # Query the RAG system with the demo question
            response = rag_engine.query(
                question=question,
                k=3,
                max_tokens=256,
                temperature=0.7,
                stream=True
            )
            
            console.print(f"\nDemo {i} completed", style="green")
        
        # Display completion message
        console.print(f"\nDemo completed successfully!", style="bold green")
        
    except Exception as e:
        # Handle errors during demo execution
        console.print(f"Demo failed: {e}", style="red")
        sys.exit(1)

def show_help():
    """
    Show help information for interactive mode.
    
    Displays available commands and usage instructions for the
    interactive query mode.
    """
    help_text = """
Available commands:
- help: Show this help message
- stats: Show system statistics
- quit/exit/q: Exit the program

Just type your question to get an answer about the Procyon Guide!
"""
    console.print(Panel(help_text, title="Help", border_style="blue"))

def show_stats(stats):
    """
    Show system statistics in a formatted table.
    
    Displays comprehensive system statistics including model information,
    vector store details, and performance metrics.
    
    Args:
        stats: Dictionary containing system statistics
    """
    # Create a formatted table for statistics display
    table = Table(title="RAG System Statistics", box=box.ROUNDED)
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    
    # Add basic system information
    table.add_row("Model Path", stats.get("model_path", "N/A"))
    table.add_row("Vector Store Path", stats.get("vector_store_path", "N/A"))
    table.add_row("Device", stats.get("device", "N/A"))
    
    # Add vector store statistics if available
    if stats.get("vector_store_stats"):
        vs_stats = stats["vector_store_stats"]
        table.add_row("Total Vectors", str(vs_stats.get("total_vectors", "N/A")))
        table.add_row("Total Chunks", str(vs_stats.get("total_chunks", "N/A")))
        table.add_row("Embedding Model", vs_stats.get("model_name", "N/A"))
    
    # Display the formatted table
    console.print(table)

def _run_performance_benchmark(rag_engine):
    """
    Run comprehensive performance benchmark.
    
    This function executes a series of predefined test queries to evaluate
    the RAG system's performance, measuring response times, token generation
    rates, and overall throughput.
    
    Args:
        rag_engine: Initialized RAG engine instance
    """
    console.print("Running performance benchmark...", style="blue")
    
    # Define test queries for comprehensive benchmarking
    test_queries = [
        "What is Procyon?",
        "Explain the main features of the system",
        "How does the data processing work?",
        "What are the system requirements?",
        "Describe the architecture"
    ]
    
    # Initialize performance tracking variables
    total_time = 0
    total_tokens = 0
    
    # Process each test query and measure performance
    for i, query in enumerate(test_queries, 1):
        console.print(f"\nBenchmark Query {i}/{len(test_queries)}: {query}", style="dim")
        
        # Measure query processing time
        start_time = time.time()
        response = rag_engine.query(
            question=query,
            k=3,
            max_tokens=256,
            temperature=0.7,
            stream=False
        )
        end_time = time.time()
        
        # Calculate performance metrics for this query
        query_time = end_time - start_time
        total_time += query_time
        
        # Count tokens in the generated response
        tokens = len(rag_engine.tokenizer.encode(response))
        total_tokens += tokens
        
        # Display per-query performance metrics
        console.print(f"Query time: {query_time:.2f}s, Tokens: {tokens}", style="dim")
    
    # Calculate overall performance metrics
    avg_time = total_time / len(test_queries)
    avg_tokens = total_tokens / len(test_queries)
    tokens_per_second = total_tokens / total_time
    
    # Display comprehensive benchmark results
    console.print("\n[bold green]Benchmark Results:[/bold green]")
    console.print(f"Total Queries: {len(test_queries)}")
    console.print(f"Total Time: {total_time:.2f}s")
    console.print(f"Average Query Time: {avg_time:.2f}s")
    console.print(f"Average Tokens per Query: {avg_tokens:.1f}")
    console.print(f"Throughput: {tokens_per_second:.1f} tokens/second")
    
    # Provide performance assessment based on throughput
    if tokens_per_second > 15:
        console.print("Performance: Excellent", style="green")
    elif tokens_per_second > 10:
        console.print("Performance: Good", style="yellow")
    else:
        console.print("Performance: Needs optimization", style="red")

if __name__ == "__main__":
    # Entry point for the CLI application
    cli()