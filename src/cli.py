"""
RAG CLI - Retrieval-Augmented Generation Command Line Interface

This module provides a comprehensive command-line interface for the RAG system
using OpenVINO GenAI for optimized inference on Intel hardware. The CLI offers
a complete suite of tools for document processing, model management, performance
monitoring, and interactive querying capabilities.

The interface is designed to meet the UL Benchmarks technical test requirements
while providing an intuitive and powerful user experience. It supports both
one-shot queries and interactive sessions, with built-in error handling,
automatic model format detection, and comprehensive system diagnostics.

Key Features:
    - Interactive and batch query processing
    - Automatic model format detection and correction
    - Document processing and vector store creation
    - Model conversion with OpenVINO GenAI optimization
- Performance monitoring and benchmarking
    - Hardware compatibility detection
    - Comprehensive error handling and recovery
    - Environment variable configuration support

Architecture:
    The CLI is built using the Click framework for robust command-line argument
    parsing and Rich for enhanced terminal output. It integrates with all system
    components including the GenAI pipeline, vector store, document processor,
    and performance monitor.

Commands:
    - query: Execute RAG queries with document retrieval
    - setup: Process documents and create vector database
    - convert-model: Download and convert models to OpenVINO format
- performance: Monitor system performance and run benchmarks
    - hardware: Display hardware information and compatibility
    - fix-model: Automatically fix model format issues
- demo: Run demonstration queries

Configuration:
    The CLI supports configuration via environment variables loaded from .env file:
    - MODEL_PATH: Path to the model directory
    - VECTOR_STORE_PATH: Path to the vector store
    - DEVICE: Inference device (CPU/GPU/AUTO)
    - K_CHUNKS: Number of retrieval chunks
    - MAX_TOKENS: Maximum generation tokens
    - TEMPERATURE: Sampling temperature

Author: Segun Oni
Version: 1.0.1
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

# Add project root to Python path for imports
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

# Core dependencies
import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.prompt import Prompt, Confirm
from rich.live import Live
from dotenv import load_dotenv

# Load environment variables from .env file
# This must be done early to ensure all modules can access the variables
load_dotenv()

# Local module imports
try:
    from genai_pipeline import GenAIRAGEngine, create_rag_engine
except ImportError as e:
    raise ImportError(f"GenAI pipeline not found: {e}. Ensure genai_pipeline.py is available.")

# Configure module-level logging with logs directory
import datetime

# Create logs directory if it doesn't exist
logs_dir = Path("./logs")
logs_dir.mkdir(exist_ok=True)

# Create timestamped log file for CLI
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
cli_log_file = logs_dir / f"rag_cli_{timestamp}.log"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(cli_log_file),
        logging.StreamHandler(sys.stdout)  # Also output to console
    ]
)
logger = logging.getLogger(__name__)

# Initialize rich console for enhanced output
console = Console()


class ModelFormatAutoFixer:
    """
    Automatic model format detection and correction utility.
    
    This class provides intelligent model path resolution and format
    detection to handle various model conversion outputs and directory
    structures. It attempts multiple strategies to locate and validate
    working model configurations.
    """
    
    @staticmethod
    def auto_fix_model_format(model_path: str) -> Optional[str]:
        """
        Automatically detect and fix model format issues.
        
        Attempts multiple strategies to locate a working model configuration:
        1. Check for OpenVINO IR format in current path
        2. Look for GenAI-specific subdirectories
        3. Test HuggingFace format compatibility
        4. Search parent directories for alternative structures
        
        Args:
            model_path (str): Initial model path to attempt fixing
            
        Returns:
            Optional[str]: Path to working model configuration, or None if all strategies fail
        """
        logger.info(f"Attempting to auto-fix model format for: {model_path}")
        model_path_obj = Path(model_path)
        
        # Strategy 1: Check if current path has OpenVINO IR format
        if ModelFormatAutoFixer._check_openvino_ir_format(model_path_obj):
            console.print("Found OpenVINO IR model at current path", style="green")
            return model_path
        
        # Strategy 2: Look for GenAI-specific directory structures
        genai_paths = ModelFormatAutoFixer._find_genai_directories(model_path_obj)
        for path in genai_paths:
            if ModelFormatAutoFixer._check_openvino_ir_format(path):
                console.print(f"Found OpenVINO IR model in GenAI directory: {path}", style="green")
                return str(path)
        
        # Strategy 3: Test HuggingFace format compatibility
        if ModelFormatAutoFixer._test_huggingface_compatibility(model_path):
            console.print("Current path compatible with HuggingFace format", style="green")
            return model_path
        
        # Strategy 4: Search for alternative model structures
        alternative_paths = ModelFormatAutoFixer._find_alternative_structures(model_path_obj)
        for path in alternative_paths:
            if ModelFormatAutoFixer._test_model_loading(str(path)):
                console.print(f"Found working model at alternative path: {path}", style="green")
                return str(path)
        
        logger.warning(f"All model format fix strategies failed for: {model_path}")
        return None
    
    @staticmethod
    def _check_openvino_ir_format(path: Path) -> bool:
        """Check if path contains OpenVINO IR format files."""
        xml_files = list(path.glob("*.xml"))
        return len(xml_files) > 0 and any(
            xml_file.name in ["model.xml", "openvino_model.xml"] 
            for xml_file in xml_files
        )
    
    @staticmethod
    def _find_genai_directories(base_path: Path) -> List[Path]:
        """Find potential GenAI-specific directory structures."""
        candidates = []
        
        # Check common GenAI subdirectory names
        genai_subdirs = ["int4-genai", "openvino_genai", "converted"]
        
        for subdir in genai_subdirs:
            candidate = base_path / subdir
            if candidate.exists():
                candidates.append(candidate)
        
        # Check parent directory for GenAI versions
        if base_path.parent.exists():
            parent_genai = base_path.parent / f"{base_path.name}-int4-genai"
            if parent_genai.exists():
                candidates.append(parent_genai)
        
        return candidates
    
    @staticmethod
    def _test_huggingface_compatibility(model_path: str) -> bool:
        """Test if model path is compatible with HuggingFace format."""
        try:
            path_obj = Path(model_path)
            
            # Check for HuggingFace format files
            has_config = (path_obj / "config.json").exists()
            has_weights = (
                any(path_obj.glob("*.safetensors")) or 
                any(path_obj.glob("pytorch_model*.bin"))
            )
            
            return has_config and has_weights
            
        except Exception as e:
            logger.debug(f"HuggingFace compatibility test failed: {e}")
            return False
    
    @staticmethod
    def _find_alternative_structures(base_path: Path) -> List[Path]:
        """Find alternative model directory structures."""
        alternatives = []
        
        # Check parent directory variations
        if base_path.parent.exists():
            parent = base_path.parent
            
            # Look for similar named directories
            for item in parent.iterdir():
                if (item.is_dir() and 
                    item.name != base_path.name and 
                    any(keyword in item.name.lower() for keyword in ["llama", "instruct", "genai", "int4"])):
                    alternatives.append(item)
        
        # Check subdirectories of current path
        if base_path.exists() and base_path.is_dir():
            for item in base_path.iterdir():
                if item.is_dir():
                    alternatives.append(item)
        
        return alternatives
    
    @staticmethod
    def _test_model_loading(model_path: str) -> bool:
        """Test if model can be loaded with OpenVINO GenAI."""
        try:
            import openvino_genai as ov_genai
            
            # Attempt minimal model loading test
            test_pipeline = ov_genai.LLMPipeline(model_path, "CPU")
            logger.debug(f"Model loading test successful for: {model_path}")
            return True
            
        except Exception as e:
            logger.debug(f"Model loading test failed for {model_path}: {e}")
            return False


class CLIContext:
    """
    Context manager for CLI operations with shared state and error handling.
    
    Provides centralized management of CLI state, configuration, and error
    handling across all commands. Ensures consistent behavior and resource
    cleanup.
    """
    
    def __init__(self):
        self.start_time = time.time()
        self.config = self._load_configuration()
        self.errors = []
        self.warnings = []
    
    def _load_configuration(self) -> Dict[str, Any]:
        """Load configuration from environment variables with defaults."""
        return {
            'model_path': os.getenv('MODEL_PATH', './models/Llama-3.1-8B-Instruct-int4-genai'),
            'vector_store_path': os.getenv('VECTOR_STORE_PATH', './data/processed_data/vector_store'),
            'device': os.getenv('DEVICE', 'AUTO'),
            'k_chunks': int(os.getenv('K_CHUNKS', '5')),
            'max_tokens': int(os.getenv('MAX_TOKENS', '512')),
            'temperature': float(os.getenv('TEMPERATURE', '0.7')),
            'pdf_path': os.getenv('PDF_PATH', './data/raw/procyon_guide.pdf'),
            'chunk_size': int(os.getenv('CHUNK_SIZE', '512')),
            'chunk_overlap': int(os.getenv('CHUNK_OVERLAP', '50')),
            'output_dir': os.getenv('OUTPUT_DIR', './data/processed_data'),
            'model_name': os.getenv('MODEL_NAME', 'meta-llama/Llama-3.1-8B-Instruct'),
            'monitoring_duration': int(os.getenv('MONITORING_DURATION', '60')),
            'enable_monitoring': os.getenv('ENABLE_MONITORING', 'true').lower() == 'true'
        }
    
    def add_error(self, error: str) -> None:
        """Add an error message to the context."""
        self.errors.append(error)
        logger.error(error)
    
    def add_warning(self, warning: str) -> None:
        """Add a warning message to the context."""
        self.warnings.append(warning)
        logger.warning(warning)
    
    def get_elapsed_time(self) -> float:
        """Get elapsed time since context creation."""
        return time.time() - self.start_time
    
    def display_summary(self) -> None:
        """Display operation summary with errors and warnings."""
        elapsed = self.get_elapsed_time()
        
        if self.errors:
            console.print(f"\nErrors encountered ({len(self.errors)}):", style="red")
            for error in self.errors:
                console.print(f"  - {error}", style="red")
        
        if self.warnings:
            console.print(f"\nWarnings ({len(self.warnings)}):", style="yellow")
            for warning in self.warnings:
                console.print(f"  - {warning}", style="yellow")
        
        console.print(f"\nOperation completed in {elapsed:.2f} seconds", style="dim")


# Global CLI context
cli_context = CLIContext()


@click.group(invoke_without_command=True)
@click.version_option(version="1.0.0", prog_name="RAG CLI")
@click.option('--model-path', 
              default=lambda: cli_context.config['model_path'],
              help='Path to the OpenVINO GenAI model directory')
@click.option('--vector-store-path', 
              default=lambda: cli_context.config['vector_store_path'],
              help='Path to the FAISS vector store directory')
@click.option('--device', 
              default=lambda: cli_context.config['device'],
              type=click.Choice(['CPU', 'GPU', 'AUTO'], case_sensitive=False),
              help='Inference device selection (CPU/GPU/AUTO)')
@click.option('--k', 'k_chunks',
              default=lambda: cli_context.config['k_chunks'],
              type=click.IntRange(1, 20),
              help='Number of document chunks to retrieve for context')
@click.option('--max-tokens', 
              default=lambda: cli_context.config['max_tokens'],
              type=click.IntRange(1, 4096),
              help='Maximum tokens to generate in response')
@click.option('--temperature', 
              default=lambda: cli_context.config['temperature'],
              type=click.FloatRange(0.0, 2.0),
              help='Sampling temperature for response generation (0.0-2.0)')
@click.option('--no-stream', 
              is_flag=True,
              help='Disable streaming output for batch response generation')
@click.option('--query', 
              help='Question to ask about the document corpus')
@click.option('--interactive', 
              is_flag=True,
              help='Start interactive query mode')
@click.option('--verbose', 
              is_flag=True,
              help='Enable verbose logging output')
@click.pass_context
def cli(ctx, model_path, vector_store_path, device, k_chunks, max_tokens, temperature, 
        no_stream, query, interactive, verbose):
    """
    RAG CLI - Retrieval-Augmented Generation Command Line Interface
    
    A comprehensive tool for document querying using Llama-3.1 8B Instruct with
    OpenVINO GenAI optimization. Provides complete functionality for document
    processing, model management, performance monitoring, and interactive querying.
    
    This CLI implements the UL Benchmarks technical test requirements with
    enhanced features for production use including automatic error recovery,
    performance monitoring, and comprehensive diagnostics.
    
    Key Features:
        - Automatic model format detection and correction
        - Interactive and batch query processing
        - Real-time performance monitoring
        - Comprehensive hardware compatibility checking
        - Document processing and vector store management
        - Model conversion with INT4 quantization
    
    Examples:
        # Basic query
        python rag_cli.py --query "What is Procyon?"
        
        # Interactive mode
        python rag_cli.py --interactive
        
        # Custom configuration
        python rag_cli.py --device GPU --temperature 0.8 --query "Explain Procyon features"
        
        # Setup system
        python rag_cli.py setup
        
        # Convert model
        python rag_cli.py convert-model
    """
    # Configure logging level
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")
    
    # Store configuration in context
    ctx.ensure_object(dict)
    ctx.obj.update({
        'model_path': model_path,
        'vector_store_path': vector_store_path,
        'device': device.upper(),
        'k_chunks': k_chunks,
        'max_tokens': max_tokens,
        'temperature': temperature,
        'no_stream': no_stream,
        'verbose': verbose
    })
    
    # Handle main command logic
    if ctx.invoked_subcommand is None:
        if query:
            # Direct query execution
            _execute_query(ctx.obj, query)
        elif interactive:
            # Interactive mode
            _run_interactive_mode(ctx.obj)
        else:
            # Show help if no action specified
            console.print(ctx.get_help())
            _display_quick_help()


def _execute_query(config: Dict[str, Any], query: str) -> None:
    """
    Execute a single RAG query with comprehensive error handling.
    
    Args:
        config (Dict[str, Any]): CLI configuration parameters
        query (str): User query to process
        
    Raises:
        SystemExit: If query execution fails
    """
    try:
        logger.info(f"Executing query: {query[:100]}...")
        
        # Validate prerequisites
        _validate_system_prerequisites(config)
        
        # Initialize RAG engine with auto-fixing
        console.print("Initializing OpenVINO GenAI RAG system...", style="bold blue")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True
        ) as progress:
            
            init_task = progress.add_task("Loading model and vector store...", total=None)
            
            try:
                # Attempt to create RAG engine
                rag_engine = create_rag_engine(
                    model_path=config['model_path'],
                    vector_store_path=config['vector_store_path'],
                    device=config['device'],
                    enable_monitoring=cli_context.config['enable_monitoring']
                )
                
            except Exception as loading_error:
                progress.update(init_task, description="Model loading failed, attempting auto-fix...")
                
                if "Could not find a model in the directory" in str(loading_error):
                    console.print("Model format issue detected", style="yellow")
                    
                    # Attempt automatic fixing
                    fixed_path = ModelFormatAutoFixer.auto_fix_model_format(config['model_path'])
                    
                    if fixed_path:
                        console.print(f"Using fixed model at: {fixed_path}", style="green")
                        rag_engine = create_rag_engine(
                            model_path=fixed_path,
                            vector_store_path=config['vector_store_path'],
                            device=config['device'],
                            enable_monitoring=cli_context.config['enable_monitoring']
                        )
                        config['model_path'] = fixed_path  # Update for future use
                    else:
                        raise RuntimeError(
                            "Auto-fix failed. Please run model conversion:\n"
                            "python rag_cli.py convert-model"
                        )
                else:
                    raise loading_error
            
            progress.update(init_task, completed=True, description="System initialized successfully")
        
        # Execute query with timing
        console.print(f"\nQuery: {query}", style="bold cyan")
        console.print("=" * 80, style="dim")
        
        start_time = time.time()
        
        try:
            if config['no_stream']:
                # Batch response generation
                with Progress(
                    SpinnerColumn(),
                    TextColumn("Generating response..."),
                    console=console,
                    transient=True
                ) as progress:
                    task = progress.add_task("Processing...", total=None)
                    
                    response = rag_engine.query(
                        question=query,
                        k=config['k_chunks'],
                        max_tokens=config['max_tokens'],
                        temperature=config['temperature'],
                        stream=False
                    )
                    
                    progress.update(task, completed=True)
                
                # Display response
                console.print("Response:", style="bold green")
                console.print(response.answer, style="green")
                
                # Display metadata
                if config['verbose']:
                    _display_response_metadata(response)
                    
            else:
                # Streaming response generation
                console.print("Response:", style="bold green")
                response_text = rag_engine.query(
                    question=query,
                    k=config['k_chunks'],
                    max_tokens=config['max_tokens'],
                    temperature=config['temperature'],
                    stream=True
                )
        
        except Exception as query_error:
            console.print(f"\nQuery execution failed: {query_error}", style="red")
            cli_context.add_error(f"Query execution failed: {query_error}")
            raise
        
        end_time = time.time()
        
        # Display timing information
        console.print("=" * 80, style="dim")
        console.print(f"Query completed in {end_time - start_time:.2f} seconds", style="dim")
        
        # Display system stats if verbose
        if config['verbose']:
            _display_system_stats(rag_engine)
            
    except Exception as e:
        console.print(f"Query failed: {e}", style="red")
        cli_context.add_error(str(e))
        cli_context.display_summary()
        sys.exit(1)


def _run_interactive_mode(config: Dict[str, Any]) -> None:
    """
    Run interactive query mode for continuous querying.
    
    Args:
        config (Dict[str, Any]): CLI configuration parameters
    """
    try:
        console.print("OpenVINO GenAI RAG - Interactive Mode", style="bold blue")
        console.print("Type 'help' for commands, 'quit' to exit", style="dim")
        console.print("=" * 60, style="dim")
        
        # Initialize RAG engine once for the session
        with console.status("Initializing RAG engine..."):
            try:
                rag_engine = create_rag_engine(
                    model_path=config['model_path'],
                    vector_store_path=config['vector_store_path'],
                    device=config['device'],
                    enable_monitoring=cli_context.config['enable_monitoring']
                )
            except Exception as e:
                # Attempt auto-fix
                fixed_path = ModelFormatAutoFixer.auto_fix_model_format(config['model_path'])
                if fixed_path:
                    rag_engine = create_rag_engine(
                        model_path=fixed_path,
                        vector_store_path=config['vector_store_path'],
                        device=config['device'],
                        enable_monitoring=cli_context.config['enable_monitoring']
                    )
                else:
                    raise e
        
        console.print("RAG engine initialized successfully!", style="green")
        
        # Interactive loop
        query_count = 0
        
        while True:
            try:
                # Get user input
                query = Prompt.ask("\nEnter your question").strip()
                
                # Handle special commands
                if query.lower() in ['quit', 'exit', 'q']:
                    console.print("Goodbye!", style="green")
                    break
                elif query.lower() == 'help':
                    _display_interactive_help()
                    continue
                elif query.lower() == 'stats':
                    _display_system_stats(rag_engine)
                    continue
                elif query.lower() == 'config':
                    _display_current_config(config)
                    continue
                elif not query:
                    continue
                
                # Process query
                query_count += 1
                console.print(f"\n[Query {query_count}] {query}", style="cyan")
                console.print("-" * 60, style="dim")
                
                start_time = time.time()
                
                if config['no_stream']:
                    response = rag_engine.query(
                        question=query,
                        k=config['k_chunks'],
                        max_tokens=config['max_tokens'],
                        temperature=config['temperature'],
                        stream=False
                    )
                    console.print(response.answer, style="green")
                else:
                    rag_engine.query(
                        question=query,
                        k=config['k_chunks'],
                        max_tokens=config['max_tokens'],
                        temperature=config['temperature'],
                        stream=True
                    )
                
                end_time = time.time()
                console.print(f"\nCompleted in {end_time - start_time:.2f}s", style="dim")
                
            except KeyboardInterrupt:
                console.print("\nGoodbye!", style="green")
                break
            except Exception as e:
                console.print(f"\nError: {e}", style="red")
                cli_context.add_error(str(e))
                
                # Ask if user wants to continue
                if not Confirm.ask("Continue with interactive mode?", default=True):
                    break
        
        console.print(f"\nSession completed - {query_count} queries processed", style="blue")
    
    except Exception as e:
        console.print(f"Interactive mode failed: {e}", style="red")
        cli_context.add_error(str(e))
        sys.exit(1)


def _validate_system_prerequisites(config: Dict[str, Any]) -> None:
    """
    Validate system prerequisites before query execution.
    
    Args:
        config (Dict[str, Any]): CLI configuration parameters
        
    Raises:
        RuntimeError: If prerequisites are not met
    """
    errors = []
    
    # Check model path
    console.print(f"Checking model path: {config['model_path']}", style="dim")
    
    # Allow HuggingFace model identifiers (containing '/')
    if '/' in config['model_path']:
        # This is a HuggingFace model identifier, skip local path check
        console.print("Using HuggingFace model identifier", style="green")
    elif not Path(config['model_path']).exists():
        errors.append(f"Model path not found: {config['model_path']}")
    
    # Check vector store path
    if not Path(config['vector_store_path']).exists():
        errors.append(f"Vector store not found: {config['vector_store_path']}")
    
    if errors:
        console.print("System prerequisites not met:", style="red")
        for error in errors:
            console.print(f"  - {error}", style="red")
        
        console.print("\nTo fix these issues:", style="yellow")
        console.print("  - Run 'python rag_cli.py setup' to create vector store", style="yellow")
        console.print("  - Run 'python rag_cli.py convert-model' to prepare model", style="yellow")
        
        raise RuntimeError("System prerequisites not met")


def _display_response_metadata(response) -> None:
    """Display detailed response metadata."""
    if hasattr(response, 'metadata'):
        table = Table(title="Response Metadata", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        for key, value in response.metadata.items():
            table.add_row(str(key).replace('_', ' ').title(), str(value))
        
        console.print(table)


def _display_system_stats(rag_engine) -> None:
    """Display comprehensive system statistics."""
    try:
        stats = rag_engine.get_system_stats()
        
        table = Table(title="System Statistics", show_header=True, header_style="bold blue")
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Details", style="dim")
        
        # Core system information
        table.add_row("Device", stats.get('device', 'Unknown'), "")
        table.add_row("Model Loaded", str(stats.get('model_loaded', False)), "")
        table.add_row("Vector Store", str(stats.get('vector_store_loaded', False)), "")
        
        # Hardware information
        if stats.get('torch_cuda_available'):
            gpu_name = stats.get('cuda_device_name', 'Unknown')
            gpu_memory = stats.get('cuda_memory_total', 0) / (1024**3)
            table.add_row("GPU", gpu_name, f"{gpu_memory:.1f}GB")
        
        console.print(table)
        
    except Exception as e:
        console.print(f"Could not retrieve system stats: {e}", style="yellow")


def _display_current_config(config: Dict[str, Any]) -> None:
    """Display current CLI configuration."""
    table = Table(title="Current Configuration", show_header=True, header_style="bold cyan")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")
    
    for key, value in config.items():
        table.add_row(str(key).replace('_', ' ').title(), str(value))
    
    console.print(table)


def _display_interactive_help() -> None:
    """Display help for interactive mode."""
    help_panel = Panel(
        """[bold cyan]Interactive Mode Commands[/bold cyan]

        [yellow]Special Commands:[/yellow]
        help    - Show this help message
        stats   - Display system statistics
        config  - Show current configuration
        quit    - Exit interactive mode

        [yellow]Query Examples:[/yellow]
        What is Procyon?
        Explain the main features
        How does data processing work?

        [yellow]Tips:[/yellow]
        - Press Ctrl+C to exit at any time
        - Questions are processed with current configuration
        - Use 'stats' to monitor system performance
                """,
        title="Help",
        border_style="blue"
    )
    console.print(help_panel)


def _display_quick_help() -> None:
    """Display quick help information."""
    help_panel = Panel(
        """[bold cyan]RAG System with OpenVINO GenAI[/bold cyan]

        [yellow]Quick Start:[/yellow]
        python rag_cli.py setup                    # Setup system
        python rag_cli.py convert-model            # Convert model
        python rag_cli.py --query "What is Procyon?"   # Ask question

        [yellow]Available Commands:[/yellow]
        setup           - Process documents and create vector store
        convert-model   - Download and convert model to OpenVINO format
        performance     - Monitor system performance
        hardware        - Display hardware information
        fix-model       - Automatically fix model format issues
        demo            - Run demonstration queries

        [yellow]Options:[/yellow]
        --interactive   - Start interactive mode
        --device GPU    - Force GPU inference
        --verbose       - Enable detailed logging
                """,
        title="Quick Help",
        border_style="green"
    )
    console.print(help_panel)


@cli.command()
@click.option('--pdf-path', 
              default=lambda: cli_context.config['pdf_path'],
              type=click.Path(exists=True),
              help='Path to the Procyon Guide PDF document')
@click.option('--chunk-size', 
              default=lambda: cli_context.config['chunk_size'],
              type=click.IntRange(100, 2048),
              help='Text chunk size in characters')
@click.option('--chunk-overlap', 
              default=lambda: cli_context.config['chunk_overlap'],
              type=click.IntRange(0, 200),
              help='Text chunk overlap in characters')
@click.option('--output-dir', 
              default=lambda: cli_context.config['output_dir'],
              type=click.Path(),
              help='Output directory for processed data')
@click.option('--force', 
              is_flag=True,
              help='Force recreation of existing vector store')
def setup(pdf_path, chunk_size, chunk_overlap, output_dir, force):
    """
    Set up the RAG system by processing documents and creating vector store.
    
    This command implements the complete document processing pipeline:
    1. PDF text extraction and validation
    2. Intelligent text chunking with overlap
    3. Embedding generation using sentence-transformers
    4. FAISS vector index creation and optimization
    5. Metadata extraction and storage
    
    The setup process is designed to handle large documents efficiently
    while maintaining semantic coherence in text chunks. Progress is
    displayed in real-time with detailed status updates.
    
    Args:
        pdf_path: Path to the Procyon Guide PDF
        chunk_size: Size of text chunks in characters
        chunk_overlap: Overlap between chunks in characters
        output_dir: Directory for storing processed data
        force: Force recreation of existing data
    """
    try:
        logger.info("Starting RAG system setup")
        console.print("Setting up RAG system...", style="bold blue")
        
        # Validate inputs
        if not Path(pdf_path).exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Check for existing data
        chunks_path = output_path / "chunks.json"
        vector_store_path = output_path / "vector_store"
        
        if not force and chunks_path.exists() and vector_store_path.exists():
            if not Confirm.ask(
                "Processed data already exists. Recreate?", 
                default=False
            ):
                console.print("Setup cancelled by user", style="yellow")
                return
        
        # Import required modules
        try:
            from document_processor import DocumentProcessor
            from vector_store import VectorStore
        except ImportError as e:
            raise ImportError(f"Required modules not found: {e}")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=console
        ) as progress:
            
            # Step 1: Process PDF document
            process_task = progress.add_task("Processing PDF document...", total=100)
            
            processor = DocumentProcessor(
                chunk_size=chunk_size, 
                chunk_overlap=chunk_overlap
            )
            
            progress.update(process_task, advance=20, description="Extracting text from PDF...")
            chunks = processor.process_document(pdf_path)
            
            progress.update(process_task, advance=40, description="Creating text chunks...")
            
            # Validate chunk extraction
            if not chunks:
                raise RuntimeError("No text chunks extracted from PDF")
            
            # Save chunks with metadata
            progress.update(process_task, advance=20, description="Saving chunks...")
            processor.save_chunks(chunks, chunks_path)
            
            progress.update(process_task, completed=100, description="PDF processing completed")
            
            # Step 2: Create vector store
            vector_task = progress.add_task("Creating vector store...", total=100)
            
            store = VectorStore()
            
            progress.update(vector_task, advance=20, description="Initializing embeddings model...")
            
            # Convert chunks to documents for vector store
            documents = []
            for i, chunk in enumerate(chunks):
                documents.append({
                    "text": chunk.text,
                    "metadata": {
                        "chunk_id": i,
                        "source": pdf_path,
                        "page": getattr(chunk, 'page', None),
                        "chunk_size": len(chunk.text)
                    }
                })
            
            progress.update(vector_task, advance=30, description="Generating embeddings...")
            store.build_index(documents)
            
            progress.update(vector_task, advance=30, description="Building FAISS index...")
            
            # Save vector store
            progress.update(vector_task, advance=15, description="Saving vector store...")
            store.save(vector_store_path)
            
            progress.update(vector_task, completed=100, description="Vector store creation completed")
        
        # Display summary
        console.print("\nSetup Summary:", style="bold green")
        
        summary_table = Table(show_header=False, box=None)
        summary_table.add_column("Item", style="cyan")
        summary_table.add_column("Details", style="green")
        
        summary_table.add_row("PDF Processed", str(pdf_path))
        summary_table.add_row("Chunks Created", str(len(chunks)))
        summary_table.add_row("Chunk Size", f"{chunk_size} characters")
        summary_table.add_row("Chunk Overlap", f"{chunk_overlap} characters")
        summary_table.add_row("Chunks File", str(chunks_path))
        summary_table.add_row("Vector Store", str(vector_store_path))
        
        console.print(summary_table)
        console.print("\nSetup completed successfully!", style="bold green")
        console.print("You can now run queries with:", style="blue")
        console.print("python rag_cli.py --query 'What is Procyon?'", style="dim")
        
    except Exception as e:
        console.print(f"Setup failed: {e}", style="red")
        cli_context.add_error(f"Setup failed: {e}")
        cli_context.display_summary()
        sys.exit(1)


@cli.command()
@click.option('--model-name', 
              default=lambda: cli_context.config['model_name'],
              help='HuggingFace model identifier to download and convert')
@click.option('--skip-download', 
              is_flag=True,
              help='Skip downloading if model already exists locally')
@click.option('--skip-validation', 
              is_flag=True,
              help='Skip model validation after conversion')
@click.option('--force', 
              is_flag=True,
              help='Force reconversion even if model exists')
def convert_model(model_name, skip_download, skip_validation, force):
    """
    Download and convert HuggingFace models to OpenVINO GenAI format.
    
    This command implements the complete model conversion pipeline required
    by the UL Benchmarks technical test:
    1. HuggingFace model download with authentication
    2. Model conversion to OpenVINO IR format
    3. INT4 quantization for optimal performance
    4. Compatibility validation with OpenVINO GenAI
    5. Performance optimization and compilation
    
    The conversion process is optimized for Intel hardware and includes
    comprehensive error handling and progress reporting.
    
    Args:
        model_name: HuggingFace model identifier
        skip_download: Skip download if model exists
        skip_validation: Skip validation after conversion
        force: Force reconversion of existing models
    """
    try:
        logger.info(f"Starting model conversion for: {model_name}")
        console.print("Converting model to OpenVINO GenAI format...", style="bold blue")
        
        # Import model converter
        try:
            from genai_model_converter import GenAIModelConverter
        except ImportError as e:
            raise ImportError(f"Model converter not found: {e}")
        
        # Check for existing conversion
        converter = GenAIModelConverter(model_name)
        model_info = converter.get_model_info()
        
        if not force and "available" in model_info:
            if not Confirm.ask(
                f"Model already converted: {model_info}. Reconvert?", 
                default=False
            ):
                console.print("Conversion cancelled by user", style="yellow")
                return
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=console
        ) as progress:
            
            # Run conversion pipeline
            convert_task = progress.add_task("Converting model...", total=100)
            
            progress.update(convert_task, advance=10, description="Initializing converter...")
            
            try:
                converted_path = converter.run_conversion_pipeline(
                    skip_download=skip_download,
                    skip_validation=skip_validation
                )
                
                progress.update(convert_task, advance=80, description="Conversion completed...")
                
                # Validate conversion if requested
                if not skip_validation:
                    progress.update(convert_task, advance=5, description="Validating conversion...")
                    
                    validation_success = converter._validate_genai_model(converted_path)
                    
                    if validation_success:
                        progress.update(convert_task, advance=5, description="Validation successful")
                        console.print("Model conversion and validation completed successfully!", style="bold green")
                    else:
                        progress.update(convert_task, advance=5, description="Validation failed")
                        console.print("Model converted but validation failed", style="yellow")
                        cli_context.add_warning("Model validation failed but conversion completed")
                else:
                    progress.update(convert_task, advance=10, description="Skipping validation")
                
                progress.update(convert_task, completed=100)
                
            except Exception as conversion_error:
                progress.update(convert_task, description="Conversion failed")
                raise conversion_error
        
        # Display conversion summary
        console.print("\nConversion Summary:", style="bold green")
        
        summary_table = Table(show_header=False, box=None)
        summary_table.add_column("Item", style="cyan")
        summary_table.add_column("Details", style="green")
        
        summary_table.add_row("Model", model_name)
        summary_table.add_row("Output Path", str(converted_path))
        summary_table.add_row("Format", "OpenVINO IR (INT4)")
        summary_table.add_row("Status", "Ready for inference")
        
        console.print(summary_table)
        console.print("\nYou can now run queries with:", style="blue")
        console.print("python rag_cli.py --query 'What is Procyon?'", style="dim")
        
    except Exception as e:
        console.print(f"Model conversion failed: {e}", style="red")
        cli_context.add_error(f"Model conversion failed: {e}")
        
        # Display troubleshooting help
        console.print("\nTroubleshooting tips:", style="yellow")
        console.print("1. Check your internet connection", style="dim")
        console.print("2. Verify your HuggingFace token in .env file", style="dim")
        console.print("3. Ensure you have access to the model", style="dim")
        console.print("4. Check available disk space (16GB+ required)", style="dim")
        
        cli_context.display_summary()
        sys.exit(1)


@cli.command()
@click.option('--model-path', 
              default='./models/Llama-3.1-8B-Instruct',
              type=click.Path(exists=True),
              help='Path to model directory that needs fixing')
@click.option('--test-loading', 
              is_flag=True,
              help='Test model loading after fixing')
def fix_model(model_path, test_loading):
    """
    Automatically detect and fix model format issues.
    
    This command provides intelligent model path resolution and format
    detection to handle various model conversion outputs and directory
    structures. It attempts multiple strategies to locate and validate
    working model configurations.
    
    The fix process includes:
    1. OpenVINO IR format detection
    2. HuggingFace format compatibility testing
    3. Alternative directory structure search
    4. Model loading validation
    
    Args:
        model_path: Path to model directory requiring fixes
        test_loading: Perform model loading test after fixing
    """
    try:
        logger.info(f"Attempting to fix model format issues for: {model_path}")
        
        console.print("OpenVINO GenAI Model Fix Utility", style="bold blue")
        console.print("=" * 50, style="dim")
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model path not found: {model_path}")
        
        console.print(f"Analyzing model at: {model_path}", style="blue")
        
        with console.status("Analyzing model format..."):
            # Run the auto-fix process
            fixed_path = ModelFormatAutoFixer.auto_fix_model_format(model_path)
        
        if fixed_path:
            console.print("Model fix completed successfully!", style="bold green")
            
            # Display fix results
            fix_table = Table(title="Fix Results", show_header=False, box=None)
            fix_table.add_column("Item", style="cyan")
            fix_table.add_column("Details", style="green")
            
            fix_table.add_row("Original Path", model_path)
            fix_table.add_row("Working Path", fixed_path)
            fix_table.add_row("Status", "Ready for use")
            
            console.print(fix_table)
            
            # Test loading if requested
            if test_loading:
                console.print("\nTesting model loading...", style="blue")
                
                try:
                    import openvino_genai as ov_genai
                    test_pipeline = ov_genai.LLMPipeline(fixed_path, "CPU")
                    
                    # Perform minimal inference test
                    test_result = test_pipeline.generate("Hello", max_length=5)
                    
                    console.print("Model loading test successful!", style="green")
                    console.print(f"Test output: {test_result}", style="dim")
                    
                except Exception as test_error:
                    console.print(f"Model loading test failed: {test_error}", style="yellow")
                    cli_context.add_warning(f"Model loading test failed: {test_error}")
            
            console.print("\nYou can now run queries with:", style="blue")
            console.print(f"python rag_cli.py --query 'What is Procyon?' --model-path '{fixed_path}'", style="dim")
            
        else:
            console.print("Auto-fix failed - unable to resolve model format issues", style="red")
            
            console.print("\nManual steps to try:", style="yellow")
            console.print("1. Run model conversion: python rag_cli.py convert-model", style="dim")
            console.print("2. Check model directory contents", style="dim")
            console.print("3. Verify model was downloaded correctly", style="dim")
            
            cli_context.add_error("Auto-fix failed to resolve model issues")
            cli_context.display_summary()
            sys.exit(1)
        
    except Exception as e:
        console.print(f"Model fix failed: {e}", style="red")
        cli_context.add_error(f"Model fix failed: {e}")
        cli_context.display_summary()
        sys.exit(1)


@cli.command()
@click.option('--duration', 
              default=lambda: cli_context.config['monitoring_duration'],
              type=click.IntRange(10, 600),
              help='Monitoring duration in seconds')
@click.option('--benchmark', 
              is_flag=True,
              help='Run comprehensive performance benchmark')
@click.option('--export', 
              type=click.Path(),
              help='Export results to JSON file')
def performance(duration, benchmark, export):
    """
    Monitor system performance and run comprehensive benchmarks.
    
    This command provides detailed performance monitoring and benchmarking
    capabilities for the RAG system. It tracks CPU usage, memory consumption,
    GPU utilization, and inference performance metrics.
    
    Monitoring includes:
    1. Real-time system resource monitoring
    2. RAG inference performance benchmarking
    3. Hardware utilization analysis
    4. Memory usage tracking
    5. Detailed performance reporting
    
    Args:
        duration: Monitoring duration in seconds
        benchmark: Run comprehensive benchmark suite
        export: Export results to JSON file
    """
    try:
        logger.info("Starting performance monitoring")
        
        # Import performance monitor
        try:
            from performance_monitor import PerformanceMonitor, create_performance_monitor
        except ImportError as e:
            raise ImportError(f"Performance monitor not found: {e}")
        
        monitor = create_performance_monitor()
        
        if benchmark:
            console.print("Running comprehensive performance benchmark...", style="bold blue")
            
            benchmark_results = {}
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                console=console
            ) as progress:
                
                # System benchmark
                sys_task = progress.add_task("System benchmark...", total=100)
                
                progress.update(sys_task, advance=25, description="Measuring CPU performance...")
                benchmark_results["cpu_usage"] = monitor.get_cpu_usage()
                
                progress.update(sys_task, advance=25, description="Measuring memory usage...")
                benchmark_results["memory_usage"] = monitor.get_memory_usage()
                
                progress.update(sys_task, advance=25, description="Measuring GPU performance...")
                if monitor.has_gpu():
                    benchmark_results["gpu_usage"] = monitor.get_gpu_usage()
                    benchmark_results["gpu_memory"] = monitor.get_gpu_memory_usage()
                
                progress.update(sys_task, advance=25, description="System benchmark completed")
                
                # RAG benchmark if system is set up
                try:
                    model_path = cli_context.config['model_path']
                    vector_store_path = cli_context.config['vector_store_path']
                    
                    if Path(model_path).exists() and Path(vector_store_path).exists():
                        rag_task = progress.add_task("RAG benchmark...", total=100)
                        
                        progress.update(rag_task, advance=20, description="Initializing RAG engine...")
                        
                        try:
                            rag_engine = create_rag_engine(
            model_path=model_path,
            vector_store_path=vector_store_path,
                                device="AUTO",
                                enable_monitoring=True
                            )
                            
                            progress.update(rag_task, advance=30, description="Running query benchmark...")
                            
                            # Benchmark query
                            test_queries = [
            "What is Procyon?",
                                "Explain the main features",
                                "How does data processing work?"
                            ]
                            
                            query_times = []
                            for query in test_queries:
                                start_time = time.time()
                                rag_engine.query(query, k=3, max_tokens=100, stream=False)
                                end_time = time.time()
                                query_times.append(end_time - start_time)
                            
                            benchmark_results["avg_query_time"] = sum(query_times) / len(query_times)
                            benchmark_results["min_query_time"] = min(query_times)
                            benchmark_results["max_query_time"] = max(query_times)
                            
                            progress.update(rag_task, completed=100, description="RAG benchmark completed")
                            
                        except Exception as rag_error:
                            progress.update(rag_task, description="RAG benchmark failed")
                            cli_context.add_warning(f"RAG benchmark failed: {rag_error}")
                
                except Exception as e:
                    cli_context.add_warning(f"RAG benchmark skipped: {e}")
            
            # Display benchmark results
            console.print("\nBenchmark Results:", style="bold green")
            
            results_table = Table(title="Performance Benchmark", show_header=True, header_style="bold magenta")
            results_table.add_column("Metric", style="cyan")
            results_table.add_column("Value", style="green")
            results_table.add_column("Unit", style="dim")
            
            for metric, value in benchmark_results.items():
                if value is not None:
                    if "time" in metric:
                        results_table.add_row(
                            metric.replace("_", " ").title(), 
                            f"{value:.3f}", 
                            "seconds"
                        )
                    elif "usage" in metric:
                        results_table.add_row(
                            metric.replace("_", " ").title(), 
                            f"{value:.1f}", 
                            "percent"
                        )
                    else:
                        results_table.add_row(
                            metric.replace("_", " ").title(), 
                            str(value), 
                            ""
                        )
            
            console.print(results_table)
            
        else:
            # Real-time monitoring
            console.print(f"Monitoring system performance for {duration} seconds...", style="bold blue")
            
            with Live(console=console, refresh_per_second=2) as live:
                monitor.start_monitoring()
                
                for i in range(duration):
                    # Create monitoring display
                    table = Table(title=f"Real-time Performance Monitor ({i+1}/{duration}s)")
                    table.add_column("Metric", style="cyan")
                    table.add_column("Current", style="green")
                    table.add_column("Average", style="yellow")
                    
                    current_stats = monitor.get_current_stats()
                    avg_stats = monitor.get_average_stats()
                    
                    table.add_row("CPU Usage", f"{current_stats.get('cpu', 0):.1f}%", f"{avg_stats.get('cpu', 0):.1f}%")
                    table.add_row("Memory Usage", f"{current_stats.get('memory', 0):.1f}%", f"{avg_stats.get('memory', 0):.1f}%")
                    
                    if monitor.has_gpu():
                        table.add_row("GPU Usage", f"{current_stats.get('gpu', 0):.1f}%", f"{avg_stats.get('gpu', 0):.1f}%")
                    
                    live.update(table)
                    time.sleep(1)
                
                monitor.stop_monitoring()
            
            # Display final statistics
            final_stats = monitor.get_stats()
            console.print("\nMonitoring completed. Final statistics:", style="bold green")
            
            stats_table = Table(show_header=True, header_style="bold blue")
            stats_table.add_column("Metric", style="cyan")
            stats_table.add_column("Value", style="green")
            
            for metric, value in final_stats.items():
                stats_table.add_row(metric.replace("_", " ").title(), str(value))
            
            console.print(stats_table)
        
        # Export results if requested
        if export:
            with open(export, 'w') as f:
                json.dump(benchmark_results if benchmark else final_stats, f, indent=2)
            console.print(f"\nResults exported to: {export}", style="blue")
        
    except Exception as e:
        console.print(f"Performance monitoring failed: {e}", style="red")
        cli_context.add_error(f"Performance monitoring failed: {e}")
        cli_context.display_summary()
        sys.exit(1)


@cli.command()
@click.option('--detailed', 
              is_flag=True,
              help='Show detailed hardware and compatibility information')
@click.option('--export', 
              type=click.Path(),
              help='Export hardware information to JSON file')
def hardware(detailed, export):
    """
    Display comprehensive hardware information and OpenVINO compatibility.
    
    This command provides detailed analysis of the system hardware and its
    compatibility with OpenVINO GenAI. It detects available compute devices,
    memory configurations, and optimization capabilities.
    
    Hardware analysis includes:
    1. CPU specifications and capabilities
    2. GPU detection and memory analysis
    3. System memory and storage information
    4. OpenVINO GenAI compatibility assessment
    5. Performance recommendations
    
    Args:
        detailed: Show comprehensive hardware details
        export: Export hardware information to JSON file
    """
    try:
        logger.info("Gathering hardware information")
        console.print("Hardware Information and Compatibility Analysis", style="bold blue")
        console.print("=" * 60, style="dim")
        
        # Import required modules for hardware detection
        try:
            import psutil
            import torch
            import platform
        except ImportError as e:
            raise ImportError(f"Required modules for hardware detection not found: {e}")
        
        hardware_info = {}
        
        # System information
        hardware_info["system"] = {
            "platform": platform.system(),
            "architecture": platform.architecture()[0],
            "processor": platform.processor(),
            "python_version": platform.python_version()
        }
        
        # CPU information
        hardware_info["cpu"] = {
            "cores_logical": psutil.cpu_count(logical=True),
            "cores_physical": psutil.cpu_count(logical=False),
            "current_usage": psutil.cpu_percent(interval=1),
            "frequency_current": psutil.cpu_freq().current if psutil.cpu_freq() else "Unknown",
            "frequency_max": psutil.cpu_freq().max if psutil.cpu_freq() else "Unknown"
        }
        
        # Memory information
        memory = psutil.virtual_memory()
        hardware_info["memory"] = {
            "total_gb": round(memory.total / (1024**3), 2),
            "available_gb": round(memory.available / (1024**3), 2),
            "used_percent": memory.percent,
            "swap_total_gb": round(psutil.swap_memory().total / (1024**3), 2)
        }
        
        # GPU information
        hardware_info["gpu"] = {}
        if torch.cuda.is_available():
            hardware_info["gpu"]["cuda_available"] = True
            hardware_info["gpu"]["device_count"] = torch.cuda.device_count()
            hardware_info["gpu"]["current_device"] = torch.cuda.current_device()
            hardware_info["gpu"]["device_name"] = torch.cuda.get_device_name(0)
            
            gpu_props = torch.cuda.get_device_properties(0)
            hardware_info["gpu"]["total_memory_gb"] = round(gpu_props.total_memory / (1024**3), 2)
            hardware_info["gpu"]["compute_capability"] = f"{gpu_props.major}.{gpu_props.minor}"
            
            # GPU memory usage
            hardware_info["gpu"]["memory_allocated_gb"] = round(torch.cuda.memory_allocated(0) / (1024**3), 2)
            hardware_info["gpu"]["memory_reserved_gb"] = round(torch.cuda.memory_reserved(0) / (1024**3), 2)
        else:
            hardware_info["gpu"]["cuda_available"] = False
        
        # OpenVINO compatibility
        hardware_info["openvino"] = {}
        try:
            import openvino_genai as ov_genai
            hardware_info["openvino"]["genai_available"] = True
            
            # Try to get OpenVINO version
            try:
                hardware_info["openvino"]["version"] = ov_genai.get_version()
            except AttributeError:
                hardware_info["openvino"]["version"] = "Unknown"
                
        except ImportError:
            hardware_info["openvino"]["genai_available"] = False
        
        # Display basic information
        basic_table = Table(title="System Overview", show_header=True, header_style="bold cyan")
        basic_table.add_column("Component", style="cyan")
        basic_table.add_column("Specification", style="green")
        
        basic_table.add_row("Operating System", f"{hardware_info['system']['platform']} ({hardware_info['system']['architecture']})")
        basic_table.add_row("CPU Cores", f"{hardware_info['cpu']['cores_physical']} physical, {hardware_info['cpu']['cores_logical']} logical")
        basic_table.add_row("Memory", f"{hardware_info['memory']['total_gb']} GB total, {hardware_info['memory']['available_gb']} GB available")
        
        if hardware_info["gpu"]["cuda_available"]:
            basic_table.add_row("GPU", f"{hardware_info['gpu']['device_name']} ({hardware_info['gpu']['total_memory_gb']} GB)")
        else:
            basic_table.add_row("GPU", "No CUDA GPU detected")
        
        basic_table.add_row("OpenVINO GenAI", "Available" if hardware_info["openvino"]["genai_available"] else "Not available")
        
        console.print(basic_table)
        
        # Detailed information
        if detailed:
            console.print("\nDetailed Hardware Analysis:", style="bold blue")
            
            # CPU details
            cpu_table = Table(title="CPU Details", show_header=True, header_style="bold magenta")
            cpu_table.add_column("Metric", style="cyan")
            cpu_table.add_column("Value", style="green")
            
            cpu_table.add_row("Current Usage", f"{hardware_info['cpu']['current_usage']}%")
            cpu_table.add_row("Current Frequency", f"{hardware_info['cpu']['frequency_current']} MHz")
            cpu_table.add_row("Max Frequency", f"{hardware_info['cpu']['frequency_max']} MHz")
            
            console.print(cpu_table)
            
            # Memory details
            memory_table = Table(title="Memory Details", show_header=True, header_style="bold magenta")
            memory_table.add_column("Metric", style="cyan")
            memory_table.add_column("Value", style="green")
            
            memory_table.add_row("Usage Percentage", f"{hardware_info['memory']['used_percent']}%")
            memory_table.add_row("Swap Memory", f"{hardware_info['memory']['swap_total_gb']} GB")
            
            console.print(memory_table)
            
            # GPU details (if available)
            if hardware_info["gpu"]["cuda_available"]:
                gpu_table = Table(title="GPU Details", show_header=True, header_style="bold magenta")
                gpu_table.add_column("Metric", style="cyan")
                gpu_table.add_column("Value", style="green")
                
                gpu_table.add_row("Compute Capability", hardware_info["gpu"]["compute_capability"])
                gpu_table.add_row("Memory Allocated", f"{hardware_info['gpu']['memory_allocated_gb']} GB")
                gpu_table.add_row("Memory Reserved", f"{hardware_info['gpu']['memory_reserved_gb']} GB")
                
                console.print(gpu_table)
            
            # OpenVINO compatibility assessment
            compat_table = Table(title="OpenVINO Compatibility", show_header=True, header_style="bold magenta")
            compat_table.add_column("Feature", style="cyan")
            compat_table.add_column("Status", style="green")
            compat_table.add_column("Recommendation", style="yellow")
            
            # CPU compatibility
            if hardware_info["cpu"]["cores_physical"] >= 4:
                compat_table.add_row("CPU Cores", "Excellent", "Optimal for OpenVINO inference")
            elif hardware_info["cpu"]["cores_physical"] >= 2:
                compat_table.add_row("CPU Cores", "Good", "Sufficient for basic inference")
            else:
                compat_table.add_row("CPU Cores", "Limited", "Consider upgrading for better performance")
            
            # Memory compatibility
            if hardware_info["memory"]["total_gb"] >= 16:
                compat_table.add_row("System Memory", "Excellent", "Sufficient for large models")
            elif hardware_info["memory"]["total_gb"] >= 8:
                compat_table.add_row("System Memory", "Good", "Suitable for medium models")
            else:
                compat_table.add_row("System Memory", "Limited", "May struggle with large models")
            
            # GPU compatibility
            if hardware_info["gpu"]["cuda_available"]:
                if hardware_info["gpu"]["total_memory_gb"] >= 8:
                    compat_table.add_row("GPU Memory", "Excellent", "Ideal for GPU acceleration")
                elif hardware_info["gpu"]["total_memory_gb"] >= 4:
                    compat_table.add_row("GPU Memory", "Good", "Suitable for smaller models")
                else:
                    compat_table.add_row("GPU Memory", "Limited", "CPU inference recommended")
            else:
                compat_table.add_row("GPU", "Not Available", "CPU-only inference")
            
            console.print(compat_table)
            
            # Performance recommendations
            console.print("\nPerformance Recommendations:", style="bold yellow")
            recommendations = []
            
            if hardware_info["gpu"]["cuda_available"] and hardware_info["gpu"]["total_memory_gb"] >= 8:
                recommendations.append("Use GPU device for optimal performance")
            else:
                recommendations.append("Use CPU device (GPU insufficient or unavailable)")
            
            if hardware_info["memory"]["total_gb"] < 16:
                recommendations.append("Consider reducing model size or using INT8 quantization")
            
            if hardware_info["cpu"]["cores_physical"] < 4:
                recommendations.append("Limit concurrent operations to avoid CPU bottlenecks")
            
            for i, rec in enumerate(recommendations, 1):
                console.print(f"{i}. {rec}", style="dim")
        
        # Export hardware information if requested
        if export:
            with open(export, 'w') as f:
                json.dump(hardware_info, f, indent=2)
            console.print(f"\nHardware information exported to: {export}", style="blue")
        
    except Exception as e:
        console.print(f"Hardware detection failed: {e}", style="red")
        cli_context.add_error(f"Hardware detection failed: {e}")
        cli_context.display_summary()
        sys.exit(1)


@cli.command()
@click.option('--quick', 
              is_flag=True,
              help='Run quick demo with preset queries')
@click.option('--interactive', 
              is_flag=True,
              help='Run interactive demo mode')
def demo(quick, interactive):
    """
    Run a comprehensive demonstration of the RAG system.
    
    This command provides a complete demonstration of the RAG system
    capabilities using sample queries and showcasing all major features.
    It validates system setup and demonstrates query processing with
    real-time performance monitoring.
    
    Demo features:
    1. System validation and health checks
    2. Sample query processing with various parameters
    3. Performance monitoring and timing analysis
    4. Interactive query capabilities
    5. Feature showcase and usage examples
    
    Args:
        quick: Run predefined queries for quick demonstration
        interactive: Start interactive demo session
    """
    try:
        logger.info("Starting RAG system demonstration")
        console.print("RAG System Demonstration with OpenVINO GenAI", style="bold blue")
        console.print("=" * 80, style="dim")
        
        # Validate system prerequisites
        required_paths = {
            'model': cli_context.config['model_path'],
            'vector_store': cli_context.config['vector_store_path']
        }
        
        missing_components = []
        for component, path in required_paths.items():
            if not Path(path).exists():
                missing_components.append(f"{component}: {path}")
        
        if missing_components:
            console.print("Demo cannot run - missing components:", style="red")
            for component in missing_components:
                console.print(f"  - {component}", style="red")
            
            console.print("\nTo fix these issues:", style="yellow")
            console.print("  - Run 'python rag_cli.py setup' to create vector store", style="yellow")
            console.print("  - Run 'python rag_cli.py convert-model' to prepare model", style="yellow")
            
            sys.exit(1)
        
        # Initialize RAG engine
        console.print("Initializing OpenVINO GenAI RAG system...", style="blue")
        
        with console.status("Loading model and vector store..."):
            try:
                rag_engine = create_rag_engine(
                    model_path=cli_context.config['model_path'],
                    vector_store_path=cli_context.config['vector_store_path'],
                    device="AUTO",
                    enable_monitoring=True
                )
            except Exception as e:
                # Attempt auto-fix
                fixed_path = ModelFormatAutoFixer.auto_fix_model_format(cli_context.config['model_path'])
                if fixed_path:
                    rag_engine = create_rag_engine(
                        model_path=fixed_path,
                        vector_store_path=cli_context.config['vector_store_path'],
                        device="AUTO",
                        enable_monitoring=True
                    )
                else:
                    raise e
        
        console.print("RAG engine initialized successfully!", style="green")
        
        # Perform system health check
        console.print("\nPerforming system health check...", style="blue")
        health = rag_engine.health_check()
        
        health_status = health["overall_status"]
        if health_status == "healthy":
            console.print("System health: Excellent", style="green")
        elif health_status == "warning":
            console.print("System health: Good (with warnings)", style="yellow")
        else:
            console.print("System health: Issues detected", style="red")
            for error in health.get("errors", []):
                console.print(f"  - {error}", style="red")
        
        if interactive:
            # Interactive demo mode
            console.print("\nStarting interactive demo mode...", style="bold blue")
            console.print("Try asking questions about Procyon!", style="dim")
            console.print("Type 'demo' for sample queries, 'quit' to exit", style="dim")
            
            demo_queries = [
        "What is Procyon?",
                "What are the main features of Procyon?",
                "How does Procyon handle data processing?",
                "What benchmarks are available in Procyon?",
                "How do I use Procyon for testing?"
            ]
            
            query_count = 0
            
            while True:
                try:
                    query = Prompt.ask("\nDemo query").strip()
                    
                    if query.lower() in ['quit', 'exit', 'q']:
                        console.print("Demo completed!", style="green")
                        break
                    elif query.lower() == 'demo':
                        console.print("Sample queries:", style="cyan")
                        for i, sample in enumerate(demo_queries, 1):
                            console.print(f"{i}. {sample}", style="dim")
                        continue
                    elif not query:
                        continue
                    
                    # Process query
                    query_count += 1
                    console.print(f"\n[Demo Query {query_count}] {query}", style="cyan")
                    console.print("-" * 60, style="dim")
                    
                    start_time = time.time()
                    
                    response = rag_engine.query(
                        question=query,
                        k=3,
                        max_tokens=256,
                        temperature=0.7,
                        stream=True
                    )
                    
                    end_time = time.time()
                    console.print(f"\nCompleted in {end_time - start_time:.2f} seconds", style="dim")
                    
                except KeyboardInterrupt:
                    console.print("\nDemo completed!", style="green")
                    break
                except Exception as e:
                    console.print(f"\nError: {e}", style="red")
        
        else:
            # Quick demo with predefined queries
            demo_queries = [
                {
                    "question": "What is Procyon?",
                    "description": "Basic information query"
                },
                {
                    "question": "What are the main features of Procyon?",
                    "description": "Feature overview query"
                },
                {
                    "question": "How does Procyon handle data processing?",
                    "description": "Technical details query"
                }
            ]
            
            console.print(f"\nRunning {len(demo_queries)} demonstration queries...", style="bold blue")
            
            total_time = 0
            
            for i, demo_item in enumerate(demo_queries, 1):
                query = demo_item["question"]
                description = demo_item["description"]
                
                console.print(f"\nDemo Query {i}/{len(demo_queries)}: {description}", style="cyan")
                console.print(f"Question: {query}", style="bold")
                console.print("-" * 80, style="dim")
                
                start_time = time.time()
                
                try:
                    response = rag_engine.query(
                        question=query,
                        k=3,
                        max_tokens=256,
                        temperature=0.7,
                        stream=False
                    )
                    
                    console.print("Response:", style="green")
                    console.print(response.answer, style="green")
                    
                except Exception as query_error:
                    console.print(f"Query failed: {query_error}", style="red")
                    cli_context.add_error(f"Demo query {i} failed: {query_error}")
                
                end_time = time.time()
                query_time = end_time - start_time
                total_time += query_time
        
                console.print(f"Query time: {query_time:.2f} seconds", style="dim")
                
                # Brief pause between queries
                if i < len(demo_queries):
                    time.sleep(1)
            
            # Display demo summary
            console.print("\nDemo Summary:", style="bold green")
            
            summary_table = Table(show_header=False, box=None)
            summary_table.add_column("Metric", style="cyan")
            summary_table.add_column("Value", style="green")
            
            summary_table.add_row("Queries Processed", str(len(demo_queries)))
            summary_table.add_row("Total Time", f"{total_time:.2f} seconds")
            summary_table.add_row("Average Time", f"{total_time/len(demo_queries):.2f} seconds/query")
            summary_table.add_row("System Status", health_status.title())
            
            console.print(summary_table)
        
        console.print("\nDemo completed successfully!", style="bold green")
        console.print("Try running your own queries with:", style="blue")
        console.print("python rag_cli.py --query 'Your question here'", style="dim")
        
    except Exception as e:
        console.print(f"Demo failed: {e}", style="red")
        cli_context.add_error(f"Demo failed: {e}")
        cli_context.display_summary()
        sys.exit(1)


@cli.command()
def version():
    """Display version information and system details."""
    try:
        console.print("RAG CLI Version Information", style="bold blue")
        console.print("=" * 40, style="dim")
        
        version_table = Table(show_header=False, box=None)
        version_table.add_column("Component", style="cyan")
        version_table.add_column("Version", style="green")
        
        version_table.add_row("RAG CLI", "1.0.0")
        version_table.add_row("Python", sys.version.split()[0])
        
        # Try to get package versions
        try:
            import torch
            version_table.add_row("PyTorch", torch.__version__)
        except ImportError:
            version_table.add_row("PyTorch", "Not installed")
        
        try:
            import openvino_genai as ov_genai
            try:
                ov_version = ov_genai.get_version()
            except AttributeError:
                ov_version = "Unknown"
            version_table.add_row("OpenVINO GenAI", ov_version)
        except ImportError:
            version_table.add_row("OpenVINO GenAI", "Not installed")
        
        try:
            import transformers
            version_table.add_row("Transformers", transformers.__version__)
        except ImportError:
            version_table.add_row("Transformers", "Not installed")
        
        try:
            import faiss
            version_table.add_row("FAISS", faiss.__version__)
        except ImportError:
            version_table.add_row("FAISS", "Not installed")
        
        console.print(version_table)
        
    except Exception as e:
        console.print(f"Version information failed: {e}", style="red")


@cli.command()
def health():
    """Perform comprehensive system health check."""
    try:
        console.print("RAG System Health Check", style="bold blue")
        console.print("=" * 40, style="dim")
        
        health_results = {
            "components": {},
            "overall_status": "unknown",
            "recommendations": []
        }
        
        # Check Python environment
        console.print("Checking Python environment...", style="blue")
        
        try:
            # Check required modules
            required_modules = [
                'torch', 'transformers', 'faiss', 'openvino_genai',
                'sentence_transformers', 'rich', 'click'
            ]
            
            missing_modules = []
            for module in required_modules:
                try:
                    __import__(module)
                except ImportError:
                    missing_modules.append(module)
            
            if missing_modules:
                health_results["components"]["python_environment"] = "unhealthy"
                health_results["recommendations"].append(f"Install missing modules: {', '.join(missing_modules)}")
            else:
                health_results["components"]["python_environment"] = "healthy"
                
        except Exception as e:
            health_results["components"]["python_environment"] = "error"
            health_results["recommendations"].append(f"Python environment check failed: {e}")
        
        # Check file system
        console.print("Checking file system...", style="blue")
        
        try:
            required_dirs = ['./models', './data', './data/raw', './data/processed_data']
            missing_dirs = []
            
            for dir_path in required_dirs:
                if not Path(dir_path).exists():
                    missing_dirs.append(dir_path)
            
            if missing_dirs:
                health_results["components"]["file_system"] = "warning"
                health_results["recommendations"].append(f"Create missing directories: {', '.join(missing_dirs)}")
            else:
                health_results["components"]["file_system"] = "healthy"
                
        except Exception as e:
            health_results["components"]["file_system"] = "error"
            health_results["recommendations"].append(f"File system check failed: {e}")
        
        # Check model availability
        console.print("Checking model availability...", style="blue")
        
        try:
            model_path = cli_context.config['model_path']
            if Path(model_path).exists():
                # Try to auto-fix and validate
                working_path = ModelFormatAutoFixer.auto_fix_model_format(model_path)
                if working_path:
                    health_results["components"]["model"] = "healthy"
                else:
                    health_results["components"]["model"] = "unhealthy"
                    health_results["recommendations"].append("Run model conversion: python rag_cli.py convert-model")
            else:
                health_results["components"]["model"] = "missing"
                health_results["recommendations"].append("Download and convert model: python rag_cli.py convert-model")
                
        except Exception as e:
            health_results["components"]["model"] = "error"
            health_results["recommendations"].append(f"Model check failed: {e}")
        
        # Check vector store
        console.print("Checking vector store...", style="blue")
        
        try:
            vector_store_path = cli_context.config['vector_store_path']
            if Path(vector_store_path).exists():
                health_results["components"]["vector_store"] = "healthy"
            else:
                health_results["components"]["vector_store"] = "missing"
                health_results["recommendations"].append("Create vector store: python rag_cli.py setup")
                
        except Exception as e:
            health_results["components"]["vector_store"] = "error"
            health_results["recommendations"].append(f"Vector store check failed: {e}")
        
        # Determine overall status
        component_statuses = list(health_results["components"].values())
        if all(status == "healthy" for status in component_statuses):
            health_results["overall_status"] = "healthy"
        elif any(status in ["unhealthy", "missing"] for status in component_statuses):
            health_results["overall_status"] = "needs_attention"
        else:
            health_results["overall_status"] = "error"
        
        # Display results
        console.print("\nHealth Check Results:", style="bold green")
        
        health_table = Table(show_header=True, header_style="bold cyan")
        health_table.add_column("Component", style="cyan")
        health_table.add_column("Status", style="green")
        
        for component, status in health_results["components"].items():
            status_style = {
                "healthy": "green",
                "warning": "yellow",
                "unhealthy": "red",
                "missing": "red",
                "error": "red"
            }.get(status, "dim")
            
            health_table.add_row(
                component.replace("_", " ").title(),
                Text(status.title(), style=status_style)
            )
        
        console.print(health_table)
        
        # Display overall status
        overall_style = {
            "healthy": "bold green",
            "needs_attention": "bold yellow",
            "error": "bold red"
        }.get(health_results["overall_status"], "dim")
        
        console.print(f"\nOverall Status: {health_results['overall_status'].replace('_', ' ').title()}", style=overall_style)
        
        # Display recommendations
        if health_results["recommendations"]:
            console.print("\nRecommendations:", style="bold yellow")
            for i, rec in enumerate(health_results["recommendations"], 1):
                console.print(f"{i}. {rec}", style="yellow")
        
    except Exception as e:
        console.print(f"Health check failed: {e}", style="red")
        cli_context.add_error(f"Health check failed: {e}")
        cli_context.display_summary()
        sys.exit(1)


def main():
    """
    Main entry point for the RAG CLI application.
    
    Handles top-level exception catching and ensures proper cleanup
    and error reporting regardless of how the application terminates.
    """
    try:
        # Run the CLI
        cli()
        
    except KeyboardInterrupt:
        console.print("\nOperation cancelled by user", style="yellow")
        logger.info("Operation cancelled by user")
        
    except Exception as e:
        console.print(f"\nUnexpected error: {e}", style="red")
        logger.error(f"Unexpected error: {e}")
        cli_context.add_error(f"Unexpected error: {e}")
        cli_context.display_summary()
        sys.exit(1)
    
    finally:
        # Display final summary if there were any issues
        if cli_context.errors or cli_context.warnings:
            cli_context.display_summary()


if __name__ == "__main__":
    main()