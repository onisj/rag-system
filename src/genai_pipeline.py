"""
OpenVINO GenAI RAG Pipeline

This module implements a comprehensive Retrieval-Augmented Generation (RAG) system
specifically designed for OpenVINO GenAI framework. The pipeline combines document
retrieval capabilities with large language model inference to provide contextually
relevant responses to user queries.

The system is optimized for Intel hardware using OpenVINO GenAI's optimized inference
engine, supporting both CPU and GPU acceleration. It integrates with a vector database
for efficient document retrieval and provides configurable generation parameters.

Key Components:
    - GenAIRAGEngine: Main RAG orchestration class
    - RAGResponse: Structured response data container
    - Vector store integration for document retrieval
    - Performance monitoring and optimization
    - Streaming and batch response generation

Architecture:
    The RAG pipeline follows a standard retrieve-then-generate approach:
    1. Query embedding and vector similarity search
    2. Context aggregation from retrieved documents
    3. Prompt construction with context and question
    4. Response generation using OpenVINO GenAI
    5. Post-processing and response formatting

Performance Features:
    - Automatic device selection (CPU/GPU/AUTO)
    - Configurable generation parameters via environment variables
    - Built-in performance monitoring and statistics
    - Memory-efficient streaming response generation
    - Optimized for Intel hardware acceleration

Usage:
    engine = GenAIRAGEngine(
        model_path="./models/llama-3.1-8b-int4-genai",
        vector_store_path="./data/vector_store"
    )
    response = engine.query("What is the main topic?")

Requirements:
    - OpenVINO GenAI package
    - Vector store implementation
    - Performance monitoring utilities
    - Rich console for enhanced output

Author: Segun Oni
Version: 1.0.0
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Union, Optional, Iterator
from dataclasses import dataclass, field

# Add project root to Python path for imports
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

# Core dependencies
import torch
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

# OpenVINO GenAI imports
try:
    import openvino_genai as ov_genai
except ImportError as e:
    raise ImportError(
        f"OpenVINO GenAI not found: {e}. "
        "Please install: pip install openvino-genai"
    )

# Local module imports
try:
    from vector_store import VectorStore
    from performance_monitor import create_performance_monitor
except ImportError as e:
    raise ImportError(
        f"Local modules not found: {e}. "
        "Ensure vector_store.py and performance_monitor.py are available."
    )

# Configure module-level logging with logs directory
import datetime

# Create logs directory if it doesn't exist
logs_dir = Path("./logs")
logs_dir.mkdir(exist_ok=True)

# Create timestamped log file for GenAI pipeline
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
pipeline_log_file = logs_dir / f"genai_pipeline_{timestamp}.log"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(pipeline_log_file),
        logging.StreamHandler()  # Also output to console
    ]
)
logger = logging.getLogger(__name__)

# Initialize rich console for enhanced output
console = Console()


@dataclass
class RAGResponse:
    """
    Structured response container for RAG system outputs.
    
    This class encapsulates all information returned by the RAG system,
    including the generated answer, source context, metadata, and performance
    metrics. It provides a standardized interface for consuming RAG results.
    
    Attributes:
        answer (str): The generated response text
        context (List[str]): List of retrieved context chunks used for generation
        sources (List[Dict[str, Any]]): Source document metadata and references
        metadata (Dict[str, Any]): Additional metadata including timing, scores, etc.
        query (str): The original user query
        generation_config (Dict[str, Any]): Parameters used for response generation
    """
    answer: str
    context: List[str] = field(default_factory=list)
    sources: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    query: str = ""
    generation_config: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization processing for response validation."""
        # Validate required fields
        if not isinstance(self.answer, str):
            raise ValueError("Answer must be a string")
        
        # Ensure lists are properly initialized
        self.context = self.context or []
        self.sources = self.sources or []
        self.metadata = self.metadata or {}
        self.generation_config = self.generation_config or {}
        
        # Add basic metadata if missing
        if 'response_length' not in self.metadata:
            self.metadata['response_length'] = len(self.answer)
        if 'context_chunks' not in self.metadata:
            self.metadata['context_chunks'] = len(self.context)


class GenAIRAGEngine:
    """
    OpenVINO GenAI Retrieval-Augmented Generation Engine
    
    This class implements a complete RAG system using OpenVINO GenAI for optimized
    inference on Intel hardware. It combines document retrieval capabilities with
    large language model generation to provide contextually relevant responses.
    
    The engine supports multiple inference devices (CPU/GPU), configurable generation
    parameters, performance monitoring, and both streaming and batch response modes.
    It is specifically optimized for Intel hardware using OpenVINO's acceleration
    capabilities.
    
    Key Features:
        - Automatic device selection and optimization
        - Configurable generation parameters via environment variables
        - Built-in performance monitoring and statistics
        - Support for streaming and batch response generation
        - Robust error handling and fallback mechanisms
        - Memory-efficient operation for large models
    
    Example:
        engine = GenAIRAGEngine(
            model_path="./models/llama-3.1-8b-int4-genai",
            vector_store_path="./data/vector_store",
            device="AUTO"
        )
        
        response = engine.query(
            question="What is the main topic?",
            k=5,
            max_tokens=512
        )
        
        print(f"Answer: {response.answer}")
        print(f"Sources: {len(response.sources)}")
    """
    
    def __init__(
        self,
        model_path: str,
        vector_store_path: str,
        device: Optional[str] = None,
        enable_monitoring: Optional[bool] = None
    ):
        """
        Initialize the OpenVINO GenAI RAG Engine.
        
        Args:
            model_path (str): Path to the converted OpenVINO GenAI model directory.
                             Should contain either OpenVINO IR files (.xml/.bin) or
                             HuggingFace format files depending on conversion method.
            vector_store_path (str): Path to the vector store containing document
                                   embeddings and metadata.
            device (Optional[str]): Target inference device. Options are:
                                  - "CPU": Force CPU inference
                                  - "GPU": Force GPU inference (if available)
                                  - "AUTO": Automatic device selection
                                  - None: Use DEVICE environment variable or AUTO
            enable_monitoring (Optional[bool]): Whether to enable performance monitoring.
                                              If None, uses ENABLE_MONITORING environment
                                              variable or defaults to True.
        
        Raises:
            ValueError: If model_path or vector_store_path are invalid
            RuntimeError: If component initialization fails
            FileNotFoundError: If required model or vector store files are missing
        """
        # Validate required parameters
        if not model_path or not isinstance(model_path, str):
            raise ValueError("model_path must be a non-empty string")
        if not vector_store_path or not isinstance(vector_store_path, str):
            raise ValueError("vector_store_path must be a non-empty string")
        
        # Store configuration
        self.model_path = str(Path(model_path).resolve())
        self.vector_store_path = str(Path(vector_store_path).resolve())
        
        # Configure device selection with environment variable fallback
        if device is None:
            self.device = os.getenv('DEVICE', 'AUTO')
        else:
            self.device = device.upper()
        
        # Configure monitoring with environment variable fallback
        if enable_monitoring is None:
            self.enable_monitoring = os.getenv('ENABLE_MONITORING', 'true').lower() == 'true'
        else:
            self.enable_monitoring = enable_monitoring
        
        # Initialize component placeholders
        self.genai_pipeline: Optional[ov_genai.LLMPipeline] = None
        self.vector_store: Optional[VectorStore] = None
        self.performance_monitor: Optional[Any] = None
        
        # Track initialization state
        self._initialized = False
        self._model_loaded = False
        self._vector_store_loaded = False
        
        # Initialize all components
        self._initialize_components()
    
    def _initialize_components(self) -> None:
        """
        Initialize all RAG engine components in the correct order.
        
        This method orchestrates the initialization of all system components:
        1. Performance monitoring setup (if enabled)
        2. Device selection and optimization
        3. OpenVINO GenAI model loading
        4. Vector store loading and validation
        
        Raises:
            RuntimeError: If any component fails to initialize
        """
        try:
            logger.info("Starting OpenVINO GenAI RAG Engine initialization")
            console.print("Initializing OpenVINO GenAI RAG Engine...", style="bold blue")
            
            # Step 1: Initialize performance monitoring
            if self.enable_monitoring:
                self._initialize_performance_monitoring()
            
            # Step 2: Select optimal inference device
            if self.device == "AUTO":
                self.device = self._select_optimal_device()
                logger.info(f"Auto-selected device: {self.device}")
            else:
                self._validate_device_availability()
            
            # Step 3: Load OpenVINO GenAI model
            self._load_genai_model()
            
            # Step 4: Load vector store
            self._load_vector_store()
            
            # Mark as fully initialized
            self._initialized = True
            
            logger.info(f"GenAI RAG Engine initialized successfully on {self.device}")
            console.print(
                f"GenAI RAG Engine initialized successfully on {self.device}",
                style="bold green"
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize GenAI RAG Engine: {e}")
            console.print(f"Error initializing GenAI RAG Engine: {e}", style="red")
            raise RuntimeError(f"Failed to initialize GenAI RAG Engine: {e}")
    
    def _initialize_performance_monitoring(self) -> None:
        """
        Initialize performance monitoring system.
        
        Sets up performance monitoring to track inference times, memory usage,
        and other system metrics during RAG operations.
        
        Raises:
            RuntimeError: If performance monitor cannot be initialized
        """
        try:
            self.performance_monitor = create_performance_monitor()
            logger.info("Performance monitoring initialized")
            console.print("Performance monitoring enabled", style="green")
            
        except Exception as e:
            logger.warning(f"Failed to initialize performance monitoring: {e}")
            console.print(f"Performance monitoring disabled: {e}", style="yellow")
            self.performance_monitor = None
    
    def _select_optimal_device(self) -> str:
        """
        Automatically select the optimal inference device.
        
        Evaluates available hardware and selects the best device for inference
        based on capabilities, memory, and performance characteristics.
        
        Returns:
            str: Selected device identifier ("GPU" or "CPU")
        """
        try:
            # Check for CUDA GPU availability
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
                
                logger.info(f"Found {gpu_count} CUDA GPU(s) with {gpu_memory:.1f}GB memory")
                
                # Require at least 8GB GPU memory for optimal performance
                if gpu_memory >= 8.0:
                    logger.info("Selected GPU for inference (sufficient memory)")
                    return "GPU"
                else:
                    logger.info("GPU has insufficient memory, falling back to CPU")
                    console.print("GPU has insufficient memory, using CPU", style="yellow")
                    return "CPU"
            else:
                logger.info("No CUDA GPU available, using CPU")
                return "CPU"
                
        except Exception as e:
            logger.warning(f"Device selection failed, defaulting to CPU: {e}")
            return "CPU"
    
    def _validate_device_availability(self) -> None:
        """
        Validate that the specified device is available.
        
        Checks that the user-specified device is actually available and
        functional for inference operations.
        
        Raises:
            RuntimeError: If specified device is not available
        """
        if self.device == "GPU":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "GPU device specified but CUDA is not available. "
                    "Use 'CPU' or 'AUTO' for automatic selection."
                )
            logger.info("GPU device validated and available")
        elif self.device == "CPU":
            logger.info("CPU device selected")
        else:
            raise ValueError(f"Invalid device specification: {self.device}")
    
    def _load_genai_model(self) -> None:
        """
        Load the OpenVINO GenAI model with enhanced path resolution.
        
        Attempts to load the model from multiple potential paths to handle
        different conversion output structures. Provides detailed error
        reporting for troubleshooting.
        
        Raises:
            RuntimeError: If model cannot be loaded from any attempted path
            FileNotFoundError: If model files are missing
        """
        try:
            logger.info(f"Loading GenAI model from: {self.model_path}")
            console.print(f"Loading GenAI model on {self.device}...", style="blue")
            
            # Define potential model paths to try
            # This handles different conversion output structures
            model_paths_to_try = [
                self.model_path,  # Direct path as specified
                str(Path(self.model_path) / "original_model"),  # HuggingFace format
                str(Path(self.model_path).parent / self.model_path.split('/')[-1]),  # Alternative structure
            ]
            
            # Remove duplicate paths and non-existent paths
            valid_paths = []
            for path in model_paths_to_try:
                if Path(path).exists() and path not in valid_paths:
                    valid_paths.append(path)
            
            if not valid_paths:
                raise FileNotFoundError(
                    f"No valid model paths found. Searched: {model_paths_to_try}"
                )
            
            # Attempt to load from each valid path
            loading_errors = []
            
            for path_index, path in enumerate(valid_paths):
                try:
                    logger.debug(f"Attempting to load from path {path_index + 1}/{len(valid_paths)}: {path}")
                    console.print(f"Trying to load from: {path}", style="dim")
                    
                    # Try different loading approaches
                    loading_attempts = [
                        # Try 1: Direct LLMPipeline loading
                        lambda: ov_genai.LLMPipeline(models_path=path, device=self.device),
                        # Try 2: With explicit tokenizer
                        lambda: ov_genai.LLMPipeline(models_path=path, tokenizer=ov_genai.Tokenizer(path), device=self.device),
                        # Try 3: CPU fallback
                        lambda: ov_genai.LLMPipeline(models_path=path, device="CPU"),
                        # Try 4: With explicit tokenizer and CPU
                        lambda: ov_genai.LLMPipeline(models_path=path, tokenizer=ov_genai.Tokenizer(path), device="CPU"),
                    ]
                    
                    for attempt_idx, attempt in enumerate(loading_attempts):
                        try:
                            logger.debug(f"Trying loading attempt {attempt_idx + 1}")
                            self.genai_pipeline = attempt()
                            
                            # Successful loading
                            self._model_loaded = True
                            logger.info(f"GenAI model loaded successfully from: {path} (attempt {attempt_idx + 1})")
                            console.print(f"GenAI model loaded successfully from: {path}", style="green")
                            console.print(f"Model compiled for device: {self.device}", style="green")
                            return
                            
                        except Exception as attempt_error:
                            logger.debug(f"Loading attempt {attempt_idx + 1} failed: {attempt_error}")
                            continue
                    
                    # If all attempts failed for this path
                    error_msg = f"Failed to load from {path} with all attempts"
                    loading_errors.append(error_msg)
                    console.print(f"Failed to load from {path} with all attempts", style="yellow")
                    
                except Exception as path_error:
                    error_msg = f"Failed to load from {path}: {path_error}"
                    loading_errors.append(error_msg)
                    logger.debug(error_msg)
                    console.print(f"Failed to load from {path}: {path_error}", style="yellow")
                    continue
            
            # If we reach here, all paths failed
            error_summary = "\n".join(loading_errors)
            raise RuntimeError(
                f"Could not load model from any attempted path. Errors:\n{error_summary}"
            )
            
        except Exception as e:
            logger.error(f"Error loading GenAI model: {e}")
            console.print(f"Error loading GenAI model: {e}", style="red")
            raise RuntimeError(f"Failed to load GenAI model: {e}")
    
    def _load_vector_store(self) -> None:
        """
        Load and validate the vector store for document retrieval.
        
        Initializes the vector store component and validates that it contains
        the necessary embeddings and metadata for document retrieval.
        
        Raises:
            RuntimeError: If vector store cannot be loaded
            FileNotFoundError: If vector store files are missing
        """
        try:
            logger.info(f"Loading vector store from: {self.vector_store_path}")
            console.print("Loading vector store...", style="blue")
            
            # Check if vector store path exists
            if not Path(self.vector_store_path).exists():
                raise FileNotFoundError(f"Vector store path does not exist: {self.vector_store_path}")
            
            # Initialize and load vector store
            self.vector_store = VectorStore()
            self.vector_store.load(self.vector_store_path)
            
            # Validate vector store contents
            if hasattr(self.vector_store, 'get_stats'):
                stats = self.vector_store.get_stats()
                logger.info(f"Vector store loaded with {stats.get('total_documents', 'unknown')} documents")
            
            self._vector_store_loaded = True
            logger.info("Vector store loaded successfully")
            console.print("Vector store loaded successfully", style="green")
            
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            console.print(f"Error loading vector store: {e}", style="red")
            raise RuntimeError(f"Failed to load vector store: {e}")
    
    def _create_rag_prompt(self, question: str, context: str) -> str:
        """
        Create a well-structured prompt for RAG inference.
        
        Constructs a prompt that effectively combines the retrieved context
        with the user question to guide the model toward generating relevant,
        accurate responses.
        
        Args:
            question (str): The user's question
            context (str): Retrieved context from document search
        
        Returns:
            str: Formatted prompt for model inference
        """
        # Use a structured prompt format optimized for RAG
        prompt = f"""You are a helpful assistant that answers questions based on the provided context. Use only the information from the context to answer the question. If the context doesn't contain enough information to answer the question, say so.

Context:
{context}

Question: {question}

Answer: """
        
        return prompt
    
    def query(
        self,
        question: str,
        k: Optional[int] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stream: bool = True
    ) -> Union[str, RAGResponse]:
        """
        Execute a RAG query with document retrieval and response generation.
        
        This is the main method for performing RAG inference. It retrieves relevant
        documents, constructs a prompt, and generates a response using the OpenVINO
        GenAI model.
        
        Args:
            question (str): The question to answer
            k (Optional[int]): Number of document chunks to retrieve for context.
                             If None, uses K_CHUNKS environment variable or default 5.
            max_tokens (Optional[int]): Maximum tokens to generate in response.
                                      If None, uses MAX_TOKENS environment variable or default 512.
            temperature (Optional[float]): Sampling temperature for generation (0.0-1.0).
                                         Higher values increase randomness.
                                         If None, uses TEMPERATURE environment variable or default 0.7.
            stream (bool): Whether to stream the response token by token.
                          If False, returns complete response at once.
        
        Returns:
            Union[str, RAGResponse]: If stream is True, returns a string with the generated answer.
                                   If stream is False, returns a structured RAGResponse object
                                   with answer, context, sources, and metadata.
        
        Raises:
            RuntimeError: If query processing fails
            ValueError: If question is empty or invalid
        """
        # Validate inputs
        if not question or not isinstance(question, str):
            raise ValueError("Question must be a non-empty string")
        
        if not self._initialized:
            raise RuntimeError("RAG engine not properly initialized")
        
        # Apply parameter defaults from environment variables
        k = k if k is not None else int(os.getenv('K_CHUNKS', '5'))
        max_tokens = max_tokens if max_tokens is not None else int(os.getenv('MAX_TOKENS', '512'))
        temperature = temperature if temperature is not None else float(os.getenv('TEMPERATURE', '0.7'))
        
        # Validate parameter ranges
        if k <= 0:
            raise ValueError("k must be positive")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if not 0.0 <= temperature <= 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0")
        
        try:
            logger.info(f"Processing query: {question[:100]}...")
            
            # Start performance monitoring if enabled
            query_start_time = None
            if self.performance_monitor:
                query_start_time = self.performance_monitor.start_timer("rag_query")
            
            # Step 1: Retrieve relevant documents
            logger.debug(f"Retrieving {k} relevant documents")
            retrieval_results = self.vector_store.search(question, k=k)
            
            # Step 2: Extract and aggregate context
            context_chunks = []
            source_metadata = []
            
            for result in retrieval_results:
                # Unpack the tuple (document_dict, score)
                document_dict, score = result
                text = document_dict.get('text', '').strip()
                if text:  # Only include non-empty chunks
                    context_chunks.append(text)
                    source_metadata.append({
                        'text': text,
                        'score': score,
                        'metadata': document_dict.get('metadata', {})
                    })
            
            # Combine context chunks
            aggregated_context = "\n\n".join(context_chunks)
            logger.debug(f"Aggregated context from {len(context_chunks)} chunks")
            
            # Step 3: Create RAG prompt
            prompt = self._create_rag_prompt(question, aggregated_context)
            
            # Step 4: Generate response
            generation_config = {
                'max_tokens': max_tokens,
                'temperature': temperature,
                'k': k
            }
            
            if stream:
                # Streaming response generation
                response_text = self._generate_streaming_response(prompt, max_tokens, temperature)
            else:
                # Batch response generation
                response_text = self._generate_response(prompt, max_tokens, temperature)
            
            # Stop performance monitoring
            if self.performance_monitor and query_start_time:
                self.performance_monitor.stop_timer("rag_query", query_start_time)
            
            # Return appropriate response format
            if stream:
                return response_text
            else:
                return RAGResponse(
                    answer=response_text,
                    context=context_chunks,
                    sources=source_metadata,
                    query=question,
                    generation_config=generation_config,
                    metadata={
                        'response_length': len(response_text),
                        'context_chunks': len(context_chunks),
                        'total_context_length': len(aggregated_context)
                    }
                )
                
        except Exception as e:
            logger.error(f"Error during query processing: {e}")
            console.print(f"Error during query: {e}", style="red")
            raise RuntimeError(f"Query failed: {e}")
    
    def _generate_response(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """
        Generate a complete response using OpenVINO GenAI.
        
        Performs batch inference to generate the complete response at once.
        This is more efficient for shorter responses but doesn't provide
        real-time feedback.
        
        Args:
            prompt (str): The formatted prompt for generation
            max_tokens (int): Maximum number of tokens to generate
            temperature (float): Sampling temperature for generation
        
        Returns:
            str: The generated response text
        
        Raises:
            RuntimeError: If response generation fails
        """
        try:
            logger.debug("Generating batch response")
            
            # Configure generation parameters
            generation_config = ov_genai.GenerationConfig(
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0.0,  # Use sampling only if temperature > 0
                top_p=0.9,  # Nucleus sampling
                repetition_penalty=1.1  # Reduce repetition
            )
            
            # Generate response
            response = self.genai_pipeline.generate(
                prompt,
                generation_config=generation_config
            )
            
            # Extract response text (handling different return types)
            if hasattr(response, 'text'):
                response_text = response.text
            elif isinstance(response, str):
                response_text = response
            else:
                response_text = str(response)
            
            # Clean up response (remove prompt echo if present)
            if response_text.startswith(prompt):
                response_text = response_text[len(prompt):].strip()
            
            logger.debug(f"Generated response of {len(response_text)} characters")
            return response_text
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            console.print(f"Error generating response: {e}", style="red")
            raise RuntimeError(f"Response generation failed: {e}")
    
    def _generate_streaming_response(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """
        Generate a streaming response using OpenVINO GenAI.
        
        Performs streaming inference to generate and display tokens in real-time.
        This provides better user experience for longer responses.
        
        Args:
            prompt (str): The formatted prompt for generation
            max_tokens (int): Maximum number of tokens to generate
            temperature (float): Sampling temperature for generation
        
        Returns:
            str: The complete generated response text
        
        Raises:
            RuntimeError: If streaming generation fails
        """
        try:
            logger.debug("Generating streaming response")
            
            # Configure generation parameters
            generation_config = ov_genai.GenerationConfig(
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0.0,
                top_p=0.9,
                repetition_penalty=1.1
            )
            
            # Generate streaming response
            full_response = ""
            token_count = 0
            
            try:
                # Use streaming generation if available
                for token in self.genai_pipeline.generate_stream(
                    prompt,
                    generation_config=generation_config
                ):
                    full_response += token
                    console.print(token, end="", style="green")
                    token_count += 1
                    
                    # Safety check for runaway generation
                    if token_count > max_tokens * 2:
                        logger.warning("Token limit exceeded, stopping generation")
                        break
                        
            except AttributeError:
                # Fallback to batch generation if streaming not available
                logger.info("Streaming not available, using batch generation")
                full_response = self._generate_response(prompt, max_tokens, temperature)
                console.print(full_response, style="green")
            
            console.print()  # Add newline after response
            
            # Clean up response
            if full_response.startswith(prompt):
                full_response = full_response[len(prompt):].strip()
            
            logger.debug(f"Generated streaming response of {len(full_response)} characters")
            return full_response
            
        except Exception as e:
            logger.error(f"Error generating streaming response: {e}")
            console.print(f"Error generating streaming response: {e}", style="red")
            raise RuntimeError(f"Streaming response generation failed: {e}")
    
    def get_system_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive system statistics and configuration.
        
        Returns detailed information about the RAG engine configuration,
        performance metrics, and operational status.
        
        Returns:
            Dict[str, Any]: Dictionary containing system statistics including:
                - Configuration parameters
                - Model information
                - Performance metrics
                - Component status
        """
        stats = {
            # Configuration
            "model_path": self.model_path,
            "vector_store_path": self.vector_store_path,
            "device": self.device,
            "enable_monitoring": self.enable_monitoring,
            
            # Status
            "initialized": self._initialized,
            "model_loaded": self._model_loaded,
            "vector_store_loaded": self._vector_store_loaded,
            
            # System information
            "torch_cuda_available": torch.cuda.is_available(),
        }
        
        # Add CUDA information if available
        if torch.cuda.is_available():
            stats.update({
                "cuda_device_count": torch.cuda.device_count(),
                "cuda_device_name": torch.cuda.get_device_name(0),
                "cuda_memory_total": torch.cuda.get_device_properties(0).total_memory,
                "cuda_memory_allocated": torch.cuda.memory_allocated(0),
            })
        
        # Add vector store statistics if available
        if self.vector_store and hasattr(self.vector_store, 'get_stats'):
            stats["vector_store_stats"] = self.vector_store.get_stats()
        
        # Add performance monitoring statistics if available
        if self.performance_monitor and hasattr(self.performance_monitor, 'get_stats'):
            stats["performance_stats"] = self.performance_monitor.get_stats()
        
        return stats
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform a comprehensive health check of the RAG system.
        
        Tests all major components to ensure they are functioning correctly.
        This is useful for diagnostic purposes and system monitoring.
        
        Returns:
            Dict[str, Any]: Health check results with status for each component
        """
        health_status = {
            "overall_status": "unknown",
            "components": {},
            "errors": [],
            "timestamp": None
        }
        
        try:
            import time
            health_status["timestamp"] = time.time()
            
            # Check model loading
            if self._model_loaded and self.genai_pipeline:
                health_status["components"]["model"] = "healthy"
            else:
                health_status["components"]["model"] = "unhealthy"
                health_status["errors"].append("Model not loaded")
            
            # Check vector store
            if self._vector_store_loaded and self.vector_store:
                try:
                    # Test vector store with a simple search
                    test_results = self.vector_store.search("test", k=1)
                    if test_results:
                        health_status["components"]["vector_store"] = "healthy"
                    else:
                        health_status["components"]["vector_store"] = "warning"
                        health_status["errors"].append("Vector store empty or not responding")
                except Exception as e:
                    health_status["components"]["vector_store"] = "unhealthy"
                    health_status["errors"].append(f"Vector store error: {e}")
            else:
                health_status["components"]["vector_store"] = "unhealthy"
                health_status["errors"].append("Vector store not loaded")
            
            # Check performance monitor
            if self.performance_monitor:
                health_status["components"]["performance_monitor"] = "healthy"
            else:
                health_status["components"]["performance_monitor"] = "disabled"
            
            # Test basic inference if possible
            if self._model_loaded and self._vector_store_loaded:
                try:
                    # Attempt a minimal inference test
                    test_response = self.query(
                        question="test",
                        k=1,
                        max_tokens=10,
                        temperature=0.1,
                        stream=False
                    )
                    if test_response and len(test_response.answer) > 0:
                        health_status["components"]["inference"] = "healthy"
                    else:
                        health_status["components"]["inference"] = "warning"
                        health_status["errors"].append("Inference returned empty response")
                except Exception as e:
                    health_status["components"]["inference"] = "unhealthy"
                    health_status["errors"].append(f"Inference test failed: {e}")
            else:
                health_status["components"]["inference"] = "unavailable"
                health_status["errors"].append("Required components not loaded for inference test")
            
            # Determine overall status
            component_statuses = list(health_status["components"].values())
            if all(status in ["healthy", "disabled"] for status in component_statuses):
                health_status["overall_status"] = "healthy"
            elif any(status == "unhealthy" for status in component_statuses):
                health_status["overall_status"] = "unhealthy"
            else:
                health_status["overall_status"] = "warning"
            
        except Exception as e:
            health_status["overall_status"] = "error"
            health_status["errors"].append(f"Health check failed: {e}")
            logger.error(f"Health check error: {e}")
        
        return health_status
    
    def cleanup(self) -> None:
        """
        Clean up resources and shut down the RAG engine.
        
        Properly releases all allocated resources including model memory,
        vector store connections, and monitoring systems. Should be called
        when the engine is no longer needed.
        """
        try:
            logger.info("Cleaning up RAG engine resources")
            
            # Clean up model resources
            if self.genai_pipeline:
                # Note: OpenVINO GenAI may not have explicit cleanup methods
                # but Python garbage collection should handle this
                self.genai_pipeline = None
                self._model_loaded = False
            
            # Clean up vector store
            if self.vector_store and hasattr(self.vector_store, 'close'):
                self.vector_store.close()
            self.vector_store = None
            self._vector_store_loaded = False
            
            # Clean up performance monitor
            if self.performance_monitor and hasattr(self.performance_monitor, 'cleanup'):
                self.performance_monitor.cleanup()
            self.performance_monitor = None
            
            # Clear CUDA cache if using GPU
            if self.device == "GPU" and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self._initialized = False
            logger.info("RAG engine cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            console.print(f"Error during cleanup: {e}", style="yellow")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with automatic cleanup."""
        self.cleanup()
    
    def __del__(self):
        """Destructor with automatic cleanup."""
        try:
            self.cleanup()
        except Exception:
            # Ignore cleanup errors during destruction
            pass


def create_rag_engine(
    model_path: str,
    vector_store_path: str,
    device: str = "AUTO",
    enable_monitoring: bool = True
) -> GenAIRAGEngine:
    """
    Factory function to create a configured GenAI RAG Engine.
    
    Provides a convenient way to create and initialize a RAG engine with
    standard configuration options. Includes error handling and validation.
    
    Args:
        model_path (str): Path to the OpenVINO GenAI model
        vector_store_path (str): Path to the vector store
        device (str): Target inference device ("CPU", "GPU", or "AUTO")
        enable_monitoring (bool): Whether to enable performance monitoring
    
    Returns:
        GenAIRAGEngine: Initialized and ready-to-use RAG engine
    
    Raises:
        RuntimeError: If engine creation or initialization fails
        
    Example:
        engine = create_rag_engine(
            model_path="./models/llama-3.1-8b-int4-genai",
            vector_store_path="./data/vector_store"
        )
        
        response = engine.query("What is the main topic?")
        print(response.answer)
    """
    try:
        logger.info("Creating GenAI RAG Engine")
        console.print("Creating GenAI RAG Engine...", style="blue")
        
        # Create and initialize engine
        engine = GenAIRAGEngine(
            model_path=model_path,
            vector_store_path=vector_store_path,
            device=device,
            enable_monitoring=enable_monitoring
        )
        
        # Perform health check
        health = engine.health_check()
        if health["overall_status"] not in ["healthy", "warning"]:
            raise RuntimeError(f"Engine health check failed: {health['errors']}")
        
        logger.info("GenAI RAG Engine created successfully")
        console.print("GenAI RAG Engine created successfully", style="green")
        
        return engine
        
    except Exception as e:
        logger.error(f"Failed to create RAG engine: {e}")
        console.print(f"Failed to create RAG engine: {e}", style="red")
        raise RuntimeError(f"Failed to create RAG engine: {e}")


def main():
    """
    Main function for command-line testing of the RAG engine.
    
    Provides a simple command-line interface for testing the RAG engine
    with sample queries and configuration options.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="OpenVINO GenAI RAG Engine Test Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage with default paths
    python genai_pipeline.py
    
    # Custom model and vector store paths
    python genai_pipeline.py --model ./models/custom-model --vector-store ./data/custom-store
    
    # Force CPU usage
    python genai_pipeline.py --device CPU
    
    # Interactive mode
    python genai_pipeline.py --interactive
        """
    )
    
    parser.add_argument(
        "--model-path",
        default="./models/Llama-3.1-8B-Instruct-int4-genai",
        help="Path to the OpenVINO GenAI model directory"
    )
    parser.add_argument(
        "--vector-store-path",
        default="./data/processed_data/vector_store",
        help="Path to the vector store directory"
    )
    parser.add_argument(
        "--device",
        choices=["CPU", "GPU", "AUTO"],
        default="AUTO",
        help="Inference device selection"
    )
    parser.add_argument(
        "--disable-monitoring",
        action="store_true",
        help="Disable performance monitoring"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Start interactive query mode"
    )
    parser.add_argument(
        "--health-check",
        action="store_true",
        help="Perform health check and exit"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Create RAG engine
        console.print("OpenVINO GenAI RAG Engine Test", style="bold blue")
        console.print("=" * 50, style="dim")
        
        engine = create_rag_engine(
            model_path=args.model_path,
            vector_store_path=args.vector_store_path,
            device=args.device,
            enable_monitoring=not args.disable_monitoring
        )
        
        # Health check mode
        if args.health_check:
            console.print("\nPerforming health check...", style="blue")
            health = engine.health_check()
            
            console.print(f"\nOverall Status: {health['overall_status']}", 
                         style="green" if health['overall_status'] == "healthy" else "yellow")
            
            for component, status in health['components'].items():
                console.print(f"  {component}: {status}")
            
            if health['errors']:
                console.print("\nErrors:", style="red")
                for error in health['errors']:
                    console.print(f"  - {error}")
            
            return
        
        # Interactive mode
        if args.interactive:
            console.print("\nEntering interactive mode. Type 'quit' to exit.", style="blue")
            
            while True:
                try:
                    question = input("\nQuery: ").strip()
                    if question.lower() in ['quit', 'exit', 'q']:
                        break
                    
                    if not question:
                        continue
                    
                    console.print(f"\nProcessing: {question}", style="blue")
                    response = engine.query(question, stream=True)
                    console.print(f"\n{'-' * 50}", style="dim")
                    
                except KeyboardInterrupt:
                    console.print("\nExiting interactive mode...", style="yellow")
                    break
                except Exception as e:
                    console.print(f"\nError: {e}", style="red")
        
        else:
            # Single test query
            test_question = "What is Procyon?"
            console.print(f"\nTest Query: {test_question}", style="blue")
            console.print("=" * 50, style="dim")
            
            response = engine.query(test_question, stream=False)
            
            console.print(f"\nAnswer: {response.answer}", style="green")
            console.print(f"Context chunks: {len(response.context)}", style="dim")
            console.print(f"Sources: {len(response.sources)}", style="dim")
        
        # Display system stats
        console.print(f"\n{'-' * 50}", style="dim")
        stats = engine.get_system_stats()
        console.print(f"Device: {stats['device']}", style="dim")
        console.print(f"Model: {stats['model_loaded']}", style="dim")
        console.print(f"Vector Store: {stats['vector_store_loaded']}", style="dim")
        
    except KeyboardInterrupt:
        console.print("\nOperation cancelled by user", style="yellow")
    except Exception as e:
        console.print(f"\nError: {e}", style="red")
        logger.error(f"Main execution error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()