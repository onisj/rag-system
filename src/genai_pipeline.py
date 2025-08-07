"""
OpenVINO GenAI RAG Pipeline

This module implements a Retrieval-Augmented Generation (RAG) system
using OpenVINO GenAI for optimized inference on Intel hardware.
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Union
from dataclasses import dataclass

# Add src to path for imports
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

import torch
from rich.console import Console

# Import OpenVINO GenAI
import openvino_genai as ov_genai

# Import local modules
from vector_store import VectorStore
from performance_monitor import create_performance_monitor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize rich console
console = Console()

@dataclass
class RAGResponse:
    """Response from RAG system"""
    answer: str
    context: List[str]
    sources: List[Dict[str, Any]]
    metadata: Dict[str, Any]

class GenAIRAGEngine:
    """
    OpenVINO GenAI RAG Engine
    
    A Retrieval-Augmented Generation system using OpenVINO GenAI
    for optimized inference on Intel hardware.
    """
    
    def __init__(
        self, 
        model_path: str,
        vector_store_path: str,
        device: str = None,
        enable_monitoring: bool = None
    ):
        """
        Initialize GenAI RAG Engine
        
        Args:
            model_path: Path to the converted GenAI model
            vector_store_path: Path to the vector store
            device: Inference device (CPU/GPU/AUTO)
            enable_monitoring: Enable performance monitoring
        """
        self.model_path = model_path
        self.vector_store_path = vector_store_path
        
        # Use environment variables if not provided
        if device is None:
            import os
            self.device = os.getenv('DEVICE', 'AUTO')
        else:
            self.device = device
            
        if enable_monitoring is None:
            import os
            self.enable_monitoring = os.getenv('ENABLE_MONITORING', 'true').lower() == 'true'
        else:
            self.enable_monitoring = enable_monitoring
        
        # Initialize components
        self.genai_pipeline = None
        self.tokenizer = None
        self.vector_store = None
        self.performance_monitor = None
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self) -> None:
        """Initialize all GenAI components"""
        try:
            console.print("Initializing OpenVINO GenAI RAG Engine...", style="bold blue")
            
            # Initialize performance monitor if enabled
            if self.enable_monitoring:
                self.performance_monitor = create_performance_monitor()
                console.print("Performance monitoring enabled", style="green")
            
            # Auto-detect optimal device
            if self.device == "AUTO":
                self.device = self._select_optimal_device()
            elif self.device == "GPU":
                self.device = self._select_optimal_device()
            
            # Load GenAI model
            self._load_genai_model()
            
            # Load tokenizer
            self._load_tokenizer()
            
            # Load vector store
            self._load_vector_store()
            
            console.print(f"GenAI RAG Engine initialized successfully on {self.device}", style="bold green")
            
        except Exception as e:
            console.print(f"Error initializing GenAI RAG Engine: {e}", style="red")
            raise RuntimeError(f"Failed to initialize GenAI RAG Engine: {e}")
    
    def _select_optimal_device(self) -> str:
        """Select optimal device for inference"""
        try:
            # Check for GPU availability
            if torch.cuda.is_available():
                return "GPU"
            else:
                return "CPU"
        except Exception:
            return "CPU"
    
    def _load_genai_model(self) -> None:
        """Load GenAI model"""
        try:
            console.print(f"Loading GenAI model on {self.device}...", style="blue")
            
            # Load from local path
            console.print(f"Loading from local path: {self.model_path}", style="blue")
            self.genai_pipeline = ov_genai.LLMPipeline(
                models_path=self.model_path,
                device=self.device
            )
            
            console.print(f"GenAI model loaded and compiled for device: {self.device}", style="green")
            
        except Exception as e:
            console.print(f"Error loading GenAI model: {e}", style="red")
            raise RuntimeError(f"Failed to load GenAI model: {e}")
    
    def _load_tokenizer(self) -> None:
        """Load tokenizer"""
        try:
            # For GenAI, we use the built-in tokenizer
            console.print("Tokenizer loaded", style="green")
            
        except Exception as e:
            console.print(f"Error loading tokenizer: {e}", style="red")
            raise RuntimeError(f"Failed to load tokenizer: {e}")
    
    def _load_vector_store(self) -> None:
        """Load vector store"""
        try:
            self.vector_store = VectorStore()
            self.vector_store.load(self.vector_store_path)
            console.print("Vector store loaded", style="green")
            
        except Exception as e:
            console.print(f"Error loading vector store: {e}", style="red")
            raise RuntimeError(f"Failed to load vector store: {e}")
    
    def _create_prompt(self, question: str, context: str) -> str:
        """Create prompt for the model"""
        prompt = f"""Context: {context}

Question: {question}

Answer:"""
        return prompt
    
    def query(
        self,
        question: str,
        k: int = None,
        max_tokens: int = None,
        temperature: float = None,
        stream: bool = True
    ) -> Union[str, RAGResponse]:
        """
        Query the RAG system
        
        Args:
            question: The question to ask
            k: Number of context chunks to retrieve
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stream: Whether to stream the response
            
        Returns:
            Generated answer
        """
        # Use environment variables for defaults if not provided
        if k is None:
            import os
            k = int(os.getenv('K_CHUNKS', '5'))
        if max_tokens is None:
            import os
            max_tokens = int(os.getenv('MAX_TOKENS', '512'))
        if temperature is None:
            import os
            temperature = float(os.getenv('TEMPERATURE', '0.7'))
            
        try:
            # Retrieve relevant documents
            results = self.vector_store.search(question, k=k)
            
            # Extract context from results
            context_chunks = [result.get('text', '') for result in results]
            context = "\n".join(context_chunks)
            
            # Create prompt
            prompt = self._create_prompt(question, context)
            
            # Generate response using GenAI
            if stream:
                return self._generate_streaming_response(prompt, max_tokens, temperature)
            else:
                return self._generate_response(prompt, max_tokens, temperature)
                
        except Exception as e:
            console.print(f"Error during query: {e}", style="red")
            raise RuntimeError(f"Query failed: {e}")
    
    def _generate_response(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Generate response using GenAI"""
        try:
            # Configure generation
            generation_config = ov_genai.GenerationConfig(
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True
            )
            
            # Generate response
            response = self.genai_pipeline.generate(
                prompt,
                generation_config=generation_config
            )
            
            return response.text
            
        except Exception as e:
            console.print(f"Error generating response: {e}", style="red")
            raise RuntimeError(f"Response generation failed: {e}")
    
    def _generate_streaming_response(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Generate streaming response using GenAI"""
        try:
            # Configure generation
            generation_config = ov_genai.GenerationConfig(
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True
            )
            
            # Generate streaming response
            response = ""
            for token in self.genai_pipeline.generate_stream(
                prompt,
                generation_config=generation_config
            ):
                response += token
                console.print(token, end="", style="green")
            
            console.print()  # New line
            return response
            
        except Exception as e:
            console.print(f"Error generating streaming response: {e}", style="red")
            raise RuntimeError(f"Streaming response generation failed: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        stats = {
            "model_path": self.model_path,
            "device": self.device,
            "vector_store_path": self.vector_store_path,
            "enable_monitoring": self.enable_monitoring
        }
        
        if self.performance_monitor:
            stats.update(self.performance_monitor.get_stats())
        
        return stats 