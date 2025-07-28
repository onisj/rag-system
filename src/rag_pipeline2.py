"""
RAG Engine - Core component for Retrieval-Augmented Generation.

The RAGEngine class integrates a large language model (LLM) with a vector search system to 
provide context-aware responses to user queries. It uses OpenVINO for optimized model inference 
on CPU or GPU, a tokenizer for text processing, and a vector store for retrieving relevant context. 
The engine supports streaming responses, performance monitoring, and automatic device selection 
for optimal hardware utilization.

Key Features:
- Combines LLM inference with vector-based document retrieval for enhanced question answering.
- Supports OpenVINO-optimized inference on CPU, GPU, or AUTO device selection.
- Integrates with a vector store for efficient context retrieval.
- Provides performance monitoring and statistics for inference latency and system usage.
- Supports streaming and non-streaming response generation.

Example:
    ```python
    rag = RAGEngine(
        model_path="/path/to/model",
        vector_store_path="/path/to/vector_store",
        device="AUTO",
        enable_monitoring=True
    )
    response = rag.query("What is RAG?", k=5, max_tokens=512, temperature=0.7)
    print(response)
    
Attributes:
    - model_path (str): Path to the INT4-converted OpenVINO model.
    - vector_store_path (str): Path to the vector store data.
    - device (str): Target device for inference (e.g., "CPU", "GPU", "AUTO").
    - enable_monitoring (bool): Whether to enable performance monitoring.
    - core (openvino.runtime.Core | None): OpenVINO runtime core.
    - model (openvino.runtime.Model | None): Loaded OpenVINO model.
    - compiled_model (openvino.runtime.CompiledModel | None): Compiled model for inference.
    - tokenizer (transformers.AutoTokenizer | None): Tokenizer for text processing.
    - vector_store (VectorStore | None): Vector store for document retrieval.
    - performance_monitor (Any | None): Performance monitoring instance.

Raises:
    RuntimeError: If initialization of OpenVINO core, model, tokenizer, or vector store fails.
"""

import openvino as ov
from transformers import AutoTokenizer
import numpy as np
from typing import List, Dict, Any, Generator
from rich.console import Console
import logging
import time
import os
from dotenv import load_dotenv
import threading

load_dotenv()

from vector_store import VectorStore
from performance_monitor import create_performance_monitor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

console = Console()

class RAGEngine:
    """
    RAG Engine that combines LLM inference with vector search
    """
    
    def __init__(self, model_path: str, vector_store_path: str, device: str = "AUTO", enable_monitoring: bool = True):
        """
        Initialize RAG Engine
        
        Args:
            # Generate tokens one by one
            current_ids = input_ids.copy()
            generated_text = ""
            generation_pos = generation_start_posmodel_path: Path to the converted INT4 model
            vector_store_path: Path to the vector store
            device: Device to run inference on (CPU/GPU/AUTO)
            enable_monitoring: Enable performance monitoring
        """
        self.model_path = model_path
        self.vector_store_path = vector_store_path
        self.device = device
        self.enable_monitoring = enable_monitoring
        
        self.core = None
        self.model = None
        self.compiled_model = None
        self.tokenizer = None
        self.vector_store = None
        self.performance_monitor = None
        self._monitor_stop_event = threading.Event()
        
        console.print("Initializing RAG Engine...", style="bold blue")
        self._initialize_components()
    
    def _initialize_components(self) -> None:
        """Initialize all components (OpenVINO model, tokenizer, vector store)"""
        try:
            # Initialize OpenVINO
            self.core = ov.Core()
            console.print(f"OpenVINO initialized. Available devices: {self.core.available_devices}", style="green")
            
            # Initialize performance monitor if enabled
            if self.enable_monitoring:
                self.performance_monitor = create_performance_monitor(self.core)
                console.print("Performance monitoring enabled", style="green")
            
            # Auto-detect optimal device or resolve generic device names
            if self.device == "AUTO":
                self.device = self._select_optimal_device()
            elif self.device == "GPU":
                self.device = self._select_optimal_device()
            
            # Load model
            self._load_model()
            
            # Load tokenizer
            self._load_tokenizer()
            
            # Load vector store
            self._load_vector_store()
            
            console.print(f"RAG Engine initialized successfully on {self.device}", style="bold green")
            
        except Exception as e:
            console.print(f"Error initializing RAG Engine: {e}", style="red")
            raise RuntimeError(f"Failed to initialize RAG Engine: {e}")
    
    def _select_optimal_device(self) -> str:
        """Select the optimal device for inference based on hardware"""
        if not self.core:
            console.print("OpenVINO core not initialized", style="red")
            return "CPU"
        
        try:
            gpu_devices = [d for d in self.core.available_devices if d.startswith("GPU")]
            
            if gpu_devices:
                console.print(f"Found GPU devices: {gpu_devices}", style="green")
                
                for gpu_device in gpu_devices:
                    try:
                        gpu_props = self.core.get_property(gpu_device, "FULL_DEVICE_NAME")
                        console.print(f"Analyzing GPU {gpu_device}: {gpu_props}", style="dim")
                        
                        if "Intel" in gpu_props or "UHD" in gpu_props:
                            console.print(f"Selected Intel UHD GPU: {gpu_device} ({gpu_props})", style="bold green")
                            return gpu_device
                        elif "NVIDIA" in gpu_props or "RTX" in gpu_props or "GeForce" in gpu_props:
                            console.print(f"Selected NVIDIA GPU: {gpu_device} ({gpu_props})", style="bold green")
                            return gpu_device
                    except Exception as e:
                        console.print(f"Could not analyze GPU {gpu_device}: {e}", style="yellow")
                        continue
                
                console.print("No compatible GPUs found, falling back to CPU", style="yellow")
                return "CPU"
            else:
                console.print("No GPU detected, using CPU", style="yellow")
                return "CPU"
                
        except Exception as e:
            console.print(f"Error in device selection: {e}", style="red")
            return "CPU"
    
    def _load_model(self) -> None:
        """Load the converted INT4 model"""
        if not self.core:
            raise RuntimeError("Cannot load model: OpenVINO core is not initialized")
        
        try:
            console.print(f"Loading INT4 model on {self.device}...", style="blue")
            
            self.model = self.core.read_model(self.model_path)
            
            compilation_config = {}
            
            if self.device.startswith("GPU"):
                compilation_config.update({
                    "PERFORMANCE_HINT": "LATENCY",
                    "INFERENCE_PRECISION_HINT": "FP16",
                })
                try:
                    gpu_props = self.core.get_property(self.device, "FULL_DEVICE_NAME")
                    if "Intel" not in gpu_props and "UHD" not in gpu_props:
                        compilation_config["NUM_STREAMS"] = 1
                except:
                    pass
                console.print("Applied GPU optimizations", style="green")
            
            self.compiled_model = self.core.compile_model(
                self.model, 
                device_name=self.device,
                config=compilation_config
            )
            
            console.print(f"Model loaded and compiled for device: {self.device}", style="green")
            
            try:
                input_info = self.compiled_model.inputs[0]
                output_info = self.compiled_model.outputs[0]
                console.print(f"Model input shape: {input_info.shape}", style="dim")
                console.print(f"Model output shape: {output_info.shape}", style="dim")
            except:
                pass
            
        except Exception as e:
            console.print(f"Error loading model: {e}", style="red")
            raise RuntimeError(f"Failed to load model: {e}")
    
    def _load_tokenizer(self) -> None:
        """Load the tokenizer with proper configuration"""
        try:
            console.print("Loading tokenizer...", style="blue")
            
            # Try to load tokenizer from the original model directory first
            original_model_dir = "./models/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
            
            if os.path.exists(original_model_dir):
                console.print("Loading tokenizer from original model directory...", style="dim")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    original_model_dir,
                    use_fast=True,
                    trust_remote_code=True,
                    clean_up_tokenization_spaces=False  # Important for Llama 3.1
                )
            else:
                model_dir = str(self.model_path).replace("/model.xml", "").replace("\\model.xml", "")
                console.print(f"Loading tokenizer from converted model directory: {model_dir}", style="dim")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_dir,
                    use_fast=True,
                    trust_remote_code=True,
                    clean_up_tokenization_spaces=False
                )
            
            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                console.print("Set pad token to EOS token", style="dim")
            
            # Test tokenizer
            test_text = "Hello world"
            test_tokens = self.tokenizer.encode(test_text)
            test_decoded = self.tokenizer.decode(test_tokens, skip_special_tokens=True)
            console.print(f"Tokenizer test: '{test_text}' -> {test_tokens} -> '{test_decoded}'", style="dim")
            
            console.print("Tokenizer loaded successfully", style="green")
            
        except Exception as e:
            console.print(f"Error loading tokenizer: {e}", style="red")
            raise RuntimeError(f"Failed to load tokenizer: {e}")
    
    def _load_vector_store(self) -> None:
        """Load the vector store"""
        try:
            console.print("Loading vector store...", style="blue")
            
            self.vector_store = VectorStore()
            self.vector_store.load(self.vector_store_path)
            
            console.print("Vector store loaded successfully", style="green")
            
        except Exception as e:
            console.print(f"Error loading vector store: {e}", style="red")
            raise RuntimeError(f"Failed to load vector store: {e}")
    
    def _retrieve_relevant_chunks(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant chunks for the query"""
        if not self.vector_store:
            raise RuntimeError("Cannot retrieve chunks: Vector store is not initialized")
        
        try:
            results = self.vector_store.search(query, k)
            chunks = [chunk for chunk, score in results]
            return chunks
            
        except Exception as e:
            console.print(f"Error retrieving chunks: {e}", style="red")
            raise RuntimeError(f"Failed to retrieve chunks: {e}")
    
    def _create_prompt(self, query: str, chunks: List[Dict[str, Any]]) -> str:
        """Create a prompt with retrieved context, optimized for token length"""
        # Limit context to avoid exceeding token limits
        context_parts = []
        total_context_length = 0
        max_context_length = 400  # Leave room for question and answer
        
        for i, chunk in enumerate(chunks, 1):
            chunk_text = f"Context {i}: {chunk['text'][:200]}..."  # Limit each chunk
            if total_context_length + len(chunk_text) > max_context_length:
                break
            context_parts.append(chunk_text)
            total_context_length += len(chunk_text)
        
        context = "\n\n".join(context_parts)
        
        # Create a shorter prompt to fit within token limits
        prompt = f"""Context:
{context}

Question: {query}

Answer:"""
        
        return prompt
    
    def _stop_monitoring_safely(self):
        """Safely stop performance monitoring"""
        if self.performance_monitor:
            try:
                self._monitor_stop_event.set()
                
                if hasattr(self.performance_monitor, '_stop_monitoring'):
                    self.performance_monitor._stop_monitoring = True
                if hasattr(self.performance_monitor, 'stop_flag'):
                    self.performance_monitor.stop_flag = True       # type: ignore
                
                self.performance_monitor.stop_monitoring()
                time.sleep(0.1)
                
                console.print("Performance monitoring stopped", style="green")
                
            except Exception as e:
                console.print(f"Warning: Error stopping performance monitor: {e}", style="yellow")
            finally:
                self.performance_monitor = None
    
    def _generate_response_simple(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> Generator[str, None, None]:
        """
        Simplified generation method that focuses on correct tokenization
        """
        if not self.tokenizer or not self.compiled_model:
            raise RuntimeError("Tokenizer or model not properly initialized")
        
        try:
            # Start monitoring if enabled
            if self.performance_monitor:
                self.performance_monitor.start_monitoring()
                console.print("Performance monitoring started", style="dim")
            
            # Tokenize the prompt
            console.print("Tokenizing prompt...", style="dim")
            
            # Tokenize to exactly 512 tokens (model's expected input size)
            inputs = self.tokenizer(
                prompt,
                return_tensors="np",
                truncation=True,
                max_length=512,  # Must match model's input shape
                padding="max_length"  # Pad to exactly 512 tokens
            )
            
            input_ids = inputs["input_ids"]
            console.print(f"Input shape: {input_ids.shape}", style="dim")
            console.print(f"Input tokens: {input_ids[0][:10]}...", style="dim")
            
            # Verify the input shape matches model expectations
            expected_shape = self.compiled_model.inputs[0].shape
            if input_ids.shape != tuple(expected_shape):
                raise RuntimeError(f"Input shape {input_ids.shape} doesn't match model expected shape {expected_shape}")
            
            console.print(f"Input shape verified: {input_ids.shape}", style="green")
            
            # Find the last non-pad token position to know where to start generating
            last_real_token_pos = -1
            for i in range(511, -1, -1):  # Search backwards
                if input_ids[0, i] != self.tokenizer.pad_token_id:
                    last_real_token_pos = i
                    break
            
            console.print(f"Last real token at position: {last_real_token_pos}", style="dim")
            generation_start_pos = last_real_token_pos + 1
            
            # Initialize current_ids with the input_ids
            current_ids = input_ids.copy()
            generation_pos = generation_start_pos
            generated_text = ""
            
            try:
                for step in range(max_tokens):
                    # Check for early termination
                    if self._monitor_stop_event.is_set():
                        break
                    
                    # Stop if we've filled all available positions
                    if generation_pos >= 512:
                        console.print("Reached maximum sequence length", style="yellow")
                        break
                    
                    # For fixed-shape models, ensure we always have exactly 512 tokens
                    assert current_ids.shape[1] == 512, f"Expected 512 tokens, got {current_ids.shape[1]}"
                    
                    # Create attention mask
                    attention_mask = np.ones_like(current_ids, dtype=np.float32)
                    
                    # Prepare model inputs based on the model's expected inputs
                    if len(self.compiled_model.inputs) == 1:
                        model_input = current_ids
                    else:
                        # Multi-input model
                        model_input = [current_ids]
                        if len(self.compiled_model.inputs) > 1:
                            model_input.append(attention_mask)
                        # Add any additional required inputs as zeros
                        for i in range(2, len(self.compiled_model.inputs)):
                            model_input.append(np.zeros_like(current_ids))
                    
                    # Run inference
                    outputs = self.compiled_model(model_input)
                    
                    # Extract logits
                    if hasattr(outputs, 'keys'):
                        first_key = next(iter(outputs.keys()))
                        logits = outputs[first_key]
                    elif isinstance(outputs, (list, tuple)):
                        logits = outputs[0]
                    else:
                        logits = outputs
                    
                    # Get the last token's logits
                    last_token_logits = logits[0, -1, :]
                    
                    # Sample next token with temperature
                    if temperature > 0:
                        last_token_logits = last_token_logits / temperature
                    
                    # Apply softmax
                    exp_logits = np.exp(last_token_logits - np.max(last_token_logits))
                    probs = exp_logits / np.sum(exp_logits)
                    
                    # Sample from distribution
                    next_token = np.random.choice(len(probs), p=probs)
                    
                    # Check for EOS token
                    if next_token == self.tokenizer.eos_token_id:
                        console.print("\nEOS token reached", style="dim")
                        break
                    
                    # Decode the token
                    try:
                        # Decode just this token
                        token_text = self.tokenizer.decode([next_token], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                        
                        # Filter out empty or problematic tokens
                        if token_text and len(token_text.strip()) > 0:
                            generated_text += token_text
                            yield token_text
                        
                    except Exception as decode_error:
                        console.print(f"Decode error for token {next_token}: {decode_error}", style="yellow")
                        continue
                    
                    # Place the new token at the current generation position
                    if generation_pos < 512:
                        current_ids[0, generation_pos] = next_token
                        generation_pos += 1
                    else:
                        # If we're at the end, shift everything left and add new token at the end
                        current_ids = np.concatenate([current_ids[:, 1:], [[next_token]]], axis=1)
                    
                    # Progress indicator (less frequent)
                    if step % 25 == 0 and step > 0:
                        console.print(f"Generated {step} tokens...", style="dim")
                
                console.print(f"\nGeneration completed. Total generated text length: {len(generated_text)}", style="green")
                
            finally:
                # Always stop monitoring
                self._stop_monitoring_safely()
                
        except Exception as e:
            self._stop_monitoring_safely()
            console.print(f"Error in generation: {e}", style="red")
            raise RuntimeError(f"Failed to generate response: {e}")
    
    def query(self, question: str, k: int = 5, max_tokens: int = 256, temperature: float = 0.7, stream: bool = True) -> str:
        """
        Main query method for RAG pipeline with reduced token limit for stability
        """
        try:
            console.print(f"\nQuestion: {question}", style="bold blue")
            
            # Step 1: Retrieve relevant chunks
            start_time = time.time()
            chunks = self._retrieve_relevant_chunks(question, k)
            retrieval_time = time.time() - start_time
            
            console.print(f"Retrieved {len(chunks)} relevant chunks in {retrieval_time:.2f}s", style="green")
            
            # Display retrieved chunks (shortened)
            for i, chunk in enumerate(chunks, 1):
                console.print(f"Chunk {i}: {chunk['text'][:100]}...", style="dim")
            
            # Step 2: Create prompt with context
            prompt = self._create_prompt(question, chunks)
            console.print(f"Prompt length: {len(prompt)} characters", style="dim")
            
            # Step 3: Generate response
            console.print(f"\nGenerating response...", style="blue")
            
            start_time = time.time()
            full_response = ""
            
            if stream:
                for token in self._generate_response_simple(prompt, max_tokens, temperature):
                    console.print(token, end="", style="white")
                    full_response += token
            else:
                for token in self._generate_response_simple(prompt, max_tokens, temperature):
                    full_response += token
            
            generation_time = time.time() - start_time
            
            console.print(f"\n\nGeneration completed in {generation_time:.2f}s", style="green")
            
            return full_response.strip()
            
        except Exception as e:
            self._stop_monitoring_safely()
            console.print(f"Error in RAG query: {e}", style="red")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        stats = {
            "device": self.device,
            "model_loaded": self.model is not None,
            "tokenizer_loaded": self.tokenizer is not None,
            "vector_store_loaded": self.vector_store is not None
        }
        
        if self.performance_monitor:
            try:
                perf_stats = self.performance_monitor.get_performance_summary()
                stats.update(perf_stats)
            except:
                stats["performance_monitoring"] = "error"
        
        return stats
    
    def __del__(self):
        """Cleanup on object destruction"""
        self._stop_monitoring_safely()

def main():
    """Main function for testing RAG engine"""
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG Engine")
    parser.add_argument("--model-path", required=True, help="Path to converted INT4 model")
    parser.add_argument("--vector-store-path", required=True, help="Path to vector store")
    parser.add_argument("--device", default="CPU", help="Device to run inference on")
    parser.add_argument("--query", help="Test query")
    parser.add_argument("--k", type=int, default=5, help="Number of chunks to retrieve")
    parser.add_argument("--max-tokens", type=int, default=256, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming")
    
    args = parser.parse_args()
    
    # Initialize RAG engine
    rag_engine = RAGEngine(
        model_path=args.model_path,
        vector_store_path=args.vector_store_path,
        device=args.device
    )
    
    # Test query if provided
    if args.query:
        response = rag_engine.query(
            question=args.query,
            k=args.k,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            stream=not args.no_stream
        )
        
        if args.no_stream:
            console.print(f"\nResponse: {response}", style="bold green")
    
    # Show stats
    stats = rag_engine.get_stats()
    console.print(f"\nRAG Engine Stats: {stats}", style="bold green")

if __name__ == "__main__":
    main()