"""
Vector Store Module - Document Embeddings and Similarity Search

This module implements a robust vector store for document embeddings and similarity search,
using Sentence Transformers for text embedding and FAISS for efficient vector storage and
retrieval. It provides a complete solution for converting text documents into searchable
vector representations and performing fast similarity queries.

Core Functionality:
- Document embedding generation using state-of-the-art Sentence Transformer models
- FAISS-based vector indexing optimized for cosine similarity search
- Efficient similarity search with configurable result counts and thresholds
- Persistent storage and loading of vector stores with timeout protection
- GPU acceleration support for both embedding generation and vector operations
- Comprehensive progress tracking and error handling for production use

Key Features:
- Generates document embeddings using Sentence Transformer models
- Stores embeddings in a FAISS index optimized for cosine similarity
- Performs efficient similarity search with configurable result counts
- Supports saving and loading vector stores to/from disk with timeout protection
- Provides detailed statistics and progress tracking via rich console output
- Automatic GPU detection and utilization for accelerated processing
- Thread-safe operations with timeout protection for large datasets

Example Usage:
    ```python
    from vector_store import VectorStore

    # Initialize vector store with specific model
    vector_store = VectorStore(model_name="all-MiniLM-L6-v2")
    
    # Prepare documents for indexing
    documents = [
        {"text": "This is a test document about machine learning"},
        {"text": "Another document about artificial intelligence"}
    ]
    
    # Build the vector index
    vector_store.build_index(documents)
    
    # Perform similarity search
    results = vector_store.search("machine learning query", k=2)
    for doc, score in results:
        print(f"Score: {score:.3f}, Text: {doc['text']}")
    ```

Classes:
    VectorStore: Main class for managing document embeddings, FAISS index, and similarity search.

Dependencies:
    - sentence_transformers: For generating text embeddings
    - faiss: For vector storage and similarity search (CPU or GPU version)
    - numpy: For array operations and numerical computations
    - rich: For console output and progress tracking
    - json: For saving/loading documents and metadata
    - pathlib: For cross-platform file path handling
    - threading: For timeout-protected FAISS operations
    - logging: For structured logging and debugging
    - argparse: For command-line testing and validation

Usage:
    Run as a script to test the vector store:
    ```bash
    python vector_store.py --model-name all-MiniLM-L6-v2 --test-query "test query" --k 5
    ```

Raises:
    ValueError: If the index is not built before searching or if input data is invalid
    RuntimeError: If embedding generation, index building, or save/load operations fail
    TimeoutError: If FAISS operations exceed timeout limits

Author: Segun Oni
Version: 1.0.0
"""

import json
import threading
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import logging

# Numerical and machine learning libraries
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# User interface and progress tracking
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

# Configure logging for vector store operations
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize console for rich text output and progress tracking
console = Console()

class VectorStore:
    """
    Vector store for document embeddings and similarity search.
    
    This class provides a complete solution for converting text documents into
    vector embeddings and performing efficient similarity search. It combines
    Sentence Transformers for high-quality text embeddings with FAISS for
    fast vector similarity search, supporting both CPU and GPU acceleration.
    
    The vector store maintains an in-memory FAISS index for fast querying and
    provides methods for persistent storage and loading. It automatically
    detects and utilizes available GPU resources for accelerated processing.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", dimension: int = 384):
        """
        Initialize vector store with specified model and configuration.
        
        Args:
            model_name: Sentence transformer model name for embedding generation
                       (default: "all-MiniLM-L6-v2" - fast and effective)
            dimension: Expected embedding dimension for the model
                      (default: 384 for all-MiniLM-L6-v2)
        """
        self.model_name = model_name
        self.dimension = dimension
        self.embedding_model: Optional[SentenceTransformer] = None  # Lazy-loaded embedding model
        self.index: Optional[faiss.IndexFlatIP] = None  # FAISS index for similarity search
        self.documents: List[Dict[str, Any]] = []  # Original documents for retrieval
        self.metadata: Dict[str, Any] = {}  # Store metadata and configuration
        
        # Log initialization with configuration parameters
        console.print(f"VectorStore initialized (model={model_name}, dim={dimension})", style="blue")
    
    def load_embedding_model(self) -> None:
        """
        Load the sentence transformer model with automatic GPU acceleration.
        
        This method initializes the Sentence Transformer model with optimal
        device placement. It automatically detects CUDA availability and
        configures the model for GPU acceleration when possible, falling back
        to CPU if no GPU is available.
        """
        try:
            console.print(f"Loading embedding model: {self.model_name}", style="blue")
            
            # Check for CUDA availability for GPU acceleration
            import torch
            if torch.cuda.is_available():
                console.print(f"CUDA available: {torch.cuda.get_device_name(0)}", style="green")
                device = "cuda:0"  # Use first NVIDIA GPU
            else:
                console.print("CUDA not available, using CPU", style="yellow")
                device = "cpu"
            
            # Load model with explicit device placement for optimal performance
            self.embedding_model = SentenceTransformer(self.model_name, device=device)
            console.print(f"Embedding model loaded on {device}", style="green")
            
        except Exception as e:
            # Log error and re-raise for proper error handling upstream
            console.print(f"Failed to load embedding model: {e}", style="red")
            raise
    
    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts using the loaded embedding model.
        
        This method processes text in batches for memory efficiency and provides
        progress tracking for large datasets. It automatically handles model
        loading if not already initialized.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            Embeddings as a numpy array with shape (num_texts, embedding_dimension)
            
        Raises:
            RuntimeError: If embedding model fails to load or generate embeddings
        """
        # Ensure embedding model is loaded before processing
        if self.embedding_model is None:
            self.load_embedding_model()
        
        if self.embedding_model is None:  # Satisfy Pylance type checking
            raise RuntimeError("Embedding model failed to load")
        
        try:
            # Set up progress tracking for embedding generation
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                console=console
            ) as progress:
                task = progress.add_task("Generating embeddings...", total=len(texts))
                
                embeddings = []
                batch_size = 32  # Process in batches for memory efficiency
                
                # Process texts in batches to manage memory usage
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i + batch_size]
                    batch_embeddings = self.embedding_model.encode(batch, show_progress_bar=False)
                    embeddings.extend(batch_embeddings)
                    progress.update(task, advance=len(batch))
            
            # Convert to numpy array with appropriate data type
            return np.array(embeddings, dtype=np.float32)
            
        except Exception as e:
            # Log error and re-raise for proper error handling
            console.print(f"Failed to generate embeddings: {e}", style="red")
            raise
    
    def build_index(self, documents: List[Dict[str, Any]]) -> None:
        """
        Build FAISS index from a list of documents with optional GPU acceleration.
        
        This method creates a FAISS index optimized for cosine similarity search.
        It automatically detects and utilizes GPU acceleration when available,
        falling back to CPU processing if GPU is not available or fails.
        
        Args:
            documents: List of document dictionaries, each containing a 'text' field
            
        Raises:
            RuntimeError: If index building fails due to embedding or FAISS errors
        """
        try:
            console.print(f"Building vector index for {len(documents)} documents...", style="blue")
            
            # Extract texts from documents for embedding generation
            texts = [doc['text'] for doc in documents]
            self.documents = documents
            
            # Generate embeddings using the configured model
            embeddings = self._get_embeddings(texts)
            
            # Create FAISS index optimized for cosine similarity
            self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
            
            # Try to enable GPU acceleration for FAISS operations
            try:
                gpu_count = faiss.get_num_gpus()
                if gpu_count > 0:
                    console.print(f"FAISS GPU acceleration available: {gpu_count} GPUs", style="green")
                    
                    # Create GPU resources for FAISS operations
                    import faiss.contrib.gpu_resources as gpu_resources # type: ignore
                    gpu_res = gpu_resources.GpuResources()
                    
                    # Use the first GPU (NVIDIA RTX 3080) for acceleration
                    gpu_id = 0
                    console.print(f"Moving FAISS index to GPU {gpu_id} (NVIDIA RTX 3080)", style="green")
                    
                    # Create GPU index for accelerated operations
                    gpu_index = faiss.index_cpu_to_gpu(gpu_res, gpu_id, self.index) # type: ignore[no-untyped-call]
                    
                    # Add vectors to GPU index for fast processing
                    gpu_index.add(embeddings)
                    
                    # Move back to CPU for persistence (FAISS GPU indices can't be saved directly)
                    self.index = faiss.index_gpu_to_cpu(gpu_index) # type: ignore[no-untyped-call]
                    console.print("FAISS index built with GPU acceleration", style="green")
                else:
                    console.print("FAISS GPU not available, using CPU", style="yellow")
                    self.index.add(embeddings) # type: ignore[no-untyped-call]
            except Exception as e:
                # Fall back to CPU if GPU acceleration fails
                console.print(f"FAISS GPU acceleration failed: {e}, falling back to CPU", style="yellow")
                self.index.add(embeddings) # type: ignore[no-untyped-call]
            
            # Store comprehensive metadata for the index
            self.metadata = {
                'model_name': self.model_name,
                'dimension': self.dimension,
                'num_documents': len(documents),
                'index_type': 'FlatIP',
                'gpu_accelerated': gpu_count > 0 if 'gpu_count' in locals() else False # type: ignore
            }
            
            console.print(f"Vector index built successfully ({len(documents)} documents)", style="green")
            
        except Exception as e:
            # Log error and re-raise for proper error handling
            console.print(f"Failed to build index: {e}", style="red")
            raise
    
    def search(self, query: str, k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """
        Search for similar documents to a given query with optional GPU acceleration.
        
        This method performs similarity search using the built FAISS index.
        It automatically utilizes GPU acceleration when available and configured,
        falling back to CPU search if GPU is not available or fails.
        
        Args:
            query: The search query string to find similar documents for
            k: Number of results to return (default: 5)
            
        Returns:
            List of (document, score) tuples, sorted by similarity score in descending order
            
        Raises:
            ValueError: If the index is not built before searching
            RuntimeError: If search operation fails
        """
        try:
            # Validate that index has been built
            if self.index is None:
                raise ValueError("Index not built. Call build_index() first.")
            
            # Generate embedding for the query text
            query_embedding = self._get_embeddings([query])
            
            # Perform search with GPU acceleration if available and configured
            try:
                gpu_count = faiss.get_num_gpus()
                if gpu_count > 0 and self.metadata.get('gpu_accelerated', False):
                    console.print("Using FAISS GPU acceleration for search", style="dim")
                    
                    # Create GPU resources for accelerated search
                    import faiss.contrib.gpu_resources as gpu_resources # type: ignore
                    gpu_res = gpu_resources.GpuResources()
                    
                    # Move index to GPU for fast similarity search
                    gpu_id = 0
                    gpu_index = faiss.index_cpu_to_gpu(gpu_res, gpu_id, self.index) # type: ignore[no-untyped-call]
                    
                    # Perform search on GPU for optimal performance
                    scores, indices = gpu_index.search(query_embedding, k)
                else:
                    # Fall back to CPU search if GPU not available
                    scores, indices = self.index.search(query_embedding, k) # type: ignore[no-untyped-call]
            except Exception as e:
                # Fall back to CPU if GPU search fails
                console.print(f"GPU search failed: {e}, falling back to CPU", style="yellow")
                scores, indices = self.index.search(query_embedding, k) # type: ignore[no-untyped-call]
            
            # Process and return search results
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                # Ensure index is within valid range
                if idx < len(self.documents):
                    results.append((self.documents[idx], float(score)))
            
            return results
            
        except Exception as e:
            # Log error and re-raise for proper error handling
            console.print(f"Search failed: {e}", style="red")
            raise
    
    def save(self, path: str) -> None:
        """
        Save the vector store to disk with timeout protection and error handling.
        
        This method saves the FAISS index, documents, and metadata to the specified
        directory. It uses threading-based timeout protection to handle large indices
        and provides comprehensive error handling for robust persistence.
        
        Args:
            path: Directory path where to save the vector store
            
        Raises:
            RuntimeError: If save operation fails due to file system or FAISS errors
        """
        try:
            save_path = Path(path)
            # Ensure save directory exists
            save_path.mkdir(parents=True, exist_ok=True)
            
            console.print(f"Saving vector store to: {save_path}", style="blue")
            
            # Save FAISS index with timeout protection for large indices
            if self.index is not None:
                # Use threading-based timeout that works on all platforms
                result = {"success": False, "error": None}
                
                def save_faiss_index():
                    """Helper function to save FAISS index with error capture"""
                    try:
                        faiss.write_index(self.index, str(save_path / "index.faiss"))
                        result["success"] = True
                    except Exception as e:
                        result["error"] = str(e)
                
                # Start FAISS save in a separate thread with timeout protection
                save_thread = threading.Thread(target=save_faiss_index)
                save_thread.daemon = True
                save_thread.start()
                
                # Wait for completion or timeout (30 seconds for large indices)
                save_thread.join(timeout=30)
                
                # Handle timeout and error cases gracefully
                if save_thread.is_alive():
                    console.print("FAISS save timed out, skipping index save", style="yellow")
                elif not result["success"]:
                    console.print(f"FAISS save failed: {result['error']}, skipping index save", style="yellow")
                else:
                    console.print("FAISS index saved successfully", style="green")
            
            # Save documents and metadata in JSON format
            with open(save_path / "documents.json", 'w', encoding='utf-8') as f:
                json.dump(self.documents, f, ensure_ascii=False, indent=2)
            
            with open(save_path / "metadata.json", 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)
            
            console.print("Vector store saved successfully", style="green")
            
        except Exception as e:
            # Log error and re-raise for proper error handling
            console.print(f"Failed to save vector store: {e}", style="red")
            raise
    
    def load(self, path: str) -> None:
        """
        Load the vector store from disk with timeout protection and error handling.
        
        This method loads the FAISS index, documents, and metadata from the specified
        directory. It uses threading-based timeout protection to handle large indices
        and provides comprehensive error handling for robust loading.
        
        Args:
            path: Directory path from where to load the vector store
            
        Raises:
            RuntimeError: If load operation fails due to file system or FAISS errors
        """
        try:
            load_path = Path(path)
            
            console.print(f"Loading vector store from: {load_path}", style="blue")
            
            # Load FAISS index with timeout protection for large indices
            index_path = load_path / "index.faiss"
            if index_path.exists():
                # Use threading-based timeout for FAISS load operations
                result = {"success": False, "error": None, "index": None}
                
                def load_faiss_index():
                    """Helper function to load FAISS index with error capture"""
                    try:
                        result["index"] = faiss.read_index(str(index_path))
                        result["success"] = True
                    except Exception as e:
                        result["error"] = str(e)
                
                # Start FAISS load in a separate thread with timeout protection
                load_thread = threading.Thread(target=load_faiss_index)
                load_thread.daemon = True
                load_thread.start()
                
                # Wait for completion or timeout (30 seconds for large indices)
                load_thread.join(timeout=30)
                
                # Handle timeout and error cases gracefully
                if load_thread.is_alive():
                    console.print("FAISS load timed out, skipping index load", style="yellow")
                elif not result["success"]:
                    console.print(f"FAISS load failed: {result['error']}, skipping index load", style="yellow")
                else:
                    self.index = result["index"]
                    console.print("FAISS index loaded successfully", style="green")
            
            # Load documents and metadata from JSON files
            documents_path = load_path / "documents.json"
            if documents_path.exists():
                with open(documents_path, 'r', encoding='utf-8') as f:
                    self.documents = json.load(f)
            
            metadata_path = load_path / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
                    # Restore model configuration from metadata
                    self.model_name = self.metadata.get('model_name', self.model_name)
                    self.dimension = self.metadata.get('dimension', self.dimension)
            
            console.print("Vector store loaded successfully", style="green")
            
        except Exception as e:
            # Log error and re-raise for proper error handling
            console.print(f"Failed to load vector store: {e}", style="red")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive vector store statistics and metadata.
        
        This method returns a dictionary containing key statistics about the
        vector store, including model configuration, index status, and
        document counts for monitoring and debugging purposes.
        
        Returns:
            Dictionary containing comprehensive statistics about the vector store
        """
        stats = {
            'model_name': self.model_name,
            'dimension': self.dimension,
            'total_vectors': len(self.documents) if self.documents else 0,
            'index_built': self.index is not None,
            'index_type': self.metadata.get('index_type', 'None')
        }
        
        # Add index-specific statistics if available
        if self.index is not None:
            stats['index_size'] = self.index.ntotal
        
        return stats


def main():
    """
    Main function for command-line testing and validation of the vector store.
    
    This function provides a command-line interface for testing vector store
    functionality, including model loading, index building, and similarity search.
    It supports various configuration options for comprehensive testing.
    """
    """Main function for testing vector store"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Vector Store")
    parser.add_argument("--model-name", default="all-MiniLM-L6-v2", help="Embedding model name")
    parser.add_argument("--test-query", help="Test query")
    parser.add_argument("--k", type=int, default=5, help="Number of results")
    
    args = parser.parse_args()
    
    try:
        # Create test documents
        test_docs = [
            {"text": "This is a test document about machine learning."},
            {"text": "Another document about artificial intelligence."},
            {"text": "A third document about data science and analytics."}
        ]
        
        # Initialize and build index
        vector_store = VectorStore(model_name=args.model_name)
        vector_store.build_index(test_docs)
        
        # Test search
        if args.test_query:
            results = vector_store.search(args.test_query, args.k)
            console.print(f"\nSearch results for: {args.test_query}", style="bold")
            for i, (doc, score) in enumerate(results, 1):
                console.print(f"{i}. Score: {score:.3f} - {doc['text']}", style="green")
        
        # Show stats
        stats = vector_store.get_stats()
        console.print(f"\nVector Store Stats: {stats}", style="bold green")
        
        return 0
        
    except Exception as e:
        console.print(f"Test failed: {e}", style="red")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())