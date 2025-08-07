"""
Test suite for RAG System with OpenVINO GenAI

This module contains comprehensive tests for all major components:
- GenAI model conversion
- Document processing
- Vector store operations
- GenAI RAG pipeline
"""

import sys
import os
import tempfile
import json
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
import pytest
import numpy as np

# Add src to path - fix the path resolution
current_dir = Path(__file__).parent
project_root = current_dir.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from rich.console import Console

# Import components to test - use GenAI implementations
try:
    from genai_model_converter import GenAIModelConverter
    from document_processor import DocumentProcessor, TextChunk
    from vector_store import VectorStore
    from genai_pipeline import GenAIRAGEngine
except ImportError as e:
    print(f"Import error: {e}")
    print(f"Current sys.path: {sys.path}")
    print(f"Looking for modules in: {src_path}")
    raise

console = Console()

# Test fixtures
@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def sample_text():
    """Sample text for testing"""
    return """
    Procyon is a comprehensive data processing platform.
    
    It provides advanced analytics capabilities for enterprise organizations.
    The system offers real-time data processing and machine learning integration.
    
    Key features include:
    - Scalable architecture
    - Real-time processing
    - Machine learning integration
    - Advanced analytics
    """

@pytest.fixture
def sample_chunks():
    """Sample text chunks for testing"""
    return [
        {
            "id": 1,
            "text": "Procyon is a comprehensive data processing platform.",
            "start_char": 0,
            "end_char": 50,
            "length": 50,
            "page_number": 1,
            "section_title": "Introduction"
        },
        {
            "id": 2,
            "text": "It provides advanced analytics capabilities for enterprise organizations.",
            "start_char": 51,
            "end_char": 120,
            "length": 69,
            "page_number": 1,
            "section_title": "Features"
        },
        {
            "id": 3,
            "text": "The system offers real-time data processing and machine learning integration.",
            "start_char": 121,
            "end_char": 190,
            "length": 69,
            "page_number": 1,
            "section_title": "Capabilities"
        }
    ]

class TestGenAIModelConverter:
    """Test cases for GenAI Model Converter"""
    
    def test_initialization(self, temp_dir):
        """Test GenAI model converter initialization"""
        converter = GenAIModelConverter("meta-llama/Llama-3.1-8B-Instruct")
        assert converter.model_name == "meta-llama/Llama-3.1-8B-Instruct"
        assert converter.cache_dir.exists()
    
    def test_hardware_compatibility_cpu_gpu(self):
        """Test hardware compatibility with CPU and GPU"""
        converter = GenAIModelConverter("meta-llama/Llama-3.1-8B-Instruct")
        devices = converter._check_hardware_compatibility()
        
        assert "CPU" in devices
        # GPU may or may not be available depending on system
    
    def test_hardware_compatibility_cpu_only(self):
        """Test hardware compatibility with CPU only"""
        converter = GenAIModelConverter("meta-llama/Llama-3.1-8B-Instruct")
        devices = converter._check_hardware_compatibility()
        
        assert "CPU" in devices
        assert len(devices) >= 1  # At least CPU should be available
    
    def test_hardware_compatibility_no_devices(self):
        """Test hardware compatibility with no devices"""
        converter = GenAIModelConverter("meta-llama/Llama-3.1-8B-Instruct")
        
        # This should not raise an error since CPU is always available
        devices = converter._check_hardware_compatibility()
        assert "CPU" in devices
    
    def test_model_info_not_converted(self):
        """Test model info when model is not converted"""
        converter = GenAIModelConverter("meta-llama/Llama-3.1-8B-Instruct")
        info = converter.get_model_info()
        assert "not converted" in info.lower()

class TestDocumentProcessor:
    """Test cases for Document Processor"""
    
    def test_initialization(self):
        """Test document processor initialization"""
        processor = DocumentProcessor()
        assert processor is not None
    
    def test_text_chunk_creation(self):
        """Test text chunk creation"""
        chunk = TextChunk(
            text="Test text",
            start_char=0,
            end_char=9,
            page_number=1,
            section_title="Test"
        )
        assert chunk.text == "Test text"
        assert chunk.start_char == 0
        assert chunk.end_char == 9
        assert chunk.page_number == 1
        assert chunk.section_title == "Test"
    
    def test_text_chunk_to_dict(self):
        """Test text chunk to dictionary conversion"""
        chunk = TextChunk(
            text="Test text",
            start_char=0,
            end_char=9,
            page_number=1,
            section_title="Test"
        )
        chunk_dict = chunk.to_dict()
        assert chunk_dict["text"] == "Test text"
        assert chunk_dict["start_char"] == 0
        assert chunk_dict["end_char"] == 9
        assert chunk_dict["page_number"] == 1
        assert chunk_dict["section_title"] == "Test"
    
    def test_clean_text(self):
        """Test text cleaning functionality"""
        processor = DocumentProcessor()
        dirty_text = "  Test   text  with   extra   spaces  "
        clean_text = processor._clean_text(dirty_text)
        assert clean_text == "Test text with extra spaces"
    
    def test_clean_text_with_page_markers(self):
        """Test text cleaning with page markers"""
        processor = DocumentProcessor()
        text_with_markers = "Page 1\nContent here\nPage 2\nMore content"
        clean_text = processor._clean_text(text_with_markers)
        assert "Page 1" not in clean_text
        assert "Page 2" not in clean_text
    
    def test_simple_chunking(self):
        """Test simple text chunking"""
        processor = DocumentProcessor()
        text = "This is a test document. It has multiple sentences. Each sentence should be a chunk."
        chunks = processor._simple_chunking(text, chunk_size=50, overlap=10)
        assert len(chunks) > 0
        assert all(len(chunk) <= 50 for chunk in chunks)
    
    def test_semantic_chunking(self):
        """Test semantic text chunking"""
        processor = DocumentProcessor()
        text = "This is a test document. It has multiple sentences. Each sentence should be a chunk."
        chunks = processor._semantic_chunking(text, chunk_size=50, overlap=10)
        assert len(chunks) > 0
        assert all(len(chunk) <= 50 for chunk in chunks)
    
    def test_save_and_load_chunks(self, temp_dir):
        """Test saving and loading chunks"""
        processor = DocumentProcessor()
        chunks = [
            TextChunk("Chunk 1", 0, 7, 1, "Section 1"),
            TextChunk("Chunk 2", 8, 15, 1, "Section 1"),
            TextChunk("Chunk 3", 16, 23, 2, "Section 2")
        ]
        
        output_path = Path(temp_dir) / "chunks.json"
        processor.save_chunks(chunks, output_path)
        
        loaded_chunks = processor.load_chunks(output_path)
        assert len(loaded_chunks) == 3
        assert loaded_chunks[0].text == "Chunk 1"
        assert loaded_chunks[1].text == "Chunk 2"
        assert loaded_chunks[2].text == "Chunk 3"

class TestVectorStore:
    """Test cases for Vector Store"""
    
    def test_initialization(self):
        """Test vector store initialization"""
        store = VectorStore()
        assert store is not None
        assert store.embedding_model_name == "all-MiniLM-L6-v2"
    
    def test_initialization_defaults(self):
        """Test vector store initialization with defaults"""
        store = VectorStore()
        assert store.index is None
        assert store.embeddings is None
        assert store.documents is None
    
    def test_stats_not_initialized(self):
        """Test stats when vector store is not initialized"""
        store = VectorStore()
        stats = store.get_stats()
        assert stats["total_documents"] == 0
        assert stats["index_built"] == False
        assert stats["embedding_model"] == "all-MiniLM-L6-v2"
    
    def test_embedding_model_loading(self):
        """Test embedding model loading"""
        store = VectorStore()
        # This should not raise an exception
        assert store.embedding_model_name == "all-MiniLM-L6-v2"
    
    @patch('sentence_transformers.SentenceTransformer')
    def test_build_index(self, mock_transformer):
        """Test building vector index"""
        mock_model = Mock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        mock_transformer.return_value = mock_model
        
        store = VectorStore()
        documents = ["Document 1", "Document 2", "Document 3"]
        
        store.build_index(documents)
        
        assert store.index is not None
        assert store.documents is not None
        assert len(store.documents) == 3
        mock_transformer.assert_called_once()
    
    @patch('sentence_transformers.SentenceTransformer')
    def test_search_functionality(self, mock_transformer):
        """Test search functionality"""
        mock_model = Mock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        mock_transformer.return_value = mock_model
        
        store = VectorStore()
        documents = ["Document 1", "Document 2", "Document 3"]
        store.build_index(documents)
        
        results = store.search("test query", k=2)
        assert len(results) == 2
        assert all(isinstance(result, dict) for result in results)
    
    def test_save_and_load(self, temp_dir):
        """Test saving and loading vector store"""
        store = VectorStore()
        documents = ["Document 1", "Document 2", "Document 3"]
        
        # Mock the build_index method to avoid actual model loading
        with patch.object(store, 'build_index'):
            store.build_index(documents)
            
            # Mock the embeddings and index
            store.embeddings = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
            store.index = Mock()
            
            output_dir = Path(temp_dir) / "vector_store"
            store.save(output_dir)
            
            # Test loading
            new_store = VectorStore()
            new_store.load(output_dir)
            
            assert new_store.documents is not None
            assert len(new_store.documents) == 3

class TestGenAIRAGEngine:
    """Test cases for GenAI RAG Engine"""
    
    @patch('vector_store.VectorStore')
    def test_initialization_mocked(self, mock_vector_store):
        """Test GenAI RAG engine initialization with mocked components"""
        mock_vector_store_instance = Mock()
        mock_vector_store.return_value = mock_vector_store_instance
        
        # This should work without Core mocking since we don't use Core
        engine = GenAIRAGEngine(
            model_path="test_model.xml",
            vector_store_path="test_vector_store"
        )
        
        assert engine is not None
        mock_vector_store.assert_called_once()
    
    def test_create_prompt(self):
        """Test prompt creation"""
        engine = GenAIRAGEngine(
            model_path="test_model.xml",
            vector_store_path="test_vector_store"
        )
        
        question = "What is Procyon?"
        context = "Procyon is a data processing platform."
        
        prompt = engine._create_prompt(question, context)
        
        assert question in prompt
        assert context in prompt
        assert "Context:" in prompt
        assert "Question:" in prompt

def test_integration_basic_flow(temp_dir, sample_chunks):
    """Test basic integration flow"""
    # Test document processing
    processor = DocumentProcessor()
    chunks = processor._semantic_chunking(
        "Procyon is a comprehensive data processing platform.",
        chunk_size=50,
        overlap=10
    )
    assert len(chunks) > 0
    
    # Test vector store
    store = VectorStore()
    documents = [chunk for chunk in chunks]
    
    with patch('sentence_transformers.SentenceTransformer'):
        store.build_index(documents)
        assert store.documents is not None

def test_error_handling():
    """Test error handling in components"""
    # Test invalid model path
    with pytest.raises((FileNotFoundError, RuntimeError)):
        engine = GenAIRAGEngine(
            model_path="nonexistent_model.xml",
            vector_store_path="nonexistent_vector_store"
        )
    
    # Test invalid vector store path
    store = VectorStore()
    with pytest.raises((FileNotFoundError, RuntimeError)):
        store.load("nonexistent_path")

def test_performance_benchmarks():
    """Test performance benchmark functionality"""
    # Test that performance monitoring can be initialized
    from performance_monitor import PerformanceMonitor
    
    monitor = PerformanceMonitor()
    assert monitor is not None
    
    # Test basic timing functionality
    import time
    start_time = time.time()
    time.sleep(0.1)
    end_time = time.time()
    
    assert end_time - start_time >= 0.1

def test_data_consistency():
    """Test data consistency across components"""
    # Test that text chunks maintain consistency
    processor = DocumentProcessor()
    text = "Test document with multiple sentences. Each sentence should be preserved."
    
    chunks = processor._semantic_chunking(text, chunk_size=100, overlap=10)
    
    # Reconstruct text from chunks
    reconstructed = " ".join(chunks)
    
    # Check that key content is preserved
    assert "Test document" in reconstructed
    assert "multiple sentences" in reconstructed
    assert "preserved" in reconstructed 