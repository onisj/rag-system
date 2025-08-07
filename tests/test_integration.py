"""
Integration Test Suite for RAG System with OpenVINO GenAI

This module tests the complete RAG system integration:
- End-to-end workflow with GenAI
- Component interactions
- Real-world scenarios
- Performance under load
- Error recovery
"""

import sys
import os
import tempfile
import shutil
import time
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest
import numpy as np

# Add timeout to prevent hanging tests
# pytestmark = pytest.mark.timeout(30)  # 30 second timeout for all tests

# Add src to path
current_dir = Path(__file__).parent
project_root = current_dir.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from rich.console import Console
from genai_model_converter import GenAIModelConverter
from document_processor import DocumentProcessor, TextChunk
from vector_store import VectorStore
from genai_pipeline import GenAIRAGEngine
import cli
from performance_monitor import PerformanceMonitor

console = Console()

@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def sample_document():
    """Create a sample document for testing"""
    return """
    Procyon Data Processing Platform
    
    Procyon is a comprehensive data processing platform designed for enterprise organizations.
    It provides advanced analytics capabilities and real-time data processing.
    
    Key Features:
    - Scalable architecture for large datasets
    - Real-time processing capabilities
    - Machine learning integration
    - Advanced analytics and reporting
    - Multi-format data support
    
    Architecture:
    The platform uses a distributed architecture with microservices for scalability.
    Data is processed through multiple stages including ingestion, transformation, and analysis.
    
    Use Cases:
    - Financial data analysis
    - Healthcare data processing
    - Manufacturing analytics
    - Retail customer insights
    """

@pytest.fixture
def sample_queries():
    """Sample queries for testing"""
    return [
        "What is Procyon?",
        "What are the key features?",
        "How does the architecture work?",
        "What are the use cases?",
        "Explain the data processing capabilities"
    ]

class TestEndToEndWorkflow:
    """Test complete end-to-end RAG workflow with GenAI"""
    
    @patch('vector_store.SentenceTransformer')
    def test_complete_rag_workflow(self, mock_transformer, sample_document):
        """Test complete RAG workflow from document to answer using GenAI"""
        # Setup mocks
        mock_transformer_instance = Mock()
        mock_transformer_instance.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        mock_transformer.return_value = mock_transformer_instance
        
        # Test document processing
        processor = DocumentProcessor()
        chunks = processor._semantic_chunking(sample_document, chunk_size=150, overlap=30)
        assert len(chunks) > 0
        
        # Test vector store
        store = VectorStore()
        documents = [{"text": chunk.text} for chunk in chunks]
        store.build_index(documents)
        assert store.index is not None
        
        # Test search functionality
        results = store.search("What is Procyon?", k=3)
        assert len(results) > 0
        
        console.print("Complete RAG workflow test passed", style="green")
    
    @patch('vector_store.SentenceTransformer')
    def test_document_processing_integration(self, mock_transformer, sample_document):
        """Test document processing integration"""
        mock_transformer_instance = Mock()
        mock_transformer_instance.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        mock_transformer.return_value = mock_transformer_instance
        
        processor = DocumentProcessor()
        store = VectorStore()
        
        # Process document
        chunks = processor._semantic_chunking(sample_document, chunk_size=150, overlap=30)
        assert len(chunks) > 0
        
        # Build vector store
        documents = [chunk for chunk in chunks]
        store.build_index(documents)
        
        # Verify integration
        assert store.documents is not None
        assert len(store.documents) == len(chunks)
        
        # Test search functionality
        results = store.search("data processing", k=2)
        assert len(results) <= 2
    
    @patch('vector_store.SentenceTransformer')
    def test_vector_store_integration(self, mock_transformer, sample_document):
        """Test vector store integration with GenAI"""
        mock_transformer_instance = Mock()
        mock_transformer_instance.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        mock_transformer.return_value = mock_transformer_instance
        
        store = VectorStore()
        processor = DocumentProcessor()
        
        # Process and store documents
        chunks = processor._semantic_chunking(sample_document, chunk_size=100, overlap=20)
        documents = [chunk for chunk in chunks]
        
        store.build_index(documents)
        
        # Test various search queries
        test_queries = ["Procyon", "architecture", "analytics", "processing"]
        
        for query in test_queries:
            results = store.search(query, k=3)
            assert len(results) <= 3
            assert all(isinstance(result, dict) for result in results)
    
    def test_cli_integration(self):
        """Test CLI integration with GenAI"""
        from click.testing import CliRunner
        
        runner = CliRunner()
        
        # Test help command
        result = runner.invoke(cli.cli, ['--help'])
        assert result.exit_code == 0
        assert "RAG CLI" in result.output
        
        # Test setup command structure
        result = runner.invoke(cli.cli, ['setup', '--help'])
        assert result.exit_code == 0
        
        # Test convert-model command structure
        result = runner.invoke(cli.cli, ['convert-model', '--help'])
        assert result.exit_code == 0
        
        # Test performance command structure
        result = runner.invoke(cli.cli, ['performance', '--help'])
        assert result.exit_code == 0

class TestPerformanceIntegration:
    """Test performance integration with GenAI"""
    
    def test_performance_monitoring_integration(self):
        """Test performance monitoring integration"""
        monitor = PerformanceMonitor()
        assert monitor is not None
        
        # Test basic monitoring
        with monitor.measure("test_operation"):
            time.sleep(0.1)
        
        stats = monitor.get_stats()
        assert "test_operation" in stats
        assert stats["test_operation"]["count"] == 1
        assert stats["test_operation"]["total_time"] > 0
    
    def test_benchmark_integration(self):
        """Test benchmark integration"""
        monitor = PerformanceMonitor()
        
        def test_function():
            time.sleep(0.05)
            return "test_result"
        
        result = monitor.benchmark(test_function, iterations=3)
        assert result["result"] == "test_result"
        assert result["iterations"] == 3
        assert result["total_time"] > 0
        assert result["avg_time"] > 0

class TestErrorHandlingIntegration:
    """Test error handling integration with GenAI"""
    
    @patch('vector_store.SentenceTransformer')
    def test_error_recovery_integration(self, mock_transformer):
        """Test error recovery integration"""
        mock_transformer_instance = Mock()
        mock_transformer_instance.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        mock_transformer.return_value = mock_transformer_instance
        
        store = VectorStore()
        processor = DocumentProcessor()
        
        # Test with empty document
        empty_document = ""
        chunks = processor._semantic_chunking(empty_document, chunk_size=100, overlap=20)
        
        # Should handle empty document gracefully
        if chunks:
            documents = [chunk for chunk in chunks]
            store.build_index(documents)
            assert store.documents is not None
    
    @patch('vector_store.SentenceTransformer')
    def test_invalid_input_handling(self, mock_transformer):
        """Test invalid input handling"""
        mock_transformer_instance = Mock()
        mock_transformer_instance.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        mock_transformer.return_value = mock_transformer_instance
        
        store = VectorStore()
        
        # Test with None input
        with pytest.raises((ValueError, TypeError)):
            store.build_index(None)
        
        # Test with empty list
        with pytest.raises((ValueError, TypeError)):
            store.build_index([])

class TestDataPersistenceIntegration:
    """Test data persistence integration with GenAI"""
    
    @patch('vector_store.SentenceTransformer')
    def test_save_load_integration(self, mock_transformer, temp_dir, sample_document):
        """Test save and load integration"""
        mock_transformer_instance = Mock()
        mock_transformer_instance.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        mock_transformer.return_value = mock_transformer_instance
        
        processor = DocumentProcessor()
        store = VectorStore()
        
        # Process document
        chunks = processor._semantic_chunking(sample_document, chunk_size=150, overlap=30)
        documents = [chunk for chunk in chunks]
        
        # Build and save vector store
        store.build_index(documents)
        output_dir = Path(temp_dir) / "vector_store"
        store.save(output_dir)
        
        # Load and verify
        new_store = VectorStore()
        new_store.load(output_dir)
        
        assert new_store.documents is not None
        assert len(new_store.documents) == len(chunks)
    
    @patch('vector_store.SentenceTransformer')
    def test_vector_store_persistence(self, mock_transformer, temp_dir, sample_document):
        """Test vector store persistence"""
        mock_transformer_instance = Mock()
        mock_transformer_instance.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        mock_transformer.return_value = mock_transformer_instance
        
        store = VectorStore()
        processor = DocumentProcessor()
        
        # Process document
        chunks = processor._semantic_chunking(sample_document, chunk_size=100, overlap=20)
        documents = [chunk for chunk in chunks]
        
        # Build index
        store.build_index(documents)
        
        # Save to temporary directory
        output_dir = Path(temp_dir) / "test_vector_store"
        store.save(output_dir)
        
        # Verify files were created
        assert output_dir.exists()
        assert (output_dir / "documents.json").exists()
        assert (output_dir / "metadata.json").exists()
        
        # Load and verify data integrity
        new_store = VectorStore()
        new_store.load(output_dir)
        
        assert new_store.documents == store.documents
        assert len(new_store.documents) == len(chunks)

class TestScalabilityIntegration:
    """Test scalability integration with GenAI"""
    
    def test_large_document_processing(self):
        """Test large document processing"""
        processor = DocumentProcessor()
        
        # Create large document
        large_document = "This is a test sentence. " * 1000  # ~30,000 characters
        
        chunks = processor._semantic_chunking(large_document, chunk_size=200, overlap=50)
        
        assert len(chunks) > 0
        assert all(len(chunk) <= 200 for chunk in chunks)
    
    @patch('vector_store.SentenceTransformer')
    def test_multiple_queries(self, mock_transformer, sample_document):
        """Test multiple queries performance"""
        mock_transformer_instance = Mock()
        mock_transformer_instance.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        mock_transformer.return_value = mock_transformer_instance
        
        store = VectorStore()
        processor = DocumentProcessor()
        
        # Process document
        chunks = processor._semantic_chunking(sample_document, chunk_size=150, overlap=30)
        documents = [chunk for chunk in chunks]
        store.build_index(documents)
        
        # Test multiple queries
        queries = [
            "What is Procyon?",
            "Explain the architecture",
            "What are the features?",
            "How does data processing work?",
            "What are the use cases?"
        ]
        
        for query in queries:
            results = store.search(query, k=3)
            assert len(results) <= 3
            assert all(isinstance(result, dict) for result in results)

def test_complete_system_integration():
    """Test complete system integration with GenAI"""
    # Test that all components can work together
    from genai_model_converter import GenAIModelConverter
    from document_processor import DocumentProcessor
    from vector_store import VectorStore
    from genai_pipeline import GenAIRAGEngine
    from performance_monitor import PerformanceMonitor
    
    # Test component initialization
    converter = GenAIModelConverter("meta-llama/Llama-3.1-8B-Instruct")
    processor = DocumentProcessor()
    store = VectorStore()
    monitor = PerformanceMonitor()
    
    assert converter is not None
    assert processor is not None
    assert store is not None
    assert monitor is not None
    
    # Test that components can be used together
    sample_text = "Procyon is a data processing platform."
    chunks = processor._semantic_chunking(sample_text, chunk_size=50, overlap=10)
    
    assert len(chunks) > 0
    assert all(len(chunk) <= 50 for chunk in chunks) 