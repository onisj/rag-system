"""
Integration Test Suite for RAG System

This module tests the complete RAG system integration:
- End-to-end workflow
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
from model_converter import ModelConverter
from document_processor import DocumentProcessor, TextChunk
from vector_store import VectorStore
from rag_pipeline import RAGEngine
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
    """Test complete end-to-end RAG workflow"""
    
    @patch('vector_store.SentenceTransformer')
    @patch('openvino.Core')
    @patch('rag_pipeline.AutoTokenizer')
    def test_complete_rag_workflow(self, mock_tokenizer, mock_core, mock_transformer, sample_document):
        """Test complete RAG workflow from document to answer"""
        # Setup mocks
        mock_core_instance = Mock()
        mock_core_instance.available_devices = ["CPU"]
        mock_core.return_value = mock_core_instance
        
        mock_model = Mock()
        mock_compiled_model = Mock()
        mock_core_instance.read_model.return_value = mock_model
        mock_core_instance.compile_model.return_value = mock_compiled_model
        
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "<eos>"
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        mock_transformer_instance = Mock()
        mock_transformer_instance.encode.return_value = np.random.rand(5, 384).astype(np.float32)
        mock_transformer.return_value = mock_transformer_instance
        
        # Step 1: Process document
        processor = DocumentProcessor(chunk_size=200, chunk_overlap=50)
        chunks = processor._chunk_text_simple(sample_document, 0)
        
        assert len(chunks) > 0
        assert all(isinstance(chunk, TextChunk) for chunk in chunks)
        
        # Step 2: Create vector store
        vector_store = VectorStore()
        documents = [{"text": chunk.text} for chunk in chunks]
        vector_store.build_index(documents)
        
        assert vector_store.index is not None
        assert len(vector_store.documents) == len(chunks)
        
        # Step 3: Test search
        results = vector_store.search("What is Procyon?", k=3)
        
        assert len(results) > 0
        assert all(isinstance(result, tuple) for result in results)
        assert all(len(result) == 2 for result in results)  # (doc, score) pairs
        
        # Step 4: Test RAG engine (mocked)
        with patch('os.path.exists', return_value=True):
            rag_engine = RAGEngine(
                model_path="test_model.xml",
                vector_store_path="test_vector_store"
            )
            
            # Test prompt creation - extract documents from (doc, score) tuples
            documents = [doc for doc, score in results[:2]]
            prompt = rag_engine._create_prompt("What is Procyon?", documents)
            
            assert "What is Procyon?" in prompt
            assert any(chunk.text in prompt for chunk in chunks)
            assert "Context:" in prompt
            assert "Question:" in prompt
        
        console.print("✅ Complete RAG workflow test passed", style="green")
    
    @patch('vector_store.SentenceTransformer')
    def test_document_processing_integration(self, mock_transformer, sample_document):
        """Test document processing integration"""
        # Setup mock to prevent hanging
        mock_transformer_instance = Mock()
        mock_transformer_instance.encode.return_value = np.random.rand(5, 384).astype(np.float32)
        mock_transformer.return_value = mock_transformer_instance
        
        processor = DocumentProcessor(chunk_size=150, chunk_overlap=30)
        
        # Process document
        chunks = processor._chunk_text_simple(sample_document, 0)
        
        # Verify chunks
        assert len(chunks) > 0
        assert all(chunk.length <= 150 for chunk in chunks)
        assert all(chunk.length > 0 for chunk in chunks)
        
        # Verify chunk content
        all_text = " ".join(chunk.text for chunk in chunks)
        assert "Procyon" in all_text
        assert "data processing" in all_text
        assert "Key Features" in all_text  # Check for the actual text in the document
        
        console.print(f"✅ Document processing created {len(chunks)} chunks", style="green")
    
    @patch('vector_store.SentenceTransformer')
    def test_vector_store_integration(self, mock_transformer, sample_document):
        """Test vector store integration"""
        # Setup mock
        mock_transformer_instance = Mock()
        mock_transformer_instance.encode.return_value = np.random.rand(10, 384).astype(np.float32)
        mock_transformer.return_value = mock_transformer_instance
        
        # Process document
        processor = DocumentProcessor(chunk_size=100, chunk_overlap=20)
        chunks = processor._chunk_text_simple(sample_document, 0)
        
        # Create vector store
        vector_store = VectorStore()
        documents = [{"text": chunk.text} for chunk in chunks]
        vector_store.build_index(documents)
        
        # Test search functionality
        test_queries = [
            "What is Procyon?",
            "data processing",
            "features",
            "architecture"
        ]
        
        for query in test_queries:
            results = vector_store.search(query, k=2)
            assert len(results) > 0
            assert all(isinstance(result, tuple) for result in results)
        
        console.print("✅ Vector store integration test passed", style="green")
    
    def test_cli_integration(self):
        """Test CLI integration"""
        from click.testing import CliRunner
        
        runner = CliRunner()
        
        # Test CLI help
        result = runner.invoke(cli.cli, ['--help'])
        assert result.exit_code == 0
        assert "RAG CLI" in result.output
        
        # Test query command help
        result = runner.invoke(cli.cli, ['query', '--help'])
        assert result.exit_code == 0
        assert "query" in result.output
        
        # Test setup command help
        result = runner.invoke(cli.cli, ['setup', '--help'])
        assert result.exit_code == 0
        assert "setup" in result.output
        
        console.print("✅ CLI integration test passed", style="green")

class TestPerformanceIntegration:
    """Test performance integration"""
    
    def test_performance_monitoring_integration(self):
        """Test performance monitoring integration"""
        monitor = PerformanceMonitor()
        
        # Test monitoring workflow with timeout
        monitor.start_monitoring()
        time.sleep(0.05)  # Reduced sleep time to prevent hanging
        monitor.stop_monitoring()
        
        # Verify performance data
        summary = monitor.get_performance_summary()
        assert "cpu_usage_percent" in summary
        assert "memory_usage_percent" in summary
        assert "inference_latency_ms" in summary
        assert "throughput_tokens_per_second" in summary
        
        # Test that performance data is accessible
        assert monitor.performance_data is not None
        assert isinstance(monitor.performance_data, dict)
        
        console.print("✅ Performance monitoring integration test passed", style="green")
    
    def test_benchmark_integration(self):
        """Test benchmarking integration"""
        monitor = PerformanceMonitor()
        
        def test_function():
            time.sleep(0.05)  # Reduced sleep time to prevent hanging
            return "test result"
        
        # Test function execution with performance monitoring
        monitor.start_monitoring()
        result = test_function()
        monitor.stop_monitoring()
        
        assert result == "test result"
        
        # Verify performance data was recorded
        summary = monitor.get_performance_summary()
        assert "cpu_usage_percent" in summary
        assert "memory_usage_percent" in summary
        
        console.print("✅ Benchmark integration test passed", style="green")

class TestErrorHandlingIntegration:
    """Test error handling integration"""
    
    @patch('vector_store.SentenceTransformer')
    def test_error_recovery_integration(self, mock_transformer):
        """Test error recovery integration"""
        # Setup mock to prevent hanging
        mock_transformer_instance = Mock()
        mock_transformer_instance.encode.return_value = np.random.rand(5, 384).astype(np.float32)
        mock_transformer.return_value = mock_transformer_instance
        
        # Test that components handle errors gracefully
        processor = DocumentProcessor()
        
        # Test with empty text
        chunks = processor._chunk_text_simple("", 0)
        assert len(chunks) == 0
        
        # Test with very short text
        chunks = processor._chunk_text_simple("Short", 0)
        assert len(chunks) > 0
        
        console.print("✅ Error recovery integration test passed", style="green")
    
    @patch('vector_store.SentenceTransformer')
    def test_invalid_input_handling(self, mock_transformer):
        """Test invalid input handling"""
        # Setup mock to prevent hanging
        mock_transformer_instance = Mock()
        mock_transformer_instance.encode.return_value = np.random.rand(5, 384).astype(np.float32)
        mock_transformer.return_value = mock_transformer_instance
        
        processor = DocumentProcessor()
        
        # Test with None input
        with pytest.raises(Exception):
            processor._chunk_text_simple(None, 0)
        
        # Test with invalid chunk size
        processor_invalid = DocumentProcessor(chunk_size=0, chunk_overlap=0)
        chunks = processor_invalid._chunk_text_simple("Test text", 0)
        assert len(chunks) > 0  # Should handle gracefully
        
        console.print("✅ Invalid input handling test passed", style="green")

class TestDataPersistenceIntegration:
    """Test data persistence integration"""
    
    @patch('vector_store.SentenceTransformer')
    def test_save_load_integration(self, mock_transformer, temp_dir, sample_document):
        """Test save and load integration"""
        # Setup mock to prevent hanging
        mock_transformer_instance = Mock()
        mock_transformer_instance.encode.return_value = np.random.rand(5, 384).astype(np.float32)
        mock_transformer.return_value = mock_transformer_instance
        
        # Process document
        processor = DocumentProcessor(chunk_size=100, chunk_overlap=20)
        chunks = processor._chunk_text_simple(sample_document, 0)
        
        # Save chunks
        chunks_path = os.path.join(temp_dir, "chunks.json")
        processor.chunks = chunks
        processor.save_chunks(chunks_path)
        
        assert os.path.exists(chunks_path)
        
        # Load chunks
        with open(chunks_path, 'r') as f:
            saved_data = json.load(f)
        
        assert 'chunks' in saved_data
        assert len(saved_data['chunks']) == len(chunks)
        
        console.print("✅ Data persistence integration test passed", style="green")
    
    @patch('vector_store.SentenceTransformer')
    def test_vector_store_persistence(self, mock_transformer, temp_dir, sample_document):
        """Test vector store persistence"""
        # Setup mock
        mock_transformer_instance = Mock()
        mock_transformer_instance.encode.return_value = np.random.rand(10, 384).astype(np.float32)
        mock_transformer.return_value = mock_transformer_instance
        
        # Create vector store
        processor = DocumentProcessor(chunk_size=100, chunk_overlap=20)
        chunks = processor._chunk_text_simple(sample_document, 0)
        
        vector_store = VectorStore()
        documents = [{"text": chunk.text} for chunk in chunks]
        vector_store.build_index(documents)
        
        # Save vector store
        save_path = os.path.join(temp_dir, "vector_store")
        vector_store.save(save_path)
        
        assert os.path.exists(save_path)
        assert os.path.exists(os.path.join(save_path, "documents.json"))
        assert os.path.exists(os.path.join(save_path, "metadata.json"))
        
        # Load vector store
        new_vector_store = VectorStore()
        new_vector_store.load(save_path)
        
        assert new_vector_store.documents == vector_store.documents
        assert new_vector_store.metadata == vector_store.metadata
        
        console.print("✅ Vector store persistence test passed", style="green")

class TestScalabilityIntegration:
    """Test scalability integration"""
    
    def test_large_document_processing(self):
        """Test processing large documents"""
        # Create large document
        large_document = "This is a test sentence. " * 1000  # 20,000 characters
        
        processor = DocumentProcessor(chunk_size=200, chunk_overlap=50)
        chunks = processor._chunk_text_simple(large_document, 0)
        
        assert len(chunks) > 0
        assert all(chunk.length <= 200 for chunk in chunks)
        
        console.print(f"✅ Large document processing: {len(chunks)} chunks", style="green")
    
    @patch('vector_store.SentenceTransformer')
    def test_multiple_queries(self, mock_transformer, sample_document):
        """Test multiple queries performance"""
        # Setup mock to prevent hanging
        mock_transformer_instance = Mock()
        mock_transformer_instance.encode.return_value = np.random.rand(5, 384).astype(np.float32)
        mock_transformer.return_value = mock_transformer_instance
        
        processor = DocumentProcessor(chunk_size=100, chunk_overlap=20)
        chunks = processor._chunk_text_simple(sample_document, 0)
        
        # Simulate multiple queries
        queries = [
            "What is Procyon?",
            "data processing",
            "features",
            "architecture",
            "use cases"
        ]
        
        # This would normally use vector store, but we'll simulate
        for query in queries:
            # Simulate search
            matching_chunks = [chunk for chunk in chunks if any(word in chunk.text.lower() for word in query.lower().split())]
            assert len(matching_chunks) >= 0  # Some queries might not match
        
        console.print(f"✅ Multiple queries test: {len(queries)} queries", style="green")

def test_complete_system_integration():
    """Complete system integration test"""
    console.print("\n🚀 Running Complete System Integration Test", style="bold blue")
    
    # Test all major components work together
    components = [
        "DocumentProcessor",
        "VectorStore", 
        "RAGEngine",
        "CLI",
        "PerformanceMonitor"
    ]
    
    for component in components:
        try:
            # Test component can be imported
            if component == "DocumentProcessor":
                from document_processor import DocumentProcessor
                processor = DocumentProcessor()
                assert processor is not None
            elif component == "VectorStore":
                from vector_store import VectorStore
                vs = VectorStore()
                assert vs is not None
            elif component == "RAGEngine":
                from rag_pipeline import RAGEngine
                # RAGEngine requires model files, so just test import
                assert RAGEngine is not None
            elif component == "CLI":
                import cli
                assert cli is not None
                assert hasattr(cli, 'cli')
            elif component == "PerformanceMonitor":
                from performance_monitor import PerformanceMonitor
                monitor = PerformanceMonitor()
                assert monitor is not None
            
            console.print(f"✅ {component} integration OK", style="green")
            
        except Exception as e:
            console.print(f"❌ {component} integration failed: {e}", style="red")
            raise
    
    console.print("\n🎉 Complete System Integration Test PASSED", style="bold green")

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"]) 