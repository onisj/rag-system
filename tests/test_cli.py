"""
Test suite for CLI Interface with OpenVINO GenAI

This module tests the command-line interface functionality:
- CLI argument parsing
- Command execution
- Interactive mode
- Error handling
- Output formatting
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch
import pytest

# Add src to path
current_dir = Path(__file__).parent
project_root = current_dir.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from rich.console import Console
import click
from click.testing import CliRunner
import cli

console = Console()

@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def sample_pdf():
    """Create a sample PDF file for testing"""
    temp_dir = tempfile.mkdtemp()
    pdf_path = os.path.join(temp_dir, "test.pdf")
    
    # Create a simple text file that can be used for testing
    with open(pdf_path, 'w') as f:
        f.write("This is a test PDF content for testing purposes.")
    
    yield pdf_path
    
    shutil.rmtree(temp_dir)

class TestCLI:
    """Test cases for CLI interface with GenAI"""
    
    def test_cli_initialization(self):
        """Test CLI initialization"""
        runner = CliRunner()
        assert runner is not None
    
    def test_help_command(self):
        """Test help command"""
        runner = CliRunner()
        result = runner.invoke(cli.cli, ['--help'])
        assert result.exit_code == 0
        assert "RAG CLI" in result.output
        assert "OpenVINO GenAI" in result.output
    
    def test_query_command_structure(self):
        """Test query command structure"""
        runner = CliRunner()
        # Test that query can be passed as an option
        result = runner.invoke(cli.cli, ['--query', 'test query', '--help'])
        assert result.exit_code == 0
    
    def test_setup_command_structure(self):
        """Test setup command structure"""
        runner = CliRunner()
        result = runner.invoke(cli.cli, ['setup', '--help'])
        assert result.exit_code == 0
        assert "setup" in result.output
    
    def test_convert_model_command_structure(self):
        """Test convert-model command structure"""
        runner = CliRunner()
        result = runner.invoke(cli.cli, ['convert-model', '--help'])
        assert result.exit_code == 0
        assert "convert-model" in result.output
    
    def test_hardware_command_structure(self):
        """Test hardware command structure"""
        runner = CliRunner()
        result = runner.invoke(cli.cli, ['hardware', '--help'])
        assert result.exit_code == 0
        assert "hardware" in result.output
    
    def test_performance_command_structure(self):
        """Test performance command structure"""
        runner = CliRunner()
        result = runner.invoke(cli.cli, ['performance', '--help'])
        assert result.exit_code == 0
        assert "performance" in result.output
    
    def test_demo_command_structure(self):
        """Test demo command structure"""
        runner = CliRunner()
        result = runner.invoke(cli.cli, ['demo', '--help'])
        assert result.exit_code == 0
        assert "demo" in result.output
    
    def test_cli_version(self):
        """Test CLI version"""
        runner = CliRunner()
        result = runner.invoke(cli.cli, ['--version'])
        assert result.exit_code == 0
        assert "1.0.0" in result.output
    
    def test_invalid_command(self):
        """Test invalid command handling"""
        runner = CliRunner()
        result = runner.invoke(cli.cli, ['invalid-command'])
        assert result.exit_code != 0
    
    def test_query_parameters(self):
        """Test query parameters"""
        runner = CliRunner()
        result = runner.invoke(cli.cli, [
            '--model-path', './models/test.xml',
            '--vector-store-path', './data/test',
            '--device', 'CPU',
            '--k', '3',
            '--max-tokens', '256',
            '--temperature', '0.7',
            '--query', 'What is Procyon?'
        ])
        # Should fail because model doesn't exist, but parameters should be parsed
        assert result.exit_code != 0
    
    def test_setup_parameters(self):
        """Test setup parameters"""
        runner = CliRunner()
        result = runner.invoke(cli.cli, [
            'setup',
            '--pdf-path', './data/test.pdf',
            '--chunk-size', '512',
            '--chunk-overlap', '50',
            '--output-dir', './data/processed'
        ])
        # Should fail because PDF doesn't exist, but parameters should be parsed
        assert result.exit_code != 0
    
    def test_convert_model_parameters(self):
        """Test convert-model parameters"""
        runner = CliRunner()
        result = runner.invoke(cli.cli, [
            'convert-model',
            '--model-name', 'meta-llama/Llama-3.1-8B-Instruct',
            '--skip-download'
        ])
        # Should fail because model conversion requires actual setup, but parameters should be parsed
        assert result.exit_code != 0

def test_cli_integration():
    """Test CLI integration with GenAI components"""
    # Test that CLI can import GenAI components
    try:
        from genai_pipeline import GenAIRAGEngine
        from genai_model_converter import GenAIModelConverter
        assert GenAIRAGEngine is not None
        assert GenAIModelConverter is not None
    except ImportError as e:
        pytest.skip(f"GenAI components not available: {e}")
    
    # Test that CLI can be invoked
    runner = CliRunner()
    result = runner.invoke(cli.cli, ['--help'])
    assert result.exit_code == 0

def test_cli_error_handling():
    """Test CLI error handling"""
    runner = CliRunner()
    
    # Test with non-existent model path
    result = runner.invoke(cli.cli, [
        '--model-path', './models/nonexistent.xml',
        '--query', 'test query'
    ])
    assert result.exit_code != 0
    
    # Test with non-existent vector store path
    result = runner.invoke(cli.cli, [
        '--vector-store-path', './data/nonexistent',
        '--query', 'test query'
    ])
    assert result.exit_code != 0

def test_cli_performance_monitoring():
    """Test CLI performance monitoring integration"""
    runner = CliRunner()
    
    # Test performance command
    result = runner.invoke(cli.cli, ['performance', '--help'])
    assert result.exit_code == 0
    
    # Test performance with benchmark
    result = runner.invoke(cli.cli, ['performance', '--benchmark', '--help'])
    assert result.exit_code == 0

def test_cli_hardware_detection():
    """Test CLI hardware detection"""
    runner = CliRunner()
    
    # Test hardware command
    result = runner.invoke(cli.cli, ['hardware', '--help'])
    assert result.exit_code == 0
    
    # Test hardware with detailed flag
    result = runner.invoke(cli.cli, ['hardware', '--detailed', '--help'])
    assert result.exit_code == 0

def test_cli_demo_functionality():
    """Test CLI demo functionality"""
    runner = CliRunner()
    
    # Test demo command
    result = runner.invoke(cli.cli, ['demo', '--help'])
    assert result.exit_code == 0
    
    # Test demo execution (should fail without proper setup)
    result = runner.invoke(cli.cli, ['demo'])
    assert result.exit_code != 0  # Should fail without model and vector store

def test_cli_streaming_options():
    """Test CLI streaming options"""
    runner = CliRunner()
    
    # Test with streaming enabled (default)
    result = runner.invoke(cli.cli, [
        '--query', 'test query',
        '--model-path', './models/test.xml'
    ])
    assert result.exit_code != 0  # Should fail without model
    
    # Test with streaming disabled
    result = runner.invoke(cli.cli, [
        '--query', 'test query',
        '--no-stream',
        '--model-path', './models/test.xml'
    ])
    assert result.exit_code != 0  # Should fail without model

def test_cli_device_selection():
    """Test CLI device selection"""
    runner = CliRunner()
    
    # Test CPU device
    result = runner.invoke(cli.cli, [
        '--device', 'CPU',
        '--query', 'test query',
        '--model-path', './models/test.xml'
    ])
    assert result.exit_code != 0  # Should fail without model
    
    # Test GPU device
    result = runner.invoke(cli.cli, [
        '--device', 'GPU',
        '--query', 'test query',
        '--model-path', './models/test.xml'
    ])
    assert result.exit_code != 0  # Should fail without model
    
    # Test AUTO device
    result = runner.invoke(cli.cli, [
        '--device', 'AUTO',
        '--query', 'test query',
        '--model-path', './models/test.xml'
    ])
    assert result.exit_code != 0  # Should fail without model 