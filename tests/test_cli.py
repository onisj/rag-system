"""
Test suite for CLI Interface

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
    """Test cases for CLI interface"""
    
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
    
    def test_query_command_structure(self):
        """Test query command structure"""
        runner = CliRunner()
        result = runner.invoke(cli.cli, ['query', '--help'])
        assert result.exit_code == 0
        assert "query" in result.output
    
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
        assert result.exit_code != 0  # Should fail
    
    def test_query_parameters(self):
        """Test query command parameters"""
        runner = CliRunner()
        result = runner.invoke(cli.cli, ['query', '--help'])
        assert result.exit_code == 0
        assert "--k" in result.output
        assert "--max-tokens" in result.output
        assert "--temperature" in result.output
        assert "--device" in result.output
    
    def test_setup_parameters(self):
        """Test setup command parameters"""
        runner = CliRunner()
        result = runner.invoke(cli.cli, ['setup', '--help'])
        assert result.exit_code == 0
        assert "--pdf-path" in result.output
        assert "--chunk-size" in result.output
        assert "--chunk-overlap" in result.output
    
    def test_convert_model_parameters(self):
        """Test convert-model command parameters"""
        runner = CliRunner()
        result = runner.invoke(cli.cli, ['convert-model', '--help'])
        assert result.exit_code == 0
        assert "--model-name" in result.output
        assert "--skip-download" in result.output

def test_cli_integration():
    """Integration test for CLI functionality"""
    runner = CliRunner()
    
    # Test that all commands are properly registered
    result = runner.invoke(cli.cli, ['--help'])
    assert result.exit_code == 0
    
    # Check for all main commands
    assert 'query' in result.output
    assert 'setup' in result.output
    assert 'convert-model' in result.output
    assert 'hardware' in result.output
    assert 'performance' in result.output
    assert 'demo' in result.output

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"]) 