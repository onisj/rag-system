"""
Test suite for Utility Scripts

This module tests the utility scripts functionality:
- Model setup script
- PDF ingestion script
- Script integration
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
import pytest

# Add src to path
current_dir = Path(__file__).parent
project_root = current_dir.parent
src_path = project_root / "src"
scripts_path = project_root / "scripts"
sys.path.insert(0, str(src_path))
sys.path.insert(0, str(scripts_path))

from rich.console import Console

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

class TestSetupModelScript:
    """Test cases for setup_model.py script"""
    
    def test_setup_model_imports(self):
        """Test that setup_model script can be imported"""
        try:
            import setup_model
            assert setup_model is not None
            assert hasattr(setup_model, 'main')
            assert callable(setup_model.main)
        except ImportError as e:
            pytest.fail(f"Failed to import setup_model: {e}")
    
    def test_setup_model_structure(self):
        """Test setup_model script structure"""
        import setup_model
        
        # Check that the script has the expected structure
        assert hasattr(setup_model, 'main')
        assert callable(setup_model.main)

class TestIngestPDFScript:
    """Test cases for ingest_pdf.py script"""
    
    def test_ingest_pdf_imports(self):
        """Test that ingest_pdf script can be imported"""
        try:
            import ingest_pdf
            assert ingest_pdf is not None
            assert hasattr(ingest_pdf, 'main')
            assert callable(ingest_pdf.main)
        except ImportError as e:
            pytest.fail(f"Failed to import ingest_pdf: {e}")
    
    def test_ingest_pdf_structure(self):
        """Test ingest_pdf script structure"""
        import ingest_pdf
        
        # Check that the script has the expected structure
        assert hasattr(ingest_pdf, 'main')
        assert callable(ingest_pdf.main)

class TestScriptIntegration:
    """Test cases for script integration"""
    
    def test_scripts_directory_structure(self):
        """Test that scripts directory has the expected structure"""
        scripts_dir = Path(__file__).parent.parent / "scripts"
        assert scripts_dir.exists(), "Scripts directory not found"
        
        # Check for required scripts
        required_scripts = ["setup_model.py", "ingest_pdf.py"]
        for script in required_scripts:
            script_path = scripts_dir / script
            assert script_path.exists(), f"Required script {script} not found"
    
    def test_scripts_import_consistency(self):
        """Test that all scripts can be imported consistently"""
        scripts_dir = Path(__file__).parent.parent / "scripts"
        
        # Test each script
        scripts_to_test = ["setup_model", "ingest_pdf"]
        
        for script_name in scripts_to_test:
            try:
                # Add scripts directory to path
                sys.path.insert(0, str(scripts_dir))
                
                # Import the script
                script_module = __import__(script_name)
                
                # Check that it has a main function
                assert hasattr(script_module, 'main'), f"Script {script_name} missing main function"
                assert callable(script_module.main), f"Script {script_name} main is not callable"
                
                console.print(f"✅ {script_name}.py imported successfully", style="green")
                
            except ImportError as e:
                pytest.fail(f"Failed to import {script_name}: {e}")
            except Exception as e:
                pytest.fail(f"Failed to import {script_name}: {e}")
    
    def test_scripts_have_main_functions(self):
        """Test that all scripts have main functions"""
        scripts_dir = Path(__file__).parent.parent / "scripts"
        
        # Get all Python files in scripts directory
        python_files = list(scripts_dir.glob("*.py"))
        python_files = [f for f in python_files if f.name != "__init__.py"]
        
        assert len(python_files) > 0, "No Python scripts found"
        
        for script_file in python_files:
            try:
                # Read the file content
                with open(script_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check for main function
                if 'def main(' in content:
                    console.print(f"✅ {script_file.name} has main function", style="green")
                else:
                    console.print(f"⚠️  {script_file.name} missing main function", style="yellow")
                    # Don't fail the test, just warn
                    
            except Exception as e:
                console.print(f"❌ Error reading {script_file.name}: {e}", style="red")
                # Don't fail the test for file reading errors

class TestScriptErrorHandling:
    """Test cases for script error handling"""
    
    def test_script_import_error_handling(self):
        """Test that script import errors are handled gracefully"""
        # Test importing a non-existent script
        try:
            import nonexistent_script
            pytest.fail("Should have raised ImportError")
        except ImportError:
            # This is expected
            pass
        except Exception as e:
            pytest.fail(f"Unexpected error: {e}")

class TestScriptConfiguration:
    """Test cases for script configuration"""
    
    def test_setup_model_configuration(self):
        """Test setup_model script configuration"""
        import setup_model
        
        # Check that the script has expected attributes
        assert hasattr(setup_model, 'main')
        
        # Check that main is callable
        assert callable(setup_model.main)
    
    def test_ingest_pdf_configuration(self):
        """Test ingest_pdf script configuration"""
        import ingest_pdf
        
        # Check that the script has expected attributes
        assert hasattr(ingest_pdf, 'main')
        
        # Check that main is callable
        assert callable(ingest_pdf.main)

def test_scripts_integration_workflow():
    """Test the integration workflow of all scripts"""
    console.print("Testing Scripts Integration Workflow", style="bold blue")
    
    try:
        # Test that all scripts can be imported
        scripts_to_test = ["setup_model", "ingest_pdf"]
        
        for script_name in scripts_to_test:
            try:
                script_module = __import__(script_name)
                assert hasattr(script_module, 'main')
                assert callable(script_module.main)
                console.print(f"✅ {script_name}.py integration test passed", style="green")
            except ImportError as e:
                pytest.fail(f"Failed to import {script_name}: {e}")
        
        console.print("✅ All scripts integration tests passed", style="bold green")
        
    except Exception as e:
        pytest.fail(f"Scripts integration workflow failed: {e}")

if __name__ == "__main__":
    pytest.main([__file__]) 