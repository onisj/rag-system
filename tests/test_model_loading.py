"""
Test Model Loading Script

This script tests the model loading with the rope_scaling fix
"""

import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.model_converter import ModelConverter

console = Console()

def test_model_loading():
    """Test model loading with the fix"""
    console.print(Panel(
        "Testing Model Loading Fix\n\n"
        "This script will test if the rope_scaling configuration fix works.",
        title="Model Loading Test",
        border_style="blue"
    ))
    
    try:
        # Initialize converter
        converter = ModelConverter("meta-llama/Llama-3.1-8B-Instruct")
        
        # Test authentication
        console.print("Testing authentication...", style="blue")
        if not converter.authenticate_huggingface():
                    console.print("Authentication failed", style="red")
        # Authentication failed
        
        # Test model download
        console.print("Testing model download...", style="blue")
        model, tokenizer = converter.download_model(skip_if_exists=True)
        
        console.print("Model loaded successfully!", style="green")
        console.print(f"Model type: {type(model)}", style="dim")
        console.print(f"Tokenizer type: {type(tokenizer)}", style="dim")
        
        # Model loaded successfully
        
    except Exception as e:
        console.print(f"Test failed: {e}", style="red")
        # Test failed

def main():
    """Main function"""
    success = test_model_loading()
    
    if success:
        console.print(Panel(
            "Model loading test passed!\n"
            "The rope_scaling fix is working correctly.",
            title="Test Passed",
            border_style="green"
        ))
        return 0
    else:
        console.print(Panel(
            "Model loading test failed!\n"
            "Please check the error message above.",
            title="Test Failed",
            border_style="red"
        ))
        return 1

if __name__ == "__main__":
    sys.exit(main()) 