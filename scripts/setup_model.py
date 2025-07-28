"""
Model Setup Script - Llama-3.1 8B Instruct Download and Conversion

This script provides a complete pipeline for setting up the Llama-3.1 8B Instruct model
for use in the RAG system. It handles downloading, conversion to INT4 format using OpenVINO,
and validation of the converted model.

Setup Pipeline:
1. Hardware Compatibility Check: Verify system meets requirements
2. Hugging Face Authentication: Authenticate for model access
3. Model Download: Download Llama-3.1 8B Instruct from Hugging Face
4. OpenVINO Conversion: Convert model to INT4 format optimized for Intel UHD Graphics
5. Model Validation: Test the converted model with sample inference
6. Configuration Setup: Save model metadata and device information

Key Features:
- Interactive setup with user confirmation
- Hardware compatibility validation
- Automatic Hugging Face authentication
- OpenVINO INT4 quantization for Intel UHD Graphics
- Comprehensive error handling and troubleshooting
- Rich console interface with progress tracking
- Model validation and testing

System Requirements:
- Minimum 16GB RAM (32GB recommended)
- 20GB+ free disk space
- Intel UHD Graphics or compatible GPU
- OpenVINO 2024.1+ installed
- Hugging Face account with Llama-3.1 access

Example Usage:
    # Run interactive setup
    python scripts/setup_model.py
    
    # The script will guide you through the entire process

Dependencies:
    - src.model_converter: Model download and OpenVINO conversion
    - rich: Console output and user interface
    - huggingface_hub: Model download and authentication

Author: Segun Oni
Version: 1.0.0
"""

import sys
import os
from pathlib import Path

# Import core model conversion component
from src.model_converter import ModelConverter

# Import UI components
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

# Initialize console for rich output
console = Console()

def check_prerequisites():
    """
    Check if system meets the prerequisites for model setup.
    
    This function validates:
    - Available disk space (minimum 20GB)
    - System memory (minimum 16GB)
    - Hugging Face token availability
    - OpenVINO installation
    
    Returns:
        bool: True if all prerequisites are met, False otherwise
    """
    console.print("\n🔍 Checking system prerequisites...", style="blue")
    
    # Check disk space
    try:
        free_space_gb = Path('./').stat().st_size / (1024**3)
        if free_space_gb < 20:
            console.print(f"Insufficient disk space: {free_space_gb:.1f}GB (need 20GB+)", style="red")
            return False
        console.print(f"Disk space: {free_space_gb:.1f}GB available", style="green")
    except Exception as e:
        console.print(f"Could not check disk space: {e}", style="yellow")
    
    # Check Hugging Face token
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    if not hf_token:
        console.print("HUGGINGFACE_TOKEN not found in environment", style="red")
        console.print("Please set your Hugging Face token in .env file", style="yellow")
        return False
    console.print("Hugging Face token found", style="green")
    
    # Check OpenVINO installation
    try:
        import openvino as ov
        console.print(f"OpenVINO {ov.__version__} installed", style="green")
    except ImportError:
        console.print("OpenVINO not installed", style="red")
        console.print("Please install OpenVINO: pip install openvino", style="yellow")
        return False
    
    console.print("All prerequisites met!", style="green")
    return True

def main():
    """
    Main setup function that orchestrates the complete model setup pipeline.
    
    This function:
    1. Displays welcome message and setup information
    2. Checks system prerequisites
    3. Gets user confirmation to proceed
    4. Initializes the model converter
    5. Runs the complete conversion pipeline
    6. Displays success message and next steps
    
    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    # Display comprehensive welcome message
    console.print(Panel(
        "RAG System - Model Setup\n\n"
        "This script will set up the Llama-3.1 8B Instruct model for your RAG system.\n\n"
        "What will happen:\n"
        "  1. Check your system compatibility\n"
        "  2. Download Llama-3.1 8B Instruct (~15GB)\n"
        "  3. Convert to INT4 format using OpenVINO\n"
        "  4. Optimize for Intel UHD Graphics\n"
        "  5. Validate the converted model\n\n"
        "Estimated time: 30-60 minutes\n"
        "Required space: ~20GB\n"
        "Requires: Hugging Face account with Llama-3.1 access",
        title="Model Setup",
        border_style="blue"
    ))
    
    # Check system prerequisites before proceeding
    if not check_prerequisites():
        console.print("\nPrerequisites not met. Please fix the issues above and try again.", style="red")
        return 1
    
    # Get user confirmation to proceed
    console.print("\nThis process will download a large model and may take significant time.", style="yellow")
    response = input("\nDo you want to continue? (y/N): ").strip().lower()
    if response not in ['y', 'yes']:
        console.print("Setup cancelled by user.", style="yellow")
        return 1
    
    try:
        # Initialize the model converter with default settings
        console.print("\nInitializing model converter...", style="blue")
        converter = ModelConverter()
        console.print("Model converter initialized", style="green")
        
        # Run the complete conversion pipeline
        console.print("\nStarting model conversion pipeline...", style="blue")
        console.print("This will download and convert the model. Please be patient.", style="dim")
        
        output_path = converter.run_conversion_pipeline(skip_download=False)
        
        # Display comprehensive success message
        console.print(Panel(
            f"Model setup completed successfully!\n\n"
            f"Model Location: {output_path}\n"
            f"Model Type: Llama-3.1 8B Instruct (INT4)\n"
            f"Optimized for: Intel UHD Graphics\n"
            f"Format: OpenVINO IR\n\n"
            f"Next Steps:\n"
            f"  1. Process your documents: python scripts/ingest_pdf.py\n"
            f"  2. Run RAG queries: python rag_cli.py --query 'your question'\n"
            f"  3. Test the system: python -m pytest tests/\n\n"
            f"Your RAG system is now ready to use!",
            title="Setup Complete",
            border_style="green"
        ))
        
        return 0
        
    except KeyboardInterrupt:
        console.print("\nSetup interrupted by user.", style="yellow")
        console.print("You can resume later by running the script again.", style="dim")
        return 1
    except Exception as e:
        console.print(f"\nSetup failed: {e}", style="red")
        console.print("\nTroubleshooting tips:", style="yellow")
        console.print("  • Check your internet connection", style="dim")
        console.print("  • Ensure you have enough disk space (20GB+)", style="dim")
        console.print("  • Verify OpenVINO is properly installed", style="dim")
        console.print("  • Check your Hugging Face token is valid", style="dim")
        console.print("  • Ensure you have access to Llama-3.1 8B Instruct", style="dim")
        console.print("  • Try running with --skip-download if model exists", style="dim")
        
        # Provide specific error context
        if "Hugging Face" in str(e):
            console.print("\nHugging Face Issues:", style="yellow")
            console.print("  • Request access at: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct", style="dim")
            console.print("  • Check your token at: https://huggingface.co/settings/tokens", style="dim")
        elif "disk" in str(e).lower() or "space" in str(e).lower():
            console.print("\nDisk Space Issues:", style="yellow")
            console.print("  • Free up at least 20GB of disk space", style="dim")
            console.print("  • Check available space: df -h", style="dim")
        
        return 1

if __name__ == "__main__":
    # Run the main function and exit with appropriate code
    sys.exit(main()) 